"""Deterministic orchestration: shot list -> Sora clips -> merged MP4 -> blob.

The producer agent decides the creative shot list (anchor lead -> field report ->
interview -> sign-off); this module executes it deterministically.

Consistency: Sora 2 has no usable seed and rejects human faces in input_reference, so the
only reliable lever is REMIX. Each on-camera speaker's FIRST clip is a fresh create; every
later clip of that same speaker is a remix of their first clip (on the same resource that
owns it), which reuses its layout/look. B-roll (face-free) uses last-frame chaining.
"""
import asyncio
import logging

import httpx

from GenBox.azurestorage import upload_video_bytes_to_blob
from GenBox.newsvideo import config, mux, sora_client
from GenBox.newsvideo.producer_agent import produce_shot_list
from GenBox.newsvideo.tracing import traceable

logger = logging.getLogger("GenBoxVideo.pipeline")

# Appended to every speaking clip so the line ends cleanly inside the clip's duration
# instead of being cut off mid-word at the hard cut.
_HOLD_TAIL = (
    "Pace the delivery so the speaker finishes the sentence about half a second before the "
    "clip ends, then stops talking and holds a calm, natural closing expression — do not cut "
    "off a word and do not start another sentence."
)


def _talking_head_prompt(shot: dict) -> str:
    """Prompt for the FIRST clip of an on-camera speaker. The studio anchor uses the fixed
    ANCHOR_BIBLE; reporters/interviewees use their own per-speaker description."""
    if shot["type"] == "anchor":
        base = config.ANCHOR_BIBLE
    else:
        desc = shot.get("speaker_description") or "A person speaking to camera at a relevant location."
        base = (
            f"{desc} On-location report for the AI World Government's news channel, eye-level "
            f"medium shot; the speaker talks directly to camera with natural, accurate lip-sync."
        )
    return (
        f"{base}\n"
        f"Shot: {shot['visual']}\n"
        f"Spoken dialogue (say exactly this, nothing else):\n"
        f"\"{shot['dialogue']}\"\n"
        f"{_HOLD_TAIL}\n"
        f"{config.BROLL_NEGATIVE}"
    )


def _remix_prompt(shot: dict) -> str:
    """Prompt for a LATER clip of a speaker, remixed from their first clip. Describe only
    the change (new line / action); remix preserves the source's look."""
    who = (
        "the same news anchor at the same GENBOX NEWS desk"
        if shot["type"] == "anchor"
        else "the same person in the same setting as the source video"
    )
    return (
        f"Keep {who} — identical face, hair, wardrobe, lighting and framing as the source "
        f"video; change only what is described next.\n"
        f"Shot: {shot['visual']}\n"
        f"Spoken dialogue (say exactly this, nothing else):\n"
        f"\"{shot['dialogue']}\"\n"
        f"{_HOLD_TAIL}\n"
        f"{config.BROLL_NEGATIVE}"
    )


def _broll_prompt(shot: dict) -> str:
    if shot.get("dialogue"):
        voiceover = (
            f"Calm voiceover: \"{shot['dialogue']}\" — finish the line about half a second "
            f"before the clip ends, then let ambient sound carry to the cut."
        )
    else:
        voiceover = "Ambient sound only, no narration."
    return (
        f"Broadcast b-roll for the AI World Government's news channel, documentary style. "
        f"{shot['visual']}.\n"
        f"{voiceover}\n"
        f"No people speaking on camera, no human faces in close-up. "
        f"{config.BROLL_NEGATIVE}"
    )


def _format_http_error(exc: httpx.HTTPStatusError) -> str:
    """Build an actionable message from an Azure error response (status + body)."""
    resp = exc.response
    if resp is None:
        return str(exc)
    body = ""
    try:
        body = resp.text or ""
    except Exception:
        body = ""
    req = resp.request
    where = f"{req.method} {req.url}" if req is not None else ""
    return f"HTTP {resp.status_code} {resp.reason_phrase} ({where}): {body[:800]}"


async def _safe(factory, label: str):
    """Run an async clip factory with one retry. Returns (result_or_None, error_detail).
    Surfaces the real HTTP status + Azure body instead of a generic message."""
    last_detail = None
    for attempt in range(2):
        try:
            return await factory(), None
        except httpx.HTTPStatusError as exc:
            last_detail = _format_http_error(exc)
            logger.warning("%s: HTTP error (attempt %d): %s", label, attempt + 1, last_detail)
        except Exception as exc:  # network/timeout/etc.
            last_detail = f"{type(exc).__name__}: {exc}"
            logger.warning("%s: error (attempt %d): %s", label, attempt + 1, last_detail)
    logger.error("%s: failed after retries: %s", label, last_detail)
    return None, last_detail


def _speaker_key(shot: dict) -> str:
    return (shot.get("speaker") or shot["type"]).lower()


@traceable(run_type="chain", name="GenBox Build Video")
async def build_news_video(decision_text: str, flat_date: str):
    """Produce one merged MP4 for this date, upload to blob. Returns (blob_url, clip_count)."""
    shot_list = await produce_shot_list(decision_text)
    shots = shot_list.get("shots", [])[:config.MAX_CLIPS]
    logger.info("build_news_video %s: %d shot(s) — %s", flat_date, len(shots), shot_list.get("title"))

    clips = []
    last_error = None
    prev_last_frame = None             # carried across consecutive frame-chained b-roll shots
    bases = {}                          # speaker_key -> (resource, base_job_id) for remixing
    for idx, shot in enumerate(shots):
        stype = shot["type"]
        seconds = shot["seconds"]
        label = f"clip{idx}:{stype}"

        # ---- b-roll: fresh create, face-free, optional frame chaining ----
        if stype == "broll":
            resource = sora_client.next_resource()
            ref = prev_last_frame
            res, err = await _safe(
                lambda r=resource, rf=ref: sora_client.create_clip(r, _broll_prompt(shot), seconds, ref_image_bytes=rf),
                label,
            )
            if res is None and ref is not None:
                logger.warning("%s: retrying b-roll without reference frame", label)
                res, err = await _safe(
                    lambda r=resource: sora_client.create_clip(r, _broll_prompt(shot), seconds),
                    label,
                )
            if res is None:
                last_error = err or last_error
                prev_last_frame = None
                continue
            data, _jid = res
            clips.append(data)
            if shot.get("frame_chain"):
                try:
                    prev_last_frame = await asyncio.to_thread(mux.extract_last_frame, data)
                except Exception as exc:
                    logger.warning("%s: last-frame extraction failed: %s", label, exc)
                    prev_last_frame = None
            else:
                prev_last_frame = None
            continue

        # ---- talking head (anchor / reporter / interview) ----
        prev_last_frame = None
        key = _speaker_key(shot)
        if key in bases:
            # Subsequent clip of a known speaker -> remix their first clip for consistency.
            resource, base_id = bases[key]
            res, err = await _safe(
                lambda r=resource, b=base_id: sora_client.remix_clip(r, b, _remix_prompt(shot), seconds),
                f"{label}:remix",
            )
            if res is None:
                logger.warning("%s: remix failed (%s); falling back to a fresh create", label, err)
                res, err = await _safe(
                    lambda r=resource: sora_client.create_clip(r, _talking_head_prompt(shot), seconds),
                    f"{label}:create-fallback",
                )
        else:
            # First clip of this speaker -> fresh create; remember it as the remix base.
            resource = sora_client.next_resource()
            res, err = await _safe(
                lambda r=resource: sora_client.create_clip(r, _talking_head_prompt(shot), seconds),
                f"{label}:create",
            )

        if res is None:
            last_error = err or last_error
            if stype == "anchor" and not clips:
                # The opening anchor lead is required; abort so we don't ship a headless segment.
                raise RuntimeError(f"Anchor lead clip failed for {flat_date}: {err}")
            continue

        data, jid = res
        clips.append(data)
        if key not in bases:
            bases[key] = (resource, jid)

    if not clips:
        raise RuntimeError(f"No clips generated for {flat_date}: {last_error}")

    merged = await asyncio.to_thread(mux.concat_clips, clips)
    url = await asyncio.to_thread(upload_video_bytes_to_blob, merged, flat_date)
    logger.info("build_news_video %s: merged %d clip(s) -> %s", flat_date, len(clips), url)
    return url, len(clips)
