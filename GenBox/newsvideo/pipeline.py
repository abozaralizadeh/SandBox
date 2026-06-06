"""Deterministic orchestration: shot list -> Sora clips -> merged MP4 -> blob.

The producer agent decides the creative shot list (anchor lead -> field report ->
interview -> sign-off); this module executes it with hard caps, single retries, stable
per-speaker seeds, and face-free frame chaining for b-roll.
"""
import asyncio
import hashlib
import logging

import httpx

from GenBox.azurestorage import upload_video_bytes_to_blob
from GenBox.newsvideo import config, mux
from GenBox.newsvideo.producer_agent import produce_shot_list
from GenBox.newsvideo.sora_client import generate_clip

logger = logging.getLogger("GenBoxVideo.pipeline")


def _talking_head_prompt(shot: dict) -> str:
    """Prompt for an on-camera speaker. The studio anchor uses the fixed ANCHOR_BIBLE;
    reporters/interviewees use their own per-speaker description."""
    if shot["type"] == "anchor":
        base = config.ANCHOR_BIBLE
    else:
        desc = shot.get("speaker_description") or "A person speaking to camera at a relevant location."
        base = (
            f"{desc} Broadcast news footage, eye-level medium shot; the speaker talks "
            f"directly to camera with natural, accurate lip-sync."
        )
    return (
        f"{base}\n"
        f"Shot: {shot['visual']}\n"
        f"Spoken dialogue (say exactly this, nothing else):\n"
        f"\"{shot['dialogue']}\"\n"
        f"{config.BROLL_NEGATIVE}"
    )


def _seed_for(shot: dict) -> int:
    """A stable seed per speaker so a given person looks consistent across their clips.
    The anchor uses the fixed ANCHOR_SEED; others are derived from their speaker label."""
    if shot["type"] == "anchor":
        return config.ANCHOR_SEED
    label = (shot.get("speaker") or shot.get("speaker_description") or shot["type"]).lower()
    return int(hashlib.md5(label.encode("utf-8")).hexdigest(), 16) % 1_000_000


def _broll_prompt(shot: dict) -> str:
    voiceover = (
        f"Voiceover: \"{shot['dialogue']}\"" if shot.get("dialogue")
        else "Ambient sound only, no narration."
    )
    return (
        f"Broadcast b-roll, documentary style. {shot['visual']}.\n"
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


async def _generate_with_retry(prompt: str, seconds: int, seed, ref_image_bytes, label: str):
    """Generate one clip with a single retry. On a 400 that looks like an input_reference
    (face) rejection, drop the reference and retry without it.

    Returns (clip_bytes_or_None, error_detail_or_None) so the caller can surface the
    real Sora/Azure failure instead of a generic message.
    """
    ref = ref_image_bytes
    last_detail = None
    for attempt in range(2):
        try:
            clip = await generate_clip(prompt, seconds, seed=seed, ref_image_bytes=ref)
            return clip, None
        except httpx.HTTPStatusError as exc:
            last_detail = _format_http_error(exc)
            if ref is not None and exc.response is not None and exc.response.status_code == 400:
                logger.warning("%s: reference rejected (400); retrying without input_reference. %s",
                               label, last_detail)
                ref = None
                continue
            logger.warning("%s: HTTP error (attempt %d): %s", label, attempt + 1, last_detail)
        except Exception as exc:  # network/timeout/etc.
            last_detail = f"{type(exc).__name__}: {exc}"
            logger.warning("%s: error (attempt %d): %s", label, attempt + 1, last_detail)
    logger.error("%s: failed after retries: %s", label, last_detail)
    return None, last_detail


async def build_news_video(decision_text: str, flat_date: str):
    """Produce one merged MP4 for this date, upload to blob. Returns (blob_url, clip_count)."""
    shot_list = await produce_shot_list(decision_text)
    shots = shot_list.get("shots", [])[:config.MAX_CLIPS]
    logger.info("build_news_video %s: %d shot(s) — %s", flat_date, len(shots), shot_list.get("title"))

    clips = []
    last_error = None
    prev_last_frame = None  # carried only across consecutive frame-chained b-roll shots
    for idx, shot in enumerate(shots):
        stype = shot["type"]
        is_broll = stype == "broll"
        seconds = shot["seconds"]
        label = f"clip{idx}:{stype}"

        if is_broll:
            prompt = _broll_prompt(shot)
            seed = None
            ref = prev_last_frame  # seed first frame from the previous b-roll's last frame
        else:
            # anchor / reporter / interview — an on-camera speaker (never a face reference)
            prompt = _talking_head_prompt(shot)
            seed = _seed_for(shot)
            ref = None

        clip, err = await _generate_with_retry(prompt, seconds, seed, ref, label)

        if clip is None:
            last_error = err or last_error
            if stype == "anchor" and not clips:
                # The opening anchor lead is required; abort so we don't ship a headless segment.
                raise RuntimeError(f"Anchor lead clip failed for {flat_date}: {err}")
            prev_last_frame = None
            continue

        clips.append(clip)

        # Prepare a chain frame for the next shot only when this b-roll asked to chain.
        if is_broll and shot.get("frame_chain"):
            try:
                prev_last_frame = await asyncio.to_thread(mux.extract_last_frame, clip)
            except Exception as exc:
                logger.warning("%s: last-frame extraction failed: %s", label, exc)
                prev_last_frame = None
        else:
            prev_last_frame = None

    if not clips:
        raise RuntimeError(f"No clips generated for {flat_date}: {last_error}")

    merged = await asyncio.to_thread(mux.concat_clips, clips)
    url = await asyncio.to_thread(upload_video_bytes_to_blob, merged, flat_date)
    logger.info("build_news_video %s: merged %d clip(s) -> %s", flat_date, len(clips), url)
    return url, len(clips)
