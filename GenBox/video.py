"""Integration surface for the GenBox news-anchor video feature.

Generation runs in a background thread (the app has no job system) so the HTTP request
never blocks. Status + single-flight lock live in Azure Tables, so any gunicorn worker
can serve the polling endpoint regardless of which worker owns the generating thread.
"""
import asyncio
import json
import logging
import threading
from datetime import datetime, timezone

from utils import get_flat_date
from GenBox.newsvideo import config
from GenBox.newsvideo.tracing import traceable
from GenBox.azurestorage import (
    get_row,
    get_video_meta,
    set_video_meta,
    try_acquire_video_lock,
    release_video_lock,
    try_acquire_audio_lock,
    release_audio_lock,
)

logger = logging.getLogger("GenBoxVideo")

# A "generating" row older than this is treated as stale (its worker likely died) and
# may be restarted. Kept in sync with the table lock TTL in azurestorage.
_STALE_SECONDS = 1800


def _flat(date) -> str:
    return get_flat_date(date) if date else get_flat_date()


def _decision_text_for(flat_date: str):
    """The anchor speaks the GenBox decision's 'output' field for that date."""
    row = get_row("assistant", flat_date)
    if not row:
        return None
    try:
        return json.loads(row["content"].strip().replace("\n", " "))["output"]
    except Exception:
        return None


def _is_stale(meta) -> bool:
    stamp = meta.get("updated_at") or meta.get("created_at")
    if not stamp:
        return True
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(stamp)).total_seconds()
    except Exception:
        return True
    return age > _STALE_SECONDS


def _status_dict(meta) -> dict:
    """Shape the {video, audio} status returned to the frontend from a metadata row."""
    if not meta:
        return {"status": "absent", "video_url": "", "audio_status": "absent", "audio_url": ""}
    return {
        "status": meta.get("status", "pending"),
        "video_url": meta.get("video_url", ""),
        "audio_status": meta.get("audio_status", "absent"),
        "audio_url": meta.get("audio_url", ""),
    }


def video_status(date=None) -> dict:
    """Read-only status for a date. Safe on any worker. Does NOT start generation."""
    if not config.video_enabled_for(date):
        return {"status": "disabled", "video_url": "", "audio_status": "disabled", "audio_url": ""}
    return _status_dict(get_video_meta(_flat(date)))


def ensure_generation_started(date=None) -> dict:
    """Idempotent, non-blocking trigger. Ensures the VIDEO and the NARRATION are generated
    INDEPENDENTLY for an eligible date (so narration backfills dates that already have a
    video), then returns the current status dict."""
    if not config.video_enabled_for(date):
        return {"status": "disabled", "video_url": "", "audio_status": "disabled", "audio_url": ""}

    flat = _flat(date)
    decision = None  # fetched lazily; shared by both branches

    meta = get_video_meta(flat)

    # ---- VIDEO (slow) ----
    need_video = True
    if meta:
        vs = meta.get("status")
        if vs == "ready":
            need_video = False
        elif vs == "generating" and not _is_stale(meta):
            need_video = False
        elif vs == "failed":
            need_video = False  # terminal; the frontend stops polling on 'failed'
    if need_video:
        decision = _decision_text_for(flat)
        if decision and try_acquire_video_lock(flat):
            set_video_meta(flat, status="generating", video_url="", error="")
            threading.Thread(target=_run_video_generation, args=(flat, decision), daemon=True).start()

    # ---- NARRATION (fast, independent) ----
    if config.tts_enabled_for(date):
        meta_a = get_video_meta(flat)
        if not (meta_a and meta_a.get("audio_status") == "ready"):
            if decision is None:
                decision = _decision_text_for(flat)
            # The audio lock both prevents duplicate work and lets a failed/stale run retry.
            if decision and try_acquire_audio_lock(flat):
                set_video_meta(flat, audio_status="generating", audio_url="")
                threading.Thread(target=_run_audio_generation, args=(flat, decision), daemon=True).start()

    return _status_dict(get_video_meta(flat))


@traceable(run_type="chain", name="GenBox Video Generation")
def _run_video_generation(flat_date: str, decision: str):
    try:
        # Imported lazily so app startup doesn't pull the agents SDK / ffmpeg.
        from GenBox.newsvideo.pipeline import build_news_video
        url, clip_count = asyncio.run(build_news_video(decision, flat_date))
        set_video_meta(flat_date, status="ready", video_url=url, clip_count=clip_count, error="")
        logger.info("GenBox video ready for %s (%d clips)", flat_date, clip_count)
    except Exception as exc:
        logger.exception("GenBox video generation failed for %s", flat_date)
        set_video_meta(flat_date, status="failed", error=str(exc)[:512], video_url="")
    finally:
        release_video_lock(flat_date)


@traceable(run_type="chain", name="GenBox Audio Generation")
def _run_audio_generation(flat_date: str, decision: str):
    try:
        from GenBox.newsvideo.tts_client import build_news_audio
        url = asyncio.run(build_news_audio(decision, flat_date))
        set_video_meta(flat_date, audio_status="ready", audio_url=url)
        logger.info("GenBox audio ready for %s", flat_date)
    except Exception as exc:
        logger.exception("GenBox audio generation failed for %s", flat_date)
        set_video_meta(flat_date, audio_status="failed", audio_url="", audio_error=str(exc)[:512])
    finally:
        release_audio_lock(flat_date)
