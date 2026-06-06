"""Configuration + constants for GenBox news-anchor video generation.

A separate Azure OpenAI resource (or several) hosts the Sora 2 video deployment, so it
has its own endpoint/key/version env vars (mirroring the existing AZURE_OPENAI_*_DALLE
split).

Sora's video API is asynchronous and job-scoped: `create` returns a video id that only
exists on the resource that served it, so `poll {id}` and `download {id}` MUST hit that
same resource. A round-robin gateway breaks this affinity. To still spread load across
several resources (e.g. distributed credits), configure the endpoints/keys as
comma-separated lists pointing DIRECTLY at each resource; the app round-robins at the
job level and pins each job's whole lifecycle to the resource it picked.
"""
import os
from datetime import date, datetime

from dotenv import load_dotenv

from utils import strtobool

load_dotenv()

# --- Sora 2 (Azure OpenAI) resource pool ---
# Endpoints and keys are comma-separated and aligned by index; a single key/model
# applies to all endpoints. Point these at the DIRECT resource endpoints, not a
# round-robin gateway, so per-job affinity holds.
SORA_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION_SORA", "preview")


def _split_csv(value):
    return [v.strip() for v in (value or "").split(",") if v.strip()]


_SORA_ENDPOINTS = _split_csv(os.getenv("AZURE_OPENAI_ENDPOINT_SORA"))
_SORA_KEYS = _split_csv(os.getenv("AZURE_OPENAI_API_KEY_SORA"))
_SORA_MODELS = _split_csv(os.getenv("AZURE_OPENAI_MODEL_SORA")) or ["sora-2"]


def sora_pool():
    """Return the Sora resource pool as a list of {endpoint, key, model}.

    Keys align to endpoints by index; if a single key/model is given it applies to all
    endpoints. Round-robining across this pool at the job level spreads usage while
    keeping each job (create/poll/download) on one resource.
    """
    pool = []
    for i, endpoint in enumerate(_SORA_ENDPOINTS):
        key = _SORA_KEYS[i] if i < len(_SORA_KEYS) else (_SORA_KEYS[0] if _SORA_KEYS else "")
        model = _SORA_MODELS[i] if i < len(_SORA_MODELS) else _SORA_MODELS[0]
        pool.append({"endpoint": endpoint, "key": key, "model": model})
    return pool


# Back-compat singulars (first resource in the pool).
_first_resource = sora_pool()
SORA_ENDPOINT = _first_resource[0]["endpoint"] if _first_resource else ""
SORA_API_KEY = _first_resource[0]["key"] if _first_resource else ""
SORA_MODEL = _first_resource[0]["model"] if _first_resource else "sora-2"

# --- video format / cost controls ---
VIDEO_SIZE = os.getenv("GENBOX_VIDEO_SIZE", "1280x720")  # landscape for the CRT screen
ALLOWED_SECONDS = (4, 8, 12)                              # Sora 2 clip durations
MAX_CLIPS = int(os.getenv("GENBOX_VIDEO_MAX_CLIPS", "6"))


def _enabled() -> bool:
    try:
        return strtobool(os.getenv("GENBOX_VIDEO_ENABLED", "true"))
    except ValueError:
        return True


def _cutoff_date():
    raw = (os.getenv("GENBOX_VIDEO_CUTOFF_DATE") or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw).date()
    except ValueError:
        return None


def _as_date(value) -> date:
    if value is None:
        return datetime.now().date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value)).date()
    except ValueError:
        return datetime.now().date()


def video_enabled_for(value=None) -> bool:
    """True only when the feature is on, Sora is configured, and the date is >= cutoff
    (decision: video for new dates only)."""
    if not _enabled():
        return False
    if not any(r["endpoint"] and r["key"] for r in sora_pool()):
        return False
    cutoff = _cutoff_date()
    if cutoff and _as_date(value) < cutoff:
        return False
    return True


# Fixed "anchor bible": identical text used for the anchor's FIRST clip so Sora renders a
# defined persona + set. Subsequent anchor clips are produced by remixing that first clip
# (see pipeline), which keeps the look consistent without face references (Azure Sora's
# input_reference rejects human faces) or a seed (Sora 2 has none).
ANCHOR_BIBLE = (
    "Television news studio. A single news anchor, a calm 40-year-old presenter with "
    "short dark hair and a neat navy-blue suit over a white shirt, sits at a glossy "
    "dark anchor desk. Behind them is a large blue holographic screen reading "
    "'GENBOX NEWS' with subtle world-map graphics. Soft broadcast key lighting, "
    "shallow depth of field, eye-level medium shot, professional 24fps cinematic look. "
    "The anchor speaks directly to camera with natural, accurate lip-sync and a steady, "
    "authoritative broadcast cadence."
)

# Appended to every prompt to keep clips free of baked-in captions/branding.
BROLL_NEGATIVE = (
    "No on-screen text, no captions, no subtitles, no chyrons, no logos, no watermarks."
)
