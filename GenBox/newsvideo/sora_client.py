"""Async client for the Sora 2 video API on Azure OpenAI.

The Azure video surface is a bespoke REST endpoint (``/openai/v1/videos``) that is
not wrapped by the OpenAI SDK ``images.*`` helpers, so we call it directly with httpx.

Workflow (async, JOB-SCOPED):
    create  -> POST   {endpoint}/openai/v1/videos?api-version=preview        -> {id}
    poll    -> GET    {endpoint}/openai/v1/videos/{id}?api-version=preview
    content -> GET    {endpoint}/openai/v1/videos/{id}/content?api-version=preview  (MP4)

A returned ``id`` only exists on the resource that served the create call, so poll and
download MUST target that SAME resource. To spread load across several resources while
respecting that affinity, we round-robin at the JOB level: ``generate_clip`` picks one
resource from the pool and runs create/poll/download all against it. (Do NOT route this
through a round-robin gateway — it would send each call to a different backend that has
never heard of the job id.)

Auth is the Azure-style ``api-key`` header (same as GenBox/prompt.py).
"""
import asyncio
import logging
import threading

import httpx

from GenBox.newsvideo import config

logger = logging.getLogger("GenBoxVideo.sora")

_SUCCESS_STATES = {"completed", "succeeded"}
_FAILURE_STATES = {"failed", "cancelled", "canceled"}

# ---------------------------------------------------------------------------
# Round-robin resource selection (job affinity preserved per job)
# ---------------------------------------------------------------------------
_rr_lock = threading.Lock()
_rr_index = 0


def next_resource() -> dict:
    """Pick the next usable {endpoint, key, model} from the pool, round-robin."""
    usable = [r for r in config.sora_pool() if r.get("endpoint") and r.get("key")]
    if not usable:
        raise RuntimeError(
            "No Sora resources configured. Set AZURE_OPENAI_ENDPOINT_SORA / "
            "AZURE_OPENAI_API_KEY_SORA (comma-separated for multiple resources)."
        )
    global _rr_index
    with _rr_lock:
        resource = usable[_rr_index % len(usable)]
        _rr_index += 1
    return resource


def _host(resource: dict) -> str:
    ep = resource.get("endpoint", "")
    return ep.split("://", 1)[-1].split("/", 1)[0]


def _videos_url(resource: dict, suffix: str = "") -> str:
    return (
        f"{resource['endpoint'].rstrip('/')}/openai/v1/videos{suffix}"
        f"?api-version={config.SORA_API_VERSION}"
    )


def _headers(resource: dict) -> dict:
    return {"api-key": resource["key"]}


def _is_seed_rejection(exc: httpx.HTTPStatusError) -> bool:
    resp = exc.response
    if resp is None or resp.status_code != 400:
        return False
    try:
        return "seed" in (resp.text or "").lower()
    except Exception:
        return False


def _multipart_parts(model: str, prompt: str, seconds: int, size: str, seed,
                     image_bytes=None) -> dict:
    """Build the multipart/form-data parts for a create call.

    The Azure Sora 2 create endpoint mirrors OpenAI's v1 ``/videos`` surface, which is
    multipart/form-data (it carries the optional ``input_reference`` file). Scalar fields
    are sent as text parts ``(None, value)`` — equivalent to ``curl -F key=value``.
    """
    parts = {
        "model": (None, model),
        "prompt": (None, prompt),
        "size": (None, size),
        "seconds": (None, str(seconds)),
    }
    if seed is not None:
        parts["seed"] = (None, str(seed))
    if image_bytes is not None:
        parts["input_reference"] = ("frame.png", image_bytes, "image/png")
    return parts


async def _post_create(resource: dict, parts: dict) -> str:
    """POST a multipart create request to a specific resource, returning the job id.
    Retries once without ``seed`` if the API rejects that field."""
    url = _videos_url(resource)
    async with httpx.AsyncClient(timeout=180) as client:
        try:
            resp = await client.post(url, headers=_headers(resource), files=parts)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if "seed" in parts and _is_seed_rejection(exc):
                logger.warning("Sora rejected 'seed'; retrying without it.")
                parts = {k: v for k, v in parts.items() if k != "seed"}
                resp = await client.post(url, headers=_headers(resource), files=parts)
                resp.raise_for_status()
            else:
                raise
        return resp.json()["id"]


async def create_video_job(resource: dict, prompt: str, seconds: int, size: str = None,
                           seed: int = None) -> str:
    """Create a text-only video job (multipart/form-data) on ``resource``. Returns job id."""
    size = size or config.VIDEO_SIZE
    return await _post_create(resource, _multipart_parts(resource["model"], prompt, seconds, size, seed))


async def create_video_job_with_reference(resource: dict, prompt: str, seconds: int,
                                          image_bytes: bytes, size: str = None,
                                          seed: int = None) -> str:
    """Create an image-guided video job on ``resource`` whose first frame is ``image_bytes``.

    The reference is sent as multipart/form-data and must match ``size``. Returns job id.
    """
    size = size or config.VIDEO_SIZE
    return await _post_create(resource, _multipart_parts(resource["model"], prompt, seconds, size, seed, image_bytes))


async def poll_until_complete(resource: dict, job_id: str, interval: float = 6.0,
                              timeout: float = 900.0) -> dict:
    """Poll a job (on the resource that created it) until terminal. Returns the job dict.

    Raises RuntimeError on failure/cancellation and TimeoutError on timeout.
    """
    url = _videos_url(resource, f"/{job_id}")
    waited = 0.0
    async with httpx.AsyncClient(timeout=60) as client:
        while waited < timeout:
            resp = await client.get(url, headers=_headers(resource))
            resp.raise_for_status()
            job = resp.json()
            status = (job.get("status") or "").lower()
            if status in _SUCCESS_STATES:
                return job
            if status in _FAILURE_STATES:
                raise RuntimeError(f"Sora job {job_id} {status}: {job.get('error')}")
            await asyncio.sleep(interval)
            waited += interval
    raise TimeoutError(f"Sora job {job_id} timed out after {timeout}s")


async def download_video_bytes(resource: dict, job_id: str) -> bytes:
    """Download the finished MP4 for a completed job (from its owning resource)."""
    url = _videos_url(resource, f"/{job_id}/content")
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.get(url, headers=_headers(resource))
        resp.raise_for_status()
        return resp.content


async def generate_clip(prompt: str, seconds: int, seed: int = None,
                        ref_image_bytes: bytes = None) -> bytes:
    """Create -> poll -> download a single clip on ONE round-robin-selected resource.

    Pinning the whole lifecycle to the chosen resource preserves job affinity while
    spreading jobs across the pool to use distributed credits.
    """
    resource = next_resource()
    logger.info("generate_clip on resource %s (model=%s)", _host(resource), resource.get("model"))
    if ref_image_bytes:
        job_id = await create_video_job_with_reference(resource, prompt, seconds, ref_image_bytes, seed=seed)
    else:
        job_id = await create_video_job(resource, prompt, seconds, seed=seed)
    await poll_until_complete(resource, job_id)
    return await download_video_bytes(resource, job_id)
