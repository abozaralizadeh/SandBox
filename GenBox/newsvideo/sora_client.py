"""Async client for the Sora 2 video API on Azure OpenAI.

The Azure video surface is a bespoke REST endpoint (``/openai/v1/videos``) that is
not wrapped by the OpenAI SDK ``images.*`` helpers, so we call it directly with httpx.

Workflow (async, JOB-SCOPED):
    create  -> POST   {endpoint}/openai/v1/videos?api-version=preview        -> {id}
    poll    -> GET    {endpoint}/openai/v1/videos/{id}?api-version=preview
    content -> GET    {endpoint}/openai/v1/videos/{id}/content?api-version=preview  (MP4)

A returned ``id`` only exists on the resource that served the create call, so poll and
download MUST target that SAME resource. To spread load across several resources while
respecting that affinity, we round-robin at the JOB level: ``create_clip`` picks one
resource from the pool and runs create/poll/download all against it, and ``remix_clip``
reuses that same resource. (Do NOT route this through a round-robin gateway — it would
send each call to a different backend that has never heard of the job id.)

Auth is the Azure-style ``api-key`` header (same as GenBox/prompt.py).
"""
import asyncio
import logging
import threading

import httpx

from GenBox.newsvideo import config
from GenBox.newsvideo.tracing import traceable, redact_resource_inputs, summarize_clip_output

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


def _multipart_parts(model: str, prompt: str, seconds: int, size: str,
                     image_bytes=None) -> dict:
    """Build the multipart/form-data parts for a create call.

    The Azure Sora 2 create endpoint mirrors OpenAI's v1 ``/videos`` surface, which is
    multipart/form-data (it carries the optional ``input_reference`` file). Scalar fields
    are sent as text parts ``(None, value)`` — equivalent to ``curl -F key=value``.
    (Note: Sora 2 has no ``seed`` parameter, so we never send one.)
    """
    parts = {
        "model": (None, model),
        "prompt": (None, prompt),
        "size": (None, size),
        "seconds": (None, str(seconds)),
    }
    if image_bytes is not None:
        parts["input_reference"] = ("frame.png", image_bytes, "image/png")
    return parts


async def _post(url: str, resource: dict, parts: dict) -> str:
    """POST a multipart request to ``url`` on ``resource``, returning the new job id."""
    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(url, headers=_headers(resource), files=parts)
        resp.raise_for_status()
        return resp.json()["id"]


async def create_video_job(resource: dict, prompt: str, seconds: int, size: str = None,
                           image_bytes: bytes = None) -> str:
    """Create a video job (multipart/form-data) on ``resource``. Returns the job id.
    Pass ``image_bytes`` to use it as the first frame (face-free content only)."""
    size = size or config.VIDEO_SIZE
    parts = _multipart_parts(resource["model"], prompt, seconds, size, image_bytes)
    return await _post(_videos_url(resource), resource, parts)


async def remix_video_job(resource: dict, base_job_id: str, prompt: str, seconds: int = None) -> str:
    """Remix a previously completed video (on the SAME resource that created it).

    Remix reuses the source video's framework, scene transitions, and visual layout while
    applying the change in ``prompt`` — the consistency mechanism that needs no face images
    or gated character access. Returns the new job id.

    Unlike ``create``, the remix endpoint takes ONLY a JSON ``prompt`` and INHERITS the
    source's model, size, and duration. Sending ``seconds`` — or posting multipart
    form-data as a create call does — is rejected with HTTP 400, so ``seconds`` is kept for
    call-site symmetry but intentionally NOT sent (the remix matches the base clip's length).
    """
    url = _videos_url(resource, f"/{base_job_id}/remix")
    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(url, headers=_headers(resource), json={"prompt": prompt})
        resp.raise_for_status()
        return resp.json()["id"]


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


@traceable(run_type="tool", name="sora.create_clip",
           process_inputs=redact_resource_inputs, process_outputs=summarize_clip_output)
async def create_clip(resource: dict, prompt: str, seconds: int,
                      ref_image_bytes: bytes = None) -> tuple:
    """Create -> poll -> download a fresh clip on ``resource``. Returns (mp4_bytes, job_id).

    The job id is returned so the caller can remix this clip later for consistency.
    """
    logger.info("create_clip on resource %s (model=%s)", _host(resource), resource.get("model"))
    job_id = await create_video_job(resource, prompt, seconds, image_bytes=ref_image_bytes)
    await poll_until_complete(resource, job_id)
    return await download_video_bytes(resource, job_id), job_id


@traceable(run_type="tool", name="sora.remix_clip",
           process_inputs=redact_resource_inputs, process_outputs=summarize_clip_output)
async def remix_clip(resource: dict, base_job_id: str, prompt: str, seconds: int) -> tuple:
    """Remix -> poll -> download on the base clip's resource. Returns (mp4_bytes, job_id)."""
    logger.info("remix_clip of %s on resource %s", base_job_id, _host(resource))
    job_id = await remix_video_job(resource, base_job_id, prompt, seconds)
    await poll_until_complete(resource, job_id)
    return await download_video_bytes(resource, job_id), job_id
