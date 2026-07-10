import asyncio
import base64
import os
import struct
import zlib
from io import BytesIO
from typing import Optional

import httpx
from openai import AsyncAzureOpenAI

from ComicBook.azurestorage import upload_image_bytes_to_blob, save_photo_to_blob

_client: Optional[AsyncAzureOpenAI] = None


class ContentModerationError(Exception):
    """The image API rejected the PROMPT via its content-safety system (HTTP 400,
    code 'moderation_blocked'). Unlike a transient/gateway error, retrying the SAME prompt
    — or falling back to prompt-only generation — fails identically, because the prompt
    content itself is the problem. Callers must REWRITE the prompt and retry."""


def _is_moderation_block(exc: Exception) -> bool:
    """True if ``exc`` is an image content-safety rejection (so callers can request a prompt
    rewrite instead of treating it as a transient failure or a dead service)."""
    if getattr(exc, "code", None) == "moderation_blocked":
        return True
    text = str(exc).lower()
    return "moderation_blocked" in text or "rejected by the safety system" in text


_DALLE3_SIZES = {"wide": "1792x1024", "tall": "1024x1792", "square": "1024x1024"}
_GPT_IMAGE_SIZES = {"wide": "1536x1024", "tall": "1024x1536", "square": "1024x1024"}


# No time limit on image generation: gpt-image edits/generations are slow (and slower under load).
# Default to the gunicorn request budget (1h) instead of the SDK's 600s so a slow-but-successful
# panel is never cut off; override with COMICBOOK_IMAGE_TIMEOUT.
_IMAGE_TIMEOUT = float(os.environ.get("COMICBOOK_IMAGE_TIMEOUT", "3600"))


def _get_client() -> AsyncAzureOpenAI:
    global _client
    if _client is None:
        _client = AsyncAzureOpenAI(
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_DALLE"],
            api_key=os.environ["AZURE_OPENAI_API_KEY_DALLE"],
            timeout=_IMAGE_TIMEOUT,
        )
    return _client


async def _call_with_retries(make_coro, attempts: int, label: str):
    """Run an image API call with a few retries on transient failures (e.g. gateway 500/timeout)."""
    delay = 2.0
    last_exc: Optional[Exception] = None
    for i in range(1, attempts + 1):
        try:
            return await make_coro()
        except Exception as exc:
            if _is_moderation_block(exc):
                # A content-safety rejection won't change on retry — surface it distinctly so
                # the caller rewrites the prompt instead of burning retries/fallbacks on it.
                raise ContentModerationError(str(exc)) from exc
            last_exc = exc
            print(f"[ComicBook:getimage] {label} attempt {i}/{attempts} failed: {str(exc)[:180]}")
            if i < attempts:
                await asyncio.sleep(delay)
                delay *= 2
    raise last_exc


def _solid_png_bytes(rgb=(214, 214, 210), w: int = 32, h: int = 32) -> bytes:
    """Build a tiny solid-color PNG with the stdlib only (no Pillow). Stretched by the panel CSS
    (object-fit: cover) it renders as a flat placeholder tile."""
    row = b"\x00" + bytes(rgb) * w          # filter byte 0 + RGB pixels
    raw = row * h
    def _chunk(typ: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + typ + data
                + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)   # 8-bit, color type 2 (RGB)
    return (b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", zlib.compress(raw))
            + _chunk(b"IEND", b""))


_placeholder_url_cache: Optional[str] = None


async def make_placeholder_image_url() -> str:
    """Return a blob URL for a neutral placeholder tile (uploaded once per process). Used when the
    image service is unavailable so a panel still renders and the run completes instead of hanging."""
    global _placeholder_url_cache
    if _placeholder_url_cache is None:
        _placeholder_url_cache = await asyncio.to_thread(upload_image_bytes_to_blob, _solid_png_bytes())
    return _placeholder_url_cache


def _get_model() -> str:
    return os.environ.get("AZURE_OPENAI_MODEL_DALLE", "gpt-image-1")


def _resolve_size(size: str) -> str:
    model = _get_model()
    size_map = _DALLE3_SIZES if model == "dall-e-3" else _GPT_IMAGE_SIZES
    return size_map.get(size, "1024x1024")


def _resolve_quality(quality: str) -> str:
    """Map a shared quality level to the model-specific parameter value.

    gpt-image-1 accepts: "low", "medium", "high"
    dall-e-3     accepts: "standard", "hd"
    """
    model = _get_model()
    if model == "dall-e-3":
        return "hd" if quality == "high" else "standard"
    return quality if quality in ("low", "medium", "high") else "medium"


async def _download_image_bytes(url: str) -> Optional[bytes]:
    if not url:
        return None
    try:
        async with httpx.AsyncClient() as http:
            resp = await http.get(url, timeout=30)
            resp.raise_for_status()
            return resp.content
    except Exception as exc:
        print(f"[ComicBook:getimage] Download failed: {exc}")
        return None


def _upload_result(result) -> str:
    model = _get_model()
    if model != "dall-e-3":
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)
        return upload_image_bytes_to_blob(image_bytes)
    image_url = result.data[0].url
    return save_photo_to_blob(image_url)


async def create_image(prompt: str, size: str = "square", quality: str = "medium") -> str:
    """Generate an image from a text prompt. Returns the blob URL."""
    model = _get_model()

    async def _gen():
        result = await _get_client().images.generate(
            model=model,
            prompt=prompt,
            size=_resolve_size(size),
            quality=_resolve_quality(quality),
        )
        return await asyncio.to_thread(_upload_result, result)

    return await _call_with_retries(_gen, attempts=1, label="generate")


async def create_image_with_reference(prompt: str, reference_url: str, size: str = "square", quality: str = "medium") -> str:
    """Generate an image using a text prompt and a reference image for style/character consistency.

    Falls back to prompt-only generation if the (slower) edits endpoint fails, so a panel
    always renders even when the image gateway times out on reference edits.
    """
    model = _get_model()
    reference_bytes = await _download_image_bytes(reference_url)
    if not reference_bytes or model == "dall-e-3":
        return await create_image(prompt, size, quality)

    async def _edit():
        ref = BytesIO(reference_bytes)
        ref.name = "reference.png"
        result = await _get_client().images.edit(
            model=model,
            prompt=prompt,
            size=_resolve_size(size),
            quality=_resolve_quality(quality),
            image=ref,
        )
        return await asyncio.to_thread(_upload_result, result)

    try:
        return await _call_with_retries(_edit, attempts=1, label="edit(1 ref)")
    except ContentModerationError:
        raise  # prompt content is the problem; prompt-only fallback would be blocked too
    except Exception as exc:
        print(f"[ComicBook:getimage] edit failed, falling back to prompt-only generate: {str(exc)[:180]}")
        return await create_image(prompt, size, quality)


async def create_image_with_references(
    prompt: str,
    image_urls: list[str],
    size: str = "square",
    quality: str = "medium",
) -> str:
    """Generate an image using a text prompt and multiple reference images (up to 16).

    The first image should be the character sheet; subsequent ones can be previous
    panels, key frames from earlier episodes, etc.  Falls back to prompt-only generation
    when no images download or the edits endpoint fails — so a panel always renders.
    """
    model = _get_model()

    if model == "dall-e-3":
        first = image_urls[0] if image_urls else ""
        return await create_image_with_reference(prompt, first, size, quality)

    downloads = await asyncio.gather(
        *[_download_image_bytes(u) for u in image_urls[:16]]
    )
    ref_bytes = [d for d in downloads if d]
    if not ref_bytes:
        return await create_image(prompt, size, quality)

    async def _edit():
        files = []
        for idx, data in enumerate(ref_bytes):
            f = BytesIO(data)
            f.name = f"ref_{idx}.png"
            files.append(f)
        result = await _get_client().images.edit(
            model=model,
            prompt=prompt,
            size=_resolve_size(size),
            quality=_resolve_quality(quality),
            image=files if len(files) > 1 else files[0],
        )
        return await asyncio.to_thread(_upload_result, result)

    try:
        return await _call_with_retries(_edit, attempts=1, label=f"edit({len(ref_bytes)} refs)")
    except ContentModerationError:
        raise  # prompt content is the problem; prompt-only fallback would be blocked too
    except Exception as exc:
        print(f"[ComicBook:getimage] multi-ref edit failed, falling back to prompt-only generate: {str(exc)[:180]}")
        return await create_image(prompt, size, quality)
