import asyncio
import base64
import os
from io import BytesIO
from typing import Optional

import httpx
from openai import AsyncAzureOpenAI

from ComicBook.azurestorage import upload_image_bytes_to_blob, save_photo_to_blob

_client: Optional[AsyncAzureOpenAI] = None

_DALLE3_SIZES = {"wide": "1792x1024", "tall": "1024x1792", "square": "1024x1024"}
_GPT_IMAGE_SIZES = {"wide": "1536x1024", "tall": "1024x1536", "square": "1024x1024"}


def _get_client() -> AsyncAzureOpenAI:
    global _client
    if _client is None:
        _client = AsyncAzureOpenAI(
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_DALLE"],
            api_key=os.environ["AZURE_OPENAI_API_KEY_DALLE"],
        )
    return _client


def _get_model() -> str:
    return os.environ.get("AZURE_OPENAI_MODEL_DALLE", "gpt-image-1")


def _resolve_size(size: str) -> str:
    model = _get_model()
    size_map = _DALLE3_SIZES if model == "dall-e-3" else _GPT_IMAGE_SIZES
    return size_map.get(size, "1024x1024")


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


async def create_image(prompt: str, size: str = "square") -> str:
    """Generate an image from a text prompt. Returns the blob URL."""
    model = _get_model()
    result = await _get_client().images.generate(
        model=model,
        prompt=prompt,
        size=_resolve_size(size),
    )
    return await asyncio.to_thread(_upload_result, result)


async def create_image_with_reference(prompt: str, reference_url: str, size: str = "square") -> str:
    """Generate an image using a text prompt and a reference image for style/character consistency."""
    model = _get_model()
    reference_bytes = await _download_image_bytes(reference_url)

    kwargs = {
        "model": model,
        "prompt": prompt,
        "size": _resolve_size(size),
    }
    if reference_bytes and model != "dall-e-3":
        reference_file = BytesIO(reference_bytes)
        reference_file.name = "reference.png"
        kwargs["image"] = reference_file

    result = await _get_client().images.edit(**kwargs)
    return await asyncio.to_thread(_upload_result, result)


async def create_image_with_references(
    prompt: str,
    image_urls: list[str],
    size: str = "square",
) -> str:
    """Generate an image using a text prompt and multiple reference images (up to 16).

    The first image should be the character sheet; subsequent ones can be previous
    panels, key frames from earlier episodes, etc.  Falls back to single-reference
    or prompt-only generation when only 0-1 images download successfully.
    """
    model = _get_model()

    if model == "dall-e-3":
        first = image_urls[0] if image_urls else ""
        return await create_image_with_reference(prompt, first, size)

    downloads = await asyncio.gather(
        *[_download_image_bytes(u) for u in image_urls[:16]]
    )
    files = []
    for idx, data in enumerate(downloads):
        if data:
            f = BytesIO(data)
            f.name = f"ref_{idx}.png"
            files.append(f)

    if not files:
        result = await _get_client().images.generate(
            model=model, prompt=prompt, size=_resolve_size(size),
        )
        return await asyncio.to_thread(_upload_result, result)

    kwargs = {
        "model": model,
        "prompt": prompt,
        "size": _resolve_size(size),
        "image": files if len(files) > 1 else files[0],
    }
    result = await _get_client().images.edit(**kwargs)
    return await asyncio.to_thread(_upload_result, result)
