import base64
import os
from io import BytesIO
from typing import Optional

import requests
from langchain_core.tools import tool
from openai import AzureOpenAI

from ComicBook.azurestorage import upload_image_bytes_to_blob, save_photo_to_blob

# Azure OpenAI client for images
client = AzureOpenAI(
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_DALLE"],
    api_key=os.environ["AZURE_OPENAI_API_KEY_DALLE"],
)


def _download_image_bytes(url: str) -> Optional[bytes]:
    if not url:
        return None
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.content
    except Exception as exc:
        print(f"[ComicBook:getimage] Failed to download reference image: {exc}")
        return None


@tool
def get_image_by_text(text: str) -> str:
    """Generate an image URL from descriptive text."""
    model = os.environ.get("AZURE_OPENAI_MODEL_DALLE", "gpt-image-1")
    result = client.images.generate(
        model=model,
        prompt=text,
        size="1024x1024",
    )

    if model != "dall-e-3":
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)
        blob_image_url = upload_image_bytes_to_blob(image_bytes)
        return blob_image_url

    image_url = result.data[0].url
    blob_image_url = save_photo_to_blob(image_url)
    return blob_image_url


@tool
def get_image_by_text_with_reference(text: str, reference_image_url: str = "") -> str:
    """
    Generate an image using text plus a reference image URL to keep character/style consistency.
    Provide a strong description and the last image URL to anchor the design.
    """
    model = os.environ.get("AZURE_OPENAI_MODEL_DALLE", "gpt-image-1")
    reference_bytes = _download_image_bytes(reference_image_url)

    kwargs = {
        "model": model,
        "prompt": text,
        "size": "1024x1024",
    }
    if reference_bytes and model != "dall-e-3":
        reference_file = BytesIO(reference_bytes)
        reference_file.name = "reference.png"  # Helps the client send a valid image mimetype
        kwargs["image"] = reference_file

    # Use generate with an image parameter to steer style/characters
    result = client.images.edit(**kwargs)

    if model != "dall-e-3":
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)
        blob_image_url = upload_image_bytes_to_blob(image_bytes)
        return blob_image_url

    image_url = result.data[0].url
    blob_image_url = save_photo_to_blob(image_url)
    return blob_image_url
