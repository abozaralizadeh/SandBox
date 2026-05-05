import asyncio
import base64
import json
import os
from typing import Optional

from langchain_core.tools import tool
from openai import AsyncAzureOpenAI

from AIBlog.azurestorage import save_photo_to_blob, upload_image_bytes_to_blob

_client: Optional[AsyncAzureOpenAI] = None


def _get_client() -> AsyncAzureOpenAI:
    global _client
    if _client is None:
        _client = AsyncAzureOpenAI(
            api_version="2024-02-01",
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_DALLE"],
            api_key=os.environ["AZURE_OPENAI_API_KEY_DALLE"],
        )
    return _client


@tool
async def get_image_by_text(text: str) -> str:
    """get the url of an image created as the input text explains, the input text should describe well the expected output"""
    model = os.environ.get("AZURE_OPENAI_MODEL_DALLE", "dall-e-3")
    result = await _get_client().images.generate(
        model=model,
        prompt=text,
        n=1,
    )

    if model != "dall-e-3":
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)
        return await asyncio.to_thread(upload_image_bytes_to_blob, image_bytes)

    image_url = json.loads(result.model_dump_json())["data"][0]["url"]
    return await asyncio.to_thread(save_photo_to_blob, image_url)
