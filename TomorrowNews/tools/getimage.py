import base64
from langchain_core.tools import tool
# Note: DALL-E 3 requires version 1.0.0 of the openai-python library or later
import logging
import os
from openai import AzureOpenAI
import json

from TomorrowNews.azurestorage import save_photo_to_blob, upload_image_bytes_to_blob

logger = logging.getLogger("TomorrowNews.getimage")

client = AzureOpenAI(
    api_version="2024-02-01",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_DALLE"],
    api_key=os.environ["AZURE_OPENAI_API_KEY_DALLE"],
)

# An image failure must never kill the edition. This tool used to let the API's exception
# propagate; LangGraph's ToolNode only absorbs `ToolInvocationError` and re-raises anything
# else, so one rejected illustration failed the `tools` node, cancelled the other images
# generating in parallel, and ended the whole newspaper run. Measured in LangSmith: that is
# exactly how the fa editions died on 2026-09-03 and 2026-09-04.
_UNAVAILABLE = (
    "IMAGE_UNAVAILABLE: the image safety system rejected this picture, and this text is NOT "
    "a URL — never place it in an img tag. Photos of identifiable people (especially children) "
    "are the usual cause. Either call this tool again describing the same scene with NO people "
    "in it, or continue the article without an image."
)

# Retrying a blocked prompt verbatim is pointless: an output-stage block is DETERMINISTIC
# (measured 9/9 on three real blocked prompts). What does pass is the same location with the
# people removed — verified in both Persian and English, so the language is not the trigger,
# the depicted minors are. So the single retry has to change the subject, not just try again.
_PEOPLE_FREE_RETRY = (
    ", wide establishing shot of the location only, empty scene, no people visible, no faces"
)


def _is_moderation_block(exc: Exception) -> bool:
    """True if ``exc`` is an image content-safety rejection. Mirrors ComicBook's predicate
    (`ComicBook/tools/getimage.py`) — the same API, the same split between a safety refusal
    and a transient failure."""
    if getattr(exc, "code", None) == "moderation_blocked":
        return True
    text = str(exc).lower()
    return "moderation_blocked" in text or "rejected by the safety system" in text


def _generate(prompt: str) -> str:
    """One image generation; returns the blob URL of the stored picture."""
    model = os.environ.get("AZURE_OPENAI_MODEL_DALLE", "dall-e-3")
    result = client.images.generate(
        model=model,  # the name of your DALL-E 3 deployment
        prompt=prompt + " (super realistic & high quality)",
        n=1
    )

    if model != "dall-e-3":
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)
        return upload_image_bytes_to_blob(image_bytes)

    image_url = json.loads(result.model_dump_json())['data'][0]['url']
    return save_photo_to_blob(image_url)


@tool
def get_image_by_text(text: str) -> str:
    """get the url of an image created as the input text explains, the input text should describe well the expected output"""
    try:
        return _generate(text)
    except Exception as exc:
        if _is_moderation_block(exc):
            logger.warning("Image blocked by content safety; retrying without people: %s", text[:120])
            try:
                return _generate(text + _PEOPLE_FREE_RETRY)
            except Exception as retry_exc:
                logger.warning("People-free retry also rejected (%s)", str(retry_exc)[:200])
        else:
            logger.warning("Image generation failed (%s): %s", type(exc).__name__, str(exc)[:200])
        return _UNAVAILABLE
