"""Text-to-speech via the same Azure OpenAI resources as Sora 2.

The TTS deployment lives on the same endpoints/keys as the Sora pool (just a different
model), so we reuse ``sora_client.next_resource`` for endpoint/key selection and call the
OpenAI v1 audio surface: ``POST {endpoint}/openai/v1/audio/speech``. TTS is a single
synchronous call (no job polling), so no per-resource affinity is needed.
"""
import asyncio
import logging

import httpx

from GenBox.azurestorage import upload_audio_bytes_to_blob
from GenBox.newsvideo import config
from GenBox.newsvideo.sora_client import next_resource, _headers, _host
from GenBox.newsvideo.tracing import traceable, summarize_bytes_output

logger = logging.getLogger("GenBoxVideo.tts")


def _speech_url(resource: dict) -> str:
    return (
        f"{resource['endpoint'].rstrip('/')}/openai/v1/audio/speech"
        f"?api-version={config.SORA_API_VERSION}"
    )


def _unsupported_field(exc: httpx.HTTPStatusError, body: dict) -> str:
    """Return the name of an optional field the deployment rejected (so we can drop it and
    retry), or None. Different TTS models accept different optional fields:
    ``tts-1`` takes ``speed`` but not ``instructions``; ``gpt-4o-mini-tts`` is the reverse.
    """
    resp = exc.response
    if resp is None or resp.status_code not in (400, 404):
        return None
    try:
        text = (resp.text or "").lower()
    except Exception:
        return None
    for field in ("instructions", "speed"):
        if field in body and field in text:
            return field
    return None


@traceable(run_type="tool", name="tts.synthesize_speech", process_outputs=summarize_bytes_output)
async def synthesize_speech(text: str, voice: str = None) -> bytes:
    """Synthesize ``text`` to speech bytes on a round-robin-selected resource.

    Sends a tone ``instructions`` hint and a faster ``speed`` best-effort, and drops
    whichever optional field the deployment rejects before retrying.
    """
    resource = next_resource()
    body = {
        "model": config.TTS_MODEL,
        "input": (text or "")[:config.TTS_MAX_CHARS],
        "voice": voice or config.TTS_VOICE,
        "response_format": config.TTS_FORMAT,
    }
    if config.TTS_INSTRUCTIONS:
        body["instructions"] = config.TTS_INSTRUCTIONS
    if config.TTS_SPEED and abs(config.TTS_SPEED - 1.0) > 1e-6:
        body["speed"] = config.TTS_SPEED          # >1.0 reads faster (tts-1 family)
    url = _speech_url(resource)
    headers = {**_headers(resource), "Content-Type": "application/json"}
    logger.info("synthesize_speech on resource %s (model=%s, speed=%s, %d chars)",
                _host(resource), config.TTS_MODEL, body.get("speed", 1.0), len(body["input"]))
    async with httpx.AsyncClient(timeout=120) as client:
        # At most one retry per optional field (instructions, speed).
        for _ in range(3):
            try:
                resp = await client.post(url, headers=headers, json=body)
                resp.raise_for_status()
                return resp.content
            except httpx.HTTPStatusError as exc:
                field = _unsupported_field(exc, body)
                if not field:
                    raise
                logger.warning("TTS deployment rejected '%s'; retrying without it.", field)
                body.pop(field, None)
    raise RuntimeError("TTS request failed after dropping unsupported optional fields")


@traceable(run_type="chain", name="GenBox Build Audio")
async def build_news_audio(text: str, flat_date: str) -> str:
    """Synthesize the narration and upload it to the video blob container. Returns the URL."""
    audio = await synthesize_speech(text)
    url = await asyncio.to_thread(upload_audio_bytes_to_blob, audio, flat_date, config.TTS_FORMAT)
    logger.info("build_news_audio %s: %d bytes -> %s", flat_date, len(audio), url)
    return url
