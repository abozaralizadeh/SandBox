"""LangSmith tracing helpers for the GenBox news-video / narration pipeline.

``@traceable`` and ``wrap_openai`` are no-ops unless LangSmith tracing is enabled via env
(``LANGCHAIN_TRACING_V2`` / ``LANGSMITH_TRACING``), so applying them is always safe. The
redactors below keep per-resource API keys and large binary payloads OUT of the traces.
"""
from langsmith import traceable  # re-exported so callers import everything from here

__all__ = ["traceable", "redact_resource_inputs", "summarize_clip_output", "summarize_bytes_output"]


def redact_resource_inputs(inputs: dict) -> dict:
    """Drop the per-resource API key and any raw image bytes from traced inputs."""
    safe = dict(inputs)
    resource = safe.get("resource")
    if isinstance(resource, dict):
        # keep endpoint + model for debugging; NEVER log the api key
        safe["resource"] = {"endpoint": resource.get("endpoint"), "model": resource.get("model")}
    for key in ("ref_image_bytes", "image_bytes", "audio_bytes"):
        val = safe.get(key)
        if isinstance(val, (bytes, bytearray)):
            safe[key] = f"<{len(val)} bytes>"
    return safe


def summarize_clip_output(output):
    """(mp4_bytes, job_id) -> a small summary instead of megabytes of binary in the trace."""
    if isinstance(output, tuple) and output and isinstance(output[0], (bytes, bytearray)):
        return {"size_bytes": len(output[0]), "job_id": output[1] if len(output) > 1 else None}
    return output


def summarize_bytes_output(output):
    if isinstance(output, (bytes, bytearray)):
        return {"size_bytes": len(output)}
    return output
