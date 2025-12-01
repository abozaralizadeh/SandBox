import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from AIOpenProblemSolver.azurestorage import (
    get_iteration_slice,
    latest_iteration,
    save_iteration,
)
from AIOpenProblemSolver.graph import get_open_deep_search_agent

try:  # LangGraph >= 0.2
    from langgraph.types import Overwrite, StateSnapshot
except ImportError:  # pragma: no cover - fallback for older installations
    Overwrite = None  # type: ignore[assignment]
    StateSnapshot = None  # type: ignore[assignment]

try:  # LangChain >= 0.2
    from langchain_core.messages import AIMessage
except ImportError:  # pragma: no cover - fallback for minimal installs
    AIMessage = None  # type: ignore[assignment]

DEFAULT_PAGE_SIZE = int(os.getenv("AIOPS_PAGE_SIZE", "5"))
GRAPH_RECURSION_LIMIT = int(os.getenv("GRAPH_RECURSION_LIMIT", "1000"))


def _maybe_overwrite_value(value: Any) -> Any:
    if Overwrite is not None and isinstance(value, Overwrite):
        return value.value
    return value


def _is_assistant_role(role: Optional[str]) -> bool:
    return bool(role) and role.lower() in {"ai", "assistant"}


def _stringify_content(content: Any) -> Optional[str]:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, (int, float, bool)):
        return str(content)
    if isinstance(content, dict):
        text = content.get("text") or content.get("content")
        if isinstance(text, str):
            return text
        if isinstance(text, list):
            return _stringify_content(text)
    if isinstance(content, list):
        pieces: List[str] = []
        for item in content:
            text = _stringify_content(item)
            if isinstance(text, str) and text.strip():
                pieces.append(text.strip())
        if pieces:
            return "\n".join(pieces)
    return None


def _assistant_message_content(message: Any) -> Optional[str]:
    msg = _maybe_overwrite_value(message)
    if AIMessage is not None and isinstance(msg, AIMessage):
        content = _stringify_content(getattr(msg, "content", None))
        if content:
            return content

    msg_role = getattr(msg, "type", None) or getattr(msg, "role", None)
    if _is_assistant_role(msg_role):
        content = _stringify_content(getattr(msg, "content", None))
        if content:
            return content

    if isinstance(msg, dict):
        role = msg.get("type") or msg.get("role")
        if _is_assistant_role(role):
            content = _stringify_content(msg.get("content"))
            if content:
                return content

    if isinstance(msg, tuple) and len(msg) == 2:
        role, content = msg
        if _is_assistant_role(role):
            text = _stringify_content(content)
            if text:
                return text

    return None


def _extract_assistant_reply(messages: Any) -> Optional[str]:
    if messages is None or isinstance(messages, str):
        return None

    try:
        iterable = list(messages)
    except TypeError:
        return None

    fallback_json: Optional[str] = None
    fallback_plain: Optional[str] = None
    for candidate in reversed(iterable):
        text = _assistant_message_content(candidate)
        if isinstance(text, str) and text.strip():
            return text

        fallback_source = candidate
        if isinstance(candidate, tuple) and len(candidate) == 2:
            fallback_source = candidate[1]

        fallback_text = _stringify_content(fallback_source)
        if isinstance(fallback_text, str):
            stripped = fallback_text.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                fallback_json = fallback_text
                break
            if fallback_plain is None and stripped:
                fallback_plain = fallback_text

    return fallback_json or fallback_plain


def _extract_message_container(payload: Any) -> Optional[Dict[str, Any]]:
    obj = _maybe_overwrite_value(payload)
    if StateSnapshot is not None and isinstance(obj, StateSnapshot):
        obj = obj.values

    if isinstance(obj, dict):
        return obj

    if isinstance(obj, (list, tuple)):
        for item in reversed(obj):
            container = _extract_message_container(item)
            if container:
                return container

    return None


def _rowkey_now() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _rowkey_to_date(rowkey: str) -> Optional[datetime]:
    for fmt in ("%Y%m%d_%H%M%S", "%Y%m%d_%H", "%Y%m%d"):
        try:
            return datetime.strptime(rowkey, fmt)
        except ValueError:
            continue
    return None


def _decode_metadata(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return {}


def _format_entity(entity: Dict[str, Any]) -> Dict[str, Any]:
    metadata = _decode_metadata(entity.get("metadata"))
    progress_percent = metadata.get("progress_percent")
    try:
        progress_percent = float(progress_percent)
    except (TypeError, ValueError):
        progress_percent = None

    if progress_percent is not None:
        progress_percent = max(0.0, min(100.0, progress_percent))

    progress_comment = metadata.get("progress_comment", "")
    if not isinstance(progress_comment, str):
        progress_comment = str(progress_comment or "")
    progress_comment = progress_comment.strip()

    return {
        "rowKey": entity.get("RowKey"),
        "problem": entity.get("problem"),
        "summary": entity.get("summary", ""),
        "html_content": entity.get("html_content", ""),
        "metadata": metadata,
        "created_at": entity.get("created_at"),
        "timestamp": entity.get("Timestamp"),
        "progress_percent": progress_percent,
        "progress_comment": progress_comment,
    }


def _history_context(problem: str, limit: int = 5) -> str:
    history, _ = get_iteration_slice(problem, offset=0, limit=limit)
    if not history:
        return "No prior progress recorded."

    lines: List[str] = []
    for entity in reversed(history):
        rowkey = entity.get("RowKey", "")
        summary = entity.get("summary", "").strip()
        timestamp = _rowkey_to_date(rowkey)
        if timestamp:
            label = timestamp.strftime("%Y-%m-%d %H:%M UTC")
        else:
            label = rowkey
        lines.append(f"- {label}: {summary or 'Summary unavailable'}")
    return "\n".join(lines)


async def _run_iteration(problem: str, today: datetime) -> Dict[str, Any]:
    agent = await get_open_deep_search_agent()
    history_snippet = _history_context(problem)

    system_prompt = f"""
You are Open Deep Search, a rigorous autonomous mathematical research agent.
Objective: Make tangible progress on the open problem provided.
Problem Statement: {problem}

You operate iteratively. Review the historical progress (if any), plan new avenues, research using the provided tools,
and provide a detailed update capturing proofs, counterexamples, heuristics, or partial results.
Treat this like you are collaborating with a research group—be truthful about uncertainties.

Historical Progress:
{history_snippet}

When you respond, output valid JSON with the keys:
- summary: concise, plain-text overview of today's advances (<= 4 sentences).
- html_content: HTML describing today's work. Use semantic tags (e.g., <section>, <h2>, <p>, <ul>) and include references inline.
- next_steps: array of 2-5 concrete follow-up actions.
- references: array of citation strings formatted as "Title — URL".
- progress_percent: number between 0 and 100 representing cumulative progress toward fully solving the problem.
- progress_comment: short (<= 120 characters) status note contextualizing the progress_percent value.

Never wrap the JSON in code fences.
"""

    user_prompt = f"""
Date (UTC): {today.strftime('%Y-%m-%d')}
Task: Continue the research and report today's progress. Cite every external claim with links.
"""

    agent_input = {"messages": [("system", system_prompt.strip()), ("user", user_prompt.strip())]}
    final_state = await agent.ainvoke(agent_input, {"recursion_limit": GRAPH_RECURSION_LIMIT})

    payload = _extract_message_container(final_state)

    final_message: Optional[str] = None
    if payload:
        messages = _maybe_overwrite_value(payload.get("messages") or [])
        final_message = _extract_assistant_reply(messages)

    if final_message is None:
        raise RuntimeError("The research agent did not return any content.")

    try:
        parsed = json.loads(final_message)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Agent response was not valid JSON: {final_message}") from exc

    html_content = parsed.get("html_content", "")
    summary = parsed.get("summary", "")
    raw_progress = parsed.get("progress_percent", None)
    try:
        progress_percent = float(raw_progress)
    except (TypeError, ValueError):
        progress_percent = None

    if progress_percent is not None:
        progress_percent = max(0.0, min(100.0, progress_percent))

    progress_comment = parsed.get("progress_comment", "")
    if not isinstance(progress_comment, str):
        progress_comment = str(progress_comment or "")
    progress_comment = progress_comment.strip()

    metadata = {
        "next_steps": parsed.get("next_steps", []),
        "references": parsed.get("references", []),
        "raw_response": parsed,
        "progress_percent": progress_percent,
        "progress_comment": progress_comment,
    }

    rowkey = _rowkey_now()
    save_iteration(
        problem=problem,
        rowkey=rowkey,
        html_content=html_content,
        summary=summary,
        metadata=metadata,
    )

    return _format_entity(
        {
            "RowKey": rowkey,
            "problem": problem,
            "summary": summary,
            "html_content": html_content,
            "metadata": json.dumps(metadata),
            "created_at": datetime.utcnow().isoformat(),
            "progress_percent": progress_percent,
            "progress_comment": progress_comment,
        }
    )


async def ensure_latest_iteration(problem: str) -> Dict[str, Any]:
    latest = latest_iteration(problem)
    if latest:
        timestamp = _rowkey_to_date(latest.get("RowKey", ""))
        if timestamp and timestamp.date() == datetime.utcnow().date():
            return _format_entity(latest)
    return await _run_iteration(problem, datetime.utcnow())


async def get_problem_history(
    problem: str,
    *,
    offset: int = 0,
    limit: int = DEFAULT_PAGE_SIZE,
    ensure_latest: bool = False,
) -> Dict[str, Any]:
    if ensure_latest and offset == 0:
        await ensure_latest_iteration(problem)

    slice_entries, next_offset = get_iteration_slice(problem, offset=offset, limit=limit)
    formatted = [_format_entity(entity) for entity in slice_entries]
    latest_progress = None
    latest_comment = ""
    if offset == 0 and formatted:
        latest_progress = formatted[0].get("progress_percent")
        latest_comment = formatted[0].get("progress_comment", "")
    return {
        "entries": formatted,
        "next_offset": next_offset,
        "progress_percent": latest_progress,
        "progress_comment": latest_comment,
    }
