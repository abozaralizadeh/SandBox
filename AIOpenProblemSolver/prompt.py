import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from AIOpenProblemSolver.azurestorage import (
    get_iteration_slice,
    latest_iteration,
    parse_iteration_rowkey,
    release_iteration_lock,
    rowkey_for_date,
    save_iteration,
    try_acquire_iteration_lock,
)
from AIOpenProblemSolver.graph import get_open_deep_search_agent
from AIOpenProblemSolver.memory import load_or_seed_memory, render_memory, update_memory

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


def _rowkey_for_today(problem: str, today: datetime) -> str:
    """The RowKey today's iteration must be saved under: the one this day already has, if
    any, so a re-run REPLACES the day's entry instead of adding a second one (the notebook
    shows one entry per day). Only a day with no entry yet gets a fresh key."""
    flat_date = today.strftime("%Y%m%d")
    return rowkey_for_date(problem, flat_date) or today.strftime("%Y%m%d_%H%M%S")


def _rowkey_to_date(rowkey: str) -> Optional[datetime]:
    return parse_iteration_rowkey(rowkey)


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


def _recent_summaries(problem: str, limit: int = 5) -> str:
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
    agent, browser_aclose = await get_open_deep_search_agent()
    history_snippet = _recent_summaries(problem)
    # The distilled record of what has already been tried — the part that keeps the agent
    # from spending today re-running an experiment it already ran (see memory.py).
    memory = load_or_seed_memory(problem)
    memory_snippet = render_memory(memory)

    system_prompt = f"""
You are Open Problem Solver, a creative autonomous mathematician working to solve one of the most important open problems in mathematics.

## Your Mission
Make tangible, ORIGINAL progress on the following problem. Do not merely summarize what others have done — push the frontier yourself.

## Problem Statement
{problem}

## Your Process
1. Review the RESEARCH MEMORY and historical progress (below) to understand what has already been attempted.
2. Identify the most promising direction that has NOT been fully explored.
3. Formulate a specific conjecture or approach for today's work.
4. USE YOUR COMPUTATIONAL TOOLS (python_math_sandbox, symbolic_calculator) to:
   - Test conjectures with concrete numerical examples
   - Explore pattern formation across parameter ranges
   - Search for counterexamples to claims
   - Verify proof steps computationally
   - Perform symbolic manipulations and simplifications
5. Develop rigorous arguments based on what you discover.
6. Use web search ALWAYS to verify a specific theorem, look up a known result, or check whether your approach has been attempted.

## Research Memory (accumulated across all previous iterations)
{memory_snippet}

## Historical Progress (recent daily summaries)
{history_snippet}

## Working With Your Memory (mandatory)
- The memory above is your own record from previous days. Treat it as fact.
- Do NOT re-run an experiment listed under "COMPUTATIONS ALREADY RUN" — reuse its stated result. If you genuinely need to revisit one, say why and change it substantively (wider range, different method, sharper hypothesis).
- Do NOT re-attempt an approach listed under "ALREADY TRIED AND FAILED" unless you have a concrete new idea that defeats the specific reason it failed — and state that reason and your fix explicitly.
- Build on "ESTABLISHED SO FAR" rather than re-deriving it, and prefer picking up an "OPEN LEAD" over inventing a fresh direction, unless you can argue the lead is exhausted.

## Expectations for Today
- Run at least 3-5 computational experiments using python_math_sandbox — each one NEW relative to the memory above
- Formulate at least one original conjecture or proof strategy
- If a direction seems unproductive, pivot and explain why
- Clearly mark what is proven vs. conjectured vs. speculative
- Be bold but honest about the significance of your findings

## Output Format
When you are finished, output valid JSON (no code fences) with these keys:
- summary: concise, plain-text overview of today's advances (<= 4 sentences). Focus on what YOU discovered or proved, not what you read online.
- html_content: HTML describing today's work. Use semantic tags (e.g., <section>, <h2>, <p>, <ul>, <pre>, <code>). Include your computations, conjectures, proof sketches, and any relevant code snippets.
- next_steps: array of 2-5 concrete follow-up actions, each describing a specific mathematical investigation to pursue.
- experiments: array of one-line strings, one per computation you actually ran today, each stating WHAT was computed (with the concrete ranges/parameters) and WHAT came out — e.g. "Counted sign changes of S(t) for t up to 10^5 via Riemann-Siegel — matched the predicted log log t growth". This is what stops tomorrow from recomputing today's work, so be specific.
- dead_ends: array of one-line strings for every approach you tried today that did NOT work, each naming the approach AND the concrete reason it failed. Empty array if nothing failed — but a day with no failures is rare, so be honest.
- established: array of one-line strings for results you consider settled after today (proven, or verified strongly enough to build on). Mark proven vs. verified.
- references: array of citation strings formatted as "Title — URL". Include only sources you actually consulted.
- progress_percent: number between 0 and 100 representing cumulative progress toward fully solving the problem. Be conservative and honest — a genuine novel partial result might be 0.1-1%. Do not inflate.
- progress_comment: short (<= 120 characters) status note contextualizing the progress_percent value. Describe what was achieved, not what was attempted.

Never wrap the JSON in code fences.
"""

    user_prompt = f"""
Date (UTC): {today.strftime('%Y-%m-%d')}
Task: Make original progress on the problem today. Think creatively, compute extensively, and develop your own mathematical insights. Cite external sources only when you actually use them.
"""

    agent_input = {"messages": [("system", system_prompt.strip()), ("user", user_prompt.strip())]}
    try:
        final_state = await agent.ainvoke(agent_input, {"recursion_limit": GRAPH_RECURSION_LIMIT})
    finally:
        # Tear down the Playwright browser/driver inside this event loop, before
        # Flask's per-request loop closes — prevents 'Event loop is closed' on GC.
        # Swallow teardown errors so they can't mask an agent exception in flight.
        try:
            await browser_aclose()
        except Exception as cleanup_error:
            print("Browser cleanup failed:", cleanup_error)

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
        "experiments": parsed.get("experiments", []),
        "dead_ends": parsed.get("dead_ends", []),
        "established": parsed.get("established", []),
        "raw_response": parsed,
        "progress_percent": progress_percent,
        "progress_comment": progress_comment,
    }

    rowkey = _rowkey_for_today(problem, today)
    save_iteration(
        problem=problem,
        rowkey=rowkey,
        html_content=html_content,
        summary=summary,
        metadata=metadata,
    )

    # Fold today's findings into the long-term memory (compacting it when it has grown
    # past its budget). Best-effort by design — the day's iteration is already saved.
    await update_memory(problem, parsed, stamp=today.strftime("%Y-%m-%d"), base=memory)

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


def _has_iteration_for_today(problem: str) -> Optional[Dict[str, Any]]:
    latest = latest_iteration(problem)
    if latest:
        timestamp = _rowkey_to_date(latest.get("RowKey", ""))
        if timestamp and timestamp.date() == datetime.utcnow().date():
            return latest
    return None


async def ensure_latest_iteration(problem: str) -> Optional[Dict[str, Any]]:
    """Run today's iteration unless it exists or another worker is already running it.

    An iteration takes minutes and is triggered from a blocking request, so without the
    single-flight lock every concurrent visitor — across 4 gunicorn workers — starts its
    own run and the day ends up with several notebook entries. Callers that lose the race
    get None and simply serve the history that is already there."""
    existing = _has_iteration_for_today(problem)
    if existing:
        return _format_entity(existing)

    if not try_acquire_iteration_lock(problem):
        return None   # another worker is generating today's iteration right now

    try:
        # Re-check inside the lock: the run we queued behind may have just finished.
        existing = _has_iteration_for_today(problem)
        if existing:
            return _format_entity(existing)
        return await _run_iteration(problem, datetime.utcnow())
    finally:
        release_iteration_lock(problem)


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
