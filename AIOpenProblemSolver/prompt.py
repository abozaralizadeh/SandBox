import json
import re
import os
import threading
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
from AIOpenProblemSolver.memory import (
    MAX_LINE_DAYS,
    active_line,
    line_is_overdue,
    load_or_seed_memory,
    render_attack_lines,
    render_memory,
    update_memory,
)

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


# Words that say nothing about WHICH object is being studied, so they must not count as
# evidence that the agent is circling the same thing. The list is deliberately long: a first
# cut without it flagged 32 of 34 windows on real history, because "formula", "coefficient",
# "progress" and the problem's own name appear nearly every day and mean nothing.
_RUT_STOPWORDS = frozenset("""
about above after again against along among around because before being below between both
could derived derive derivation during each either every exact exactly first found further
general given gives having identity implies indeed instead into itself large later least
level limit local lower making matching might model models normalized number numerical
numerically order orders other output problem proof proved proving range ratio ratios
result results same second several should showed shows simple since slope small still
stronger structure term terms tested testing their there these third this those
three through today took toward under until using value values verified version where
which while whose within without would yield yields
analysis analytic approach argument asymptotic asymptotics bound bounded bounds class closed
coefficient coefficients compact computation computations computed conditional conjecture
conjectured connection constant continuous criterion decomposition derivative derivatives
discrete dominant established estimate evidence exhibit expansion explicit extended extension
finite formula formulas function functions global high-precision higher implication
inequality inequalities integral method negative positive positivity precise precision
problem product progress proves quantity recurrence region relation representation sequence series
sharp sharper statement strong sufficient summand symbolic theorem theory transform uniform
""".split())


def _problem_words(problem: str) -> set:
    """The problem's own vocabulary. "riemann" appears in almost every summary of the Riemann
    work, so it is evidence of nothing."""
    return {w for w in re.findall(r"[a-z][a-z-]{4,}", (problem or "").lower())}


def _distinctive_terms(summaries: List[str], skip: set, min_share: float = 0.7) -> set:
    """Content words shared by most of a window of summaries."""
    if not summaries:
        return set()
    counts: Dict[str, int] = {}
    for text in summaries:
        words = {w for w in re.findall(r"[a-z][a-z-]{4,}", text)
                 if w not in _RUT_STOPWORDS and w not in skip}
        for w in words:
            counts[w] = counts.get(w, 0) + 1
    threshold = max(2, int(len(summaries) * min_share))
    return {w for w, c in counts.items() if c >= threshold}


def _rut_terms(problem: str, days: int) -> tuple:
    """Terms the agent has been circling for TWO windows running, and how many days that spans.

    A deterministic backstop for the attack-line ledger, which believes the label the agent
    gives its own work. This reads what it actually wrote instead. The test is deliberately a
    change detector, not a repetition detector: it asks whether the distinctive vocabulary of
    the last `days` is the SAME as the vocabulary of the `days` before that. One window on its
    own is always repetitive, which is just what focused work looks like.
    """
    history, _ = get_iteration_slice(problem, offset=0, limit=days * 2)
    summaries = [(e.get("summary") or "").lower() for e in history]
    summaries = [t for t in summaries if t]
    if len(summaries) < days + 2:
        return (), 0

    skip = _problem_words(problem)
    recent = _distinctive_terms(summaries[:days], skip)
    earlier = _distinctive_terms(summaries[days:], skip)
    carried = recent & earlier
    return tuple(sorted(carried))[:6], len(summaries)


def _latest_progress(problem: str) -> str:
    """The last cumulative progress figure, for the prompt to anchor against.

    Without it the agent invents a fresh number every day, so the page's headline bounced
    0.3 -> 0.1 -> 0.01 -> 0.02 and looked like the work was going backwards.
    """
    try:
        history, _ = get_iteration_slice(problem, offset=0, limit=1)
    except Exception:
        return "unknown"
    if not history:
        return "none yet (this is the first iteration)"
    meta = _decode_metadata(history[0].get("metadata", ""))
    value = meta.get("progress_percent")
    try:
        return f"{float(value):g}%"
    except (TypeError, ValueError):
        return "unknown"


async def _run_iteration(problem: str, today: datetime) -> Dict[str, Any]:
    agent, browser_aclose = await get_open_deep_search_agent()
    history_snippet = _recent_summaries(problem)
    # The distilled record of what has already been tried — the part that keeps the agent
    # from spending today re-running an experiment it already ran (see memory.py).
    memory = load_or_seed_memory(problem)
    memory_snippet = render_memory(memory)
    lines_snippet = render_attack_lines(memory)
    previous_progress = _latest_progress(problem)

    # Two independent reads on "are we going in circles": the ledger (what the agent said it
    # was working on) and the summaries (what it actually wrote). Either one can raise the flag.
    rut_words, rut_days = _rut_terms(problem, days=max(MAX_LINE_DAYS, 7))
    stuck_by_ledger = line_is_overdue(memory)
    # One distinctive word (not generic, not the problem's own name) surviving two full
    # windows means the same object is still being turned over. On the real history that
    # fires across the two genuine ruts and stays quiet while the work is actually moving.
    stuck_by_text = len(rut_words) >= 1

    if stuck_by_ledger or stuck_by_text:
        current = active_line(memory)
        evidence = []
        if stuck_by_ledger and current:
            evidence.append(
                f"the line \"{current.get('label')}\" has had {current.get('iterations')} "
                f"consecutive days"
            )
        if stuck_by_text:
            evidence.append(
                f"across the last {rut_days} days the same subject keeps coming back "
                f"({', '.join(rut_words)}), so the vocabulary has not moved in two weeks"
            )
        pivot_directive = f"""
## STOP. YOU ARE IN A RUT. (this overrides the preference for continuing an open lead)
Evidence: {'; and '.join(evidence)}.

You have been refining one object instead of attacking the problem. Each day you proved a
property of something you invented the day before, and the distance to the actual problem
grew. That is not progress, however rigorous each step was.

Today you may NOT continue that line. Choose exactly one:
  (a) CLOSE IT OUT. State the single theorem the line was meant to deliver, prove it or show
      it is false, and say plainly what it gives you toward the problem. No new sub-objects.
  (b) ABANDON IT. Say in one line why it will not reach the problem, add it to dead_ends,
      and open a genuinely different attack. Different means a different technique family,
      not a different parameter of the same construction.

Set `focus` to the new line's label. If you pick (a) and finish, `focus` is the new line you
open afterwards. Ignoring this and producing another refinement of the same object is the
one outcome that counts as a wasted day.
"""
    else:
        pivot_directive = ""

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

## Lines of Attack (the agenda, tracked across days)
{lines_snippet}

## Historical Progress (recent daily summaries)
{history_snippet}
{pivot_directive}
## Working With Your Memory (mandatory)
- The memory above is your own record from previous days. Treat it as fact.
- Do NOT re-run an experiment listed under "COMPUTATIONS ALREADY RUN" — reuse its stated result. If you genuinely need to revisit one, say why and change it substantively (wider range, different method, sharper hypothesis).
- Do NOT re-attempt an approach listed under "ALREADY TRIED AND FAILED" unless you have a concrete new idea that defeats the specific reason it failed — and state that reason and your fix explicitly.
- Build on "ESTABLISHED SO FAR" rather than re-deriving it, and prefer picking up an "OPEN LEAD" over inventing a fresh direction, unless you can argue the lead is exhausted.
- A follow-up is only worth taking if it shortens the distance to the PROBLEM. Refining an object you invented yourself, and then refining that, is how a research programme dies: each step is rigorous and the whole goes nowhere. Before adopting a lead, say what it would give you toward the problem if it worked out. If you cannot, drop it and pick something else.
- Every few days, step back and ask the blunt question: if today's result were fully proved, would anyone be closer to settling the problem? If the answer is no several days running, the line is wrong, not just slow.

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
- focus: a SHORT label (<= 80 chars) naming the line of attack you worked today, e.g. "Turan inequalities for xi coefficients". Reuse the current line's label verbatim if you continued it; use a new label when you open a new line. This is how the agenda above is tracked, so be consistent.
- relation_to_problem: one line stating what today's result gives you toward the problem itself, if anything. Say "nothing direct yet, this is groundwork for X" when that is the truth. Do not dress up a property of an auxiliary object as progress on the problem.
- progress_percent: number between 0 and 100 representing CUMULATIVE progress toward fully solving the problem, counting everything established to date, not just today. The last recorded value was {previous_progress}. It should normally be greater than or equal to that. Going below it means you are retracting earlier work, which is legitimate but must be said in progress_comment. Be conservative and honest: a genuine novel partial result moves this by a fraction of a percent. Do not inflate.
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
        "focus": str(parsed.get("focus") or "").strip(),
        "relation_to_problem": str(parsed.get("relation_to_problem") or "").strip(),
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


def _run_iteration_in_background(problem: str) -> None:
    """Own the lock for the whole run, in this thread's own event loop.

    `_run_iteration` drives Playwright, so it needs a loop of its own here (the same shape as
    GenBox's `_run_video_generation`); the browser is torn down inside that loop by
    `_run_iteration`'s own `finally`."""
    import asyncio

    try:
        asyncio.run(_run_iteration(problem, datetime.utcnow()))
    except Exception as exc:  # noqa: BLE001 - a failed iteration must not take the app down
        print(f"AIOPS iteration failed for {problem!r}: {type(exc).__name__}: {exc}")
    finally:
        release_iteration_lock(problem)


async def ensure_latest_iteration(problem: str) -> Optional[Dict[str, Any]]:
    """Start today's iteration if it is missing, WITHOUT blocking the request.

    An iteration takes 2-10 minutes (measured in LangSmith). Azure App Service cuts any HTTP
    request at ~230s, so running it inline meant the reader always got a 500 — "Failed to load
    progress" — even though the work carried on server-side and the entry appeared later. Worse,
    every timed-out attempt released the lock on its way out and the next visitor started
    ANOTHER 5-10 minute run: 11 overlapping iterations on 2026-09-07 alone.

    So this only kicks the work off in a background daemon thread and returns immediately; the
    page renders the stored history and picks the new entry up on a later load. The
    single-flight lock is still what stops 4 gunicorn workers starting four runs, and the
    background thread now owns it for the run's whole life instead of only for the request's.
    """
    existing = _has_iteration_for_today(problem)
    if existing:
        return _format_entity(existing)

    try:
        if not try_acquire_iteration_lock(problem):
            return None   # another worker is generating today's iteration right now
    except Exception as exc:  # noqa: BLE001 - storage hiccup must not 500 the page
        print(f"AIOPS could not take the iteration lock: {type(exc).__name__}: {exc}")
        return None

    # Re-check inside the lock: the run we queued behind may have just finished.
    existing = _has_iteration_for_today(problem)
    if existing:
        release_iteration_lock(problem)
        return _format_entity(existing)

    threading.Thread(target=_run_iteration_in_background, args=(problem,), daemon=True).start()
    return None


async def get_problem_history(
    problem: str,
    *,
    offset: int = 0,
    limit: int = DEFAULT_PAGE_SIZE,
    ensure_latest: bool = False,
) -> Dict[str, Any]:
    generating = False
    if ensure_latest and offset == 0:
        # Non-blocking: returns None while today's iteration runs in the background.
        generating = await ensure_latest_iteration(problem) is None

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
        # True when today's entry is still being produced in the background, so the page can
        # say "in progress" rather than implying the archive is all there is.
        "generating": generating,
    }
