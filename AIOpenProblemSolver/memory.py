"""Accumulated research memory for the Open Problem Solver.

The agent gets a fresh context every day, so the only thing standing between it and
re-running yesterday's experiment is what we put in its prompt. A rolling window of the
last few one-line summaries is not enough: summaries describe *what was achieved*, and
what prevents wasted work is the record of what was **tried and failed**.

So each iteration folds its own report into a per-problem memory document with four
buckets:

* `established` — results proven/verified, safe to build on without redoing.
* `dead_ends`   — approaches that did NOT work, with the reason. The anti-repetition core.
* `experiments` — computations already run and their outcome, so they are not recomputed.
* `open_leads`  — promising directions not yet explored (tomorrow's starting points).

The document only grows, so it is **compacted** once it crosses `AIOPS_MEMORY_MAX_ITEMS`
or `AIOPS_MEMORY_MAX_CHARS`: an LLM pass merges near-duplicates and generalizes clusters
of related entries while being told to preserve every distinct dead end and established
result. If that pass fails for any reason, a deterministic trim runs instead, so memory is
bounded even when the model is unavailable. Compaction never raises into the iteration —
a failed compaction must not lose a day's research.
"""
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from AIOpenProblemSolver.azurestorage import load_memory_record, save_memory_record

BUCKETS = ("established", "dead_ends", "experiments", "open_leads")

# Compaction thresholds. Items are short one-liners, so these are generous; the point is
# to bound the prompt, not to keep it tiny.
MAX_ITEMS = int(os.getenv("AIOPS_MEMORY_MAX_ITEMS", "80"))
MAX_CHARS = int(os.getenv("AIOPS_MEMORY_MAX_CHARS", "12000"))
# Per-bucket caps used by the deterministic fallback trim, in priority order: dead ends
# and established results are what prevent repeated work, so they survive longest.
FALLBACK_CAPS = {"dead_ends": 30, "established": 25, "experiments": 20, "open_leads": 10}
# Compacted entries legitimately get longer than raw ones (a merged entry enumerates what
# it covers), so this cap has to leave room for that — total size is bounded by MAX_CHARS.
ITEM_MAX_CHARS = int(os.getenv("AIOPS_MEMORY_ITEM_MAX_CHARS", "600"))


def _empty() -> Dict[str, Any]:
    return {b: [] for b in BUCKETS} | {"version": 0, "iterations_covered": 0,
                                       "compacted_at": None, "updated_at": None}


def _strip_stamp(text: str) -> str:
    """Drop the leading `[YYYY-MM-DD] ` tag stored items carry, so an item is compared on
    its content and not on the day it was first recorded."""
    stripped = text.lstrip()
    if stripped.startswith("[") and "]" in stripped:
        return stripped[stripped.index("]") + 1:].strip()
    return stripped


def _normalize(text: str) -> str:
    """Comparison key for deduplication — stamp-, case- and punctuation-insensitive."""
    text = _strip_stamp(text).lower()
    return " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in text).split())


def _coerce_items(raw: Any) -> List[str]:
    """Accept the shapes a model actually emits for these fields: a list of strings, a
    list of {description, outcome}-ish dicts, or a single string."""
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if isinstance(raw, dict):
        raw = [raw]
    items: List[str] = []
    for entry in raw if isinstance(raw, list) else []:
        if isinstance(entry, str):
            text = entry
        elif isinstance(entry, dict):
            head = entry.get("description") or entry.get("approach") or entry.get("statement") \
                or entry.get("experiment") or entry.get("lead") or entry.get("title") or ""
            tail = entry.get("outcome") or entry.get("result") or entry.get("reason") \
                or entry.get("why") or entry.get("status") or ""
            text = f"{head} — {tail}" if head and tail else (head or tail or json.dumps(entry, ensure_ascii=False))
        else:
            text = str(entry)
        text = " ".join(str(text).split())
        if text:
            items.append(text[:ITEM_MAX_CHARS])
    return items


def load_memory(problem: str) -> Dict[str, Any]:
    record = load_memory_record(problem) or {}
    memory = _empty()
    for bucket in BUCKETS:
        memory[bucket] = _coerce_items(record.get(bucket))
    for key in ("version", "iterations_covered", "compacted_at", "updated_at"):
        if record.get(key) is not None:
            memory[key] = record[key]
    return memory


SEED_ITERATIONS = int(os.getenv("AIOPS_MEMORY_SEED_ITERATIONS", "30"))
SEED_ITEM_CHARS = 300


def _first_sentences(text: str, limit: int = SEED_ITEM_CHARS) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    stop = cut.rfind(". ")
    return (cut[:stop + 1] if stop > limit // 2 else cut).strip()


def seed_memory(problem: str) -> Dict[str, Any]:
    """Bootstrap the memory from iterations that predate it.

    A problem can already have hundreds of stored iterations, and an empty memory would
    take months to become useful. Seeding distils the recent frontier — the last
    `AIOPS_MEMORY_SEED_ITERATIONS` days' summaries as established work, plus the newest
    day's next steps as open leads. Dead ends stay empty: older iterations never recorded
    which approaches failed, and inventing them would be worse than leaving them out."""
    from AIOpenProblemSolver.azurestorage import get_iteration_slice, iteration_date_key

    memory = _empty()
    try:
        history, _ = get_iteration_slice(problem, offset=0, limit=SEED_ITERATIONS)
    except Exception as exc:
        print(f"AIOPS memory seeding failed: {exc}")
        return memory
    if not history:
        return memory

    established: List[str] = []
    seen = set()
    for entity in reversed(history):          # oldest first, so stamps read chronologically
        summary = _first_sentences(entity.get("summary", ""))
        if not summary or _normalize(summary) in seen:
            continue
        seen.add(_normalize(summary))
        day = iteration_date_key(entity.get("RowKey", ""))
        stamp = f"{day[:4]}-{day[4:6]}-{day[6:]}" if len(day) == 8 else day
        established.append(f"[{stamp}] {summary}")

    memory["established"] = established
    newest_metadata = json.loads(history[0].get("metadata") or "{}") if isinstance(history[0].get("metadata"), str) else (history[0].get("metadata") or {})
    memory["open_leads"] = _coerce_items(newest_metadata.get("next_steps"))
    memory["iterations_covered"] = len(history)
    memory["updated_at"] = datetime.utcnow().isoformat()
    return memory


def load_or_seed_memory(problem: str) -> Dict[str, Any]:
    """The memory to work from: what is stored, or a seed distilled from prior iterations
    when nothing is stored yet. The seed is not persisted here — the next `update_memory`
    saves it together with that day's findings."""
    memory = load_memory(problem)
    if any(memory.get(bucket) for bucket in BUCKETS):
        return memory
    return seed_memory(problem)


def render_memory(memory: Dict[str, Any]) -> str:
    """The prompt block. Ordered so the anti-repetition sections come first — the agent
    reads the "do not redo" material before it starts planning today's work."""
    labels = [
        ("dead_ends", "ALREADY TRIED AND FAILED — do not repeat these (pivot or vary substantially)"),
        ("experiments", "COMPUTATIONS ALREADY RUN — reuse these results instead of recomputing"),
        ("established", "ESTABLISHED SO FAR — treat as settled and build on top"),
        ("open_leads", "OPEN LEADS — unexplored directions from previous days"),
    ]
    sections: List[str] = []
    for bucket, heading in labels:
        items = memory.get(bucket) or []
        if items:
            sections.append(heading + ":\n" + "\n".join(f"- {item}" for item in items))
    if not sections:
        return "No accumulated research memory yet — this is the first iteration."
    covered = memory.get("iterations_covered") or 0
    header = f"(distilled from {covered} previous iteration{'s' if covered != 1 else ''})"
    return f"{header}\n\n" + "\n\n".join(sections)


def _merge(memory: Dict[str, Any], parsed: Dict[str, Any], stamp: str) -> Dict[str, Any]:
    """Fold one iteration's report into the memory, newest last, without duplicates."""
    incoming = {
        "established": _coerce_items(parsed.get("established") or parsed.get("established_results")),
        "dead_ends": _coerce_items(parsed.get("dead_ends") or parsed.get("failed_approaches")),
        "experiments": _coerce_items(parsed.get("experiments") or parsed.get("computations")),
        "open_leads": _coerce_items(parsed.get("next_steps")),
    }
    for bucket, items in incoming.items():
        existing = list(memory.get(bucket) or [])
        seen = {_normalize(item) for item in existing}
        for item in items:
            stamped = f"[{stamp}] {item}"
            if _normalize(item) not in seen:
                existing.append(stamped)
                seen.add(_normalize(item))
        memory[bucket] = existing

    # Today's open leads supersede yesterday's: a lead the agent stopped restating has
    # either been taken up (it is an experiment/dead end now) or been abandoned.
    if incoming["open_leads"]:
        memory["open_leads"] = [f"[{stamp}] {item}" for item in incoming["open_leads"]]

    memory["iterations_covered"] = int(memory.get("iterations_covered") or 0) + 1
    memory["updated_at"] = datetime.utcnow().isoformat()
    return memory


def _size(memory: Dict[str, Any]) -> tuple:
    items = sum(len(memory.get(b) or []) for b in BUCKETS)
    chars = len(json.dumps({b: memory.get(b) for b in BUCKETS}, ensure_ascii=False))
    return items, chars


def needs_compaction(memory: Dict[str, Any]) -> bool:
    items, chars = _size(memory)
    return items > MAX_ITEMS or chars > MAX_CHARS


def _fallback_compact(memory: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic trim: keep the most recent items in each bucket, under caps that
    favour the buckets which prevent repeated work. Used when the LLM pass is unavailable
    or returns something unusable — memory stays bounded either way."""
    for bucket, cap in FALLBACK_CAPS.items():
        items = memory.get(bucket) or []
        if len(items) > cap:
            memory[bucket] = items[-cap:]
    return memory


def _compaction_model():
    """A plain chat model for the compaction pass (no tools, no agent). Deliberately does
    not set `temperature`: the shared deployment is a reasoning model, which rejects it."""
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        model=os.environ["AZURE_OPENAI_MODEL"],
        model_provider="azure_openai",
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        max_retries=2,
    )


_COMPACTION_PROMPT = """You are compacting the long-term research memory of an autonomous mathematician working on this open problem:

{problem}

Below is the memory, which has grown too long. Rewrite it as a SHORTER document with the same four buckets.

Rules:
- PRESERVE EVERY DISTINCT dead end and established result. These are what stop the mathematician from wasting days redoing failed work — losing one is the worst possible outcome of this task.
- Merge near-duplicates and near-identical entries into a single, more general entry that still ENUMERATES the specific approaches/parameters/ranges it covers (e.g. "Numerically verified for N up to 10^6 across three zero-counting methods (Riemann-Siegel, Turing, Odlyzko-Schonhage) — no counterexample"). A merged entry that loses the specifics is worse than no merge: the mathematician must still be able to tell whether a given experiment was already run.
- Only merge entries that are genuinely about the same approach. Do NOT summarize the memory — this is a compaction, not an overview.
- Drop entries that are pure narration, vague, or fully implied by another entry.
- Keep each entry to one line, concrete and self-contained. Preserve the leading [date] tag of the newest entry a merged item came from.
- Aim for between {min_items} and {target_items} entries in total.

Current memory (JSON):
{memory_json}

Respond with ONLY a JSON object with keys "established", "dead_ends", "experiments", "open_leads", each an array of strings. No code fences, no commentary."""


async def compact_memory(problem: str, memory: Dict[str, Any]) -> Dict[str, Any]:
    """LLM compaction with a deterministic fallback. Never raises."""
    target_items = max(12, MAX_ITEMS // 2)
    before_dead_ends = len(memory.get("dead_ends") or [])
    try:
        model = _compaction_model()
        response = await model.ainvoke(_COMPACTION_PROMPT.format(
            problem=problem,
            target_items=target_items,
            min_items=max(6, target_items // 2),
            memory_json=json.dumps({b: memory.get(b) for b in BUCKETS}, ensure_ascii=False),
        ))
        content = getattr(response, "content", response)
        if isinstance(content, list):  # some providers return content blocks
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        text = str(content).strip()
        if text.startswith("```"):     # tolerate fences even though we asked for none
            text = text.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
        parsed = json.loads(text[text.index("{"):text.rindex("}") + 1])
        compacted = {b: _coerce_items(parsed.get(b)) for b in BUCKETS}
        if not any(compacted.values()):
            raise ValueError("compaction returned an empty memory")
        # A compaction that flattened the dead ends into a paragraph-sized summary defeats
        # the purpose (the agent can no longer tell what it already ruled out), so reject
        # anything that collapsed them that far and keep the detail-preserving trim instead.
        floor = 3 if before_dead_ends >= 8 else 1
        if before_dead_ends and len(compacted["dead_ends"]) < min(floor, before_dead_ends):
            raise ValueError(
                f"compaction collapsed {before_dead_ends} dead ends into {len(compacted['dead_ends'])}"
            )
        memory.update(compacted)
    except Exception as exc:
        print(f"AIOPS memory compaction failed ({exc}); falling back to deterministic trim")
        memory = _fallback_compact(memory)

    memory["version"] = int(memory.get("version") or 0) + 1
    memory["compacted_at"] = datetime.utcnow().isoformat()
    return memory


async def update_memory(
    problem: str,
    parsed: Dict[str, Any],
    stamp: Optional[str] = None,
    base: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fold today's report into the memory, compact it if it has outgrown its budget, and
    persist. Best-effort: a memory failure must never cost the iteration that produced it.

    `base` is the memory the iteration actually ran against. Pass it — re-loading here would
    re-seed from a history that now contains the row this very iteration just saved, folding
    today's summary back in as if it were prior work."""
    stamp = stamp or datetime.utcnow().strftime("%Y-%m-%d")
    try:
        memory = _merge(base if base is not None else load_or_seed_memory(problem), parsed, stamp)
        if needs_compaction(memory):
            memory = await compact_memory(problem, memory)
        save_memory_record(problem, memory)
        return memory
    except Exception as exc:
        print(f"AIOPS memory update failed: {exc}")
        return {}
