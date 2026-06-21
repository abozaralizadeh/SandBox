"""Real-world research step for GenBox decisions, using the model's NATIVE web search.

Mirrors how AIBlog (and ComicBook) search the live web: an Azure OpenAI model on the
Responses API is given the built-in ``{"type": "web_search"}`` tool, so the search runs
server-side and a single call returns a synthesized, source-cited briefing — no Tavily or
other third-party search API.

Once a day's *topic* has been chosen, this gathers the current real-world state of that
topic — recent achievements and progress, plus the open challenges, blockers, and hard
limits — so the detailed decision that follows can propose concrete solutions to real
problems instead of abstract policy.

Strictly best-effort: any failure (or a model/endpoint without web search) returns an
empty briefing so the daily decision is never blocked.
"""

import os
import re
from datetime import datetime

from dotenv import load_dotenv
from langsmith import traceable

# Ensure AZURE_OPENAI_* are available regardless of import order.
load_dotenv()

_URL_RE = re.compile(r"https?://[^\s\)\]\}>\"'`]+")
_MAX_BRIEFING_CHARS = 4000
_MAX_SOURCES = 10


def _build_llm():
    """An Azure chat model on the Responses API with the native web_search tool bound.

    Mirrors AIBlog/graph.py's config (same deployment, api version, responses/v1 output)."""
    from langchain_openai import AzureChatOpenAI

    llm = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_MODEL"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        output_version="responses/v1",
        use_responses_api=True,
        temperature=1,
        max_retries=2,
        timeout=90,
    )
    return llm.bind_tools([{"type": "web_search"}])


def _flatten_text(payload):
    """Pull the assistant's *answer* text out of a responses/v1 message.

    Content is a list of blocks; only ``text``/``output_text`` blocks hold the answer.
    We deliberately do NOT recurse into other block types (reasoning, web_search_call,
    ids, search queries) — that would pollute the briefing with tool-call internals."""
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, (list, tuple)):
        return "".join(_flatten_text(item) for item in payload)
    if isinstance(payload, dict):
        if payload.get("type") in ("text", "output_text") and isinstance(payload.get("text"), str):
            return payload["text"]
        return ""
    return ""


def _extract_sources(message, text):
    """Best-effort source URLs: prefer the web_search url_citation annotations on the
    message; fall back to any URLs the model wrote inline in the briefing."""
    urls = []

    def _walk(node):
        if isinstance(node, dict):
            if node.get("type") in ("url_citation", "citation") or "url" in node:
                u = node.get("url")
                if isinstance(u, str) and u.startswith("http"):
                    urls.append(u)
            for v in node.values():
                _walk(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                _walk(v)

    try:
        _walk(getattr(message, "content", None))
        _walk(getattr(message, "additional_kwargs", None))
        _walk(getattr(message, "response_metadata", None))
    except Exception:
        pass

    # Merge in any URLs the model wrote inline (annotations first for priority); dedup below.
    urls += _URL_RE.findall(text or "")

    # Dedup, preserve order, strip trailing punctuation, cap.
    seen, cleaned = set(), []
    for u in urls:
        u = u.rstrip(".,);]")
        if u and u not in seen:
            seen.add(u)
            cleaned.append(u)
    return [{"title": "", "url": u} for u in cleaned[:_MAX_SOURCES]]


@traceable(run_type="tool", name="GenBox Real-World Research")
def research_real_world(topic, date=None):
    """Web-search the current real-world state of an already-chosen `topic`.

    Returns {"briefing": str, "sources": [{"title", "url"}]}. `briefing` is "" when web
    search is unavailable or returns nothing — callers must treat research as optional.
    """
    topic = (topic or "").strip()
    if not topic:
        return {"briefing": "", "sources": []}
    year = (date or datetime.now()).year

    prompt = (
        f'Use web search to research the CURRENT real-world state of this topic, as of {year}:\n\n'
        f'"{topic}"\n\n'
        "Report concrete, recent, well-sourced facts and figures under three headings:\n"
        "1. Recent real-world achievements and progress.\n"
        "2. Current challenges, obstacles, and blockers.\n"
        "3. Hard limits and unsolved problems.\n\n"
        "Keep it under ~400 words and include the source URLs you relied on."
    )

    try:
        message = _build_llm().invoke(
            [
                (
                    "system",
                    "You are a research assistant. Search the live web and report only "
                    "current, well-sourced facts — no speculation.",
                ),
                ("human", prompt),
            ]
        )
    except Exception as e:  # endpoint without web search, network, auth, etc.
        print(f"GenBox research: web search failed ({e}); proceeding ungrounded")
        return {"briefing": "", "sources": []}

    briefing = _flatten_text(getattr(message, "content", "")).strip()
    if len(briefing) > _MAX_BRIEFING_CHARS:
        briefing = briefing[:_MAX_BRIEFING_CHARS].rsplit("\n", 1)[0] + "\n…(truncated)"
    sources = _extract_sources(message, briefing)
    if not briefing:
        print("GenBox research: empty web-search result; proceeding ungrounded")
    return {"briefing": briefing, "sources": sources}
