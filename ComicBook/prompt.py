import os
from datetime import datetime
from typing import Optional
import re

from AIBlog.prompt import _extract_text_from_last_message
from ComicBook.azurestorage import (
    ensure_active_arc,
    get_episode_by_date,
    get_recent_episodes,
    save_episode,
)
from ComicBook.graph import react_graph
from ComicBook.comic_graph import comic_graph
from utils import get_flat_date

MODE = os.environ.get("COMICBOOK_MODE", "comic_graph")


def _summarize_recent(arc_id: str, limit: int = 3) -> str:
    episodes = get_recent_episodes(arc_id, limit=limit)
    if not episodes:
        return "No prior episodes—this is a fresh arc kickoff."

    lines = []
    for ep in episodes:
        recap = ep.get("story_summary", "")
        panels = ep.get("panel_notes", "")
        day = ep.get("episode_number", "?")
        row_date = ep.get("RowKey", "")
        lines.append(
            f"Day {day} ({row_date}): {recap}\nPanel highlights: {panels}"
        )
    return "\n".join(lines)


def _extract_first_image_url(html: str) -> str:
    if not html:
        return ""
    match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', html, re.IGNORECASE)
    if match:
        return match.group(1)
    return ""


def _last_image_url(arc_id: str) -> str:
    episodes = get_recent_episodes(arc_id, limit=1)
    if not episodes:
        return ""
    html = episodes[0].get("html_content", "")
    return _extract_first_image_url(html)


def _build_context_message(arc: dict, target_date: datetime) -> str:
    arc_title = arc.get("title", "ComicBook Arc")
    logline = arc.get("logline", "Story arc guided by AI.")
    start_date = arc.get("start_date", "")
    target_days = arc.get("target_days", 7)
    episode_count = int(arc.get("episodes_count", 0))
    prior = _summarize_recent(arc["RowKey"])
    return (
        f"You are producing the ComicBook daily strip.\n"
        f"Arc title: {arc_title}\n"
        f"Logline: {logline}\n"
        f"Start date: {start_date}\n"
        f"Target length: {target_days} days.\n"
        f"Today's date: {target_date.strftime('%Y-%m-%d')} (episode {episode_count + 1}).\n"
        f"Recent continuity:\n{prior}\n"
        "Respect tone, characters, and callbacks while keeping today's strip satisfying on its own."
    )


def _parse_final_messages(state) -> tuple[str, str, str]:
    messages = state["messages"]
    html = messages[-1].content
    script_summary = ""
    panel_notes = ""
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if not script_summary and ("Panel" in content or "panel" in content):
            script_summary = content
        if not panel_notes and "image" in content.lower():
            panel_notes = content
        if script_summary and panel_notes:
            break
    return html, script_summary, panel_notes


def get_comicbook(parsed_date: Optional[datetime] = None):
    target_date = parsed_date or datetime.utcnow()
    flat_date = get_flat_date(target_date)

    if cached := get_episode_by_date(flat_date):
        return cached.get("html_content", ""), target_date, cached.get("PartitionKey")

    arc = ensure_active_arc(target_date=target_date, min_days=7, max_days=10)
    arc_title = arc.get("title", "ComicBook Arc")
    arc_logline = arc.get("logline", "AI-driven comic adventure.")
    arc_target = int(arc.get("target_days", 7))
    arc_day = int(arc.get("episodes_count", 0)) + 1
    previous_beats = _summarize_recent(arc["RowKey"])
    last_image_url = _last_image_url(arc["RowKey"])

    context_message = _build_context_message(arc, target_date)
    initial_state = {
        "messages": [("system", context_message)],
        "arc_title": arc_title,
        "arc_logline": arc_logline,
        "arc_day": arc_day,
        "arc_target": arc_target,
        "previous_beats": previous_beats,
        "last_image_url": last_image_url,
    }

    latest_layout_state = None
    storyteller_notes = ""
    cartoonist_notes = ""

    if MODE == "comic_graph":
        for event in comic_graph.stream(initial_state, subgraphs=True):
            if "storyteller" in event:
                storyteller_notes = event["storyteller"]["messages"][-1].content
            if "cartoonist" in event:
                cartoonist_notes = event["cartoonist"]["messages"][-1].content
            if "layout" in event:
                latest_layout_state = event["layout"]

        # Fallback: if streaming path missed layout (e.g., tool routing quirks), run a single invoke to get the final layout.
        if not latest_layout_state:
            final_state = comic_graph.invoke(initial_state)
            latest_layout_state = {"messages": final_state["messages"]}
        
        html, _, _ = _parse_final_messages(latest_layout_state)

    elif MODE == "react_graph":

        COMIC_CREATOR_PROMPT = f"""
        You are the **ComicBook Creator**, an all-in-one AI responsible for directing, writing, illustrating, and publishing a daily comic strip.

        You are currently working on the Story Arc: "{arc_title}"
        Logline: {arc_logline}
        Progress: Day {arc_day} of {arc_target}
        Last Image URL (for consistency): {last_image_url}
        Recent Story Beats:
        {previous_beats}

        You have access to image generation tools. You must execute the following workflow step-by-step:

        ### STEP 1: DIRECT (The Director)
        - Plan today's episode so it advances the arc while feeling self-contained.
        - Lock the tone (whimsical, noir, cosmic, etc.) and visual motifs.
        - Outline 4-6 panels with short beats and stakes.

        ### STEP 2: WRITE (The Storyteller)
        - Write a concise recap (2-3 lines).
        - Write the panel-by-panel script including dialogue, captions, and detailed visual descriptions (setting, lighting, mood).

        ### STEP 3: ILLUSTRATE (The Cartoonist)
        - For every panel in your script, generate an image using your tools.
        - **CRITICAL:** To maintain character consistency, if a `{last_image_url}` is provided above, you MUST use the `get_image_by_text_with_reference` tool using that URL as the reference.
        - If no previous image exists, use `get_image_by_text`.
        - Do not make up URLs. You must use the actual URLs returned by the tool observations.

        ### STEP 4: PUBLISH (The Layout Artist)
        - Once you have the generated image URLs, assemble the strip into a single responsive HTML block.
        - **Style:** Use a consistent color paper style and real comicbook styling (CSS).
        - **Layout:** Panels must be readable on mobile (stacked) and desktop.
        - **Content:** Include the Arc Title, Date, Recap, the Images, and Dialogue.
        - **Footer:** Close with a teaser for the next day.

        **FINAL OUTPUT:** Return ONLY the raw HTML code. Do not wrap it in markdown code fences.
        """

        for event in react_graph.stream(
            {"messages": [("system", COMIC_CREATOR_PROMPT)]},
            {"recursion_limit": 1000}
        ):
            print("event: ", event)
            for value in event.values():
                print("React Agent:", value["messages"][-1].content)

        last_message = value["messages"][-1]
        # When the web_search tool is active, responses arrive as structured chunks.
        html = _extract_text_from_last_message(last_message)

    summary = storyteller_notes or "Daily comic script."
    panels = cartoonist_notes or "Panels auto-generated."

    saved = save_episode(
        arc=arc,
        episode_date=target_date,
        html_content=html,
        storyboard_summary=summary,
        panel_notes=panels,
    )

    return html, target_date, saved.get("PartitionKey")
