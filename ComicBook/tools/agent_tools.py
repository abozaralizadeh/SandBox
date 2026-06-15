from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict

from agents.tool import function_tool

from ComicBook.azurestorage import (
    close_arc as _storage_close_arc,
    get_recent_arc_summaries,
    get_recent_episodes,
    save_arc_story_outline,
    save_key_panel,
    start_new_arc as _storage_start_new_arc,
    update_arc_metadata,
)
from ComicBook.helpers import (
    _assemble_html,
    _normalize,
    _parse_arc_theme,
    _summarize_episodes,
)
from ComicBook.tools.getimage import (
    create_image,
    create_image_with_reference,
    create_image_with_references,
)

logger = logging.getLogger("ComicBook")


def build_comic_tools(state: Dict[str, Any], target_date: datetime) -> Dict[str, Any]:
    """Build the agent function-tools as closures over the pipeline's mutable state.

    `state` is the same dict the pipeline reads after the agents run, so tool mutations
    (arc, episode_number, generated panel urls, assembled html, ...) remain visible to
    run_comic_pipeline. Returns a name -> tool dict used to wire the Director and Cartoonist.
    """
    @function_tool
    async def get_arc_status() -> Dict[str, Any]:
        """Get the current story arc status including title, characters, episode count, and recent summaries."""
        logger.info("TOOL get_arc_status called")
        a = state["arc"]
        if not a:
            logger.info("  -> No active arc found")
            past_arcs = get_recent_arc_summaries(limit=10)
            return {
                "status": "no_active_arc",
                "message": "No active story arc. You must create a new one.",
                "past_arcs": past_arcs,
            }
        recent_eps = get_recent_episodes(a["RowKey"], limit=5, hydrate_html=False)
        logger.info("  -> Active arc: '%s' (%s), %s episodes so far",
                     a.get("title", ""), a["RowKey"], a.get("episodes_count", 0))
        planned_eps = int(a.get("planned_episodes", 0) or a.get("target_days", 0) or 8)
        episodes_done = int(a.get("episodes_count", 0))
        episode_today = state["episode_number"]
        return {
            "status": "active",
            "arc_id": a["RowKey"],
            "title": a.get("title", ""),
            "logline": a.get("logline", ""),
            "genre": a.get("genre", ""),
            "planned_episodes": planned_eps,
            "episodes_so_far": episodes_done,
            "episode_number_today": episode_today,
            "is_today_the_finale": episode_today == planned_eps,
            "is_arc_complete_before_today": episodes_done >= planned_eps,
            "start_date": a.get("start_date", ""),
            "characters": a.get("characters", ""),
            "art_style": a.get("art_style", ""),
            "recent_episodes": _summarize_episodes(recent_eps),
            "character_sheet_url": a.get("character_sheet_url", ""),
            "story_outline": get_arc_story_outline(a) if a.get("story_outline") or a.get("story_outline_blob_name") else "",
        }

    @function_tool
    async def start_new_arc(
        title: str,
        logline: str,
        genre: str,
        planned_episodes: int,
        characters: str,
        art_style: str,
        color_theme: str,
    ) -> Dict[str, Any]:
        """Create a brand new story arc with original characters, world, art style, and color theme.
        color_theme must be a JSON string with keys: page_bg, title_color, title_shadow,
        recap_bg, recap_border, caption_bg, caption_border, caption_text, speech_bg,
        speech_border, sfx_color, teaser_color, header_border, font_import (Google Fonts URL), heading_font, body_font."""
        logger.info("TOOL start_new_arc called: title='%s', genre='%s', planned_episodes=%s, art_style='%s'",
                     title, genre, planned_episodes, art_style)
        # Backstop guard (tool-refusal layer): never let a recently-used art style ship again.
        recent_styles = {
            _normalize(a.get("art_style", ""))
            for a in get_recent_arc_summaries(limit=10)
            if a.get("art_style")
        }
        if _normalize(art_style) in recent_styles:
            logger.warning("  -> REFUSED start_new_arc: art_style '%s' collides with a recent arc", art_style)
            return {
                "error": "art_style_collision",
                "message": (
                    f"Art style '{art_style}' was used by a recent arc. Pick a visibly different "
                    f"art style and call start_new_arc again."
                ),
                "recent_art_styles": sorted(recent_styles),
            }
        new_arc = _storage_start_new_arc(
            title=title,
            logline=logline,
            target_days=planned_episodes,
            start_date=target_date,
        )
        update_arc_metadata(
            new_arc["RowKey"],
            genre=genre,
            planned_episodes=planned_episodes,
            characters=characters,
            art_style=art_style,
            color_theme=color_theme,
        )
        new_arc.update(
            genre=genre,
            planned_episodes=planned_episodes,
            characters=characters,
            art_style=art_style,
            color_theme=color_theme,
        )
        state["arc"] = new_arc
        state["episode_number"] = 1
        logger.info("  -> New arc created: %s", new_arc["RowKey"])
        return {
            "status": "created",
            "arc_id": new_arc["RowKey"],
            "title": title,
            "logline": logline,
            "genre": genre,
            "planned_episodes": planned_episodes,
            "characters": characters,
            "art_style": art_style,
            "color_theme": color_theme,
        }

    @function_tool
    async def get_recent_arcs() -> Dict[str, Any]:
        """Return summaries (title, logline, genre, art_style) of the most recent story arcs,
        so a candidate new arc can be judged for originality against them."""
        logger.info("TOOL get_recent_arcs called")
        return {"past_arcs": get_recent_arc_summaries(limit=10)}

    @function_tool
    async def end_current_arc(conclusion_note: str) -> Dict[str, Any]:
        """Close the current story arc when the story has reached its natural conclusion."""
        logger.info("TOOL end_current_arc called: %s", conclusion_note[:100])
        a = state["arc"]
        if not a:
            logger.warning("  -> No active arc to close!")
            return {"error": "No active arc to close"}
        planned_eps = int(a.get("planned_episodes", 0) or a.get("target_days", 0) or 0)
        episodes_done = int(a.get("episodes_count", 0))
        if planned_eps > 0 and episodes_done < planned_eps:
            logger.warning(
                "  -> REFUSED: arc '%s' has only %d/%d planned episodes published — cannot close yet",
                a.get("title", a["RowKey"]), episodes_done, planned_eps,
            )
            return {
                "error": "arc_not_complete",
                "message": (
                    f"Cannot close arc yet: only {episodes_done} of {planned_eps} planned "
                    f"episodes have been published. Today's episode (#{state['episode_number']}) "
                    f"belongs to THIS arc. Plan it now and do NOT close the arc. "
                    f"end_current_arc may only be called when episodes_so_far >= planned_episodes."
                ),
                "episodes_so_far": episodes_done,
                "planned_episodes": planned_eps,
                "episode_number_today": state["episode_number"],
            }
        _storage_close_arc(a["RowKey"], end_date=target_date)
        update_arc_metadata(a["RowKey"], conclusion=conclusion_note)
        logger.info("  -> Arc '%s' closed", a.get("title", a["RowKey"]))
        state["arc"] = None
        state["episode_number"] = 1
        return {
            "status": "closed",
            "arc_id": a["RowKey"],
            "conclusion": conclusion_note,
            "message": "Arc closed. Call start_new_arc to begin a fresh story.",
        }

    @function_tool
    async def save_story_outline(story_outline: str) -> Dict[str, Any]:
        """Save a comprehensive story outline for the current arc. This outline guides all future episodes for plot consistency."""
        logger.info("TOOL save_story_outline called (outline length=%d)", len(story_outline))
        a = state["arc"]
        if not a:
            logger.warning("  -> No active arc to save outline for!")
            return {"error": "No active arc. Create one first with start_new_arc."}
        save_arc_story_outline(a["RowKey"], story_outline)
        a["story_outline"] = story_outline
        logger.info("  -> Story outline saved for arc '%s' (%d chars)", a.get("title", ""), len(story_outline))
        return {
            "status": "saved",
            "arc_id": a["RowKey"],
            "outline_length": len(story_outline),
        }

    @function_tool
    async def generate_character_sheet(description: str, style: str) -> Dict[str, Any]:
        """Generate a character and environment reference sheet image for visual consistency across all panels."""
        logger.info("TOOL generate_character_sheet called (style='%s', desc length=%d)", style, len(description))

        a = state["arc"]
        if a:
            cached_url = a.get("character_sheet_url", "")
            if cached_url:
                logger.info("  -> Returning cached character sheet: %s", cached_url[:120])
                return {"status": "success", "reference_url": cached_url, "style": style, "cached": True}

        # Build the prompt from the full arc character roster when available, so
        # characters who appear in future episodes are included in the sheet even
        # if they are absent from today's script.
        arc_chars = (state["arc"].get("characters", "") if state["arc"] else "") or description
        prompt = (
            f"Character and environment reference sheet, {style} art style. "
            f"Show ALL characters listed below clearly with full body, distinct visual features, "
            f"face clearly visible, each labeled by name. "
            f"Include the primary setting/environment as background. "
            f"Clean grid-style composition with generous spacing between characters. "
            f"No text overlays, no speech bubbles. "
            f"Characters: {arc_chars}"
        )
        try:
            url = await create_image(prompt, size="wide", quality="high")
            logger.info("  -> Character sheet generated: %s", url[:120])
            if a:
                update_arc_metadata(a["RowKey"], character_sheet_url=url)
                a["character_sheet_url"] = url
                logger.info("  -> Character sheet URL saved to arc '%s'", a["RowKey"])
            return {"status": "success", "reference_url": url, "style": style}
        except Exception as exc:
            logger.error("  -> Character sheet generation FAILED: %s", exc)
            return {"error": str(exc)}

    @function_tool
    async def generate_panel_image(
        prompt: str,
        reference_url: str,
        size: str = "square",
    ) -> Dict[str, Any]:
        """Generate a single comic panel image. Use the character reference sheet URL for consistency."""
        if size not in ("wide", "tall", "square"):
            size = "square"
        no_text_suffix = ". No text, no speech bubbles, no captions, no letters, no words, no writing."
        if "no text" not in prompt.lower():
            prompt = prompt.rstrip(".") + no_text_suffix
        logger.info("TOOL generate_panel_image called (size='%s', has_ref=%s, prompt='%s')",
                     size, bool(reference_url), prompt[:80])
        try:
            seen: set[str] = set()
            image_urls: list[str] = []
            def _add(u: str) -> None:
                if u and u not in seen:
                    seen.add(u)
                    image_urls.append(u)
            _add(reference_url)                          # character sheet — always first
            for u in state["key_panel_urls"][-3:]:       # mid-arc character key panels
                _add(u)
            for u in state["generated_panel_urls"][-2:]: # panels from this session
                _add(u)
            for u in state["prev_episode_images"][-2:]:  # arc history anchor
                _add(u)

            if len(image_urls) > 1:
                url = await create_image_with_references(prompt, image_urls, size)
            elif image_urls:
                url = await create_image_with_reference(prompt, image_urls[0], size)
            else:
                url = await create_image(prompt, size)

            state["generated_panel_urls"].append(url)
            logger.info("  -> Panel image generated (%d refs): %s", len(image_urls), url[:120])
            return {"status": "success", "image_url": url, "size": size}
        except Exception as exc:
            logger.error("  -> Panel image generation FAILED: %s", exc)
            return {"error": str(exc)}

    @function_tool
    async def mark_key_panel(image_url: str, character_name: str, reason: str) -> Dict[str, Any]:
        """Mark a generated panel as a permanent visual reference for a character introduced mid-arc.
        Call this immediately after generate_panel_image whenever a panel shows a character who does
        NOT appear on the original character sheet. The panel URL will be included as a reference
        image for all future generate_panel_image calls in this arc."""
        logger.info("TOOL mark_key_panel: character='%s', reason='%s'", character_name, reason)
        a = state["arc"]
        if not a:
            return {"error": "No active arc"}
        if not image_url:
            return {"error": "Missing image_url"}
        if image_url not in state["key_panel_urls"]:
            state["key_panel_urls"].append(image_url)
            state["key_panels"].append({
                "url": image_url,
                "character": character_name,
                "episode": state["episode_number"],
            })
            save_key_panel(a["RowKey"], image_url, character_name, state["episode_number"])
            logger.info("  -> Key panel stored for '%s' (arc=%s)", character_name, a["RowKey"])
        return {"status": "marked", "character": character_name, "url": image_url}

    @function_tool
    async def assemble_layout(
        arc_title: str,
        episode_number: int,
        date: str,
        recap: str,
        teaser: str,
        panels_json: str,
    ) -> Dict[str, Any]:
        """Assemble the final comic page HTML from generated panel images, dialogue, and captions."""
        logger.info("TOOL assemble_layout called: arc='%s', ep=%s, date='%s'", arc_title, episode_number, date)
        try:
            panels = json.loads(panels_json)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.error("  -> Invalid panels_json: %s", exc)
            return {"error": f"Invalid panels_json: {exc}"}
        if not panels:
            logger.error("  -> No panels provided in panels_json")
            return {"error": "No panels provided"}
        logger.info("  -> Assembling HTML with %d panels", len(panels))
        for i, p in enumerate(panels):
            logger.info("    Panel %d: size=%s, has_img=%s, dialogue=%s",
                         i + 1, p.get("size", "?"), bool(p.get("image_url")),
                         p.get("dialogue", "")[:50])
        arc_theme = _parse_arc_theme(state["arc"])
        actual_title = state["arc"].get("title", arc_title) if state["arc"] else arc_title
        actual_ep = state["episode_number"]
        html = _assemble_html(actual_title, actual_ep, date, recap, teaser, panels, theme=arc_theme)
        state["assembled_panels"] = panels
        state["assembled_recap"] = recap
        state["assembled_teaser"] = teaser
        logger.info("  -> Layout assembled, HTML length=%d", len(html))
        return {"status": "success", "html": html, "panel_count": len(panels)}

    return {
        "get_arc_status": get_arc_status,
        "get_recent_arcs": get_recent_arcs,
        "start_new_arc": start_new_arc,
        "end_current_arc": end_current_arc,
        "save_story_outline": save_story_outline,
        "generate_character_sheet": generate_character_sheet,
        "generate_panel_image": generate_panel_image,
        "mark_key_panel": mark_key_panel,
        "assemble_layout": assemble_layout,
    }
