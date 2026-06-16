from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict

from agents.tool import function_tool

from ComicBook.azurestorage import (
    close_arc as _storage_close_arc,
    get_arc_glossary,
    get_arc_story_outline,
    get_recent_arc_summaries,
    get_recent_episodes,
    save_arc_glossary,
    save_arc_story_outline,
    save_key_panel,
    start_new_arc as _storage_start_new_arc,
    update_arc_metadata,
)
from ComicBook.helpers import (
    _apply_reteller_output,
    _assemble_html,
    _build_reteller_payload,
    _normalize,
    _parse_arc_theme,
    _summarize_episodes,
)
from ComicBook.tools.getimage import (
    create_image,
    create_image_with_reference,
    create_image_with_references,
    make_placeholder_image_url,
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
            state["image_failures"] = state.get("image_failures", 0) + 1
            logger.error("  -> Character sheet generation FAILED (%d): %s — using placeholder",
                         state["image_failures"], str(exc)[:150])
            # Do NOT cache a placeholder as the arc's sheet; just let this run proceed.
            url = await make_placeholder_image_url()
            return {"status": "success", "reference_url": url, "style": style, "placeholder": True}

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

        # Circuit breaker: once the image service has failed repeatedly this run, stop calling it
        # and serve a placeholder so the Cartoonist proceeds instead of looping on a dead service.
        if state.get("image_failures", 0) >= 2:
            url = await make_placeholder_image_url()
            logger.warning("  -> image service unavailable (%d prior failures) — using placeholder",
                           state.get("image_failures", 0))
            return {"status": "success", "image_url": url, "size": size, "placeholder": True}

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
            state["image_failures"] = 0
            logger.info("  -> Panel image generated (%d refs): %s", len(image_urls), url[:120])
            return {"status": "success", "image_url": url, "size": size}
        except Exception as exc:
            state["image_failures"] = state.get("image_failures", 0) + 1
            logger.error("  -> Panel image generation FAILED (%d): %s — using placeholder",
                         state["image_failures"], str(exc)[:150])
            url = await make_placeholder_image_url()
            return {"status": "success", "image_url": url, "size": size, "placeholder": True}

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
        state["html_en"] = html
        logger.info("  -> Layout assembled, HTML length=%d", len(html))
        return {"status": "success", "panel_count": len(panels)}

    # ------------------------------------------------------------------
    # Cartoonist / Reteller briefs and localized assembly (read shared state;
    # used by the handoff chain so downstream agents pull what they need)
    # ------------------------------------------------------------------

    @function_tool
    async def get_cartoonist_brief() -> Dict[str, Any]:
        """Return the FULL arc character roster, the arc's art style, and any mid-arc character key
        panels already registered. Call this first so the character sheet covers every character who
        appears at any point in the arc, and so you do not re-mark already-registered characters."""
        logger.info("TOOL get_cartoonist_brief called")
        a = state["arc"]
        registered = [
            {"character": p.get("character", ""), "episode": p.get("episode", ""), "url": p.get("url", "")}
            for p in state.get("key_panels", []) if p.get("url")
        ]
        return {
            "characters": a.get("characters", "") if a else "",
            "art_style": a.get("art_style", "") if a else "",
            "already_registered_key_panels": registered,
        }

    @function_tool
    async def get_localization_brief(target_language: str) -> Dict[str, Any]:
        """For a target language ('it' or 'fa'), return the FIXED-panel manifest (panel numbers,
        sizes, and the English dialogue/caption/sfx as INTENT reference only), the English story
        outline, any existing localized outline, and the arc glossary. Call this BEFORE writing the
        native retelling for that language."""
        lang = "it" if str(target_language).lower().startswith("it") else "fa"
        logger.info("TOOL get_localization_brief called (lang=%s)", lang)
        a = state["arc"]
        panels = state.get("assembled_panels", [])
        manifest = _build_reteller_payload(
            panels,
            state.get("assembled_recap", ""),
            state.get("assembled_teaser", ""),
            title=(a.get("title", "") if a else ""),
        )
        local_outline = ""
        english_outline = ""
        glossary: Dict[str, Any] = {}
        if a:
            local_outline = a.get(f"story_outline_{lang}", "") or get_arc_story_outline(a, lang=lang)
            english_outline = get_arc_story_outline(a)
            glossary = get_arc_glossary(a, lang)
        return {
            "lang_code": lang,
            "manifest": manifest,
            "english_outline": english_outline,
            "local_outline": local_outline,
            "glossary": glossary,
        }

    @function_tool
    async def save_local_outline(target_language: str, outline: str) -> Dict[str, Any]:
        """Save the story outline adapted into the target language. Do this on episode 1 only, when
        get_localization_brief shows no existing local_outline."""
        lang = "it" if str(target_language).lower().startswith("it") else "fa"
        logger.info("TOOL save_local_outline called (lang=%s, len=%d)", lang, len(outline))
        a = state["arc"]
        if not a:
            return {"error": "No active arc"}
        save_arc_story_outline(a["RowKey"], outline, lang=lang)
        a[f"story_outline_{lang}"] = outline
        return {"status": "saved", "lang": lang, "length": len(outline)}

    @function_tool
    async def assemble_localized(
        target_language: str,
        native_panels_json: str,
        recap: str,
        teaser: str,
        title: str,
        updated_glossary_json: str = "",
    ) -> Dict[str, Any]:
        """Assemble and store the localized comic page from your native retelling.
        native_panels_json is a JSON array of {"number","dialogue","caption","sfx"} in the target
        script — it is mapped onto the FIXED English panel images (you do not change images/positions).
        updated_glossary_json is the full glossary JSON to persist for future episodes."""
        lang = "it" if str(target_language).lower().startswith("it") else "fa"
        logger.info("TOOL assemble_localized called (lang=%s)", lang)
        a = state["arc"]
        panels = state.get("assembled_panels", [])
        if not panels:
            return {"error": "No assembled English panels to localize"}
        try:
            native_panels = json.loads(native_panels_json)
        except (json.JSONDecodeError, TypeError) as exc:
            return {"error": f"Invalid native_panels_json: {exc}"}
        if updated_glossary_json:
            try:
                ug = json.loads(updated_glossary_json)
                if ug and a:
                    save_arc_glossary(a["RowKey"], lang, ug)
                    a[f"glossary_{lang}"] = json.dumps(ug, ensure_ascii=False)
                    logger.info("  -> Glossary for %s updated: %d entries", lang, len(ug))
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("  -> invalid updated_glossary_json for %s: %s", lang, exc)
        native = {"title": title, "recap": recap, "teaser": teaser, "panels": native_panels}
        t_panels, t_recap, t_teaser, t_title = _apply_reteller_output(
            panels,
            state.get("assembled_recap", ""),
            state.get("assembled_teaser", ""),
            native,
            title=(a.get("title", "") if a else title),
        )
        html = _assemble_html(
            t_title, state["episode_number"], target_date.strftime("%Y-%m-%d"),
            t_recap, t_teaser, t_panels, lang=lang, theme=_parse_arc_theme(a),
        )
        state[f"html_{lang}"] = html
        logger.info("  -> Localized (%s) HTML assembled: %d chars", lang, len(html))
        return {"status": "success", "lang": lang, "panel_count": len(t_panels)}

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
        "get_cartoonist_brief": get_cartoonist_brief,
        "get_localization_brief": get_localization_brief,
        "save_local_outline": save_local_outline,
        "assemble_localized": assemble_localized,
    }
