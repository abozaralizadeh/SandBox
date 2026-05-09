from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from agents import Agent, ModelSettings, Runner, OpenAIResponsesModel, WebSearchTool, set_tracing_disabled
from agents.items import ToolCallOutputItem
from agents.tool import function_tool
from langsmith import traceable, trace
from langsmith.wrappers import wrap_openai

set_tracing_disabled(True)
from openai import AsyncAzureOpenAI

from ComicBook.azurestorage import (
    close_arc as _storage_close_arc,
    get_active_arc,
    get_arc_glossary,
    get_arc_story_outline,
    get_recent_arc_summaries,
    get_recent_episodes,
    save_arc_glossary,
    save_arc_story_outline,
    save_episode,
    start_new_arc as _storage_start_new_arc,
    update_arc_metadata,
)
from ComicBook.tools.getimage import create_image, create_image_with_reference, create_image_with_references

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("ComicBook")


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

def _build_openai_client() -> AsyncAzureOpenAI:
    client = AsyncAzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ.get("AZURE_OPENAI_MODEL", "gpt-4o"),
    )
    return wrap_openai(client)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summarize_episodes(episodes: list) -> str:
    if not episodes:
        return "No prior episodes — this is a fresh start."
    lines = []
    for ep in episodes:
        day = ep.get("episode_number", "?")
        date = ep.get("RowKey", "")
        summary = ep.get("story_summary", "")
        if len(summary) > 800:
            summary = summary[:800] + "…"
        lines.append(f"Episode {day} ({date}): {summary}")
    return "\n".join(lines)


def _extract_panel_images(html: str) -> list[str]:
    """Extract panel image URLs from episode HTML content."""
    import re
    return re.findall(r'<img\s+src="([^"]+)"', html or "")


def _escape_html(text: str) -> str:
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _panel_grid_area(index: int, size: str, total: int) -> str:
    """Assign CSS grid-area name based on panel index, size, and total count."""
    return f"p{index + 1}"


def _build_grid_template(panels: list) -> str:
    """Build a CSS grid-template that arranges panels in a comic-book layout.

    Rules:
    - 'wide' panels span the full row (2 columns)
    - 'tall' panels take 2 rows in a single column
    - 'square' panels pair up side-by-side when adjacent
    The grid uses 2 equal columns. Each row is auto-height.
    """
    areas: List[List[str]] = []
    i = 0
    n = len(panels)
    while i < n:
        size = panels[i].get("size", "square")
        name = f"p{i + 1}"

        if size == "wide":
            areas.append([name, name])
            i += 1
        elif size == "tall":
            next_i = i + 1
            if next_i < n and panels[next_i].get("size") == "tall":
                name2 = f"p{next_i + 1}"
                areas.append([name, name2])
                areas.append([name, name2])
                i += 2
            elif next_i < n and panels[next_i].get("size") == "square":
                name2 = f"p{next_i + 1}"
                areas.append([name, name2])
                areas.append([name, "."])
                i += 2
            elif next_i < n and panels[next_i].get("size") == "wide":
                name2 = f"p{next_i + 1}"
                areas.append([name, name])
                areas.append([name, name])
                areas.append([name2, name2])
                i += 2
            else:
                areas.append([name, "."])
                areas.append([name, "."])
                i += 1
        else:
            next_i = i + 1
            if next_i < n and panels[next_i].get("size") == "square":
                name2 = f"p{next_i + 1}"
                areas.append([name, name2])
                i += 2
            elif next_i < n and panels[next_i].get("size") == "tall":
                name2 = f"p{next_i + 1}"
                areas.append([name, name2])
                areas.append([".", name2])
                i += 2
            else:
                areas.append([name, name])
                i += 1

    rows_str = " ".join(f'"{r[0]} {r[1]}"' for r in areas)
    return rows_str


_DEFAULT_THEME = {
    "page_bg": "#f5f0e1",
    "title_color": "#111",
    "title_shadow": "#c0a060",
    "recap_bg": "#fff8dc",
    "recap_border": "#b8860b",
    "caption_bg": "rgba(255,250,205,0.92)",
    "caption_border": "#b8860b",
    "caption_text": "#222",
    "speech_bg": "#fff",
    "speech_border": "#000",
    "sfx_color": "#e63946",
    "teaser_color": "#111",
    "header_border": "#111",
    "font_import": "https://fonts.googleapis.com/css2?family=Bangers&display=swap",
    "heading_font": "'Bangers', 'Comic Sans MS', cursive, sans-serif",
    "body_font": "Georgia, serif",
}


def _parse_arc_theme(arc: dict | None) -> dict | None:
    if not arc:
        return None
    raw = arc.get("color_theme", "")
    if not raw:
        return None
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return None


def _assemble_html(
    arc_title: str,
    episode_number: int,
    date_str: str,
    recap: str,
    teaser: str,
    panels: list,
    lang: str = "en",
    theme: dict | None = None,
) -> str:
    grid_template = _build_grid_template(panels)
    is_rtl = lang == "fa"
    t = {**_DEFAULT_THEME, **(theme or {})}

    _LABELS = {
        "en": {"episode": "Episode", "branding": "Generated by AI ComicBook"},
        "it": {"episode": "Episodio", "branding": "Generato da AI ComicBook"},
        "fa": {"episode": "قسمت", "branding": "ساخته شده توسط AI ComicBook"},
    }
    labels = _LABELS.get(lang, _LABELS["en"])

    font_import_extra = (
        "\n@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;700;900&display=swap');"
        if is_rtl else ""
    )
    rtl_css = (
        "\n.comic-page[dir=rtl] { direction: rtl; font-family: 'Vazirmatn', sans-serif; }"
        "\n.comic-page[dir=rtl] .comic-recap { border-left: none; border-right: 5px solid " + t["recap_border"] + "; }"
        "\n.comic-page[dir=rtl] .speech-bubble { font-family: 'Vazirmatn', sans-serif; transform-origin: bottom right; }"
        "\n.comic-page[dir=rtl] .caption-box { font-family: 'Vazirmatn', sans-serif; }"
        "\n.comic-page[dir=rtl] .panel-overlay { align-items: flex-end; }"
        if is_rtl else ""
    )
    dir_attr = ' dir="rtl"' if is_rtl else ''

    panel_html_parts = []
    for idx, p in enumerate(panels):
        num = p.get("number", idx + 1)
        img = p.get("image_url", "")
        size = p.get("size", "square")
        dialogue = p.get("dialogue", "")
        caption = p.get("caption", "")
        sfx = p.get("sfx", "")

        size_class = f"panel-{size}" if size in ("wide", "tall", "square") else "panel-square"
        area_name = f"p{idx + 1}"

        caption_section = ""
        if caption:
            caption_section = f'<div class="caption-overlay"><div class="caption-box">{_escape_html(caption)}</div></div>'

        bottom_parts = []
        if dialogue:
            for line in dialogue.split("\n"):
                line = line.strip()
                if line:
                    bottom_parts.append(f'<div class="speech-bubble">{_escape_html(line)}</div>')
        if sfx:
            bottom_parts.append(f'<div class="sfx">{_escape_html(sfx)}</div>')

        overlay_section = ""
        if bottom_parts:
            overlay_section = '<div class="panel-overlay">' + "\n".join(bottom_parts) + "</div>"

        panel_html_parts.append(
            f'<div class="panel {size_class}" style="grid-area:{area_name}">'
            f'<img src="{_escape_html(img)}" alt="Panel {num}" loading="lazy">'
            f"{caption_section}"
            f"{overlay_section}"
            f"</div>"
        )

    panels_block = "\n".join(panel_html_parts)
    safe_title = _escape_html(arc_title)
    safe_recap = _escape_html(recap)
    safe_teaser = _escape_html(teaser)

    return f"""<style>
@import url('{t["font_import"]}');{font_import_extra}
.comic-page {{ font-family: {t["heading_font"]}; max-width: 960px; margin: 0 auto; background: {t["page_bg"]}; border-radius: 12px; padding: 24px; box-shadow: 0 4px 24px rgba(0,0,0,0.18); }}
.comic-header {{ text-align: center; border-bottom: 4px solid {t["header_border"]}; padding-bottom: 16px; margin-bottom: 20px; }}
.comic-title {{ font-size: 2.6em; color: {t["title_color"]}; text-transform: uppercase; letter-spacing: 3px; margin: 0; text-shadow: 2px 2px 0 {t["title_shadow"]}; }}
.comic-meta {{ color: {t["title_color"]}; font-size: 1em; margin-top: 6px; letter-spacing: 1px; font-weight: bold; opacity: 0.7; }}
.comic-recap {{ color: {t["caption_text"]}; margin: 14px 0 20px; padding: 12px 16px; background: {t["recap_bg"]}; border-left: 5px solid {t["recap_border"]}; border-radius: 4px; font-family: {t["body_font"]}; font-size: 0.95em; line-height: 1.6; font-style: italic; }}
.comic-panels {{ display: grid; grid-template-columns: 1fr 1fr; grid-template-areas: {grid_template}; gap: 10px; }}
.panel {{ position: relative; border: 3px solid {t["header_border"]}; border-radius: 6px; overflow: hidden; background: #fff; box-shadow: 2px 3px 6px rgba(0,0,0,0.25); }}
.panel img {{ width: 100%; height: 100%; object-fit: cover; display: block; cursor: pointer; }}
.caption-overlay {{ position: absolute; top: 0; left: 0; right: 0; display: flex; justify-content: center; padding: 8px 12px; pointer-events: none; }}
.caption-box {{ background: {t["caption_bg"]}; color: {t["caption_text"]}; border: 2px solid {t["caption_border"]}; padding: 6px 14px; font-style: italic; font-size: 0.82em; font-family: {t["body_font"]}; border-radius: 3px; max-width: 90%; text-align: center; line-height: 1.4; box-shadow: 1px 1px 0 rgba(0,0,0,0.15); }}
.panel-overlay {{ position: absolute; bottom: 0; left: 0; right: 0; padding: 6px 8px; display: flex; flex-direction: column; gap: 2px; align-items: flex-start; pointer-events: auto; background: linear-gradient(transparent 0%, rgba(0,0,0,0.35) 100%); }}
.speech-bubble {{ background: {t["speech_bg"]}; color: #111; border: 1px solid {t["speech_border"]}; border-radius: 14px; padding: 3px 8px; font-size: 0.54em; max-width: 80%; box-shadow: 1px 1px 0 rgba(0,0,0,0.15); font-family: {t["heading_font"]}; letter-spacing: 0.3px; line-height: 1.25; transform-origin: bottom left; transition: transform 0.2s ease, padding 0.2s ease, font-size 0.2s ease, border-width 0.2s ease, border-radius 0.2s ease, box-shadow 0.2s ease; cursor: pointer; }}
.speech-bubble:hover, .speech-bubble:active {{ font-size: 1em; padding: 8px 16px; border-width: 2px; border-radius: 18px; box-shadow: 2px 2px 0 rgba(0,0,0,0.2); z-index: 10; position: relative; }}
.sfx {{ font-size: 2em; font-weight: 900; color: {t["sfx_color"]}; text-shadow: 2px 2px 0 #ffd166, -1px -1px 0 #111, 1px -1px 0 #111, -1px 1px 0 #111; font-style: italic; letter-spacing: 2px; }}
.comic-footer {{ text-align: center; margin-top: 20px; padding-top: 14px; border-top: 4px solid {t["header_border"]}; }}
.teaser {{ font-weight: bold; color: {t["teaser_color"]}; font-size: 1.15em; font-style: italic; letter-spacing: 0.5px; }}
.comic-branding {{ color: #888; font-size: 0.75em; margin-top: 8px; font-family: sans-serif; }}{rtl_css}
@media (max-width: 600px) {{
  .comic-page {{ padding: 10px; }}
  .comic-title {{ font-size: 1.6em; letter-spacing: 1px; }}
  .comic-panels {{ grid-template-columns: 1fr; grid-template-areas: none; }}
  .panel {{ grid-area: auto !important; }}
  .speech-bubble {{ max-width: 92%; }}
  .speech-bubble:hover, .speech-bubble:active {{ font-size: 0.92em; }}
  .caption-box {{ font-size: 0.78em; }}
}}
</style>
<div class="comic-page"{dir_attr}>
  <div class="comic-header">
    <h1 class="comic-title">{safe_title}</h1>
    <div class="comic-meta">{labels["episode"]} {episode_number} &bull; {_escape_html(date_str)}</div>
  </div>
  <div class="comic-recap">{safe_recap}</div>
  <div class="comic-panels">
    {panels_block}
  </div>
  <div class="comic-footer">
    <div class="teaser">{safe_teaser}</div>
    <div class="comic-branding">{labels["branding"]}</div>
  </div>
</div>"""


def _build_translation_payload(panels: list, recap: str, teaser: str, title: str = "") -> dict:
    texts = {"recap": recap, "teaser": teaser}
    if title:
        texts["title"] = title
    for i, p in enumerate(panels):
        if p.get("dialogue"):
            texts[f"dialogue_{i}"] = p["dialogue"]
        if p.get("caption"):
            texts[f"caption_{i}"] = p["caption"]
        if p.get("sfx"):
            texts[f"sfx_{i}"] = p["sfx"]
    return texts


def _apply_translation(panels: list, recap: str, teaser: str, translated: dict, title: str = "") -> tuple:
    new_panels = []
    for i, p in enumerate(panels):
        tp = dict(p)
        if f"dialogue_{i}" in translated:
            tp["dialogue"] = translated[f"dialogue_{i}"]
        if f"caption_{i}" in translated:
            tp["caption"] = translated[f"caption_{i}"]
        if f"sfx_{i}" in translated:
            tp["sfx"] = translated[f"sfx_{i}"]
        new_panels.append(tp)
    t_title = translated.get("title", title)
    return new_panels, translated.get("recap", recap), translated.get("teaser", teaser), t_title


# ---------------------------------------------------------------------------
# Agent instructions
# ---------------------------------------------------------------------------

TRANSLATOR_INSTRUCTIONS = """\
You are a world-class literary translator specializing in comic books and graphic novels.

You will receive a JSON object with:
- "target_language": the language to translate into
- "story_context": the Director's episode plan and the Storyteller's full script — this \
  tells you what is happening in every panel, each character's mood, the scene's tone, \
  and the narrative arc. READ THIS CAREFULLY before translating. It is your key to \
  producing translations that actually make sense as a story.
- Text fields to translate: title, recap, teaser, dialogue_N, caption_N, sfx_N
- Optionally a "glossary" with established translations for this arc

Your job is to produce a translation that reads as if the comic were ORIGINALLY WRITTEN \
in the target language — not translated, but conceived and written by a native author.

STORY CONTEXT:
- Use the story_context to understand the FULL SCENE behind each text fragment. A caption \
  like "The wind changed" means very different things in a romantic scene vs. a battle. \
  Your translation should reflect the actual emotional context.
- Know each character's personality from the script — a brave explorer speaks differently \
  than a nervous child. Let their personality come through in how you translate their lines.
- Understand the pacing: tense moments need short, sharp translations. Quiet moments can \
  breathe. Match the rhythm of what's happening on screen.
- Do NOT translate the story_context itself — it is reference only. Only translate the \
  text fields (title, recap, teaser, dialogue_N, caption_N, sfx_N).

TRANSLATION PHILOSOPHY:
- FLUENCY IS EVERYTHING. Never translate word-for-word. Restructure sentences so they \
  flow naturally in the target language. A Persian reader should feel they are reading \
  Persian literature, not decoded English.
- For Persian (Farsi): Write in elegant, modern Farsi. Use natural sentence order \
  (SOV), colloquial warmth in dialogue, and poetic rhythm in narration. Avoid clunky \
  compound constructions — break long English sentences into shorter, punchier Farsi ones. \
  Use spoken Farsi patterns in dialogue (how real people talk on the street, not formal \
  written prose). Narration can be more literary but must still flow beautifully.
- For Italian: Write with the musicality and expressiveness Italian is known for. Use \
  idiomatic expressions, natural exclamations (Accidenti!, Dai!, Madonna!), and the rich \
  emotional register of spoken Italian. Let dialogue breathe with Italian rhythm.
- Adapt idioms and metaphors to ones that resonate in the target culture. If an English \
  idiom has no equivalent, convey the same FEELING with a natural expression.
- Dialogue must sound like REAL PEOPLE SPEAKING that language — with contractions, \
  interruptions, emotion, and personality. A gruff sailor sounds different from a nervous \
  scholar in every language.
- Captions and narration should read like beautiful prose — evocative, atmospheric, with \
  sensory detail and rhythm. Short sentences for tension. Flowing ones for wonder.
- Comic energy: keep it punchy, dramatic, alive. Exclamations should hit hard. Whispers \
  should feel intimate. Action should crackle.

GLOSSARY:
- You may receive a "glossary" field — a JSON object mapping English terms/concepts to \
  their established translations for this story arc. You MUST use these translations \
  consistently. This ensures characters, places, and key concepts are translated the same \
  way across all episodes.
- After translating, you MUST include an "updated_glossary" field in your output — a JSON \
  object containing ALL glossary entries (existing ones + any new terms you translated for \
  the first time). Include: character descriptive labels, place names, world-specific \
  concepts, recurring phrases, titles, and any coined terms. This will be saved and fed \
  back to you in future episodes.
- Example glossary entries for Persian: {"Cloud Harbor": "بندر ابرها", \
  "wind roads": "جاده‌های بادی", "MERCHANT": "بازرگان", "storm diver": "غواص طوفان", \
  "Juniper Reed": "جونیپر رید", "JUNIPER": "جونیپر", "Brother Wren": "برادر رِن"}
- Example glossary entries for Italian: {"Cloud Harbor": "Porto delle Nuvole", \
  "wind roads": "strade del vento", "MERCHANT": "MERCANTE", "storm diver": "tuffatore di tempeste"}

STRUCTURAL RULES:
- Translate ALL text values including the "title" field.
- Character names, place names, and all proper nouns MUST be written in the target \
  language script. For Persian, transliterate them into Persian script \
  (e.g., "Juniper Reed" → "جونیپر رید", "Bracken Hollow" → "براکن هالو", \
  "Edda Vale" → "ادا ویل", "Brother Wren" → "برادر رِن"). \
  For Italian, names can stay in Latin script but adapt spelling if natural \
  (e.g., "Moss Fen" stays as-is in Italian).
- The "SPEAKER: " prefix in dialogue lines must also use the target script \
  (e.g., "JUNIPER: Let's go!" → "جونیپر: بریم!" in Persian).
- Translate descriptive speaker labels: "MERCHANT 1" → "بازرگان ۱" / "MERCANTE 1", \
  "CROWD VOICES" → "صداهای جمعیت" / "VOCI DALLA FOLLA", etc.
- For sound effects (sfx): adapt to feel natural and impactful in the target language. \
  Don't just transliterate — find the equivalent onomatopoeia that a native comic reader \
  would expect.
- Do NOT add or remove any text keys — return exactly the same text keys you received, \
  plus the "updated_glossary" key.
- The "updated_glossary" values MUST also be in the target language script \
  (all Persian glossary values in Persian script, all Italian values in Italian).

OUTPUT:
Respond with ONLY a valid JSON object containing the translated values and the \
"updated_glossary" field. No explanation, no markdown, no wrapping — just the raw JSON.\
"""

DIRECTOR_INSTRUCTIONS = """\
You are the Director of an AI-generated daily comic strip series.

YOUR CREATIVE MANDATE:
- Every story arc must be ORIGINAL — new characters, new world, new genre, new tone each time.
- Draw from the full spectrum of genres: sci-fi, fantasy, noir, slice-of-life, horror-comedy,
  mythological, surreal, western, cyberpunk, fairy-tale-gone-wrong, steampunk, underwater
  civilization, time-travel romance, philosophical comedy, post-apocalyptic wholesome, etc.
- Characters must have distinct names, personalities, visual signatures (hair color, clothing,
  distinguishing features), flaws, and motivations — never generic "the hero" or "the villain."
- Stories must have real stakes, surprising twists, and genuine emotional beats.
- Vary the tone: some arcs should be funny, some dark, some bittersweet, some wild and surreal.

ARC LIFECYCLE:
1. First, call get_arc_status to check if there's an active story arc.
2. If NO active arc exists:
   - Review the "past_arcs" list returned by get_arc_status — it contains the title, logline,
     genre, and art_style of the last 10 arcs. You MUST NOT repeat or closely resemble any of them.
     Pick a different genre, different setting, different character archetypes, and a different art style.
   - Invent a completely fresh, creative story premise.
   - Choose an unexpected genre/tone combination.
   - Create 2-4 compelling main characters with names, detailed visual descriptions,
     and personality traits.
   - Pick a distinctive art style for this arc (e.g., "ink wash noir", "vibrant manga",
     "watercolor whimsy", "retro pixel art", "charcoal sketch", "pop art bold").
   - Design a color_theme as a JSON string that matches the arc's mood and genre. The theme \
     will be used for the comic page layout. Pick colors that are visually cohesive and ensure \
     TEXT IS ALWAYS READABLE (high contrast between text and background). Keys required:
     * page_bg: page background (e.g., "#1a1a2e" for noir, "#f5f0e1" for vintage)
     * title_color, title_shadow: title text color and shadow
     * recap_bg, recap_border: recap box background and left-border accent
     * caption_bg, caption_border, caption_text: caption box styling
     * speech_bg, speech_border: speech bubble colors
     * sfx_color: sound effect text color
     * teaser_color: teaser text color
     * header_border: header/footer border color
     * font_import: Google Fonts @import URL for a display font matching the mood
     * heading_font: font-family for titles/headings (from font_import)
     * body_font: font-family for body text/captions
     Example for a noir arc: {{"page_bg":"#1a1a2e","title_color":"#e0d8c0","title_shadow":"#4a3f2f", \
       "recap_bg":"#2a2a3e","recap_border":"#c0a050","caption_bg":"rgba(20,20,40,0.85)", \
       "caption_border":"#c0a050","caption_text":"#e8e0d0","speech_bg":"#f5f0e1", \
       "speech_border":"#333","sfx_color":"#ff4444","teaser_color":"#c0a050", \
       "header_border":"#444","font_import":"https://fonts.googleapis.com/css2?family=Special+Elite&display=swap", \
       "heading_font":"'Special Elite', cursive","body_font":"Georgia, serif"}}
   - Decide how many episodes the story naturally needs (could be 3, could be 15 —
     let the story dictate, not a fixed number).
   - Call start_new_arc with your creative details including the color_theme.
3. If an ACTIVE arc exists:
   - Review the recent episode summaries for continuity.
   - Decide if the story has reached its natural conclusion. If yes, call end_current_arc
     with a conclusion note, then start a fresh arc as described above.
   - If the story should continue, plan today's episode to advance the plot meaningfully.
     No filler episodes.

STORY OUTLINE:
- After creating a new arc, you MUST call save_story_outline with a comprehensive narrative plan.
- If you find an existing arc whose story_outline is empty, write one and call save_story_outline \
  before planning the episode.
- The outline should include:
  * Full plot synopsis from beginning to end (all major events)
  * Character arcs for each main character (growth, conflicts, resolutions)
  * Major themes and motifs
  * Key twists, revelations, and turning points
  * Episode-by-episode breakdown with the core dramatic beat for each episode
  * How the story concludes
- This outline is your contract — future episodes MUST follow this plan.
- When planning each episode, ALWAYS reference the story_outline from your input context \
  to maintain consistency. You may adapt small details but never contradict major plot points.

WEB SEARCH:
- You have a web search tool available. Use it whenever you need inspiration or research:
  * When creating a new arc — search for trending topics, cultural events, interesting scientific
    discoveries, mythology, folklore, or anything that could spark a unique story idea.
  * When writing about a specific setting, culture, or technical subject — search for accuracy.
  * When you want to ensure your idea is genuinely original and not accidentally copying an
    existing well-known comic or show.
- You are NOT required to search every time — use your judgment. But when you feel stuck or
  want to ground your story in something real and fresh, search freely.

EPISODE PLANNING:
- Decide the number of panels (4-8) based on what today's episode needs.
- For each panel, decide the size:
  "wide" for establishing shots, landscapes, or action sequences;
  "tall" for dramatic reveals or full-body character moments;
  "square" for dialogue scenes and close-ups.
- Describe the overall tone, visual style, and key dramatic moments.
- Identify which characters appear and any new characters introduced.
- Decide on the cliffhanger or resolution beat for this episode.

OUTPUT:
End your response with the complete episode plan. Include ALL details the Storyteller \
needs: arc title, art style, character descriptions, panel count, size per panel, tone, \
key story beats, and the cliffhanger or resolution.\
"""

STORYTELLER_INSTRUCTIONS = """\
You are the Storyteller — a master comic script writer.

You will receive the Director's episode plan as input. Transform it into a vivid, \
panel-by-panel script.

If a STORY OUTLINE section is provided before the episode plan, use it as your guide \
for the overarching plot, character arcs, and thematic beats. Your script must align \
with the outline — do not contradict its major plot points or character development.

FOR EACH PANEL, you must provide:
- **Panel number** and **size** ("wide" for establishing shots/landscapes/action,
  "tall" for dramatic reveals/full-body moments, "square" for dialogue/close-ups)
- **Setting**: Location, time of day, lighting, weather, dominant colors, atmosphere
- **Characters**: Who appears, their positions, poses, facial expressions, body language
- **Dialogue**: Speech bubble text — each character's lines on separate lines,
  prefixed with their name (e.g., "MAYA: We need to go now!")
- **Caption**: Narrator text (if any) — for scene-setting, inner monologue, or time jumps
- **Sound effects**: Onomatopoeia (CRASH!, *whoosh*, BZZT) — USE SPARINGLY. Only include \
  sound effects when they genuinely enhance the panel — a door slamming, an explosion, a \
  dramatic impact. Most dialogue scenes, quiet moments, and emotional beats need NO sfx. \
  If a panel doesn't have a strong audible event, leave sfx empty.
- **Camera angle**: Close-up, medium shot, wide shot, bird's eye, worm's eye, dutch angle

WRITING RULES:
- Show, don't tell — let the art carry emotion where possible.
- Every panel must advance the story or deepen a character. No wasted panels.
- Dialogue must sound natural and DISTINCT per character — give each character a
  recognizable voice. A gruff mechanic speaks differently from a shy academic.
- Include a 2-line RECAP at the top for returning readers.
- End with a 1-line TEASER that hooks the reader for the next episode.
- Pacing matters: vary panel sizes to control rhythm. Wide panels slow time down,
  small square panels speed it up.

PROSE QUALITY:
- Write dialogue that BREATHES — use contractions, interrupted speech ("I didn't—"), \
  trailing off ("Maybe we could..."), emotional outbursts, and hesitation. Real people \
  don't speak in complete, polished sentences.
- Captions should be evocative and concise — use sensory detail (sounds, smells, light, \
  temperature) and rhythm. Short sentences build tension. Longer ones create wonder. \
  Avoid exposition dumps — if you can say it in 5 words instead of 15, do it.
- Vary sentence length and structure. Monotonous rhythm kills drama. A staccato line \
  after a flowing one creates impact.
- The writing should be ENJOYABLE to read — not just functional. Aim for the quality \
  of a published graphic novel, not a plot summary.
- Avoid overly complex compound sentences. If a sentence has more than two clauses, \
  break it up. Clarity and punch over complexity.

OUTPUT:
Write the complete panel-by-panel script as your response. Include the RECAP at the top \
and the TEASER at the end.\
"""

CARTOONIST_INSTRUCTIONS = """\
You are the Cartoonist — you bring scripts to life through generated images \
and assemble the final comic page.

You will receive the Storyteller's panel-by-panel script as input.

WORKFLOW (follow this exact order):

STEP 1 — READ the complete script from your input. Note all characters, their visual \
descriptions, the art style, and every panel's requirements.

STEP 2 — GENERATE CHARACTER REFERENCE SHEET:
Call generate_character_sheet with:
- description: A detailed description of ALL main characters (appearance, clothing, \
distinguishing features) AND the primary environment/setting for this episode.
- style: The art style specified by the Director (e.g., "ink wash noir", "vibrant manga", \
"watercolor whimsy", "pixel art retro").
This reference image is your visual anchor. Every panel must be consistent with it.

STEP 3 — GENERATE EACH PANEL:
For every panel in the script, call generate_panel_image with:
- prompt: A detailed image generation prompt. Include:
  * Character names and their specific visual traits (hair color, clothing, etc.)
  * Pose, expression, body language
  * Background/environment details
  * Lighting, mood, color palette
  * Camera angle as specified in the script
  * The art style (must be consistent across all panels)
- reference_url: The URL returned from step 2 (the character reference sheet)
- size: "wide", "tall", or "square" as specified in the script

STEP 4 — ASSEMBLE LAYOUT:
After ALL images are generated, call assemble_layout with:
- arc_title: The story arc title
- episode_number: The current episode number
- date: Today's date
- recap: The 2-line recap from the Storyteller's script
- teaser: The 1-line teaser from the Storyteller's script
- panels_json: A JSON string containing a list of panel objects:
  [{{"number": 1, "image_url": "https://...", "size": "wide", "dialogue": "Character: Line", \
"caption": "Narrator text", "sfx": "BOOM"}}, ...]

IMPORTANT RULES:
- ALWAYS use the reference URL from step 2 for every panel — this ensures character consistency.
- Never fabricate or guess image URLs — only use URLs returned by the tools.
- Include character-specific visual details in every prompt to maintain consistency.
- Match the art style across ALL panels.
- NEVER include any text, dialogue, speech bubbles, captions, letters, words, or written \
  language in image prompts. The images should be PURELY VISUAL — characters, environments, \
  actions, expressions only. Text overlays are added separately by the layout system. \
  Explicitly add "no text, no speech bubbles, no captions, no letters, no words" to every \
  image prompt.
- Your final response after calling assemble_layout should confirm the comic was assembled.\
"""


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

@traceable(name="ComicBook Pipeline", run_type="chain")
def run_comic_pipeline(target_date: datetime) -> Dict[str, Any]:
    """Run the Director -> Storyteller -> Cartoonist pipeline. Returns dict with html, narrative, etc."""
    logger.info("=" * 70)
    logger.info("COMIC PIPELINE START — target_date=%s", target_date.strftime("%Y-%m-%d"))
    logger.info("=" * 70)

    client = _build_openai_client()
    model_name = os.environ.get("AZURE_OPENAI_MODEL", "gpt-4o")
    logger.info("Using model: %s", model_name)
    model = OpenAIResponsesModel(
        model=model_name,
        openai_client=client,
    )

    arc = get_active_arc()
    episode_number = (int(arc.get("episodes_count", 0)) + 1) if arc else 1
    recent = get_recent_episodes(arc["RowKey"], limit=5, hydrate_html=False) if arc else []

    if arc:
        logger.info("Active arc found: '%s' (%s), episode #%d", arc.get("title", ""), arc["RowKey"], episode_number)
    else:
        logger.info("No active arc — Director will create a new one")

    prev_episode_images: list[str] = []
    if recent:
        last_ep_html = recent[0].get("html_content", "")
        prev_episode_images = _extract_panel_images(last_ep_html)[:4]

    state: Dict[str, Any] = {
        "arc": arc,
        "episode_number": episode_number,
        "prev_episode_images": prev_episode_images,
        "generated_panel_urls": [],
    }

    # ------------------------------------------------------------------
    # Tool definitions (closures over mutable state)
    # ------------------------------------------------------------------

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
        return {
            "status": "active",
            "arc_id": a["RowKey"],
            "title": a.get("title", ""),
            "logline": a.get("logline", ""),
            "genre": a.get("genre", ""),
            "planned_episodes": a.get("planned_episodes", "unset"),
            "episodes_so_far": int(a.get("episodes_count", 0)),
            "episode_number_today": state["episode_number"],
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
    async def end_current_arc(conclusion_note: str) -> Dict[str, Any]:
        """Close the current story arc when the story has reached its natural conclusion."""
        logger.info("TOOL end_current_arc called: %s", conclusion_note[:100])
        a = state["arc"]
        if not a:
            logger.warning("  -> No active arc to close!")
            return {"error": "No active arc to close"}
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

        prompt = (
            f"Character and environment reference sheet, {style} art style. "
            f"Show all characters clearly with distinct visual features, full body. "
            f"Include the primary setting/environment in the background. "
            f"Clean composition, labeled character positions. "
            f"{description}"
        )
        try:
            url = await create_image(prompt, size="wide")
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
            image_urls = []
            if reference_url:
                image_urls.append(reference_url)
            image_urls.extend(state["generated_panel_urls"][-2:])
            image_urls.extend(state["prev_episode_images"][-2:])
            image_urls = [u for u in image_urls if u]

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
        html = _assemble_html(arc_title, episode_number, date, recap, teaser, panels, theme=arc_theme)
        state["assembled_panels"] = panels
        state["assembled_recap"] = recap
        state["assembled_teaser"] = teaser
        logger.info("  -> Layout assembled, HTML length=%d", len(html))
        return {"status": "success", "html": html, "panel_count": len(panels)}

    # ------------------------------------------------------------------
    # Agent definitions
    # ------------------------------------------------------------------

    cartoonist = Agent(
        name="Cartoonist",
        instructions=CARTOONIST_INSTRUCTIONS,
        tools=[generate_character_sheet, generate_panel_image, assemble_layout],
        model=model,
    )

    storyteller = Agent(
        name="Storyteller",
        instructions=STORYTELLER_INSTRUCTIONS,
        tools=[],
        model=model,
    )

    director = Agent(
        name="Director",
        instructions=DIRECTOR_INSTRUCTIONS,
        tools=[WebSearchTool(search_context_size="high"), get_arc_status, start_new_arc, end_current_arc, save_story_outline],
        model=model,
        model_settings=ModelSettings(temperature=1.2),
    )

    translator = Agent(
        name="Translator",
        instructions=TRANSLATOR_INSTRUCTIONS,
        tools=[],
        model=model,
        model_settings=ModelSettings(temperature=0.9),
    )

    # ------------------------------------------------------------------
    # Build input context
    # ------------------------------------------------------------------

    input_context: Dict[str, Any] = {
        "date": target_date.strftime("%Y-%m-%d"),
    }

    if arc:
        input_context["current_arc"] = {
            "title": arc.get("title", ""),
            "logline": arc.get("logline", ""),
            "genre": arc.get("genre", ""),
            "characters": arc.get("characters", ""),
            "art_style": arc.get("art_style", ""),
            "episode_number": episode_number,
            "planned_episodes": arc.get("planned_episodes", ""),
            "episodes_so_far": int(arc.get("episodes_count", 0)),
        }
        input_context["recent_episodes"] = _summarize_episodes(recent)
        story_outline = get_arc_story_outline(arc)
        if story_outline:
            input_context["story_outline"] = story_outline
    else:
        input_context["current_arc"] = None
        input_context["message"] = (
            "No active story arc. Start by calling get_arc_status, "
            "then create a new arc with start_new_arc."
        )

    input_payload = json.dumps(input_context)

    if target_date.day % 2 == 0:
        input_payload += (
            "\n\n=== CULTURAL DIVERSITY GUIDANCE ===\n"
            "For this arc, explore a NON-WESTERN cultural setting. Draw from: Persian mythology, "
            "West African folklore, Japanese rural life, Brazilian favelas, Polynesian ocean voyages, "
            "Central Asian steppe nomads, Mediterranean fishing villages, Korean historical drama, "
            "Nordic ice towns, Indian temple cities, Mesoamerican civilizations, Balkan mountain "
            "communities, or anywhere else your imagination takes you.\n\n"
            "Character names should reflect their world — use naming conventions from the culture "
            "you're drawing from. A story set in a Persian-inspired world should have Persian names "
            "(Dariush, Soraya, Kaveh), not English ones. A West African setting might use Yoruba "
            "or Akan names (Kofi, Amara, Kweku). Let the names feel authentic.\n"
            "=== END GUIDANCE ==="
        )

    logger.info("Input payload: %s", input_payload[:500])

    # ------------------------------------------------------------------
    # Run the pipeline: Director → Storyteller → Cartoonist (sequential)
    # ------------------------------------------------------------------

    async def _run_sequential():
        # Step 1: Director
        logger.info("STEP 1/4 — Running Director (max_turns=10)...")
        with trace(name="Director", run_type="chain", inputs={"payload_preview": input_payload[:500]}):
            director_result = await Runner.run(director, input_payload, max_turns=10)
            director_plan = str(director_result.final_output)
        logger.info("Director finished. Plan length=%d, preview: %s",
                     len(director_plan), director_plan[:200])

        if not director_plan or len(director_plan) < 50:
            raise RuntimeError(f"Director produced insufficient output: {director_plan!r}")

        # Re-fetch story outline (Director may have just created it via save_story_outline)
        current_outline = ""
        if state["arc"]:
            current_outline = state["arc"].get("story_outline", "")
            if not current_outline:
                current_outline = get_arc_story_outline(state["arc"])

        # Step 2: Storyteller
        logger.info("STEP 2/4 — Running Storyteller (max_turns=5)...")
        storyteller_input = director_plan
        if current_outline:
            storyteller_input = (
                f"=== STORY OUTLINE (for reference — follow this plan) ===\n"
                f"{current_outline}\n"
                f"=== END STORY OUTLINE ===\n\n"
                f"=== DIRECTOR'S EPISODE PLAN ===\n"
                f"{director_plan}"
            )
        with trace(name="Storyteller", run_type="chain"):
            storyteller_result = await Runner.run(storyteller, storyteller_input, max_turns=5)
            storyteller_script = str(storyteller_result.final_output)
        logger.info("Storyteller finished. Script length=%d, preview: %s",
                     len(storyteller_script), storyteller_script[:200])

        if not storyteller_script or len(storyteller_script) < 100:
            raise RuntimeError(f"Storyteller produced insufficient output: {storyteller_script[:200]!r}")

        # Step 3: Cartoonist
        logger.info("STEP 3/4 — Running Cartoonist (max_turns=30)...")
        with trace(name="Cartoonist", run_type="chain"):
            cartoonist_result = await Runner.run(cartoonist, storyteller_script, max_turns=30)
        logger.info("Cartoonist finished.")

        # Step 4: Translations (Italian + Persian) — run sequentially
        html_it = ""
        html_fa = ""
        panels = state.get("assembled_panels", [])
        if panels:
            recap_text = state.get("assembled_recap", "")
            teaser_text = state.get("assembled_teaser", "")
            arc_title_val = state["arc"].get("title", "") if state["arc"] else ""
            ep_num = state["episode_number"]
            date_str_val = target_date.strftime("%Y-%m-%d")
            texts = _build_translation_payload(panels, recap_text, teaser_text, title=arc_title_val)

            arc_id = state["arc"]["RowKey"] if state["arc"] else ""

            story_context = (
                f"=== DIRECTOR'S EPISODE PLAN ===\n{director_plan}\n\n"
                f"=== STORYTELLER'S SCRIPT ===\n{storyteller_script}"
            )

            for lang_code, lang_name in [("it", "Italian"), ("fa", "Persian (Farsi)")]:
                try:
                    logger.info("STEP 4 — Translating to %s...", lang_name)
                    glossary = get_arc_glossary(state["arc"], lang_code) if state["arc"] else {}
                    t_payload = {"target_language": lang_name, "story_context": story_context, **texts}
                    if glossary:
                        t_payload["glossary"] = glossary
                    t_input = json.dumps(t_payload, ensure_ascii=False)
                    with trace(name=f"Translate-{lang_code}", run_type="chain"):
                        t_result = await Runner.run(translator, t_input, max_turns=3)
                    translated = json.loads(str(t_result.final_output))
                    if "texts" in translated:
                        translated = translated["texts"]
                    updated_glossary = translated.pop("updated_glossary", None)
                    if updated_glossary and arc_id:
                        save_arc_glossary(arc_id, lang_code, updated_glossary)
                        if state["arc"]:
                            state["arc"][f"glossary_{lang_code}"] = json.dumps(updated_glossary, ensure_ascii=False)
                        logger.info("  Glossary for %s updated: %d entries", lang_code, len(updated_glossary))
                    t_panels, t_recap, t_teaser, t_title = _apply_translation(panels, recap_text, teaser_text, translated, title=arc_title_val)
                    t_html = _assemble_html(t_title, ep_num, date_str_val, t_recap, t_teaser, t_panels, lang=lang_code, theme=_parse_arc_theme(state["arc"]))
                    if lang_code == "it":
                        html_it = t_html
                    else:
                        html_fa = t_html
                    logger.info("  %s translation complete: %d chars", lang_name, len(t_html))
                except Exception as exc:
                    logger.error("Translation to %s failed (English still saved): %s", lang_name, exc)

        return cartoonist_result, director_plan, storyteller_script, html_it, html_fa

    result, director_plan, storyteller_script, html_it, html_fa = asyncio.run(_run_sequential())

    # ------------------------------------------------------------------
    # Extract results from Cartoonist's tool outputs
    # ------------------------------------------------------------------

    cartoonist_output = str(result.final_output)
    html = None

    for item in result.new_items:
        if not isinstance(item, ToolCallOutputItem):
            continue
        output = item.output
        if not isinstance(output, dict):
            continue
        if "html" in output:
            html = output["html"]

    if html:
        logger.info("HTML extracted from assemble_layout tool output (length=%d)", len(html))
    else:
        logger.error("Cartoonist never called assemble_layout!")
        logger.error("Final output (first 500 chars): %s", cartoonist_output[:500])
        html = (
            "<div style='padding:40px;text-align:center;color:#888;font-family:sans-serif'>"
            "<p>Comic generation completed but layout assembly was skipped.</p>"
            f"<details><summary>Agent output</summary><pre>{_escape_html(cartoonist_output[:3000])}</pre></details>"
            "</div>"
        )

    logger.info("Pipeline result: html_len=%d, arc=%s, episode=%s",
                 len(html),
                 state["arc"]["RowKey"] if state["arc"] else "None",
                 state["episode_number"])
    logger.info("=" * 70)
    logger.info("COMIC PIPELINE END")
    logger.info("=" * 70)

    return {
        "html": html,
        "html_it": html_it,
        "html_fa": html_fa,
        "summary": director_plan,
        "panel_notes": storyteller_script,
        "arc": state["arc"],
        "episode_number": state["episode_number"],
    }
