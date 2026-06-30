from __future__ import annotations

import json
import logging
from typing import List

# Pure, stateless helpers shared by the pipeline (agents.py) and the agent tools
# (tools/agent_tools.py): episode summaries, art-style normalization, comic-page
# HTML assembly, and the Reteller manifest mapping.
logger = logging.getLogger("ComicBook")


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


def _normalize(s: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace — for fuzzy art_style/genre collision checks."""
    import re
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", (s or "").lower())).strip()


def _extract_panel_images(html: str) -> list[str]:
    """Extract panel image URLs from episode HTML content."""
    import re
    return re.findall(r'<img\s+src="([^"]+)"', html or "")


def _escape_html(text: str) -> str:
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _escape_html_multiline(text: str) -> str:
    """Escape HTML and render line breaks as <br>.

    The model frequently writes multi-line text (notably the long episode-1 setup
    recap, in every language) using LITERAL backslash-n sequences rather than real
    newlines. Plain _escape_html leaves those untouched, so the page shows a stray
    "\\n". Here we normalize both literal "\\r\\n"/"\\n"/"\\r" escapes and real
    CR/LF into <br> so the recap reads as the intended separate lines.
    """
    escaped = _escape_html(text)
    escaped = escaped.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
    escaped = escaped.replace("\r\n", "\n").replace("\r", "\n")
    return escaped.replace("\n", "<br>")


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


# ---------------------------------------------------------------------------
# Readability guard — never render light text on a light box (or dark on dark).
# The Director picks the box colors; an LLM can easily pick a low-contrast pair, so we
# verify each text/background pair and flip the TEXT to near-black/near-white when it
# doesn't contrast enough. Colors that are already readable are left untouched.
# ---------------------------------------------------------------------------

_DARK_TEXT = "#161616"
_LIGHT_TEXT = "#f4f4f4"


def _parse_color(value) -> tuple | None:
    """Parse a CSS color (#rgb, #rrggbb, rgb(...), rgba(...)) to (r, g, b); None if unknown."""
    if not value or not isinstance(value, str):
        return None
    s = value.strip().lower()
    if s.startswith("#"):
        h = s[1:]
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        if len(h) == 6:
            try:
                return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            except ValueError:
                return None
        return None
    if s.startswith("rgb"):
        import re
        nums = re.findall(r"[\d.]+", s)
        if len(nums) >= 3:
            try:
                return int(float(nums[0])), int(float(nums[1])), int(float(nums[2]))
            except ValueError:
                return None
    return None


def _rel_luminance(rgb: tuple) -> float:
    def _chan(c: float) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * _chan(r) + 0.7152 * _chan(g) + 0.0722 * _chan(b)


def _contrast_ratio(c1: tuple, c2: tuple) -> float:
    l1, l2 = _rel_luminance(c1), _rel_luminance(c2)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def _readable_text(text_color, bg_color, min_ratio: float = 4.5) -> str:
    """Keep text_color if it contrasts enough with bg_color; otherwise return near-black or
    near-white — whichever reads better on that background."""
    bg = _parse_color(bg_color)
    if bg is None:
        return text_color  # unknown background — can't assess, leave the Director's choice
    txt = _parse_color(text_color)
    if txt is not None and _contrast_ratio(txt, bg) >= min_ratio:
        return text_color
    return _DARK_TEXT if _rel_luminance(bg) > 0.4 else _LIGHT_TEXT


def _ensure_readable_theme(t: dict) -> dict:
    """Return a copy of the theme with every box's text color guaranteed readable on its box."""
    t = dict(t)
    # Large/bold text (title, teaser) tolerates a lower ratio than small body text.
    t["title_color"] = _readable_text(t.get("title_color"), t.get("page_bg"), min_ratio=3.0)
    t["teaser_color"] = _readable_text(t.get("teaser_color"), t.get("page_bg"), min_ratio=3.0)
    t["caption_text"] = _readable_text(t.get("caption_text"), t.get("caption_bg"))
    # recap and speech bubbles get their own derived text colors so each reads on its own box.
    t["recap_text"] = _readable_text(t.get("recap_text", t.get("caption_text")), t.get("recap_bg"))
    t["speech_text"] = _readable_text(t.get("speech_text", "#161616"), t.get("speech_bg"))
    return t


def _assemble_html(
    arc_title: str,
    episode_number: int,
    date_str: str,
    recap: str,
    teaser: str,
    panels: list,
    lang: str = "en",
    theme: dict | None = None,
    subtitle: str = "",
) -> str:
    grid_template = _build_grid_template(panels)
    is_rtl = lang == "fa"
    t = {**_DEFAULT_THEME, **(theme or {})}
    t = _ensure_readable_theme(t)

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

        # Fixed positions for every language: caption at the top, dialogue/sfx stacked at the
        # bottom (RTL handled by rtl_css). The Reteller controls the words, not the placement.
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
    safe_recap = _escape_html_multiline(recap)
    safe_teaser = _escape_html(teaser)
    subtitle_block = (
        f'\n    <div class="comic-subtitle">{_escape_html(subtitle)}</div>' if subtitle else ""
    )

    return f"""<style>
@import url('{t["font_import"]}');{font_import_extra}
.comic-page {{ font-family: {t["heading_font"]}; max-width: 960px; margin: 0 auto; background: {t["page_bg"]}; border-radius: 12px; padding: 24px; box-shadow: 0 4px 24px rgba(0,0,0,0.18); }}
.comic-header {{ text-align: center; border-bottom: 4px solid {t["header_border"]}; padding-bottom: 16px; margin-bottom: 20px; }}
.comic-title {{ font-size: 2.6em; color: {t["title_color"]}; text-transform: uppercase; letter-spacing: 3px; margin: 0; text-shadow: 2px 2px 0 {t["title_shadow"]}; }}
.comic-subtitle {{ color: {t["title_color"]}; font-family: {t["body_font"]}; font-size: 1.15em; font-style: italic; margin-top: 6px; opacity: 0.85; letter-spacing: 0.5px; }}
.comic-meta {{ color: {t["title_color"]}; font-size: 1em; margin-top: 6px; letter-spacing: 1px; font-weight: bold; opacity: 0.7; }}
.comic-recap {{ color: {t["recap_text"]}; margin: 14px 0 20px; padding: 12px 16px; background: {t["recap_bg"]}; border-left: 5px solid {t["recap_border"]}; border-radius: 4px; font-family: {t["body_font"]}; font-size: 0.95em; line-height: 1.6; font-style: italic; }}
.comic-panels {{ display: grid; grid-template-columns: 1fr 1fr; grid-template-areas: {grid_template}; gap: 10px; }}
.panel {{ position: relative; border: 3px solid {t["header_border"]}; border-radius: 6px; overflow: hidden; background: #fff; box-shadow: 2px 3px 6px rgba(0,0,0,0.25); }}
.panel img {{ width: 100%; height: 100%; object-fit: cover; display: block; cursor: pointer; }}
.caption-overlay {{ position: absolute; top: 0; left: 0; right: 0; display: flex; justify-content: center; padding: 8px 12px; pointer-events: none; }}
.caption-box {{ background: {t["caption_bg"]}; color: {t["caption_text"]}; border: 2px solid {t["caption_border"]}; padding: 6px 14px; font-style: italic; font-size: 0.82em; font-family: {t["body_font"]}; border-radius: 3px; max-width: 90%; text-align: center; line-height: 1.4; box-shadow: 1px 1px 0 rgba(0,0,0,0.15); }}
.panel-overlay {{ position: absolute; bottom: 0; left: 0; right: 0; padding: 6px 8px; display: flex; flex-direction: column; gap: 2px; align-items: flex-start; pointer-events: auto; background: linear-gradient(transparent 0%, rgba(0,0,0,0.35) 100%); }}
.speech-bubble {{ background: {t["speech_bg"]}; color: {t["speech_text"]}; border: 1px solid {t["speech_border"]}; border-radius: 14px; padding: 3px 8px; font-size: 0.54em; max-width: 80%; box-shadow: 1px 1px 0 rgba(0,0,0,0.15); font-family: {t["heading_font"]}; letter-spacing: 0.3px; line-height: 1.25; transform-origin: bottom left; transition: transform 0.2s ease, padding 0.2s ease, font-size 0.2s ease, border-width 0.2s ease, border-radius 0.2s ease, box-shadow 0.2s ease; cursor: pointer; }}
.speech-bubble:hover, .speech-bubble:active {{ font-size: 1em; padding: 8px 16px; border-width: 2px; border-radius: 18px; box-shadow: 2px 2px 0 rgba(0,0,0,0.2); z-index: 10; position: relative; }}
.sfx {{ font-size: 2em; font-weight: 900; color: {t["sfx_color"]}; text-shadow: 2px 2px 0 #ffd166, -1px -1px 0 #111, 1px -1px 0 #111, -1px 1px 0 #111; font-style: italic; letter-spacing: 2px; }}
.comic-footer {{ text-align: center; margin-top: 20px; padding-top: 14px; border-top: 4px solid {t["header_border"]}; }}
.teaser {{ font-weight: bold; color: {t["teaser_color"]}; font-size: 1.15em; font-style: italic; letter-spacing: 0.5px; }}
.comic-branding {{ color: #888; font-size: 0.75em; margin-top: 8px; font-family: sans-serif; }}{rtl_css}
@media (max-width: 600px) {{
  .comic-page {{ padding: 10px; }}
  .comic-title {{ font-size: 1.6em; letter-spacing: 1px; }}
  .comic-subtitle {{ font-size: 0.95em; }}
  .comic-panels {{ grid-template-columns: 1fr; grid-template-areas: none; }}
  .panel {{ grid-area: auto !important; }}
  .speech-bubble {{ max-width: 92%; }}
  .speech-bubble:hover, .speech-bubble:active {{ font-size: 0.92em; }}
  .caption-box {{ font-size: 0.78em; }}
}}
</style>
<div class="comic-page"{dir_attr}>
  <div class="comic-header">
    <h1 class="comic-title">{safe_title}</h1>{subtitle_block}
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


def _build_reteller_payload(panels: list, recap: str, teaser: str, title: str = "") -> dict:
    """Build the per-panel manifest the Reteller rewrites natively.

    The English fields are INTENT REFERENCE ONLY — the Reteller may restructure them freely.
    Per-language fields (target_language, story_context, story_outline_local, glossary) are
    added by the caller in the language loop.
    """
    manifest = []
    for i, p in enumerate(panels):
        manifest.append({
            "number": p.get("number", i + 1),
            "size": p.get("size", "square"),
            "en_dialogue": p.get("dialogue", ""),
            "en_caption": p.get("caption", ""),
            "en_sfx": p.get("sfx", ""),
        })
    return {"title_en": title, "recap_en": recap, "teaser_en": teaser, "panels": manifest}


def _apply_reteller_output(panels: list, recap: str, teaser: str, native: dict, title: str = "") -> tuple:
    """Map the natively-written text onto the shared panels, matched by panel number.

    The English panels own the fixed image_url/size and the fixed box positions; only the
    text (dialogue/caption/sfx) changes per language. English text must NEVER leak onto a
    target-language page: a panel the Reteller didn't cover renders art-only (empty text),
    not the English skeleton.
    """
    native_panels = native.get("panels", []) if isinstance(native, dict) else []
    by_number = {}
    for np in native_panels:
        if isinstance(np, dict) and np.get("number") is not None:
            by_number[np["number"]] = np

    new_panels = []
    for i, p in enumerate(panels):
        num = p.get("number", i + 1)
        match = by_number.get(num)
        if match is None and i < len(native_panels) and isinstance(native_panels[i], dict):
            match = native_panels[i]  # positional fallback when numbers don't line up
        if match is None:
            logger.warning("Reteller: no native text matched panel %s — rendering art only.", num)
            match = {}
        new_panels.append({
            "number": num,
            "size": p.get("size", "square"),
            "image_url": p.get("image_url", ""),
            "dialogue": match.get("dialogue", "") if isinstance(match, dict) else "",
            "caption": match.get("caption", "") if isinstance(match, dict) else "",
            "sfx": match.get("sfx", "") if isinstance(match, dict) else "",
        })
    t_title = native.get("title", title) if isinstance(native, dict) else title
    t_recap = native.get("recap", recap) if isinstance(native, dict) else recap
    t_teaser = native.get("teaser", teaser) if isinstance(native, dict) else teaser
    return new_panels, t_recap, t_teaser, t_title
