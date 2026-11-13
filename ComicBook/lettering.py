"""Build the prompt that makes the image model letter a panel.

Why the model and not code: a balloon has to point at the right character, and only something
that can SEE the panel knows which figure is which. A Pillow version of this placed balloons in
the quietest region and guessed the tails, which put one character's line over another's head
roughly half the time. The model reads the cast description, finds that person in the picture
and aims the tail at them, in English, Italian and Persian alike.

This module is pure string work: no network, no storage, safe to import anywhere.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# Lettering is one extra image edit per panel per language, so it is worth a kill switch.
# Set COMICBOOK_LETTERING=false to fall straight back to the HTML overlay.
LANG_RULES = {
    "en": "Hand-lettered comic capitals in black ink, evenly spaced and correctly spelled.",
    "it": ("The text is ITALIAN. Hand-lettered comic capitals in black ink. Reproduce every "
           "accented letter exactly as given (à è é ì ò ù) and never translate it."),
    "fa": ("The text is PERSIAN (Farsi), written in Arabic script and read RIGHT TO LEFT. Render "
           "it as properly joined cursive Persian with correct letter shapes, right-aligned "
           "inside each balloon. Do NOT transliterate it, do NOT use Latin letters, do NOT "
           "reverse the order of the letters, and do NOT translate it."),
}


def cast_block(characters: str, limit: int = 6) -> str:
    """Condense the arc's character roster into 'NAME: what they look like' lines.

    The roster is free text written by the Director, shaped like
    'ADA BELL — ... Visual: straight black hair in a thick single braid ... Exact outfit: ...'.
    We keep the FULL name (dialogue labels use a part of it, and a nickname in quotes is often
    the part that gets used) plus the visual sentences, which is what the model needs to find
    the person in the frame. The personality prose is dropped.
    """
    if not characters:
        return ""
    # A record starts with an all-caps name, possibly several words and a "NICKNAME" in quotes,
    # followed by a dash.
    head = re.compile(
        r'([\u201c\u201d"\u2019\']?[A-Z][A-Z\u2019\u201c\u201d\'"\.\-]*'
        r'(?:\s+[\u201c\u201d"\u2019\']?[A-Z][A-Z\u2019\u201c\u201d\'"\.\-]*)*)'
        r'\s*[\u2014\u2013]\s*')
    marks = list(head.finditer(characters))
    lines: List[str] = []
    for i, m in enumerate(marks):
        name = re.sub(r"\s+", " ", m.group(1)).strip()
        if len(name) < 3:
            continue
        body = characters[m.end(): marks[i + 1].start() if i + 1 < len(marks) else len(characters)]
        look = re.search(r"Visual:\s*(.+?)(?:\s*Exact outfit:\s*(.+))?$", body, re.S)
        if look:
            desc = (look.group(1) or "").strip()
            if look.group(2):
                desc += " Wearing: " + look.group(2).strip()
        else:
            desc = body.strip()
        desc = re.sub(r"\s+", " ", desc).strip().rstrip(".")
        if desc:
            lines.append(f"- {name}: {desc[:300]}")
        if len(lines) >= limit:
            break
    return "\n".join(lines)


def _split_speaker(raw: str) -> tuple:
    """'ADA: line' -> ('ADA', 'line'). Works for localized labels too, which are not Latin."""
    m = re.match(r"^\s*([^:\n]{1,40}?)\s*:\s*(.+)$", (raw or "").strip(), re.S)
    if m and len(m.group(1).split()) <= 4:
        return m.group(1).strip(), m.group(2).strip()
    return "", (raw or "").strip()


def build_lettering_prompt(panel: Dict[str, Any], cast: str, lang: str = "en") -> Optional[str]:
    """The edit prompt for one panel, or None when there is nothing to letter."""
    dialogue = panel.get("dialogue") or ""
    lines_in = [l for l in (dialogue.split("\n") if isinstance(dialogue, str) else dialogue) if l.strip()]
    caption = (panel.get("caption") or "").strip()
    sfx = (panel.get("sfx") or "").strip()
    if not (lines_in or caption or sfx):
        return None

    parts = [
        "Add comic-book lettering to this image. Do NOT change the artwork, the characters, "
        "their poses, the colours or the composition in any way. Only add lettering.",
    ]
    if lines_in:
        parts.append(
            "\nDraw these speech balloons in classic comic style: opaque white, clean black "
            "outline, rounded, each with a short tail pointing at the character who says it."
        )
        if cast:
            parts.append(
                "\nThe cast, so you can tell who is who in the picture (a speaker's name may be "
                "written in the local language, match it to this list):\n" + cast
            )
        parts.append("")
        for raw in lines_in:
            speaker, text = _split_speaker(raw)
            who = f"the character called {speaker}" if speaker else "the character who is speaking"
            parts.append(f"- A balloon for {who}, tail pointing at them, containing exactly this "
                         f"and nothing else: {text}")
    if caption:
        first = re.split(r"(?<=[.!?])\s", caption)[0].strip()
        parts.append(
            f"\nDraw one rectangular narration box in a top corner, in the page's own caption "
            f"colours, containing exactly this and nothing else: {first}"
        )
    if sfx:
        parts.append(f"\nDraw the sound effect {sfx} as big stylised comic lettering over the "
                     f"action, angled, with a bold outline.")

    parts.append("\n" + LANG_RULES.get(lang, LANG_RULES["en"]))
    parts.append(
        "Copy the wording exactly: nothing added, nothing dropped, nothing abbreviated and "
        "nothing paraphrased. No signature, no watermark, no page number, no extra text of any "
        "kind. Balloons sit on empty background and must never cover a face."
    )
    return "\n".join(parts)
