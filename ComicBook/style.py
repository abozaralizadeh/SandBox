"""Art-style specification for the comic pipeline.

Pure data + deterministic helpers — no LLM calls, no storage access, no imports from
`agents.py` or `tools/`, so both can import this without a cycle (same role as `helpers.py`).

WHY THIS MODULE EXISTS
----------------------
Arcs used to carry a 2-3 word `art_style` label ("watercolour whimsy"), and that label never
reached the image API programmatically — only if the Cartoonist happened to retype it into a
panel prompt. A short adjective phrase leaves nearly every pixel decision to gpt-image's own
prior, which is a single strong attractor: soft airbrushed digital painting, mid-saturation,
ambient shading, naturalistic 6-7-head figures. So every arc landed in the same place no matter
what the label said.

A StyleCard replaces the label with a full production spec, and `compose_image_prompt` /
`compose_sheet_prompt` paste it into EVERY image prompt from code, so the style no longer
depends on an agent remembering to mention it.

Two design constraints that shape the schema:

1. gpt-image has NO negative-prompt channel. "avoid glossy 3D" reliably summons glossy 3D.
   So the avoid-language is split: `contrastive_assertions` are POSITIVE claims that make the
   default look impossible and DO go to the image API; `generic_tells` are the negative
   phrasings and are AUDIT-ONLY — never sent.
2. The phrase "character reference sheet" is itself a style command with a stronger visual
   prior than any style adjective (clean flat turnaround on white), and that sheet anchors
   every panel of the arc. `sheet_conceit` replaces it with an artefact the medium would
   actually produce — a photographed shelf of figures, a printed sheet of stamps.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from pydantic import BaseModel

# Bump when the card schema or the prompt composition changes in a way that should
# re-commission the style of an arc that is already running (see the restyle hatch).
STYLE_CARD_VERSION = 1


# A coarse partition of HOW an image is physically produced. Deliberately NOT a list of
# looks — "photographic" implies neither Ghibli nor Lego. This is the skeleton the LRU
# rotation needs in order to be deterministic; the ArtDirector supplies all the flesh, and
# may return a family outside this tuple, which is stored verbatim and treated by the LRU as
# never-used (so inventing one is rewarded, not punished).
STYLE_FAMILIES = (
    "photographic",
    "sculptural-object",
    "print-reproduction",
    "painterly-traditional",
    "animation-cel",
    "graphic-vector",
    "textile-craft",
    "historical-fine-art",
    "digital-native",
)

# Figure-construction convention: the single axis a reader feels first, and the one that
# most cheaply separates a minifigure from a cel-animated child from a heroic comic figure.
# Two arcs in the same family can still look identical; two arcs in the same CONSTRUCTION
# almost always do.
CONSTRUCTION_BUCKETS = (
    "super-deformed-chibi",   # 2-3 heads tall
    "cartoon-round",          # 4-5 heads, rubber-hose / bouncy
    "stylized-natural",       # 6-7 heads
    "heroic-idealized",       # 8-9 heads, exaggerated musculature
    "photoreal-anatomic",
    "toy-articulated",        # rigid joints, moulded seams, no soft tissue
    "abstract-geometric",
    "silhouette-flat",
)


NO_TEXT_CLAUSE = (
    "No text, no speech bubbles, no captions, no letters, no words, no writing, "
    "no labels, no signatures, no watermarks."
)


# Tokens that summon the AI-default look. The Cartoonist adds these reflexively ("cinematic
# lighting, highly detailed, vibrant, 4k") and they fight the style block from inside the same
# prompt. Stripping them is not hardcoding a style — it removes anti-style noise.
_GENERIC_TOKEN_PATTERNS = (
    r"cinematic(?:\s+lighting)?",
    r"hyper[-\s]?realistic",
    r"ultra[-\s]?realistic",
    r"photo[-\s]?realistic",
    r"highly\s+detailed",
    r"hyper[-\s]?detailed",
    r"intricately\s+detailed",
    r"extremely\s+detailed",
    r"\b[48]k\b",
    r"digital\s+art",
    r"digital\s+painting",
    r"concept\s+art",
    r"trending\s+on\s+\w+",
    r"artstation",
    r"octane(?:\s+render)?",
    r"unreal\s+engine",
    r"masterpiece",
    r"award[-\s]?winning",
    r"volumetric(?:\s+lighting)?",
    r"bokeh",
    r"dramatic\s+lighting",
    r"vibrant\s+colou?rs?",
    r"stunning",
    r"breathtaking",
    r"beautifully\s+rendered",
    r"high\s+quality",
    r"ultra[-\s]?hd",
)

_GENERIC_TOKEN_RE = re.compile(
    r"\b(?:" + "|".join(_GENERIC_TOKEN_PATTERNS) + r")\b", re.IGNORECASE
)


class StyleCard(BaseModel):
    """The full production spec for one arc's look.

    Every field is REQUIRED and typed `str` or `list[str]`. The Agents SDK converts
    `output_type` into a strict JSON schema, and Optional / defaults / dicts / unions can make
    that conversion raise at agent-construction time. Keep it flat and required.
    """

    # --- identity / comparison axes (not sent to the image API) ---------------------
    style_name: str            # 2-5 words. Logs, UI, and collision checks only. May name lineage.
    style_family: str          # from STYLE_FAMILIES, or a novel one the ArtDirector coined
    construction_bucket: str   # from CONSTRUCTION_BUCKETS
    lineage_note: str          # where this look comes from; context for humans, never rendered

    # --- the spec ------------------------------------------------------------------
    sheet_conceit: str             # what artefact THIS medium uses to show a whole cast at once
    medium_and_process: str        # physical substrate + tool + process
    linework: str                  # or the explicit absence of line
    color_and_palette: str         # named hues, saturation, how colour is laid down
    shading_and_light: str         # flat / cel / hatched / global-illumination / hard key
    texture_and_surface: str       # grain, weave, dot, plastic sheen, paper tooth
    character_construction: str    # prose: proportions, faces, hands, eyes, silhouette
    composition_and_framing: str   # camera convention, depth, staging, border treatment

    # --- what actually reaches the image API ---------------------------------------
    render_directive: str              # 90-160 words, pasted VERBATIM into every image prompt.
                                       # MUST be free of studio/brand/franchise/living-artist names:
                                       # the safety layer rejects style-mimicry by name and the
                                       # panel comes back blank.
    contrastive_assertions: List[str]  # 4-6 POSITIVE surface claims. Sent to the image API.

    # --- audit-only / page-level ----------------------------------------------------
    generic_tells: List[str]   # "if you can see this, the render failed". NEVER sent to the API.
    page_palette: List[str]    # 4-6 hex colours for the surrounding page CSS
    render_quality: str        # "medium" | "high" — texture-heavy styles need "high"


class ObservedStyle(BaseModel):
    """What a blind cataloguer reports after looking at a rendered image.

    The agent that fills this in is shown the image and NOTHING else — not the arc, not the
    declared style, not what it was supposed to be. That blindness is structural, not a matter
    of prompt politeness: an auditor told "does this match the declared style?" has been handed
    the answer and will agree.
    """

    medium_and_process: str
    linework: str
    color_and_palette: str
    shading_and_light: str
    texture_and_surface: str
    character_construction: str
    construction_bucket: str
    style_family: str
    one_line_catalogue_entry: str


class StyleAudit(BaseModel):
    """The verdict, written by an agent that sees the two cards and NEVER the image."""

    verdict: str                   # "pass" | "fail"
    overall_score: float           # 0.0 - 1.0
    axis_scores: List[str]         # "medium_and_process: 0.3 - reads as soft digital painting"
    worst_axis: str
    generic_tells_present: List[str]
    intensified_directive: str     # complete replacement render_directive; "" when passing
    conceit_advice: str            # a different sheet artefact to try; "" when passing


# ---------------------------------------------------------------------------
# Fallbacks — pure code, no LLM. Every arc that predates the StyleCard must keep working.
# ---------------------------------------------------------------------------

def neutral_style_card() -> StyleCard:
    """A card that renders exactly today's behaviour: no style block at all.

    Used when there is no arc yet (the Cartoonist can be asked for an image before
    `state["arc"]` exists). `compose_image_prompt` degrades to subject + no-text.
    """
    return StyleCard(
        style_name="",
        style_family="",
        construction_bucket="",
        lineage_note="",
        sheet_conceit="",
        medium_and_process="",
        linework="",
        color_and_palette="",
        shading_and_light="",
        texture_and_surface="",
        character_construction="",
        composition_and_framing="",
        render_directive="",
        contrastive_assertions=[],
        generic_tells=[],
        page_palette=[],
        render_quality="",
    )


def legacy_style_card(art_style: str) -> StyleCard:
    """Synthesize a minimal card from a pre-StyleCard arc's `art_style` string.

    Deliberately weak: it reproduces almost exactly the prompt these arcs get today, so
    shipping the StyleCard machinery does NOT restyle a story that is already halfway
    through its run. Only the restyle hatch changes an in-flight arc on purpose.
    """
    art_style = (art_style or "").strip()
    if not art_style:
        return neutral_style_card()
    card = neutral_style_card()
    card.style_name = art_style
    card.render_directive = f"{art_style} art style."
    return card


def load_style_card(arc: Dict[str, Any] | None) -> StyleCard:
    """Resolve an arc entity to a StyleCard: stored card → legacy string → neutral.

    Never raises: a malformed or partial stored card falls back to the legacy string rather
    than taking down a day's comic.
    """
    if not arc:
        return neutral_style_card()

    raw = arc.get("style_card", "")
    if raw:
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
            return StyleCard.model_validate(data)
        except Exception:  # noqa: BLE001 — a bad card must degrade, not fail the run
            pass

    return legacy_style_card(arc.get("art_style", ""))


# ---------------------------------------------------------------------------
# Prompt composition — the mechanical fix. Style enters every image prompt HERE, in code,
# rather than depending on an agent to retype it.
# ---------------------------------------------------------------------------

def scrub_generic_style_tokens(text: str) -> str:
    """Strip the quality-adjective tokens that summon gpt-image's default look."""
    cleaned = _GENERIC_TOKEN_RE.sub("", text or "")
    cleaned = re.sub(r"\s*,\s*(?=,)", "", cleaned)      # collapse ", ,"
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)     # " ," -> ","
    cleaned = re.sub(r"(?:,\s*){2,}", ", ", cleaned)
    return cleaned.strip().strip(",").strip()


def _surface_rules(card: StyleCard, override_line: str) -> str:
    """The trailing restatement. gpt-image weights early tokens most, so the directive leads —
    but a compact surface assertion at the tail measurably resists drift, and the no-text
    clause (which already works today) stays last."""
    parts: List[str] = []
    if card.contrastive_assertions:
        parts.append(
            f"SURFACE RULES — not optional; {override_line}: "
            + "; ".join(a.strip().rstrip(".") for a in card.contrastive_assertions)
            + "."
        )
    if card.character_construction:
        parts.append(f"Figures are built to this convention: {card.character_construction}")
    if card.medium_and_process:
        parts.append(f"Every square centimetre is {card.medium_and_process}")
    return "\n".join(parts)


def compose_image_prompt(card: StyleCard, subject: str) -> str:
    """STYLE BLOCK (verbatim) → SUBJECT → SURFACE RULES → no-text clause.

    The "overrides the look of any reference image" framing is deliberate: panels go through
    images.edit with reference images attached, which pulls the render toward whatever those
    references look like. This keeps the references supplying character IDENTITY while the
    style block stays authoritative for the LOOK — which also protects the arc when a grey
    placeholder has entered the reference stack.
    """
    subject = (subject or "").strip().rstrip(".")
    blocks: List[str] = []

    if card.render_directive:
        blocks.append(
            "STYLE — this is how the image must be rendered. It overrides the look of any "
            "reference image; references supply character identity only.\n"
            f"{card.render_directive}"
        )

    blocks.append(f"SUBJECT: {subject}." if subject else "SUBJECT: as described.")

    rules = _surface_rules(card, "these override any other reading of the subject")
    if rules:
        blocks.append(rules)

    blocks.append(NO_TEXT_CLAUSE)
    return "\n\n".join(blocks)


def compose_sheet_prompt(card: StyleCard, characters: str) -> str:
    """The arc's reference sheet, framed as an ARTEFACT OF THE MEDIUM.

    Never the words "character reference sheet": that phrase carries its own strong prior
    (clean flat turnaround on white) which outranks any style adjective, and because this
    image is reference #1 for every panel of the arc, that prior is what propagates. Also
    drops the old prompt's "each labeled by name", which contradicted its own "no text
    overlays" instruction and seeded a text-tendency into every panel reference.
    """
    characters = (characters or "").strip()

    # A conceit like "the whole troupe" invites the model to pad the cast out with invented
    # figures — and because this sheet is reference #1 for every panel, invented characters
    # would then haunt the entire arc. The roster is closed, explicitly.
    roster_rule = (
        "Show ONLY the characters listed below and no others — do not invent, duplicate, or add "
        "any extra figures to fill the composition."
    )

    if card.sheet_conceit:
        subject = (
            f"{card.sheet_conceit.strip().rstrip('.')}. It presents every one of the following "
            f"characters head-to-toe, well separated, faces clearly visible, plus the story's "
            f"primary setting behind them. {roster_rule}"
        )
    else:
        # Legacy / neutral card: keep close to the historical prompt so in-flight arcs are
        # not restyled by this change alone.
        subject = (
            "A reference image showing all of the following characters head-to-toe, well "
            "separated, faces clearly visible, with the story's primary setting behind them. "
            + roster_rule
        )

    blocks: List[str] = []
    if card.render_directive:
        blocks.append(
            "STYLE — this is how the image must be rendered, in full, edge to edge.\n"
            f"{card.render_directive}"
        )
    blocks.append(f"SUBJECT: {subject}\n\nCharacters: {characters}")

    rules = _surface_rules(card, "these override any impulse toward a neutral illustrated line-up")
    if rules:
        blocks.append(rules)

    blocks.append(NO_TEXT_CLAUSE)
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Rotation — computed from real arc history, so the anti-repetition guarantee is
# deterministic rather than something an LLM is asked to remember.
# ---------------------------------------------------------------------------

def starved_families(recent_arcs: List[dict], lookback: int = 10) -> List[str]:
    """Families not used recently, least-recently-used first.

    `recent_arcs` is newest-first (as `get_recent_arc_summaries` returns). Arcs predating the
    StyleCard have no family; they are skipped rather than counted, which biases the result
    toward "everything is starved" — harmless, and self-correcting after ~10 new arcs.

    Families the ArtDirector invented appear here too once they are in storage, and any family
    never seen sorts to the front.
    """
    used_order: List[str] = []
    for arc in recent_arcs[:lookback]:
        fam = (arc.get("style_family") or "").strip().lower()
        if fam and fam not in used_order:
            used_order.append(fam)

    pool: List[str] = list(STYLE_FAMILIES)
    for fam in used_order:
        if fam not in pool:
            pool.append(fam)

    # Never-used first (preserving the tuple's order), then by how long ago it was used.
    never_used = [f for f in pool if f not in used_order]
    stale_first = [f for f in reversed(used_order) if f in pool]
    return never_used + stale_first


def _token_overlap(a: str, b: str) -> float:
    """Jaccard overlap of the content words of two phrases."""
    stop = {"a", "an", "the", "of", "on", "in", "with", "and", "or", "to", "is", "are", "by"}
    ta = {w for w in re.findall(r"[a-z0-9]+", (a or "").lower()) if w not in stop}
    tb = {w for w in re.findall(r"[a-z0-9]+", (b or "").lower()) if w not in stop}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def card_collision(card: StyleCard, recent_arcs: List[dict]) -> Dict[str, Any] | None:
    """Is this card too close to a recent arc? Returns the offending axis, or None.

    Deliberately NOT a string comparison of style names: a different label is not a different
    image, which is exactly how the series ended up with a dozen names for one look. What is
    compared is how the image is physically made and how figures are built.

    Windows differ per axis because the axes differ in how strongly they carry a look:
      - construction (3 arcs) — the axis a reader feels first; two arcs sharing it look alike
        almost regardless of anything else.
      - family (2 arcs) — coarse; repeating it soon is bad, but it is a wide bucket.
      - medium overlap (5 arcs) — catches "risograph zine print" vs "risograph print zine".
    """
    fam = (card.style_family or "").strip().lower()
    con = (card.construction_bucket or "").strip().lower()

    for arc in recent_arcs[:3]:
        if con and con == (arc.get("style_construction") or "").strip().lower():
            return {
                "axis": "construction_bucket",
                "value": card.construction_bucket,
                "message": (
                    f"Figure construction '{card.construction_bucket}' was used by one of the "
                    f"last three arcs. This is the axis readers notice first — two arcs that "
                    f"build figures the same way look like the same artist however different "
                    f"the medium is. Choose a different construction."
                ),
            }

    for arc in recent_arcs[:2]:
        if fam and fam == (arc.get("style_family") or "").strip().lower():
            return {
                "axis": "style_family",
                "value": card.style_family,
                "message": (
                    f"Production family '{card.style_family}' was used by one of the last two "
                    f"arcs. Search inside a starved family instead."
                ),
            }

    for arc in recent_arcs[:5]:
        prev_medium = arc.get("medium_and_process") or ""
        if prev_medium and _token_overlap(card.medium_and_process, prev_medium) >= 0.6:
            return {
                "axis": "medium_and_process",
                "value": card.medium_and_process,
                "message": (
                    "This is the same physical process as a recent arc, described in different "
                    f"words (that arc: '{prev_medium[:160]}'). Change how the image is actually "
                    "made, not the vocabulary."
                ),
            }

    return None


def banned_constructions(recent_arcs: List[dict], n: int = 3) -> List[str]:
    """Figure-construction buckets used by the last `n` arcs — off-limits for the next one.

    Nine families alone give a long cycle, but two arcs both in `painterly-traditional` can
    still look identical. This is the second rotation axis that prevents that.
    """
    out: List[str] = []
    for arc in recent_arcs[:n]:
        bucket = (arc.get("style_construction") or "").strip().lower()
        if bucket and bucket not in out:
            out.append(bucket)
    return out
