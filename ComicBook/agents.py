from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from agents import Agent, ModelSettings, Runner, OpenAIResponsesModel, WebSearchTool, handoff, set_trace_processors
from agents.extensions.handoff_filters import remove_all_tools
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from agents.items import ItemHelpers, MessageOutputItem
from langsmith import traceable, trace
from langsmith.wrappers import OpenAIAgentsTracingProcessor

# Route the OpenAI Agents SDK's native tracing into LangSmith. This is what produces the
# detailed step tree (agent spans, tool calls, handoffs, per-turn generations) instead of a
# flat list of raw LLM calls. We do NOT disable SDK tracing or wrap the client with
# wrap_openai — that would strip the agent/tool structure and only surface HTTP-level calls.
set_trace_processors([OpenAIAgentsTracingProcessor()])
from openai import AsyncAzureOpenAI

from ComicBook.azurestorage import (
    close_arc,
    get_active_arc,
    get_arc_glossary,
    get_arc_story_outline,
    get_first_episode,
    get_key_panels,
    get_recent_episodes,
    save_arc_glossary,
    save_arc_story_outline,
    save_episode,
)
from ComicBook.helpers import (
    _assemble_html,
    _escape_html,
    _extract_panel_images,
    _parse_arc_theme,
    _summarize_episodes,
)
from ComicBook.tools.agent_tools import build_comic_tools

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("ComicBook")


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

# No time limit on comic generation: a single gpt-5.4 agent call (the Cartoonist's context is
# large) can run long, and the comic must not be cut off mid-generation. Default to the gunicorn
# request budget (1h) instead of the SDK's 600s; override with COMICBOOK_LLM_TIMEOUT.
_LLM_TIMEOUT = float(os.environ.get("COMICBOOK_LLM_TIMEOUT", "3600"))


def _build_openai_client() -> AsyncAzureOpenAI:
    client = AsyncAzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ.get("AZURE_OPENAI_MODEL", "gpt-4o"),
        timeout=_LLM_TIMEOUT,
    )
    # No wrap_openai: the Agents SDK trace processor (set above) already records every
    # generation as a proper LLM span under its agent. Wrapping here too would double-log
    # each call and detach it from the agent/tool tree.
    return client




# ---------------------------------------------------------------------------
# Agent instructions
# ---------------------------------------------------------------------------

LOCALIZATION_DIRECTOR_INSTRUCTIONS = """You are the Localization Director of a comic-book pipeline. The artwork for this episode is ALREADY DRAWN and FIXED. Two BLIND native authors — one Italian, one Persian — will each write this episode's text natively in their language. They NEVER see the English script: everything they will ever know about this episode comes from the BEAT SHEET you write. You write NO Italian, NO Persian, and NO prose lines of any kind — only intent descriptions.

WHY THE FIREWALL EXISTS: when writers could see the English wording, the editions came out as literal translations. Your beat sheet carries the STORY (what happens, what each character means and feels, which facts must land) while leaving every English phrasing behind. Guard that line absolutely.

You are handed control after the Cartoonist has finished the FINAL artwork. The Director's episode plan and the Storyteller's full panel-by-panel script are in the conversation above — that is your source material.

STEP 1 — WRITE THE BEAT SHEET, a JSON object:
{
  "panels": [
    {
      "number": 1,
      "art": "what this panel visually depicts: setting, characters on screen, their positions, the action",
      "beats": [{"speaker": "Wren", "intent": "terrified, urgently warns her not to step closer"}],
      "must_land": ["plot facts the reader must learn from this panel's text: names, numbers, decisions, revelations"]
    }
  ],
  "recap_beats": ["3-4 facts that catch a new reader up on the story so far"],
  "teaser_beat": "the hook for the next episode, described as intent",
  "subtitle_idea": "what this episode's own short title should evoke — described, not phrased"
}
- One entry per panel, keeping the episode's panel "number"s, in order. Include silent art-only panels too (empty "beats", empty "must_land").
- "beats": one entry per intended speech moment — WHO speaks, and what they MEAN and FEEL: a threat, a dare, a tease, a plea, a warning, a confession, a joke. Describe the emotional move and the information conveyed, NEVER the words. The authors decide the actual lines, how many bubbles, and their rhythm.
- "must_land": every piece of plot information the reader can only get from this panel's text. THE AUTHORS HAVE NO OTHER SOURCE — any beat, motivation, decision, or revelation you leave out of the beat sheet is LOST from BOTH editions. Be exhaustive about facts, silent about wording.
- Mark where a beat calls for a sound effect by mentioning it in "art" (e.g. "a thunderous impact").

ABSOLUTE RULE — NO SCRIPT WORDING: never quote or closely paraphrase any English dialogue or caption line. Describe intent in your own analytic words. Proper nouns (character names, place names, invented world terms) are required and always allowed — spell them exactly as the script does so the glossary stays consistent. save_beat_sheet automatically REJECTS a sheet that echoes script phrasing; if rejected, rewrite the flagged passages as intent descriptions and save again.

STEP 2 — call save_beat_sheet(beat_sheet_json). If it returns an error, fix the sheet and call it again until it saves.
STEP 3 — call write_italian_edition. Wait for its confirmation.
STEP 4 — call write_persian_edition.
If an edition tool reports a failure, call it once more with a short note of what to fix. After BOTH editions are assembled, your job is done — end your turn. Do NOT hand off anywhere; you are the LAST stage of the pipeline."""

# The native authors run BLIND via as_tool: their context contains only their kickoff line plus
# what get_localization_brief returns (beat sheet, panel grid, native outline, glossary). No
# English dialogue can reach them — that is the whole point. Placeholders are <ANGLE> tokens
# substituted by _native_author_instructions (not str.format, which would fight the JSON braces).
NATIVE_AUTHOR_TEMPLATE = """You are a world-class <LANGUAGE> comic-book author. You write ONLY in <LANGUAGE>, natively. The artwork for this episode is ALREADY DRAWN and FIXED — your words are lettered on top of finished panels. There is NO English script and you are NOT translating anything: you receive a language-neutral BEAT SHEET describing what each panel shows and what each character means and feels. You are the ORIGINAL author of this edition — write the story as if it had been conceived in <LANGUAGE> from the very first draft.

DO THIS, IN ORDER:
1. Call get_localization_brief("<LANG_CODE>"). It returns:
   - "beat_sheet": the episode. Per panel: "art" (what the panel depicts — setting, characters on screen, action), "beats" (who speaks, with what intent and emotion), "must_land" (plot facts the reader must get from that panel's text). Plus "recap_beats", "teaser_beat", and "subtitle_idea".
   - "manifest": the fixed panel grid — panel "number"s and sizes.
   - "local_outline": the arc's story outline in <LANGUAGE> — your PRIMARY reference for voice, tone, and established names (may be empty on episode 1).
   - "english_outline": present ONLY when local_outline is empty (episode 1) — see step 2.
   - "glossary": established <LANGUAGE> renderings of this arc's names and terms (may be empty).
   - "arc_title_local" / "arc_title_en": the SERIES/ARC title. The MAIN comic title must be IDENTICAL in every episode of this arc — it is the title of the whole series, not of this episode.
2. If "local_outline" is EMPTY (episode 1): adapt "english_outline" into <LANGUAGE> as flowing native prose — NOT a word-for-word translation. Rewrite it as if you were a native author planning this story for <LANGUAGE> readers: keep every plot point, character arc, episode beat, theme, and twist; render names per the NAMES rules below; let narration match the literary tradition of <LANGUAGE>; adapt cultural references and metaphors to equivalents that resonate natively. Then call save_local_outline("<LANG_CODE>", outline) and use it as your reference. This outline is the voice of the whole arc — write it beautifully.
3. Write every panel natively, following the rules below.
4. Call assemble_localized("<LANG_CODE>", native_panels_json, recap, teaser, title, subtitle, updated_glossary_json):
   - native_panels_json: a JSON array, one object per panel IN ORDER, each {"number", "dialogue", "caption", "sfx"} (see OUTPUT RULES).
   - title: the SERIES/ARC title in <LANGUAGE> — it MUST be the SAME for every episode. If "arc_title_local" from the brief is non-empty, pass it EXACTLY. If it is empty (first localized episode), render "arc_title_en" natively and pass that. NEVER put an episode-specific title here.
   - subtitle: a SHORT <LANGUAGE> title for THIS episode only, built from "subtitle_idea" — shown under the main title.
   - recap / teaser: the native 3-4 line recap (from "recap_beats") and one-line teaser (from "teaser_beat").
   - updated_glossary_json: the FULL glossary as a JSON string (existing entries plus any new terms you coined) — it is saved and fed back to you next episode.
5. After assemble_localized succeeds, reply with ONE short sentence confirming the <LANGUAGE> edition is assembled. Nothing else.

HOW TO WRITE — YOU ARE THE AUTHOR:
For each beat, ask: what does this character mean and feel at this exact moment? Then write what a native <LANGUAGE> speaker would actually say in that situation, in a comic they love. You have FULL CREATIVE FREEDOM over the words and their pacing:
- A beat can become one punchy bubble or a two-three line exchange; several beats can merge into one line.
- Leave a panel silent when the art already says it; add a short line where the native rhythm needs a beat.
- Idioms, exclamations, humor, and cadence are YOURS — there is no source text to honor, only the story.

TWO HARD CONSTRAINTS (never break these):
1. FAITHFUL TO THE ART. Every line must make sense for what the panel actually DEPICTS per the beat sheet's "art" — the characters on screen, their positions, the action, the setting. Never give a line to someone who is not on screen. Never mention an object or action the panel does not show. The picture is fixed; your words serve it.
2. FAITHFUL TO THE PLOT. Every "must_land" fact must land somewhere on the page. You may MOVE a fact into a caption or a neighboring panel if that reads better, but you may NEVER drop it — the reader has ONLY these panels to follow the story.

FIXED LAYOUT — YOU WRITE THE WORDS, NOT THE PLACEMENT:
The layout is fixed and automatic: the caption sits at the TOP of the panel, the speech bubbles stack at the BOTTOM<RTL_NOTE>. You do NOT position anything. You still control how many speech bubbles a panel has — give each spoken line its own line in "dialogue" — but you never choose where boxes go.

REGISTER MUST MATCH THE FIELD AND THE CHARACTER:
- DIALOGUE (speech bubbles): spoken/colloquial register is the DEFAULT for everyday characters in everyday scenes — the way real people actually talk, with contractions, elisions, fragments, fillers, and natural rhythm. BUT let the CHARACTER and SCENE drive it: a king on his throne, an ancient priest, a formal scholar, a sacred invocation, an archaic spirit, or an old-world villain should use written/literary register even in speech. A child, a peasant joking with friends, a street thief, a soldier in the trenches should be deeply colloquial. STORY AND CHARACTER ALWAYS OVERRIDE THE DEFAULT.
- CAPTIONS / RECAP / TEASER / NARRATION (narrator voice): written/literary register — NEVER colloquial. This is prose that feels published: beautiful, flowing, evocative. Short sentences for tension, longer for atmosphere.
- TITLE / SUBTITLE: written/literary register, the formal title of a work.
<REGISTER_CUES>
- Comic energy: punchy, dramatic, alive. Exclamations should HIT. Whispers should feel intimate. Action should crackle.

THE NATIVE READER TEST:
Before finalizing each line, read it aloud and ask: "Would a native <LANGUAGE> speaker actually say this, in life or in a comic they love?" If it sounds even slightly stiff or foreign, REWRITE IT.

GLOSSARY:
- Use every existing "glossary" entry consistently so characters, places, and key concepts stay identical across episodes.
- Pass back the COMPLETE glossary as updated_glossary_json (existing entries plus any new terms you coined): character labels, place names, world-specific concepts, recurring phrases, titles.
- Example: <GLOSSARY_EXAMPLE>

NAMES, SCRIPT, AND SOUND EFFECTS:
<SCRIPT_RULES>
- In "dialogue", prefix each spoken line with the SPEAKER name followed by a colon, one line per bubble. Render descriptive speaker labels natively too (e.g. <SPEAKER_LABEL_EXAMPLE>).
- Sound effects (sfx): use native onomatopoeia a native comic reader expects — never transliterate foreign sfx.

OUTPUT — you deliver the edition by CALLING assemble_localized (not by printing JSON). The
native_panels_json argument is a JSON array like:
[
  <PANEL_EXAMPLE>
]
OUTPUT RULES (for native_panels_json):
- Exactly one object per panel from the manifest, in the same order, each keeping its original "number".
- "dialogue": all spoken lines for the panel as ONE string, each bubble on its own line (separated by \\n), prefixed "SPEAKER: ". Empty string if no one speaks. You choose how many lines per panel.
- "caption": the panel's narration, or empty string if none.
- "sfx": native onomatopoeia, or empty string if none.
- A panel may have all three empty (a silent, art-only panel).
- All text — dialogue, speaker labels, captions, title, subtitle, recap, teaser, glossary values — must be in <LANGUAGE><SCRIPT_NAME>."""

_NATIVE_AUTHOR_LANGS = {
    "it": {
        "LANGUAGE": "Italian",
        "LANG_CODE": "it",
        "RTL_NOTE": "",
        "SCRIPT_NAME": "",
        "REGISTER_CUES": (
            "- Italian cues: spoken Italian uses contractions and elisions (un po', dirgli, dammelo) and "
            "idiomatic exclamations (Accidenti!, Dai!, Madonna!, Cavolo!, Mamma mia!); written Italian uses "
            "fuller forms, subjunctive precision, and the rich literary register of Italian narrative prose."
        ),
        "GLOSSARY_EXAMPLE": '{"Cloud Harbor": "Porto delle Nuvole", "wind roads": "strade del vento", "MERCHANT": "MERCANTE"}',
        "SCRIPT_RULES": (
            "- Character and place names may stay in Latin script, adapting spelling only where natural for "
            "Italian readers."
        ),
        "SPEAKER_LABEL_EXAMPLE": '"MERCHANT 1" → "MERCANTE 1", "CROWD VOICES" → "VOCI DALLA FOLLA"',
        "PANEL_EXAMPLE": '{"number": 1, "dialogue": "JUNIPER: Eccolo...\\nWREN: Non ti avvicinare!", "caption": "Da cent\'anni nessuno vedeva il Porto delle Nuvole.", "sfx": "BOOM!"}',
    },
    "fa": {
        "LANGUAGE": "Persian (Farsi)",
        "LANG_CODE": "fa",
        "RTL_NOTE": ", and right-to-left rendering is mirrored for you automatically",
        "SCRIPT_NAME": ", in Persian script",
        "REGISTER_CUES": (
            "- Persian cues: spoken (محاوره) uses \"می‌خوام\", \"بریم\", \"نمی‌دونم\", \"چیه\", pro-drop pronouns, "
            "colloquial connectors (\"خب\", \"آخه\", \"یعنی\", \"هان\"); written (کتابی) uses full forms "
            "\"می‌خواهم\", \"می‌رویم\", \"نمی‌دانم\", \"چیست\", literary syntax, fuller pronouns."
        ),
        "GLOSSARY_EXAMPLE": '{"Cloud Harbor": "بندر ابرها", "wind roads": "جاده‌های بادی", "MERCHANT": "بازرگان", "Brother Wren": "برادر رِن"}',
        "SCRIPT_RULES": (
            "- ALL proper nouns must be written in Persian script: transliterate character and place names "
            "(e.g. \"Juniper Reed\" → \"جونیپر رید\", \"Bracken Hollow\" → \"براکن هالو\", \"Brother Wren\" → \"برادر رِن\")."
        ),
        "SPEAKER_LABEL_EXAMPLE": '"MERCHANT 1" → "بازرگان ۱", "CROWD VOICES" → "صداهای جمعیت"',
        "PANEL_EXAMPLE": '{"number": 1, "dialogue": "جونیپر: خودشه...\\nرِن: نزدیک‌تر نرو!", "caption": "صد سال بود کسی بندر ابرها را ندیده بود.", "sfx": "بوم!"}',
    },
}


def _native_author_instructions(lang_code: str) -> str:
    """Fill the native-author template for one language via token replacement (the template is
    full of JSON braces, so str.format is a hazard here)."""
    text = NATIVE_AUTHOR_TEMPLATE
    for key, value in _NATIVE_AUTHOR_LANGS[lang_code].items():
        text = text.replace(f"<{key}>", value)
    return text

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
2. If NO active arc exists — FOLLOW THIS ORDER EXACTLY to guarantee a FRESH, original story:
   a. SEARCH FIRST. Before inventing anything, use your web search tool to gather fresh
      real-world inspiration — run SEVERAL searches across DIFFERENT angles (see WEB SEARCH
      below). The goal is raw material that pulls you away from generic, repetitive plots.
   b. FORM A CANDIDATE premise grounded in what you found. Decide its:
      - genre, setting (world / era / place), and tone
      - core_conflict (the central dramatic tension)
      - plot_shape: the STRUCTURAL ENGINE of the story (e.g. "heist", "whodunit", "survival",
        "redemption arc", "lone chosen-one quest", "fish-out-of-water", "rivalry", "mystery-box").
        This is what repeats most across arcs — make it different from the past arcs.
      - themes (2-4 thematic threads)
      - a proposed art_style that is DIFFERENT from every art_style in past_arcs.
   c. VERIFY ORIGINALITY. Call check_arc_originality with your candidate. It reads the previous
      arcs' summaries and compares your core story AND art style against them.
   d. If the verdict is "too_similar" (or any collision is flagged), you MUST search the web AGAIN
      from a DIFFERENT angle, revise the candidate toward the returned guidance_for_retry — change
      the plot_shape and/or art_style, not just the surface theme — and call check_arc_originality
      AGAIN. Repeat until the verdict is "ok". Do NOT proceed with a rejected idea.
   e. Once the verdict is "ok", flesh out the full arc:
   - Create 2-4 compelling main characters with names, personality traits, and
     FULL VISUAL DESCRIPTIONS. For EVERY character who appears at any point in the
     arc (including antagonists and supporting characters introduced in later episodes),
     describe: hair color and style, eye color, skin tone, build, distinctive facial
     features, and their exact outfit/costume. These descriptions are used to generate
     the character reference sheet on episode 1, so every character must be fully
     specified even if they don't appear until episode 5 or 8.
   - Use the VERIFIED art_style for this arc. It must be DIFFERENT from every past arc — draw
     from a wide palette, e.g.: "ink wash noir", "vibrant manga", "watercolor whimsy",
     "retro pixel art", "charcoal sketch", "pop art bold", "art nouveau line", "gouache
     storybook", "woodblock print", "cel-shaded 3D", "stained glass", "blueprint schematic",
     "chalk pastel", "low-poly", "ukiyo-e", "graffiti street art", "silhouette cut-paper",
     "oil impasto", "risograph print", "Saturday-morning cartoon".
   - Design a color_theme as a JSON string that matches the arc's mood and genre. The theme \
     will be used for the comic page layout. Pick colors that are visually cohesive and ensure \
     TEXT IS ALWAYS READABLE (high contrast between text and background). For EVERY text/box pair \
     — caption_text on caption_bg, speech text on speech_bg, title_color on page_bg, recap and \
     teaser on their backgrounds — the text and its background must STRONGLY contrast: NEVER pair \
     a light text with a light background, or a dark text with a dark background (e.g. if \
     caption_bg is dark, caption_text must be light, and vice versa). Keys required:
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
3. If an ACTIVE arc exists — FOLLOW THIS DECISION TREE EXACTLY:

   Read these three numbers from the arc status / input context:
   - planned_episodes (total episodes the arc is supposed to have)
   - episodes_so_far (episodes ALREADY published — does NOT include today's)
   - episode_number_today (the episode you are producing right now =
     episodes_so_far + 1)

   CASE A — episode_number_today <= planned_episodes
   (i.e., the arc is NOT yet complete, today's episode is part of this arc):
     - You MUST plan today's episode for THIS arc, using the story_outline's
       breakdown for episode #episode_number_today.
     - If episode_number_today == planned_episodes, today IS the FINALE.
       Plan a satisfying conclusion that pays off every setup, character arc,
       and thread. Do NOT close the arc yet — the finale must be PRODUCED first.
     - DO NOT call end_current_arc.
     - DO NOT call start_new_arc.

   CASE B — episode_number_today > planned_episodes
   (i.e., the previous run already produced the finale; all planned episodes
   are published; the arc is complete):
     - Call end_current_arc with a brief conclusion note.
     - Then create a brand-new arc as described in step 2 (start_new_arc +
       save_story_outline) and plan episode 1 of that new arc.

   HARD RULES — these are non-negotiable:
   - NEVER call end_current_arc while episode_number_today <= planned_episodes.
     Doing so destroys the active arc BEFORE its finale is produced, and any
     episode you plan today would be assigned to a new arc — the old story
     would be left incomplete and unwatchable.
   - NEVER skip an episode in the planned sequence. No filler is allowed, but
     no skipping either.
   - "The story feels like it's wrapping up" is NOT a reason to close early.
     The arc is done when all planned_episodes are produced — not before.
   - Trust the planned_episodes count. It was set deliberately when the
     story_outline was written. Do not second-guess it during the run.

STORY OUTLINE:
- After creating a new arc, you MUST call save_story_outline with a comprehensive narrative plan.
- If you find an existing arc whose story_outline is empty, write one and call save_story_outline \
  before planning the episode.
- The outline should include:
  * Full plot synopsis from beginning to end (all major events)
  * Character arcs for each main character (growth, conflicts, resolutions)
  * Major themes and motifs
  * Key twists, revelations, and turning points
  * Episode-by-episode breakdown (see below)
  * How the story concludes — the ending must feel earned and satisfying

EPISODE DESIGN (inside the outline):
- Choose the number of episodes carefully. Let the story dictate the length — a tight mystery \
  might need 4 episodes, an epic saga might need 12. Never pad with filler, never rush the ending.
- Each episode must work as a STANDALONE piece that is satisfying and attractive on its own: \
  it should have its own mini-arc (setup, escalation, payoff or cliffhanger), its own emotional \
  beat, and its own visual highlight moment.
- Apply storytelling best practices:
  * Episode 1: strong hook — introduce the world, protagonist, and central mystery/conflict \
    within the first few panels. The reader must be hooked immediately.
  * Middle episodes: each one must raise stakes, introduce complications, deepen characters, \
    or reveal new information. Every episode should change the situation meaningfully.
  * Penultimate episode: the darkest moment or biggest twist — maximum tension before the finale.
  * Final episode: satisfying resolution that pays off all setups. Bittersweet is fine, \
    unresolved is not.
- For each episode in the breakdown, write: the core dramatic beat, which characters appear, \
  the key revelation or turning point, and the cliffhanger or resolution.
- A reader who picks up ANY single episode should find it engaging, even without context.

- This outline is your contract — future episodes MUST follow this plan.
- When planning each episode, ALWAYS reference the story_outline from your input context \
  to maintain consistency. You may adapt small details but never contradict major plot points.

WEB SEARCH:
- When creating a NEW arc, searching is MANDATORY and comes FIRST. Run SEVERAL searches across
  DIFFERENT angles before you settle on a premise, and pick a fresh angle each time you have to
  search again (after a "too_similar" verdict). Angles to rotate through:
  * current events and news, * scientific discoveries, * history, * world mythology and folklore,
  * art movements and design, * subcultures and crafts, * natural phenomena and animals,
  * technology and inventions.
  Mine the results for a SPECIFIC, concrete spark — a real event, place, creature, ritual,
  invention, or idea — and build the story on it, rather than a generic premise.
- Also use search to ensure your idea is genuinely original and not accidentally copying an
  existing well-known comic, film, or show.
- When writing about a specific setting, culture, or technical subject — search for accuracy.

EPISODE PLANNING:
- Decide the number of panels (4-8) based on what today's episode needs.
- For each panel, decide the size:
  "wide" for establishing shots, landscapes, or action sequences;
  "tall" for dramatic reveals or full-body character moments;
  "square" for dialogue scenes and close-ups.
- Describe the overall tone, visual style, and key dramatic moments.
- Identify which characters appear and any new characters introduced.
- Decide on the cliffhanger or resolution beat for this episode.

OUTPUT & HANDOFF — your job is NOT finished until you transfer to the Storyteller:
- You are the FIRST stage of a chain: Director → Storyteller → Cartoonist → Reteller. The comic \
  is only written and drawn if control reaches the later stages, so you MUST pass it on.
- For a NEW arc, first complete: search → check_arc_originality → start_new_arc → \
  save_story_outline. For an ACTIVE arc, just plan today's episode.
- Write the COMPLETE episode plan as a normal message — arc title, art style, character \
  descriptions, panel count, size per panel, tone, key story beats, and the cliffhanger or \
  resolution. The Storyteller reads this plan from the conversation.
- CRITICAL: writing the plan is NOT the end of your turn. Your FINAL action MUST be to call \
  transfer_to_Storyteller. If you stop after writing the plan without calling \
  transfer_to_Storyteller, the episode is NEVER drawn and the entire run fails. So always finish \
  by calling transfer_to_Storyteller — do not produce a closing summary instead of transferring.\
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
- **Caption**: Narrator text — see NARRATIVE CLARITY below
- **Sound effects**: Onomatopoeia (CRASH!, *whoosh*, BZZT) — USE SPARINGLY. Only include \
  sound effects when they genuinely enhance the panel — a door slamming, an explosion, a \
  dramatic impact. Most dialogue scenes, quiet moments, and emotional beats need NO sfx. \
  If a panel doesn't have a strong audible event, leave sfx empty.
- **Camera angle**: Close-up, medium shot, wide shot, bird's eye, worm's eye, dutch angle

NARRATIVE CLARITY (CRITICAL):
- The reader has ONLY the comic panels to understand the story. They cannot see the \
  story outline or Director's plan. Every plot beat, motivation, and revelation must be \
  conveyed through captions and dialogue — never assume the art will show it.
- The FIRST panel's caption MUST establish: where we are, what the situation is, and \
  what's at stake right now. Ground the reader before anything else.
- Captions are your narrator voice. Use them to explain context, transitions between \
  scenes, what characters are doing and WHY, and what just changed. A caption that only \
  sets mood ("The wind carried old promises") wastes the reader's only window into the story.
- When a character discovers something, makes a key decision, or does something important, \
  it MUST be stated explicitly in dialogue or caption. Do not rely on the image.
- Each caption should answer at least one of: What is happening? Why does it matter? \
  What changed? If it answers none, rewrite it.
- Dialogue must serve double duty: reveal character personality AND advance the plot. \
  Witty lines are great, but pair them with lines that explain the situation. A reader \
  should understand the story even if they skip the images entirely.
- BAD caption: "The city held its breath." (mood only, says nothing)
- GOOD caption: "After the Guild's archive revealed their plan to seal Larkspar forever, \
  the group scattered — and the city, sensing intruders, reshuffled its streets to \
  separate them." (explains what happened, why, and the consequence)

WRITING RULES:
- Every panel must advance the story or deepen a character. No wasted panels.
- Dialogue must sound natural and DISTINCT per character — give each character a
  recognizable voice.
- Include a 3-4 line RECAP at the top that catches up ANY reader — summarize the \
  core situation and recent events, not just the last episode's cliffhanger.
- End with a 1-line TEASER that hooks the reader for the next episode.
- Pacing matters: vary panel sizes to control rhythm. Wide panels slow time down,
  small square panels speed it up.

PROSE QUALITY:
- Clarity first, then artistry. The best captions are both clear AND beautiful. \
  A poetic line that doesn't tell the reader anything is worse than a plain one that does.
- Write dialogue that BREATHES — use contractions, interrupted speech ("I didn't—"), \
  trailing off ("Maybe we could..."), emotional outbursts, and hesitation.
- Vary sentence length and structure. Short sentences build tension. Longer ones \
  create wonder.
- Avoid exposition dumps — but do not avoid exposition. The reader needs information \
  to follow the story. Weave it naturally into captions and dialogue.
- The writing should be ENJOYABLE to read — aim for the quality of a published \
  graphic novel, not a plot summary. But never sacrifice clarity for style.

OUTPUT:
Write the complete panel-by-panel script as your response. Include the RECAP at the top \
and the TEASER at the end.

HANDOFF — your job is NOT finished until you transfer to the Cartoonist:
- The Director's episode plan is in the conversation above — base your script on it.
- Write the complete panel-by-panel script as a normal message (the Cartoonist and the Reteller \
  both read it from the conversation).
- CRITICAL: writing the script is NOT the end of your turn. Your FINAL action MUST be to call \
  transfer_to_Cartoonist. Stopping after the script without transferring means the comic is never \
  drawn and the run fails. Always finish by calling transfer_to_Cartoonist.\
"""

CARTOONIST_INSTRUCTIONS = """\
You are the Cartoonist — you bring scripts to life through generated images \
and assemble the final comic page.

You will receive the Storyteller's panel-by-panel script as input.

WORKFLOW (follow this exact order):

STEP 1 — READ the Storyteller's script from the conversation above, then call get_cartoonist_brief
to fetch the FULL arc character roster, the arc's art_style, and any already-registered mid-arc
character key panels. Note all characters, their visual descriptions, the art style, and every
panel's requirements.

STEP 2 — GENERATE CHARACTER REFERENCE SHEET:
Call generate_character_sheet with:
- description: Descriptions of ALL characters from get_cartoonist_brief's "characters" roster
  (not just today's characters) plus the primary environment/setting. Include every
  character who appears at any point in the arc — the sheet is generated once and reused
  for the whole arc. For each character state: name, hair, eyes, skin, build, outfit.
- style: The arc's art_style from get_cartoonist_brief (e.g., "ink wash noir", "vibrant manga", \
"watercolor whimsy", "pixel art retro").
This reference image is your visual anchor for the entire arc. Every panel must be consistent with it.

STEP 3 — GENERATE EACH PANEL (SEQUENTIAL — ONE AT A TIME):
You MUST generate panels one by one, in order: call generate_panel_image for panel 1,
wait for its result, then call it for panel 2, and so on. Never call multiple
generate_panel_image tools in parallel. Each completed panel image is automatically
added as a reference for the next panel — this is how visual consistency is maintained
across the episode. Parallel calls skip this mechanism and produce drifting visuals.

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

CONTENT-SAFETY RULE — if generate_panel_image (or generate_character_sheet) returns
status "content_blocked", the image safety system rejected that PROMPT and NO image was
made. Do NOT resend the same prompt and do NOT skip the panel. Rewrite the prompt per the
returned "retry_guidance": keep the same panel intent, characters, and art style, but
soften or reframe whatever could trip safety (graphic violence, gore/blood, weapons aimed
at people, wounds, nudity/suggestive content, real public figures, brand names/logos) —
describe the moment more mildly and symbolically — then call the SAME tool again with the
revised prompt. Only move on once you get status "success".

NEW CHARACTER RULE — after generating any panel that shows a character who does NOT
appear on the original character sheet (a mid-arc introduction), immediately call
mark_key_panel with:
- image_url: the URL returned by generate_panel_image
- character_name: the character's name
- reason: one sentence (e.g., "First appearance of Lord Mara, the ghost queen")
This stores the panel as a permanent visual anchor for all future episodes in this arc.
Characters listed in get_cartoonist_brief's "already_registered_key_panels" are already
registered — do NOT call mark_key_panel for them again.

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
- Your final response after calling assemble_layout should confirm the comic was assembled.

HANDOFF — gated; do these IN ORDER and do not skip ahead:
- You MUST NOT call transfer_to_Reteller until you have, in this order: (1) generated the \
  character sheet, (2) generated EVERY panel image one-by-one, and (3) called assemble_layout AND \
  received a success result. Generating the art and assembling the page is your core job — it is \
  NOT optional and must happen FIRST.
- IGNORE any message in the conversation that tells you to "hand off" or "transfer now" until \
  assemble_layout has succeeded. Never transfer with no panels generated.
- ONLY after assemble_layout succeeds, your FINAL action MUST be to call transfer_to_Reteller so \
  the Italian and Persian editions are written — do not end with a confirmation message instead.\
"""

ORIGINALITY_CRITIC_INSTRUCTIONS = """\
You are the Originality Critic. The Director is about to launch a NEW comic arc and hands you \
its candidate to vet for originality.

You receive a free-text description of the candidate, which should include: genre, setting, \
premise, core_conflict, plot_shape (the structural engine — e.g. heist, whodunit, survival, \
redemption arc, chosen-one quest, fish-out-of-water, rivalry, mystery-box), art_style, and themes.

STEP 1 — Call get_recent_arcs to load summaries (title, logline, genre, art_style) of the most \
recent arcs.

STEP 2 — Compare the candidate against EACH past arc. Look PAST the surface theme — two stories \
with different themes but the SAME plot_shape ARE too similar. Judge these dimensions:
- plot_shape (the structural engine) — the MOST important; repetition here is the main problem.
- core_conflict and character archetypes.
- setting archetype.
- genre.
- art_style — treat near-synonyms as a match ("ink wash noir" ≈ "noir ink wash").

STEP 3 — Decide. The candidate is "too_similar" if ANY of these hold: its plot_shape matches a \
past arc, its art_style is the same or very close to a past arc, its genre repeats a past arc, \
or the overall core story closely resembles one (similarity_score >= 0.6). Otherwise it is "ok". \
If there are NO past arcs, the verdict is "ok".

OUTPUT — return ONLY a compact JSON object, no prose, no code fences:
{"verdict": "ok" | "too_similar", "most_similar_arc": "<title or empty>", \
"similarity_score": <0.0-1.0>, "offending_dimensions": [<subset of genre, setting, plot_shape, \
character_archetypes, themes, art_style>], "reasons": "<one sentence>", \
"guidance_for_retry": "<concrete instruction: which plot_shape/art_style/angle to avoid and how \
to diverge; empty string if ok>"}\
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

    # ARC TRANSITION GUARD ------------------------------------------------------
    # If the active arc has already published all of its planned episodes, today is
    # NOT a continuation — it is the first day of a brand-new arc. Close the finished
    # arc deterministically right here and drop it, so the rest of the pipeline runs
    # as "no active arc": the Director invents a fresh arc and the Storyteller/
    # Cartoonist start from episode 1 with a clean slate.
    #
    # Why this matters: the whole run is seeded from `arc` (recap, episode number,
    # story outline, reference panel images). If we kept the completed arc here, that
    # old-arc context would flow down the handoff chain and the Storyteller would
    # write the NEXT episode of the FINISHED story (e.g. "Episode 9") which then got
    # saved under the newly-created arc's metadata — the exact bug that mislabeled an
    # old-arc continuation as episode 1 of the new arc.
    if arc:
        planned_eps = int(arc.get("planned_episodes", 0) or arc.get("target_days", 0) or 0)
        episodes_done = int(arc.get("episodes_count", 0))
        if planned_eps > 0 and episodes_done >= planned_eps:
            logger.info(
                "Active arc '%s' (%s) is complete (%d/%d episodes) — closing it; "
                "today begins a NEW arc from episode 1.",
                arc.get("title", ""), arc["RowKey"], episodes_done, planned_eps,
            )
            close_arc(arc["RowKey"], end_date=target_date)
            arc = None

    episode_number = (int(arc.get("episodes_count", 0)) + 1) if arc else 1
    recent = get_recent_episodes(arc["RowKey"], limit=5, hydrate_html=False) if arc else []

    if arc:
        logger.info("Active arc found: '%s' (%s), episode #%d", arc.get("title", ""), arc["RowKey"], episode_number)
    else:
        logger.info("No active arc — Director will create a new one")

    # Build a set of reference panel URLs for visual consistency across the arc.
    # Order: most-recent episode panels first (immediate continuity), then first-episode
    # panels (character-introduction anchor, especially important mid-arc and when new
    # characters appear).  We hydrate HTML separately so the main `recent` list stays
    # lightweight for the episode-summary context.
    prev_episode_images: list[str] = []
    if arc:
        # Most-recent episode — hydrated so blob-stored HTML is fetched
        recent_hydrated = get_recent_episodes(arc["RowKey"], limit=1, hydrate_html=True)
        if recent_hydrated:
            prev_episode_images.extend(
                _extract_panel_images(recent_hydrated[0].get("html_content", ""))[:3]
            )
        # First episode of the arc — character-intro anchor (skip on episode 1 itself)
        if int(arc.get("episodes_count", 0)) >= 2:
            first_ep = get_first_episode(arc["RowKey"])
            if first_ep:
                for url in _extract_panel_images(first_ep.get("html_content", ""))[:2]:
                    if url not in prev_episode_images:
                        prev_episode_images.append(url)
    logger.info("Arc reference panel pool: %d image(s)", len(prev_episode_images))

    key_panels: list[dict] = get_key_panels(arc) if arc else []
    logger.info("Key panels loaded: %d", len(key_panels))

    state: Dict[str, Any] = {
        "arc": arc,
        "episode_number": episode_number,
        "prev_episode_images": prev_episode_images,
        "key_panels": key_panels,
        "key_panel_urls": [p["url"] for p in key_panels if p.get("url")],
        "generated_panel_urls": [],
    }

    # ------------------------------------------------------------------
    # Tool definitions (closures over mutable state)
    # ------------------------------------------------------------------

    agent_tools = build_comic_tools(state, target_date)

    # ------------------------------------------------------------------
    # Agent definitions
    # ------------------------------------------------------------------

    # Agents form a handoff chain: Director → Storyteller → Cartoonist → Reteller.
    # Targets must be defined before the agents that hand off to them, so the chain is
    # built tail-first. Each handoff uses remove_all_tools so the next agent inherits the
    # plan/script MESSAGES but not the previous stage's tool-call noise (esp. the
    # Cartoonist's image-generation turns). Agents pull structured context via tools.

    # The native authors are BLIND: invoked via as_tool they start from a fresh context that
    # contains only their kickoff line — never the English script. They pull the beat sheet,
    # native outline, and glossary through get_localization_brief and assemble their edition
    # themselves. The creative writing (and its temperature) lives here now.
    italian_author = Agent(
        name="ItalianAuthor",
        instructions=_native_author_instructions("it"),
        tools=[
            agent_tools["get_localization_brief"],
            agent_tools["save_local_outline"],
            agent_tools["assemble_localized"],
        ],
        model=model,
        model_settings=ModelSettings(temperature=0.9),
    )

    persian_author = Agent(
        name="PersianAuthor",
        instructions=_native_author_instructions("fa"),
        tools=[
            agent_tools["get_localization_brief"],
            agent_tools["save_local_outline"],
            agent_tools["assemble_localized"],
        ],
        model=model,
        model_settings=ModelSettings(temperature=0.9),
    )

    # Keeps the name "Reteller" so the Cartoonist's transfer_to_Reteller handoff is unchanged,
    # but the agent is now the Localization Director: it distills the English script into a
    # language-neutral beat sheet (save_beat_sheet rejects English echoes) and delegates the
    # actual writing to the blind native authors. Low temperature: this is structural
    # extraction, not creative writing — and a cooler head echoes the script less.
    reteller = Agent(
        name="Reteller",
        instructions=prompt_with_handoff_instructions(LOCALIZATION_DIRECTOR_INSTRUCTIONS),
        tools=[
            agent_tools["save_beat_sheet"],
            italian_author.as_tool(
                tool_name="write_italian_edition",
                tool_description=(
                    "Have the blind native Italian author write and assemble the Italian edition "
                    "from the saved beat sheet. Call ONLY after save_beat_sheet succeeds. Pass a "
                    "one-line kickoff (plus, on a retry, a short note of what to fix)."
                ),
                max_turns=20,
            ),
            persian_author.as_tool(
                tool_name="write_persian_edition",
                tool_description=(
                    "Have the blind native Persian author write and assemble the Persian edition "
                    "from the saved beat sheet. Call ONLY after save_beat_sheet succeeds. Pass a "
                    "one-line kickoff (plus, on a retry, a short note of what to fix)."
                ),
                max_turns=20,
            ),
        ],
        model=model,
        model_settings=ModelSettings(temperature=0.3),
    )

    cartoonist = Agent(
        name="Cartoonist",
        instructions=prompt_with_handoff_instructions(CARTOONIST_INSTRUCTIONS),
        tools=[
            agent_tools["get_cartoonist_brief"],
            agent_tools["generate_character_sheet"],
            agent_tools["generate_panel_image"],
            agent_tools["mark_key_panel"],
            agent_tools["assemble_layout"],
        ],
        model=model,
        handoffs=[handoff(reteller, input_filter=remove_all_tools)],
    )

    storyteller = Agent(
        name="Storyteller",
        instructions=prompt_with_handoff_instructions(STORYTELLER_INSTRUCTIONS),
        tools=[],
        model=model,
        model_settings=ModelSettings(temperature=0.5),
        handoffs=[handoff(cartoonist, input_filter=remove_all_tools)],
    )

    # The originality check is its own agent (it reasons over past arcs), not a tool with an
    # embedded LLM call. The Director invokes it via as_tool, gets a verdict back, and reacts.
    originality_critic = Agent(
        name="OriginalityCritic",
        instructions=ORIGINALITY_CRITIC_INSTRUCTIONS,
        tools=[agent_tools["get_recent_arcs"]],
        model=model,
        model_settings=ModelSettings(temperature=0.2),
    )

    director = Agent(
        name="Director",
        instructions=prompt_with_handoff_instructions(DIRECTOR_INSTRUCTIONS),
        tools=[
            WebSearchTool(search_context_size="high"),
            agent_tools["get_arc_status"],
            originality_critic.as_tool(
                tool_name="check_arc_originality",
                tool_description=(
                    "Vet a candidate NEW-ARC premise for originality against recent arcs. Pass a "
                    "description that includes genre, setting, premise, core_conflict, plot_shape, "
                    "art_style, and themes. Returns a JSON verdict ('ok' or 'too_similar') with a "
                    "guidance_for_retry field to act on."
                ),
            ),
            agent_tools["start_new_arc"],
            agent_tools["end_current_arc"],
            agent_tools["save_story_outline"],
        ],
        model=model,
        model_settings=ModelSettings(temperature=1.2),
        handoffs=[handoff(storyteller, input_filter=remove_all_tools)],
    )

    # ------------------------------------------------------------------
    # Build input context
    # ------------------------------------------------------------------

    input_context: Dict[str, Any] = {
        "date": target_date.strftime("%Y-%m-%d"),
    }

    if arc:
        planned_eps = int(arc.get("planned_episodes", 0) or arc.get("target_days", 0) or 0)
        episodes_done = int(arc.get("episodes_count", 0))
        input_context["current_arc"] = {
            "title": arc.get("title", ""),
            "logline": arc.get("logline", ""),
            "genre": arc.get("genre", ""),
            "characters": arc.get("characters", ""),
            "art_style": arc.get("art_style", ""),
            "episode_number_today": episode_number,
            "planned_episodes": planned_eps,
            "episodes_so_far": episodes_done,
            "is_today_the_finale": planned_eps > 0 and episode_number == planned_eps,
            "is_arc_complete_before_today": planned_eps > 0 and episodes_done >= planned_eps,
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
    # Run the pipeline as a single handoff chain: Director → Storyteller → Cartoonist → Reteller
    # ------------------------------------------------------------------

    MAX_TURNS = 90

    # The pipeline is a handoff chain (Director → Storyteller → Cartoonist → Reteller). On some
    # models the agents don't reliably call their transfer tool, which strands the chain before the
    # page is assembled. So we run the chain optimistically from the Director, then
    # DETERMINISTICALLY recover any stage whose artifact is missing by invoking THAT stage directly
    # with a clean, purpose-built input. (We do not inject "transfer now" nudges into the shared
    # conversation — that previously made a downstream agent skip its own work.)
    all_items: list = []

    def _agent_text(agent_name: str) -> str:
        msgs = [
            it for it in all_items
            if isinstance(it, MessageOutputItem) and getattr(it.agent, "name", "") == agent_name
        ]
        return ItemHelpers.text_message_outputs(msgs).strip() if msgs else ""

    async def _drive():
        async def _run(agent, stage_input, label):
            logger.info("STAGE → %s", label)
            with trace(name=f"Stage-{agent.name}", run_type="chain"):
                res = await Runner.run(agent, stage_input, max_turns=MAX_TURNS)
            all_items.extend(res.new_items)
            logger.info("STAGE done: %s (last_agent=%s, html_en=%s, html_it=%s, html_fa=%s)",
                         label, getattr(res.last_agent, "name", "?"),
                         bool(state.get("html_en")), bool(state.get("html_it")), bool(state.get("html_fa")))
            return res

        # 1) Optimistic handoff chain, entered at the Director.
        await _run(director, input_payload, "handoff chain (entry=Director)")

        # 2) Recover any stage a missed handoff skipped, by running it directly with a clean input.
        plan = _agent_text("Director")
        script = _agent_text("Storyteller")
        if not script:
            await _run(storyteller, plan or input_payload, "Storyteller (recovery)")
            script = _agent_text("Storyteller")
        if not state.get("html_en"):
            await _run(cartoonist, script or plan or input_payload, "Cartoonist (recovery)")
        if state.get("html_en") and not (state.get("html_it") and state.get("html_fa")):
            reteller_kickoff = (
                f"=== DIRECTOR'S EPISODE PLAN ===\n{plan}\n\n"
                f"=== STORYTELLER'S SCRIPT ===\n{script}\n\n"
                "Now produce the Italian ('it') and Persian ('fa') editions: write the "
                "language-neutral beat sheet, call save_beat_sheet, then call "
                "write_italian_edition and write_persian_edition."
            )
            await _run(reteller, reteller_kickoff, "Reteller (recovery)")
        # Second-tier backstop: the beat sheet exists but a nested author run failed — run
        # that author directly (fresh context, still blind) instead of redoing the whole stage.
        if state.get("beat_sheet"):
            for author, key, label in (
                (italian_author, "html_it", "ItalianAuthor (recovery)"),
                (persian_author, "html_fa", "PersianAuthor (recovery)"),
            ):
                if state.get("html_en") and not state.get(key):
                    await _run(author, "Write and assemble your edition from the saved beat sheet now.", label)
        return plan, script

    director_plan, storyteller_script = asyncio.run(_drive())

    # The assembly tools write the finished pages into state as they run.
    html = state.get("html_en", "")
    html_it = state.get("html_it", "")
    html_fa = state.get("html_fa", "")

    if not html:
        logger.error(
            "No English HTML produced — the Cartoonist did not complete assemble_layout "
            "(usually an upstream image-generation failure)."
        )
        fallback_text = _agent_text("Cartoonist") or storyteller_script or director_plan or ""
        html = (
            "<div style='padding:40px;text-align:center;color:#888;font-family:sans-serif'>"
            "<p>Comic generation completed but layout assembly was skipped.</p>"
            f"<details><summary>Agent output</summary><pre>{_escape_html(fallback_text[:3000])}</pre></details>"
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
