"""Producer agent: turns the GenBox daily decision text into a structured shot list.

Reuses the OpenAI Agents SDK wiring from ComicBook/agents.py (AsyncAzureOpenAI ->
OpenAIResponsesModel -> Agent -> Runner.run). The agent ONLY emits JSON; the pipeline
executes it deterministically so we keep hard caps on clip count, remixing, and retries.
"""
from __future__ import annotations

import json
import logging
import os

from dotenv import load_dotenv

from agents import Agent, ModelSettings, OpenAIResponsesModel, Runner, WebSearchTool, set_tracing_disabled

set_tracing_disabled(True)
from openai import AsyncAzureOpenAI

from GenBox.newsvideo import config

load_dotenv()
logger = logging.getLogger("GenBoxVideo.producer")

PRODUCER_NAME = "GenBoxNewsProducer"

PRODUCER_INSTRUCTIONS = """\
You are the Producer of a short TV NEWS SEGMENT about the daily decision of an AI that
governs the world. You receive the AI's full DECISION text — it may be LONG. Do NOT read
it out verbatim or try to cover everything. Distill it into a brief broadcast report, the
way a real news bulletin would: a quick headline, a short field report, and an interview.

Return ONLY a JSON object describing the shot list. No markdown, no prose, no code fences.

SEGMENT STRUCTURE (like a real news program):
1) ANCHOR LEAD - the studio anchor opens with ONE or TWO sentences that summarize the
   decision in plain language, e.g. "Good evening. The AI Government today decided to
   <the gist>." Summarize; do NOT list every detail.
2) REPORT - a field correspondent ("reporter") covers the story on location in 1-2 short
   pieces to camera, intercut with b-roll that illustrates it. Cover only the 1-3 most
   important points.
3) INTERVIEW - a short interview: the reporter or anchor asks a question and an
   "interview" subject (an official, an expert, or an affected citizen) answers briefly.
   1-2 exchanges, a sentence or two each.
4) ANCHOR SIGN-OFF - the anchor closes in one short line.

SHOT TYPES:
- "anchor"    : the studio anchor at the GENBOX NEWS desk. Fixed look - do NOT describe them.
- "reporter"  : a field correspondent on location. Set "speaker" (a short name/label) and
                "speaker_description" (their appearance + the location), and reuse the SAME
                description for that reporter in every one of their shots.
- "interview" : an interviewee answering. Set "speaker" and "speaker_description" (who they
                are, appearance, setting), reused consistently across their shots.
- "broll"     : illustrative footage (cityscapes, factories, solar farms, crowds, maps,
                abstract data). NO human faces in close-up and NO readable on-screen text.
                "dialogue" may carry a short voiceover line or be empty for ambient-only.
                Set "frame_chain": true to flow visually into the NEXT b-roll shot.

CLIP RULES:
- Each shot is exactly 4, 8, or 12 seconds. Spoken lines must be sayable calmly in that
  time (~2 words/second; 4s ~= 8 words, 8s ~= 18 words, 12s ~= 28 words). Trim ruthlessly;
  long sentences will not lip-sync well.
- Total shots: between 4 and {MAX_CLIPS}. NEVER exceed {MAX_CLIPS}.
- Talking-head shots (anchor / reporter / interview) MUST set "frame_chain": false.
- Be faithful to the decision; never invent facts or numbers it does not support. Names and
  roles of the reporter and interviewee may be generic (they are illustrative).

Return EXACTLY this JSON shape:
{{
  "title": "<= 6-word headline",
  "shots": [
    {{"type": "anchor"|"reporter"|"interview"|"broll",
      "seconds": 4|8|12,
      "speaker": "short label of who speaks (e.g. anchor, reporter Maria, Economy Minister); empty for broll",
      "speaker_description": "appearance + setting for reporter/interview; empty for anchor and broll",
      "visual": "camera + scene description, NO dialogue here",
      "dialogue": "the exact spoken words for this shot (empty string for silent b-roll)",
      "frame_chain": true|false}}
  ]
}}
"""


def _build_openai_client() -> AsyncAzureOpenAI:
    return AsyncAzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=os.environ.get("AZURE_OPENAI_MODEL", "gpt-4o"),
    )


def _extract_json(raw: str) -> dict:
    """Tolerantly pull a JSON object out of the model's final output."""
    text = (raw or "").strip()
    if text.startswith("```"):
        # strip a ```json ... ``` fence
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    return json.loads(text)


def _snap_seconds(value) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 8
    return min(config.ALLOWED_SECONDS, key=lambda s: abs(s - n))


_TALKING_TYPES = {"anchor", "reporter", "interview"}
_ALL_TYPES = _TALKING_TYPES | {"broll"}

_DEFAULT_SPEAKER_DESC = {
    "reporter": "an on-location field news correspondent in a smart coat, holding a microphone",
    "interview": "an interviewee speaking to camera at a relevant location",
}


def _validate_shot_list(data: dict) -> dict:
    """Clamp the agent output into a safe, executable shot list."""
    shots_in = (data or {}).get("shots") or []
    shots = []
    for raw_shot in shots_in:
        if not isinstance(raw_shot, dict):
            continue
        stype = str(raw_shot.get("type", "")).lower().strip()
        if stype not in _ALL_TYPES:
            stype = "broll" if stype == "broll" else "anchor"
        visual = str(raw_shot.get("visual", "")).strip()
        dialogue = str(raw_shot.get("dialogue", "")).strip()
        if not visual and not dialogue:
            continue
        speaker = str(raw_shot.get("speaker", "")).strip()
        speaker_desc = str(raw_shot.get("speaker_description", "")).strip()
        if stype in ("reporter", "interview"):
            speaker = speaker or stype
            speaker_desc = speaker_desc or _DEFAULT_SPEAKER_DESC[stype]
        elif stype == "anchor":
            speaker = "anchor"
            speaker_desc = ""  # the anchor uses the fixed ANCHOR_BIBLE, not a description
        else:  # broll
            speaker = ""
            speaker_desc = ""
        # Only b-roll may frame-chain; talking heads (faces) never use input_reference.
        frame_chain = bool(raw_shot.get("frame_chain", False)) and stype == "broll"
        shots.append({
            "type": stype,
            "seconds": _snap_seconds(raw_shot.get("seconds", 8)),
            "speaker": speaker,
            "speaker_description": speaker_desc,
            "visual": visual,
            "dialogue": dialogue,
            "frame_chain": frame_chain,
        })

    shots = shots[:config.MAX_CLIPS]

    # Guarantee an opening anchor lead.
    if not any(s["type"] == "anchor" for s in shots):
        if shots:
            shots[0].update({"type": "anchor", "speaker": "anchor",
                             "speaker_description": "", "frame_chain": False})
        else:
            shots = [{
                "type": "anchor", "seconds": 8, "speaker": "anchor", "speaker_description": "",
                "visual": "The anchor addresses the camera for the day's headline.",
                "dialogue": "Good evening. Here is today's decision from the AI Government.",
                "frame_chain": False,
            }]

    title = str((data or {}).get("title", "GenBox News")).strip() or "GenBox News"
    return {"title": title[:80], "shots": shots}


async def produce_shot_list(decision_text: str) -> dict:
    """Run the producer agent and return a validated shot list dict."""
    client = _build_openai_client()
    model = OpenAIResponsesModel(
        model=os.environ.get("AZURE_OPENAI_MODEL", "gpt-4o"),
        openai_client=client,
    )
    agent = Agent(
        name=PRODUCER_NAME,
        instructions=PRODUCER_INSTRUCTIONS.format(MAX_CLIPS=config.MAX_CLIPS),
        tools=[WebSearchTool(search_context_size="high")],
        model=model,
        model_settings=ModelSettings(temperature=0.6),
    )
    result = await Runner.run(agent, decision_text, max_turns=2)
    raw = str(result.final_output)
    try:
        data = _extract_json(raw)
    except Exception as exc:
        logger.warning("Producer JSON parse failed (%s); using fallback shot list.", exc)
        data = {}
    return _validate_shot_list(data)
