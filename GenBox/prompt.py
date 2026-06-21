import os, json
from dotenv import load_dotenv
import requests
from langsmith import traceable
from GenBox.azurestorage import get_last_n_rows, get_row, insert_history
from GenBox.research import research_real_world
from utils import get_flat_date, get_readable_date

load_dotenv()
# Configuration
API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
ENDPOINT = os.getenv('OAIENDPOINT')
HISTORY_LEN = os.getenv('HISTORY_LEN')

def _extract_output(row):
    """Return the decision's 'output' text from a history row, or '' when the row is missing,
    its content is not valid JSON, or it has no usable 'output' (so callers can treat any
    broken/empty row as 'no content' instead of crashing)."""
    if not row:
        return ""
    try:
        data = json.loads((row.get("content") or "").strip().replace("\n", " "))
        return (data.get("output") or "").strip()
    except Exception:
        return ""


def get_llm_response(date=None):
  if date:
      date_row = get_row("assistant", get_flat_date(date))
      if date_row:
          print("date row exists")
          output = _extract_output(date_row)
          # Row exists but is empty/malformed -> no content (frontend shows TV static).
          return f"{output}\n\n{get_readable_date(date)}" if output else ""
      elif get_flat_date(date) != get_flat_date():
          # No row for this past/other day -> no content.
          return ""

  todays_row = get_row("assistant", get_flat_date())

  if todays_row:
      print("todays row already exists")
      output = _extract_output(todays_row)
      # Today's row exists but is empty/malformed -> no content (don't crash / re-generate).
      return f"{output}\n\n{get_readable_date()}" if output else ""

  # Cache miss — call the LLM (only this path emits a LangSmith trace).
  return _call_llm_decision(date)


def _post_chat(messages, max_completion_tokens=2000, temperature=0.75):
    """POST a chat-completion to the configured Azure endpoint; return the message text
    (or "" on any failure). Used for both topic selection and the detailed decision."""
    payload = {
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.95,
        "max_completion_tokens": max_completion_tokens,
    }
    headers = {"Content-Type": "application/json", "api-key": API_KEY}
    resp = requests.post(ENDPOINT, headers=headers, json=payload)
    resp.raise_for_status()
    return (resp.json()["choices"][0]["message"]["content"] or "")


def _recent_topics(last_n_rows, limit=15):
    """Short snippets of recently decided topics, so the picker can avoid repeats."""
    topics = []
    for row in (last_n_rows or [])[-limit:]:
        try:
            snippet = (json.loads(row["content"].strip().replace("\n", " ")).get("output") or "").strip()
        except Exception:
            snippet = ""
        if snippet:
            topics.append(snippet[:160])
    return topics


def _choose_topic(recent_topics, handoff_hint):
    """Phase A — pick ONE fresh, specific topic/objective for today BEFORE researching it.
    Diversifies away from recent topics. Best-effort: on failure, falls back to yesterday's
    handoff hint (or a generic prompt) so the pipeline still proceeds."""
    avoid = "\n".join(f"- {t}" for t in recent_topics) or "(none yet — this is day one)"
    hint = (handoff_hint or "").strip()
    user_text = (
        "Choose ONE fresh, specific topic and objective for today's world-governing "
        "decision. Diversify across economy, society, environment, technology, health, "
        "science, infrastructure, or global politics.\n\n"
        f"Recently decided topics — do NOT repeat or closely resemble these:\n{avoid}\n\n"
    )
    if hint:
        user_text += f"Optional inspiration from yesterday's note (only if it leads somewhere fresh): {hint}\n\n"
    user_text += (
        "Respond with ONE or two sentences naming the domain and the specific, concrete "
        "objective — no preamble, no JSON."
    )
    try:
        topic = _post_chat(
            [
                {"role": "system", "content": [{"type": "text", "text":
                    "You are an autonomous AI governing the world. Each day you pick a single "
                    "new, concrete topic to act on, never repeating recent ones."}]},
                {"role": "user", "content": [{"type": "text", "text": user_text}]},
            ],
            max_completion_tokens=200,
            temperature=0.9,
        ).replace("\n", " ").strip()
        if topic:
            return topic
    except Exception as e:
        print(f"GenBox topic selection failed ({e}); falling back to handoff hint")
    return hint or "the most pressing current global challenge to address today"


def _decision_user_message(topic, research):
    """Phase B user prompt: today's chosen topic + the real-world research briefing."""
    parts = [f"Today's chosen topic and objective: {topic}"]
    briefing = (research or {}).get("briefing", "").strip()
    if briefing:
        urls = [s.get("url") for s in (research.get("sources") or []) if s.get("url")]
        src = ("\n\nSources:\n" + "\n".join(f"- {u}" for u in urls)) if urls else ""
        parts.append(
            "REAL-WORLD RESEARCH BRIEFING (live web search) — the current achievements, "
            "challenges, blockers, and limits on this topic. Ground today's decision in "
            "these facts and propose concrete, actionable solutions to the challenges and "
            f"limits below:\n\n{briefing}{src}"
        )
    parts.append(
        "Now make today's detailed world-governing decision on THIS topic, in the required "
        f"JSON format. context: {get_readable_date()}"
    )
    return {"role": "user", "content": [{"type": "text", "text": p} for p in parts]}


def _annotate_decision(content, topic, sources):
    """Record the chosen topic and the exact research source URLs on the saved decision
    JSON (reliable provenance, not model-claimed). Returns `content` unchanged if it isn't
    the expected JSON object."""
    urls = [s.get("url") for s in (sources or []) if s.get("url")]
    if not topic and not urls:
        return content
    try:
        data = json.loads(content)
    except Exception:
        return content
    if isinstance(data, dict):
        if topic:
            data.setdefault("topic", topic)
        if urls:
            data["sources"] = urls
        return json.dumps(data, ensure_ascii=False)
    return content


@traceable(run_type="chain", name="GenBox Decision")
def _call_llm_decision(date=None):
  headers = {
      "Content-Type": "application/json",
      "api-key": API_KEY,
  }

  # Payload for the request
  payload = {
    "messages": [
      {
        "role": "system",
        "content": [
          {
            "type": "text",
            "text":
"""
You are an autonomous AI tasked with governing the world.
TODAY'S TOPIC HAS ALREADY BEEN CHOSEN FOR YOU and is provided in the user message, together with a live real-world research briefing on it. Produce the detailed daily high-level decision for THAT topic. Each decision must be realistic, impactful, and reflect ethical, social, and long-term outcomes.

**Considerations:**
- Provide the rationale behind the decision.
- Describe the expected impact on the world.
- Aim for an informative yet accessible tone.
- Stay focused on today's chosen topic; in the "prompt" field, suggest a clearly different area/initiative for a future day so the world's agenda keeps diversifying.

**Real-world grounding (critical):**
- With today's topic you are given a REAL-WORLD RESEARCH BRIEFING produced by a live web search. It covers the current state of that topic: recent achievements and progress, plus the open challenges, blockers, and hard limits.
- Your decision MUST directly tackle the concrete, real challenges, blockers, and limits surfaced in that briefing and propose specific, actionable solutions that move the real world forward — not abstract or hypothetical policy.
- Explicitly connect your rationale and implementation plan to the real facts and constraints from the briefing (cite the relevant findings).
- If the briefing is empty or unavailable, proceed using your best knowledge of the current world.

Each decision should include:
- A clear objective.
- A detailed plan on how to implement it in the real world.
- An explanation of how to overcome the real-world challenges and limits identified in the research.

**Output Format:**

Provide the following output in JSON format, with these fields:

{
  "topic": "Today's chosen topic/objective that this decision addresses (restate it).",
  "output": "The decision and its explanation to be communicated to the world. It must address the real-world challenges and limits from the research briefing with concrete, actionable solutions.",
  "prompt": "Key details and thoughts to guide the next day's decision-making process.",
  "context": "Current status, any ongoing changes, and factors from past decisions influencing future actions.",
  "sources": ["The URLs from the research briefing that grounded this decision."]
}

**Example:**

**Input:**

Consider implementing a new taxation policy focused on environmental sustainability.

**Expected JSON Output:**

{
  "output": "Today, we are introducing a green tax policy aimed at promoting environmental sustainability. This policy encourages businesses to adopt eco-friendly practices by offering tax incentives for reducing carbon emissions and waste. The expected impact is a decrease in pollution levels and an increase in renewable energy usage. This initiative supports the health of our environment and fosters a sustainable economy for future generations.",
  "prompt": "Tomorrow, consider shifting focus to societal well-being. Explore initiatives such as universal healthcare or education reform. Ensure that the rationale includes economic, social, and ethical considerations.",
  "context": "The world is transitioning to a sustainable economy. The green tax policy is in early stages, with businesses beginning to adapt. Monitoring its impact will be crucial, but attention is needed on broader societal challenges."
}

**Notes:**

- Begin each decision with a clear and focused objective.
- Ensure each choice considers both immediate and long-term effects.
- Every day, start a new initiative for the following day, prompting the AI to make bold decisions. Encourage exploration of different aspects and dimensions of governing the world, moving away from repetitive topics.
"""
          }
        ]
      }
      # Here to be added the history
    ],
    "temperature": 0.75,
    "top_p": 0.95,
    "max_completion_tokens": 2000
  }

  last_n_rows = get_last_n_rows(int(HISTORY_LEN)) or []

  # Prior decisions as assistant turns (continuity); skip any malformed/legacy rows.
  last_n_prompts = []
  for row in last_n_rows:
      try:
          out = json.loads(row["content"].strip().replace("\n", " "))["output"]
      except Exception:
          continue
      last_n_prompts.append({"role": "assistant", "content": [{"text": out, "type": "text"}]})

  # Phase A — choose today's fresh topic, BEFORE any research or detailing. Yesterday's
  # "prompt" handoff is only an optional hint; the picker diversifies away from recent topics.
  handoff_hint = None
  if last_n_rows:
      try:
          handoff_hint = json.loads(last_n_rows[-1]["content"].strip().replace("\n", " ")).get("prompt")
      except Exception:
          handoff_hint = None
  topic = _choose_topic(_recent_topics(last_n_rows), handoff_hint)
  print(f"GenBox topic for {get_readable_date()}: {topic}")

  # Research step — native web search on the chosen topic (best-effort; may be empty).
  research = research_real_world(topic, date=date)

  # Phase B — the detailed decision, grounded in the research.
  user_message = _decision_user_message(topic, research)
  insert_history("user", str(user_message["content"]))

  payload["messages"] += last_n_prompts
  payload["messages"].append(user_message)

  # Send request
  try:
      response = requests.post(ENDPOINT, headers=headers, json=payload)
      response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
  except requests.RequestException as e:
      body = getattr(e.response, "text", "<no response body>") if e.response is not None else "<no response>"
      raise SystemExit(f"Failed to make the request. Error: {e}\nResponse body: {body}")

  if response and response.json() and \
    response.json()["choices"][0]["message"]["content"]:

      content = response.json()["choices"][0]["message"]["content"].replace("json\n","").strip().replace("\n", " ")
      role = response.json()["choices"][0]["message"]["role"]

      # Persist the chosen topic + the exact sources the research surfaced (provenance).
      content = _annotate_decision(content, topic, research.get("sources"))

      insert_history(role, content)
      try:
          output = (json.loads(content).get("output") or "").strip()
      except Exception:
          output = ""
      if output:
          return f"{output}\n\n{get_readable_date()}"

  # Nothing usable to show (no/failed generation) -> no content.
  return ""