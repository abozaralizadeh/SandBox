# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this repo is

SandBox is a single Flask app (`main.py`) hosting six independent AI subprojects, deployed to
Azure App Service (web app "genbox", domain SandBoxes.Live) via `azure-pipelines.yml` on every
push to `main`. There is **no scheduler anywhere**: all content is generated lazily when the
first visitor of the day requests it, then cached in Azure Table Storage. GenBox publishes on a
cadence (`GENBOX_GENERATION_INTERVAL_DAYS`, default 1) — still lazily, but only on dates that
land on the interval grid; see `GenBox/schedule.py`.

Run locally:

```bash
python main.py                          # Flask dev server on :5050
gunicorn --bind=0.0.0.0 --timeout 3600 --workers 4 --threads 2 main:app   # prod (startup.sh)
langgraph dev                           # LangGraph server for the graphs in langgraph.json
```

The 1-hour gunicorn timeout is deliberate — it matches the ComicBook generation budget
(`COMICBOOK_LLM_TIMEOUT` / `COMICBOOK_IMAGE_TIMEOUT`, both default 3600s). Don't lower it.

## Subprojects at a glance

| Directory | What it makes | Agent framework | Entry point (called from main.py) |
|---|---|---|---|
| `TomorrowNews/` | Speculative "tomorrow's newspaper" (en/fa/it) from real RSS news | LangGraph (`StateGraph` agent↔ToolNode loop; legacy supervisor variants exist) | `prompt.gettomorrownews(date, lang)` |
| `AIBlog/` | Daily AI-research blog post with DALL·E banner | LangGraph `create_react_agent` over `TokenAwareAzureChatOpenAI` | `prompt.getaiblog(date)` (async) |
| `ComicBook/` | Daily comic strip with persistent multi-episode arcs, it/fa retellings | **OpenAI Agents SDK** (handoff chain, NOT LangGraph) | `prompt.get_comicbook(date, lang)` |
| `GenBox/` | Daily "AI world government" decision + Sora 2 news video + TTS narration | Plain HTTP for decision text; OpenAI Agents SDK for the video Producer | `prompt.get_llm_response(date)`, `video.ensure_generation_started(date)` |
| `AIOpenProblemSolver/` | Daily research iterations on open math problems (one entry per day; `memory.py` keeps the cross-day record of tried/failed work) | `deepagents.create_deep_agent` (LangGraph-based; ReAct fallback) | `prompt.get_problem_history(...)` |
| `TrAIde/` | Read-only dashboard for the separate trAIde trading bot | None — pure consumer | `azurestorage.*` getters, `market.get_candles` |

`TrAIde/` contains **no LLM code**. The producing agents live in the separate repo at
`/Users/abozar/Documents/Projects/trAIde`; its `dashboard_publisher.py` writes a sanitized,
privacy-safe projection (indexed equity starting at 100, no dollar amounts) to Azure, and this
repo only reads it. Never add account data or absolute amounts to the dashboard payload.

## Core architecture pattern (all generators)

1. Flask route checks Azure **Table Storage** keyed by flat date (`YYYYMMDD`, sometimes
   `YYYYMMDD_HH` or `YYYYMMDD_00_{lang}`; helpers in `utils.py`).
2. Cache miss → generate: synchronously in-request (TomorrowNews, AIBlog, ComicBook), in a
   background daemon thread (GenBox text/video/audio), or on `ensure_latest=true`
   (AIOpenProblemSolver).
3. HTML larger than **32,000 chars** doesn't fit a table property — it is offloaded to Blob
   Storage and the table row keeps an `html_blob_name` pointer, hydrated on read. Preserve this
   in any storage change.
4. Cross-worker single-flight is done with **Azure-table entity-create locks** with TTLs
   (ComicBook `generation_lock` 1h; GenBox decision/video/audio locks, stale after 1800s;
   AIOpenProblemSolver `iteration_lock`, `AIOPS_ITERATION_LOCK_TTL` 5400s — its iteration runs
   from a *blocking* request, so without the lock every concurrent visitor started its own).
   There are 4 gunicorn workers — never assume in-process state is shared.
5. **A failed generation must never be persisted.** `run_comic_pipeline` returns `failed: True`
   when the Cartoonist never produced a page, and `get_comicbook` then stores nothing. A stored
   fallback page would be served from cache for that date forever (so the day could never
   regenerate), would consume an episode slot via `episodes_count`, and would blank the arc's
   `last_story_summary` — the "where the story stands" note the next day's Director reads.
   Failures are counted per date (`generation_failure` partition) and capped at 3 attempts so a
   persistently failing day stops re-running the whole image pipeline on every page view.
6. Content endpoints (`/tomorrownewscontent`, `/aiblogcontent`, `/comicbookcontent`, `/traide/*`)
   are **Referer-guarded** against hotlinking; keep the guard on new data endpoints.

## Gaps in the archives (every generator)

Nothing schedules these generators: content exists for a date only if a visitor arrived that
day and the app was up. Every archive therefore has **holes**, and any downtime adds more. Two
rules keep that from breaking, and both must hold for anything new:

- **Only the LIVE date generates.** A request for any earlier date serves what is stored or
  reports the gap; it never generates. Each edition reports the day it was made (TomorrowNews
  predicts tomorrow from *that day's* real news, AIBlog covers research published that day), so a
  back-dated edition would be a lie — and for ComicBook actively destructive, since `save_episode`
  advances the arc's `last_episode_date`/`last_story_summary` and a back-dated episode would
  leave the running arc describing something that is not its newest chapter. `save_episode`
  additionally numbers episodes **by date order** and only ever moves the arc's pointer forward,
  so a stray back-dated write cannot corrupt continuity. GenBox's live date is its schedule slot
  (`GenBox/schedule.py`), not simply today.
- **Navigation follows storage, not the calendar.** Stepping ±1 day walks straight into a hole,
  which is why every ◀/▶ pair steps over an index of dates that actually exist:
  `/aiblogindex`, `/tomorrownewsindex` (grouped per language — `fa`/`it` start much later than
  `en`, and the early archive is hourly, so editions are matched on the full timestamp),
  `/comicbookindex`, `/genbox-channels`, and AIOpenProblemSolver's paginated timeline. All clamp
  at both ends, recover when landing on a gap date (a shared link), and fall back to plain
  ±1-day stepping if the index endpoint is unreachable. Calendars grey out dates with no content.
  A date with no content returns a placeholder page plus an `Edition-Missing: 1` header.

## Agent-design conventions (repo-wide)

- **Tools never call the LLM.** A `@function_tool` (or LangChain tool) does deterministic work
  only. Anything that needs to reason is its own `Agent`, exposed via `as_tool` or a handoff
  (see ComicBook's OriginalityCritic and the Italian/Persian authors).
- **Prefer SDK-native handoff chains** over imperative `Runner.run` sequencing. The ComicBook
  pipeline is a single `Runner.run(Director)` handoff chain (Director → Storyteller →
  Cartoonist → Reteller) with *deterministic recovery*: after the optimistic run, any stage
  whose artifact is missing is re-run directly. Keep the recovery when touching the pipeline.
- **Localization is done by blind native authors, not translators**: the Reteller/native agents
  write from an English-echo-guarded beat sheet in fresh context, never from the English text.
- **Guard misbehaving agents in three layers**: prompt instruction + input flags + the tool
  itself refusing (e.g. `end_current_arc` refusing early arc-closes, `commit_style_card` refusing
  a style that collides with a recent arc's family/construction/process). A prompt-only guard is
  not enough.
- **A quality gate that judges its own subject will pass it.** Where an agent checks work, make
  the blindness structural: the beat sheet is written by an agent that never saw the English
  script; `StyleForensics` catalogues the sheet image knowing nothing about the intended style,
  and `StyleAuditor` compares the two descriptions without ever seeing the image.

## ComicBook art style (`ComicBook/style.py`)

The arc's look is a `StyleCard`, not a label, and it is injected into every image prompt **by
code** (`compose_image_prompt` / `compose_sheet_prompt`) — never by asking the Cartoonist to
retype it. Constraints that were learned the hard way; do not undo them:

- **gpt-image has no negative-prompt channel.** "avoid glossy 3D" summons glossy 3D. Only
  positive `contrastive_assertions` are sent; the avoid-list (`generic_tells`) is audit-only.
- **`render_directive` must be brand-free.** Naming a studio/franchise/living artist trips the
  image safety layer and returns a blank panel. Describe technique, material and proportions
  instead — it also renders more distinctly.
- **Never prompt for "a character reference sheet".** That phrase has a stronger visual prior
  (clean flat turnaround on white) than any style adjective, and that sheet is reference #1 for
  every panel of the arc, so its look propagates everywhere. Use `sheet_conceit` — an artefact
  the medium itself would produce.
- **Style names are not the anti-repetition test**; `commit_style_card` compares production
  family, figure construction and physical process, computed LRU-style from arc history.
- New style fields must be added to the explicit `select=[...]` in `get_recent_arc_summaries` or
  they come back empty with no error, silently breaking the rotation.
- `COMICBOOK_PANEL_QUALITY` (default `high`) and `COMICBOOK_RESTYLE_ARC` (re-style a running arc
  today rather than waiting for the next arc boundary) are the two rollout dials.

## Handoffs on reasoning models

Do NOT use the SDK's `remove_all_tools` as a handoff `input_filter` here. It strips
`ReasoningItem` while keeping the assistant messages, and on a reasoning deployment every
message is bound to the reasoning item from the same turn — so replaying that history is
rejected with `400 ... Item 'msg_...' of type 'message' was provided without its required
'reasoning' item`. That killed the whole chain at the Director→Storyteller handoff *after* the
arc and outline had been written, and looked intermittent because it only fires when the
handing-off agent emitted a message alongside its reasoning. `_strip_tools_keep_reasoning` in
`agents.py` drops the same tool items but preserves reasoning items.

## Model capability

Reasoning-family deployments accept **only the default `temperature` of 1** and reject every
other value with a 400 (production `AZURE_OPENAI_MODEL` is `gpt-5.6-luna`). The error text
misleads twice over: it blames the parameter rather than the value, and it differs by surface —
Responses says *"Unsupported parameter: 'temperature' is not supported with this model"*, chat
completions says *"Unsupported value: 'temperature' does not support 0.8"*. Measured on
`gpt-5.6-luna`: 1 succeeds on both surfaces, 0.6/0.8/1.3 fail on both.

So a deliberate creative temperature must be **dropped**, not clamped — omitting it sends
exactly the 1 the model requires. `temperature_kwargs(value)` / `supports_custom_temperature()`
in `llm_runtime.py` do that for the whole repo (ComicBook's `_model_settings` delegates to
them); override with `LLM_MODEL_SUPPORTS_TEMPERATURE` or the older
`COMICBOOK_MODEL_SUPPORTS_TEMPERATURE`. The prefix rule over-matches on purpose: `gpt-5.4`
chat completions *does* accept 0.6-0.9, but dropping never 400s.

Beware which client you are using — the guard is only needed where the value reaches the wire.
LangChain's `init_chat_model` strips temperature for these models on its own (which is why
AIOpenProblemSolver kept running through the switch), while a directly-constructed
`AzureChatOpenAI(temperature=…)` (TomorrowNews) and the Agents SDK's `ModelSettings(temperature=…)`
(GenBox's Producer) both send it and 400.

## LangSmith tracing

- LangGraph graphs must be **named** (`compile(name=...)`, `create_react_agent(..., name=...)`,
  `create_deep_agent(..., name=...)`) or every trace shows up as "LangGraph". All current graphs
  carry project names ("Tomorrow News (en/fa/it)", "AIBlog", "AI Open Problem Solver").
- ComicBook routes the OpenAI Agents SDK tracer into LangSmith via
  `set_trace_processors([OpenAIAgentsTracingProcessor()])` in `ComicBook/agents.py`. Do NOT add
  `wrap_openai` or `@traceable` around the same calls — it duplicates traces.
- **Hosted tools (WebSearchTool) emit no SDK span** and no `on_tool_start/end` hooks fire for
  them. `WebSearchTracingHooks` in `ComicBook/agents.py` compensates by emitting a `custom_span`
  per `web_search_call` from `on_llm_end`; pass `hooks=_WEB_SEARCH_HOOKS` to any new
  `Runner.run` that binds `WebSearchTool`.
- gpt-image calls (`images.generate/edit`) are invisible to both tracers;
  `ComicBook/tools/getimage.py` wraps them in `generation_span` (no-op without an active trace).
- GenBox's producer runs with `RunConfig(tracing_disabled=True)` **on purpose** (per-run
  disable; the process-global switch would kill ComicBook's tracing too) and uses `wrap_openai`
  instead. `GenBox/newsvideo/tracing.py` redacts per-resource API keys and binary payloads —
  route new traced GenBox code through it.

## Debug & environment

- `DEBUG=true` — skip cache reads (TomorrowNews/AIBlog force regeneration); in ComicBook,
  isolates all reads/writes to `arc_debug` / debug-lock partitions so local runs never touch
  prod data. `DEBUG_SAVE=false` (with DEBUG) makes ComicBook/AIBlog a pure dry run (no writes).
  Prod (DEBUG unset) always persists. Use these for any local pipeline test.
- One shared `connection_string` for all Azure Storage; per-project table/blob names are
  lowercase env vars (`comicbook_table_name`, `aiblog_blob_name`, `genbox_table_name`,
  `aiops_table_name`, `traide_table_name`, …).
- Chat models come from `AZURE_OPENAI_{API_KEY,ENDPOINT,MODEL,API_VERSION}`; image models from a
  separate `*_DALLE` resource; Sora/TTS from `*_SORA` vars (comma-separated lists for a
  resource pool). `.env` is loaded by `python-dotenv` and by `langgraph.json`.
- Sora's API is **job-scoped**: a video id only exists on the resource that created it, so each
  clip's create→poll→download lifecycle must stay pinned to one resource
  (`newsvideo/sora_client.py` handles affinity + failover). Never put the Sora endpoints behind
  a round-robin gateway.
- The same rule binds the **Responses API**, and here it BITES: in production
  `AZURE_OPENAI_ENDPOINT` is the APIM load balancer
  (`https://pocs-abozar-apim.azure-api.net/abopenailb/` in the `genbox` App Service settings —
  *not* the single resource the local `.env` points at), which round-robins three independent
  Azure OpenAI resources. With the API default `store=true`, each turn's output items carry ids
  minted by the serving resource (`fc_*`, `rs_*`) and the SDK replays them as the next turn's
  input; another resource rejects them with *"The requested item was created under a different
  Azure OpenAI resource"*. Every multi-turn run — ComicBook's whole handoff chain, GenBox's
  Producer — therefore failed on 2 of 3 backends and was retried by the gateway. Fixed by
  `llm_runtime.py`: every `Runner.run` passes `store=False`, which carries the conversation in
  the request and mints no ids. On the `gpt-5.6-luna` reasoning deployment `store=False` returns
  the reasoning item with `encrypted_content` automatically, so cross-resource replay keeps the
  reasoning that `_strip_tools_keep_reasoning` protects — no `response_include` needed. Never
  add a `Runner.run` here without that run config.

## Gotchas

- `sitecustomize.py` and the top of `main.py` both strip `/agents/python` from `sys.path` —
  Azure App Service ships outdated stdlib shims there that shadow modern libraries. Keep both.
- `AIBlog/tools/searchinternet.py` requires `TAVILY_API_KEY` at import time.
- ComicBook panel prompts carry a reference stack built in priority order: character sheet →
  one mid-arc key panel → the last `COMICBOOK_RECENT_PANEL_REFS` (default 5) panels already drawn
  in this episode → one prior-episode anchor, truncated to the endpoint's 16-image ceiling. The
  5-panel window is only possible because panels are generated **sequentially** — a parallel call
  has nothing finished to reference. More references buy character/scene consistency and cost
  style fidelity (each one pulls the render toward its own look), so treat the count as a dial.
- ComicBook panel images are served through `/cbimg`, a lazy WebP transcode/302 proxy with an
  open-relay guard (`imageproxy.py`) — comic HTML must go through `rewrite_comic_images()`.
- The image API distinguishes **moderation blocks** (`ContentModerationError` — rewrite the
  prompt, don't retry) from transient failures (retry/fallback). Preserve that split.
  `COMICBOOK_IMAGE_ATTEMPTS` (default 3) drives the transient retry; it was effectively 1
  (no retries) once, and a single connection blip then tripped the run's 2-failure circuit
  breaker and rendered every remaining panel as a grey placeholder.
- `TomorrowNews/ReAct.py`, `multiagent.py`, `supervisor.py` are legacy/alternate architectures;
  the Flask path uses `graph.py`'s per-language graphs. `langgraph.json` still exposes the
  supervisor for the dev server.
- Mobile ComicBook layout is a single column with text **below** each photo (flex order in the
  `@600` media block); desktop keeps the overlay look. Don't regress this when touching CSS.
