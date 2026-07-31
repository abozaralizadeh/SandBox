# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this repo is

SandBox is a single Flask app (`main.py`) hosting six independent AI subprojects, deployed to
Azure App Service (web app "genbox", domain SandBoxes.Live) via `azure-pipelines.yml` on every
push to `main`. There is **no scheduler anywhere**: all content is generated lazily when the
first visitor of the day requests it, then cached in Azure Table Storage.

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
| `AIOpenProblemSolver/` | Daily research iterations on open math problems | `deepagents.create_deep_agent` (LangGraph-based; ReAct fallback) | `prompt.get_problem_history(...)` |
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
   (ComicBook `generation_lock` 1h; GenBox decision/video/audio locks, stale after 1800s).
   There are 4 gunicorn workers — never assume in-process state is shared.
5. Content endpoints (`/tomorrownewscontent`, `/aiblogcontent`, `/comicbookcontent`, `/traide/*`)
   are **Referer-guarded** against hotlinking; keep the guard on new data endpoints.

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

## Model capability

Reasoning-family deployments (`gpt-5.x`, `o1`/`o3`/`o4` — and `AZURE_OPENAI_MODEL` is currently
`gpt-5.5`) reject `temperature` with a 400. Every ComicBook agent sets a deliberate temperature,
so they go through `_model_settings(model_name, temp)` in `agents.py`, which omits the parameter
where it cannot be sent. Override with `COMICBOOK_MODEL_SUPPORTS_TEMPERATURE`.

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

## Gotchas

- `sitecustomize.py` and the top of `main.py` both strip `/agents/python` from `sys.path` —
  Azure App Service ships outdated stdlib shims there that shadow modern libraries. Keep both.
- `AIBlog/tools/searchinternet.py` requires `TAVILY_API_KEY` at import time.
- ComicBook panel images are served through `/cbimg`, a lazy WebP transcode/302 proxy with an
  open-relay guard (`imageproxy.py`) — comic HTML must go through `rewrite_comic_images()`.
- The image API distinguishes **moderation blocks** (`ContentModerationError` — rewrite the
  prompt, don't retry) from transient failures (retry/fallback). Preserve that split.
- `TomorrowNews/ReAct.py`, `multiagent.py`, `supervisor.py` are legacy/alternate architectures;
  the Flask path uses `graph.py`'s per-language graphs. `langgraph.json` still exposes the
  supervisor for the dev server.
- Mobile ComicBook layout is a single column with text **below** each photo (flex order in the
  `@600` media block); desktop keeps the overlay look. Don't regress this when touching CSS.
