"""Run-level LLM settings shared by every Agents-SDK run in this repo.

`AZURE_OPENAI_ENDPOINT` in production is an **APIM load balancer** that round-robins three
independent Azure OpenAI resources (the App Service setting is
`https://pocs-abozar-apim.azure-api.net/abopenailb/`, not the single resource the local
`.env` points at). The Responses API is **stateful**: with the API default `store=true`,
every turn's output items come back with ids minted by the resource that served them
(`fc_*` tool calls, `rs_*` reasoning) and the SDK replays those items as the next turn's
input. Any other resource rejects them:

    400 - "The requested item was created under a different Azure OpenAI resource.
           Use the same resource that created the item to access it."

So on a multi-turn run — ComicBook's whole Director→Storyteller→Cartoonist→Reteller chain,
or the Producer's search-then-answer turn — every turn after the first had a 2-in-3 chance
of being rejected and retried against the next backend. (Measured on the sibling trAIde bot
against this same pool: 59% of all backend calls wasted; 0% after this change.)

`store=false` carries the whole conversation in the request and mints no ids, so any backend
can serve any turn. Verified on `gpt-5.6-luna`, which is a reasoning deployment: with
`store=false` the reasoning item comes back carrying `encrypted_content` **automatically**
(no `response_include` needed), and replaying it to a *different* resource is accepted — so
the reasoning continuity that `_strip_tools_keep_reasoning` exists to protect still holds.

Applied through `RunConfig.model_settings`, which merges over each agent's own settings
(only non-None fields override), so the deliberate per-agent temperatures set by
`_model_settings()` are untouched.
"""
import os
import re

from agents import ModelSettings, RunConfig

# For a Runner.run that has no RunConfig of its own.
STATELESS_MODEL_SETTINGS = ModelSettings(store=False)
STATELESS_RUN_CONFIG = RunConfig(model_settings=STATELESS_MODEL_SETTINGS)

# The LangChain half of the same rule. LangGraph subprojects reach the Responses API through
# `AzureChatOpenAI(use_responses_api=True)` / `output_version="responses/v1"`, and they replay
# the previous turn's `rs_*` / `fc_*` / `resp_*` ids on the next call — which the load
# balancer's other backends reject with "The requested item was created under a different
# Azure OpenAI resource". Measured: with `store` unset a replayed turn fails on a different
# resource and succeeds on the minting one; with `store=False` it succeeds on both.
# AIOpenProblemSolver lost 16 daily runs to this between 2026-08-29 and 2026-09-04.
STATELESS_CHAT_KWARGS = {"store": False}

# ---------------------------------------------------------------------------
# Temperature support
# ---------------------------------------------------------------------------
# Reasoning-family deployments (gpt-5 and later, o1/o3/o4) accept ONLY the default temperature of 1
# and reject every other value. The error text is misleading -- it blames the parameter, not
# the value -- and differs by surface, which is why this looks like two unrelated bugs:
#
#   Responses API      400 "Unsupported parameter: 'temperature' is not supported with this model."
#   chat completions   400 "Unsupported value: 'temperature' does not support 0.8 ..."
#
# Verified on gpt-5.6-luna: temperature=1 succeeds on both surfaces, 0.6 / 0.8 / 1.3 all fail
# on both. So a deliberate creative temperature has to be DROPPED here, not clamped -- omitting
# it is exactly equivalent to sending the 1 the model insists on. Reasoning deployments expose
# `reasoning.effort` instead as the knob that actually varies output.
#
# Do NOT rely on LangChain's `init_chat_model` to strip it: it does so only for model names it
# recognises. It knew gpt-5, which is why AIOpenProblemSolver survived the switch to
# gpt-5.6-luna — and it did not know gpt-6, so on 2026-09-24 AIOPS 400'd along with everything
# else. A directly-constructed `AzureChatOpenAI(temperature=...)` and the Agents SDK's
# `ModelSettings(temperature=...)` never strip it at all.
#
# The rule is by GENERATION, not a list of names: the previous list ("gpt-5", "o1", "o3", "o4")
# was correct until production moved to gpt-6-luna, and then every chat call in the app 400'd
# at once (measured on gpt-6-luna: identical to 5.6 — 1 or omitted succeeds on both surfaces,
# 0.7 fails on both). Any gpt-N with N >= 5 is treated as reasoning. It deliberately
# over-matches: gpt-5.4 chat completions was measured to ACCEPT 0.6-0.9, but erring toward
# dropping never errors; set LLM_MODEL_SUPPORTS_TEMPERATURE=true for a known-good deployment.
_O_SERIES_PREFIXES = ("o1", "o3", "o4")
_GPT_GENERATION = re.compile(r"^gpt-(\d+)")


def _is_reasoning_family(name: str) -> bool:
    if name.startswith(_O_SERIES_PREFIXES):
        return True
    match = _GPT_GENERATION.match(name)
    return bool(match) and int(match.group(1)) >= 5


def supports_custom_temperature(model_name: str = None) -> bool:
    """True when the deployment accepts a temperature other than the default 1."""
    override = (os.environ.get("LLM_MODEL_SUPPORTS_TEMPERATURE")
                or os.environ.get("COMICBOOK_MODEL_SUPPORTS_TEMPERATURE")
                or "").strip().lower()
    if override in ("true", "1", "yes"):
        return True
    if override in ("false", "0", "no"):
        return False
    name = (model_name if model_name is not None else os.environ.get("AZURE_OPENAI_MODEL", ""))
    name = (name or "").strip().strip('"').lower()
    return not _is_reasoning_family(name)


def temperature_kwargs(value: float, model_name: str = None) -> dict:
    """`{"temperature": value}` where the deployment allows it, `{}` where it does not.

    Keeps the intended temperature visible in the calling code while staying valid on a
    reasoning deployment: `AzureChatOpenAI(**temperature_kwargs(1.3), ...)`.
    """
    return {"temperature": value} if supports_custom_temperature(model_name) else {}
