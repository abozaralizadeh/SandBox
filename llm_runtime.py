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
from agents import ModelSettings, RunConfig

# For a Runner.run that has no RunConfig of its own.
STATELESS_MODEL_SETTINGS = ModelSettings(store=False)
STATELESS_RUN_CONFIG = RunConfig(model_settings=STATELESS_MODEL_SETTINGS)
