import os
import types
from typing import Any, List

from langchain.chat_models import init_chat_model

try:
    from deepagents import create_deep_agent
except ImportError as exc:  # pragma: no cover - runtime guard
    raise ImportError(
        "deepagents package is required for the Open Problem Solver agent."
    ) from exc

from AIOpenProblemSolver.tools.browseweb import get_browse_web_tools
from AIOpenProblemSolver.tools.searchinternet import (
    ddg_search,
    ddg_search_results,
    tavily_search,
)


def _ensure_env(var_name: str) -> None:
    if var_name not in os.environ:
        raise EnvironmentError(f"Required environment variable '{var_name}' is missing.")


async def get_open_deep_search_agent():
    _ensure_env("AZURE_OPENAI_API_KEY")
    _ensure_env("AZURE_OPENAI_ENDPOINT")

    llm = init_chat_model(
        model=os.environ["AZURE_OPENAI_MODEL"],
        model_provider="azure_openai",
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        max_tokens=None,
        timeout=None,
        max_retries=3,
    )

    search_tools: List = [ddg_search_results, ddg_search, tavily_search]
    browse_tools = await get_browse_web_tools()
    tools = [_truncate_tool_output(tool) for tool in [*search_tools, *browse_tools]]

    instructions = """
You are Open Deep Search, an autonomous mathematician tasked with advancing frontier research problems.

- Read and respect the historical context supplied in the conversation.
- Formulate a plan that decomposes the open problem into sub-goals before acting.
- Use web search and browsing tools aggressively to gather modern papers, blog posts, lecture notes, code, and datasets.
- Track every claim with inline citations referencing the retrieved sources.
- Surface contradictions, gaps, and partial results explicitly; speculate responsibly.
- Keep tool observations concise—summarize relevant passages instead of pasting entire webpages or PDFs. Limit each tool response to a few paragraphs with citations.
- Finish each research cycle with a concise briefing that highlights what changed, the supporting evidence, and next actions.
""".strip()

    try:
        agent = create_deep_agent(
            model=llm,
            tools=tools,
            instructions=instructions,
        )
        return agent
    except TypeError as exc:
        if "post_model_hook" not in str(exc):
            raise
        from langgraph.prebuilt import create_react_agent

        return create_react_agent(
            llm,
            tools=tools,
            prompt=instructions,
        )


MAX_TOOL_OUTPUT_CHARS = int(os.getenv("AIOPS_TOOL_MAX_CHARS", "12000"))


def _truncate_value(value: Any, limit: int = MAX_TOOL_OUTPUT_CHARS) -> Any:
    if isinstance(value, str):
        if len(value) > limit:
            return value[:limit] + "\n\n[Tool output truncated to remain within context limits.]"
        return value
    if isinstance(value, dict):
        return {k: _truncate_value(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [_truncate_value(item, limit) for item in value]
    return value


def _truncate_tool_output(tool):
    if hasattr(tool, "__wrapped_trunc__"):
        return tool

    def wrap_sync(original):
        def inner(*args, **kwargs):
            result = original(*args, **kwargs)
            return _truncate_value(result)
        return inner

    def wrap_async(original):
        async def inner(*args, **kwargs):
            result = await original(*args, **kwargs)
            return _truncate_value(result)
        return inner

    if hasattr(tool, "invoke"):
        original_invoke = tool.invoke
        object.__setattr__(tool, "invoke", types.MethodType(wrap_sync(original_invoke), tool))

    if hasattr(tool, "ainvoke"):
        original_ainvoke = tool.ainvoke
        object.__setattr__(tool, "ainvoke", types.MethodType(wrap_async(original_ainvoke), tool))

    if hasattr(tool, "_run"):
        original_run = tool._run

        def _run(self, *args, **kwargs):
            result = original_run(self, *args, **kwargs)
            return _truncate_value(result)

        object.__setattr__(tool, "_run", types.MethodType(_run, tool))

    if hasattr(tool, "_arun"):
        original_arun = tool._arun

        async def _arun(self, *args, **kwargs):
            result = await original_arun(self, *args, **kwargs)
            return _truncate_value(result)

        object.__setattr__(tool, "_arun", types.MethodType(_arun, tool))

    try:
        object.__setattr__(tool, "__wrapped_trunc__", True)
    except Exception:
        pass
    return tool
