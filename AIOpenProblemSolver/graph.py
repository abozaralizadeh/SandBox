import os
from typing import List

from langchain_openai import AzureChatOpenAI

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

    llm = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_MODEL"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        max_tokens=None,
        timeout=None,
        max_retries=3,
    )

    search_tools: List = [ddg_search_results, ddg_search, tavily_search]
    browse_tools = await get_browse_web_tools()
    tools = [*search_tools, *browse_tools]

    instructions = (
        "You are an open deep-search research agent tackling frontier mathematics problems. "
        "Break the problem into sub-goals, use the available search and browsing tools extensively, "
        "and produce rigorous, well-structured progress updates with citations. "
        "Iterate until you reach a meaningful advancement or identify next steps."
    )

    try:
        from langgraph.prebuilt import create_openai_deep_search_agent

        agent = create_openai_deep_search_agent(llm=llm, tools=tools)
    except (ImportError, AttributeError):
        try:
            from deepagents import create_deep_agent

            agent = create_deep_agent(
                model=llm,
                tools=tools,
                instructions=instructions,
            )
        except (ImportError, TypeError):
            from langgraph.prebuilt import create_react_agent

            agent = create_react_agent(
                llm,
                tools=tools,
                prompt=instructions,
            )
    return agent
