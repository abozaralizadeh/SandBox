import getpass
import logging
import os

from typing import Annotated

from AIBlog.tools.settitle import set_title
from AIBlog.tools.getimage import get_image_by_text
from AIBlog.tools.searchinternet import *
from AIBlog.tools.browseweb import *
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from llm_runtime import STATELESS_CHAT_KWARGS

from AIBlog.token_controller import TokenAwareAzureChatOpenAI

if "AZURE_OPENAI_API_KEY" not in os.environ:
    raise Exception("No AZURE_OPENAI_API_KEY found in environment!")

if "AZURE_OPENAI_ENDPOINT" not in os.environ:
    raise Exception("No AZURE_OPENAI_ENDPOINT found in environment!")

logger = logging.getLogger("AIBlog.graph")


def _resilient_tool(tool):
    """Make a tool report its failure as text instead of raising.

    The browse tools fail routinely — a paper 404s, a host aborts the navigation
    (`Page.goto: net::ERR_ABORTED`) — and LangGraph's ToolNode absorbs only
    `ToolInvocationError`, re-raising everything else and ending the whole post. LangSmith
    counted 20 blog runs lost that way between 2026-06 and 2026-09.

    Wrapping each tool (the pattern AIOpenProblemSolver uses for `_truncate_tool_output`)
    rather than the node keeps hosted-tool dicts like `{"type": "web_search"}` untouched —
    they have no `invoke` and are passed straight through to `create_react_agent`.
    """
    if isinstance(tool, dict) or getattr(tool, "__wrapped_resilient__", False):
        return tool

    def _report(exc):
        logger.warning("Tool call failed, continuing without it: %s: %s",
                       type(exc).__name__, str(exc)[:200])
        return (f"TOOL_FAILED ({type(exc).__name__}): {str(exc)[:300]} — this source is "
                "unavailable. Do not retry it; use another source and continue.")

    if hasattr(tool, "invoke"):
        original = tool.invoke

        def invoke(*args, **kwargs):
            try:
                return original(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - any tool failure degrades to text
                return _report(exc)

        object.__setattr__(tool, "invoke", invoke)

    if hasattr(tool, "ainvoke"):
        original_a = tool.ainvoke

        async def ainvoke(*args, **kwargs):
            try:
                return await original_a(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                return _report(exc)

        object.__setattr__(tool, "ainvoke", ainvoke)

    try:
        object.__setattr__(tool, "__wrapped_resilient__", True)
    except Exception:
        pass
    return tool


async def get_react_agent():
    savetitletool = set_title
    imagetool = get_image_by_text
    tools = [{"type": "web_search"}, imagetool, savetitletool]
    browse_tools, browser_aclose = await get_browsewebtools()
    tools += browse_tools
    tools = [_resilient_tool(t) for t in tools]

    max_input_tokens = int(os.environ.get("AZURE_OPENAI_MAX_INPUT_TOKENS", "270000"))
    tool_token_limit = int(
        os.environ.get("AZURE_OPENAI_TOOL_MESSAGE_TOKEN_LIMIT", "270000")
    )
    summary_chunk_tokens = int(
        os.environ.get("AZURE_OPENAI_SUMMARY_CHUNK_TOKENS", "50000")
    )
    max_map_chunks = int(
        os.environ.get("AZURE_OPENAI_MAX_MAP_CHUNKS", "5")
    )
    summary_target_tokens = int(
        os.environ.get("AZURE_OPENAI_SUMMARY_TARGET_TOKENS", "50000")
    )

    llm = TokenAwareAzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_MODEL"],  # or your deployment
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],  # or your api version
        temperature=1,
        max_tokens=None,
        timeout=None,
        max_retries=3,
        output_version=os.environ.get("AZURE_OPENAI_OUTPUT_VERSION", "responses/v1"),
        **STATELESS_CHAT_KWARGS,
        max_input_tokens=max_input_tokens,
        tool_message_token_limit=tool_token_limit,
        summary_chunk_tokens=summary_chunk_tokens,
        summary_target_tokens=summary_target_tokens,
        max_map_chunks=max_map_chunks,
        # other params...
    )

    from langgraph.prebuilt import ToolNode, create_react_agent

    # Name the agent so LangSmith traces show "AIBlog" instead of the default
    # "LangGraph" root run name. `tools` MUST be passed as the plain list: it contains the
    # hosted `{"type": "web_search"}` dict, which create_react_agent understands (it splits
    # hosted tools from executable ones) but `ToolNode` rejects outright with "The first
    # argument must be a string or a callable". Wrapping this list in a ToolNode to attach an
    # error handler therefore raised at CONSTRUCTION, before any graph existed — AIBlog 500'd
    # with no LangSmith trace at all from 2026-09-04 until it was reverted. Resilience comes
    # from _resilient_tool below instead, which is applied per tool and leaves the dict alone.
    react_agent = create_react_agent(llm, tools=tools, name="AIBlog")
    return react_agent, browser_aclose
