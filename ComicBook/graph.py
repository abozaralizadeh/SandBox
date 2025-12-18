import os
from datetime import datetime
from typing import Annotated

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command
from langchain_openai import AzureChatOpenAI
from typing_extensions import Literal, TypedDict

from ComicBook.tools.getimage import (
    get_image_by_text,
    get_image_by_text_with_reference,
)


class ComicState(TypedDict):
    messages: Annotated[list, add_messages]
    arc_title: str
    arc_logline: str
    arc_day: int
    arc_target: int
    previous_beats: str
    last_image_url: str


def _ensure_env(var_name: str):
    if var_name not in os.environ:
        raise EnvironmentError(f"Missing required environment variable '{var_name}'.")


_ensure_env("AZURE_OPENAI_API_KEY")
_ensure_env("AZURE_OPENAI_ENDPOINT")

llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_MODEL"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    # Some deployments only allow temperature=1; use default-safe value.
    temperature=1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

tools=[get_image_by_text, get_image_by_text_with_reference, {"type": "web_search"}]

from langchain.agents import create_agent
react_graph = create_agent(llm, tools=tools)

#comic_graph.get_graph().draw_mermaid_png(output_file_path="comic_graph.png")
