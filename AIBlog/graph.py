import getpass
import os

from typing import Annotated

from AIBlog.tools.getimage import get_image_by_text
from AIBlog.tools.getnews import get_todays_news_feed
from AIBlog.tools.searchinternet import askinternettool, searchinternettool
from AIBlog.tools.browseweb import *
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_openai import AzureChatOpenAI

if "AZURE_OPENAI_API_KEY" not in os.environ:
    raise Exception("No AZURE_OPENAI_API_KEY found in environment!")

if "AZURE_OPENAI_ENDPOINT" not in os.environ:
    raise Exception("No AZURE_OPENAI_ENDPOINT found in environment!")

newstool = get_todays_news_feed
imagetool = get_image_by_text
tools = [searchinternettool, imagetool]
tools += browsewebtools

llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_MODEL"],  # or your deployment
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],  # or your api version
    temperature=1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

from langgraph.prebuilt import create_react_agent
react_agent = create_react_agent(llm, tools=tools)