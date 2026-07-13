import os
from typing import Annotated

from TomorrowNews.tools.getimage import get_image_by_text
from TomorrowNews.tools.getnews import get_todays_news_feed, create_news_feed_tool, RSS_URLS
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

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_MODEL"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    output_version="responses/v1",
    use_responses_api=True,
    temperature=1.3,
    timeout=None,
    max_retries=2,
)

imagetool = get_image_by_text


def create_news_graph(news_tool):
    tools = [news_tool, imagetool]
    # web_search is an OpenAI server-side tool: bound to the model but never
    # routed through ToolNode (it produces content blocks, not tool_calls).
    llm_with_tools = llm.bind_tools(tools + [{"type": "web_search"}])

    graph_builder = StateGraph(State)

    def agent(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph_builder.add_node("agent", agent)
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_conditional_edges("agent", tools_condition)
    graph_builder.add_edge("tools", "agent")
    graph_builder.set_entry_point("agent")
    return graph_builder.compile()


news_graphs = {
    lang: create_news_graph(create_news_feed_tool(rss_url))
    for lang, rss_url in RSS_URLS.items()
}

news_graph = news_graphs["en"]
