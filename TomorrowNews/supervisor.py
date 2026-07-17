import getpass
import os
from datetime import datetime, timedelta
from typing import Annotated
from typing_extensions import Literal
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.types import Command
from TomorrowNews.tools.getimage import get_image_by_text
from TomorrowNews.tools.getnews import get_todays_news_feed
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_openai import AzureChatOpenAI

if "AZURE_OPENAI_API_KEY" not in os.environ:
    raise Exception("No AZURE_OPENAI_API_KEY found in environment!")

if "AZURE_OPENAI_ENDPOINT" not in os.environ:
    raise Exception("No AZURE_OPENAI_ENDPOINT found in environment!")

class State(TypedDict):
    messages: Annotated[list, add_messages]

#graph_builder = StateGraph(State)
graph_builder = StateGraph(MessagesState)

newstool = get_todays_news_feed
imagetool = get_image_by_text

llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_MODEL"],  # or your deployment
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],  # or your api version
    temperature=1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

tools = [newstool, imagetool]
llm_with_tools = llm.bind_tools(tools)

@tool
def transfer_to_html_developer():
    """Transfer to HTML Developer"""
    return

@tool
def transfer_to_photographer():
    """Transfer to Photographer"""
    return

@tool
def transfer_to_editor():
    """Transfer to Editor"""
    return

@tool
def transfer_to_journalist():
    """Transfer to Journalist"""
    return

def journalist(
    state: MessagesState,
) -> Command[Literal["editor"]]:
    system_prompt = (
        f"""
        Based on the Editor's analysis, develop in-depth stories and headlines for the "Tomorrow News" edition.
        Focus on predicting plausible future events as a result of today’s news.
        Cover diverse topics such as geopolitics, environment, health, and culture.
        Each story should be detailed enough to fill the column space, resembling a genuine newspaper article in length and depth. 
        Ensure that your writing is engaging, informative, and reflects the nuances of future predictions.
        """
    )
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    ai_msg = llm_with_tools.bind_tools([transfer_to_editor]).invoke(messages)
    # If there are tool calls, the LLM needs to hand off to another agent
    if len(ai_msg.tool_calls) > 0:
        tool_call_id = ai_msg.tool_calls[-1]["id"]
        # NOTE: it's important to insert a tool message here because LLM providers are expecting
        # all AI messages to be followed by a corresponding tool result message
        tool_msg = {
            "role": "tool",
            "content": "Successfully transferred",
            "tool_call_id": tool_call_id,
        }
        return Command(goto="editor", update={"messages": [ai_msg, tool_msg]})

    # If the expert has an answer, return it directly to the user
    return {"messages": [ai_msg]}


def photographer(
    state: MessagesState,
) -> Command[Literal["editor"]]:
    system_prompt = (
        """
Role: Expert Newspaper Photographer
Task: Use the image generation tool (get_image_by_text) to create realistic and visually compelling photos that align with the newspaper headlines.
Instructions:
Think Like a Newspaper Photographer: Approach each image as if you're capturing a real-world scene for a major publication. Consider the context, mood, and details that would make the photo believable and engaging.
Provide Detailed Descriptions: Describe each photo thoroughly, including the setting, subjects, emotions, lighting, and any other elements that contribute to the realism and relevance of the image.
Focus on Realism: Ensure that the images are as realistic as possible. Avoid abstract or overly artistic interpretations unless explicitly required by the headline.
Content Filtering Awareness: Be mindful of the image tool's content filtering. Aim to create images that are appropriate and unlikely to be filtered. If an image is rejected due to filtering, promptly revise the description and try again to ensure a successful result.
Iterate as Needed: If content filtering issues arise, adjust the photo description to avoid sensitive or restricted content and retry until an acceptable image is generated.
Finally when the iterations are over and you have the photos transfer to editor.
        """
    )
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    ai_msg = llm_with_tools.bind_tools([transfer_to_editor, imagetool]).invoke(messages)
    # If there are tool calls, the LLM needs to hand off to another agent
    if len(ai_msg.tool_calls) > 0:
        tool_call_id = ai_msg.tool_calls[-1]["id"]
        tool_name = ai_msg.tool_calls[-1]["name"]
        if tool_name != "get_image_by_text":
            # NOTE: it's important to insert a tool message here because LLM providers are expecting
            # all AI messages to be followed by a corresponding tool result message
            tool_msg = {
                "role": "tool",
                "content": "Successfully transferred",
                "tool_call_id": tool_call_id,
            }
            return Command(goto="editor", update={"messages": [ai_msg, tool_msg]})

    # If the expert has an answer, return it directly to the user
    return {"messages": [ai_msg]}


def html_developer(
    state: MessagesState,
) -> Command[Literal["editor"]]:
    system_prompt = (
        """
After receiving the content from the Editor, your responsibility is to design a professional HTML layout for "Tomorrow News." The layout should include:
A clear title "Tomorrow News."
A two-column format with a wide central column for the main content.
Visually striking headlines and a prominent main headline.
Well-integrated images provided by the Photographer.
Ensure readability with appropriate fonts, contrast, and a cohesive color scheme.
Prioritize responsiveness so the layout adapts well to both desktop and mobile views.
The final output must be pure HTML, structured for seamless browser rendering.

        """
    )
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    ai_msg = llm_with_tools.bind_tools([transfer_to_editor]).invoke(messages)
    # If there are tool calls, the LLM needs to hand off to another agent
    if len(ai_msg.tool_calls) > 0:
        tool_call_id = ai_msg.tool_calls[-1]["id"]
        # NOTE: it's important to insert a tool message here because LLM providers are expecting
        # all AI messages to be followed by a corresponding tool result message
        tool_msg = {
            "role": "tool",
            "content": "Successfully transferred",
            "tool_call_id": tool_call_id,
        }
        return Command(goto="editor", update={"messages": [ai_msg, tool_msg]})

    # If the expert has an answer, return it directly to the user
    return {"messages": [ai_msg]}


def editor(
    state: MessagesState,
) -> Command[Literal["journalist", "photographer", "html_developer", "__end__"]]:
    timestamp = datetime.utcnow()
    next_day = timestamp + timedelta(days=1)
    system_prompt = (
        f"""
You are the Editor, the central figure in producing the next day's edition of "Tomorrow News" (dated {next_day.strftime('%Y-%m-%d')}), starting from today's newspaper ({timestamp.strftime('%Y-%m-%d')}). 
Your role involves analyzing current news, predicting future events, and orchestrating the creation of content and design. You will:
Analyze today’s news (using tool) to forecast future events.
Delegate tasks to the appropriate agents—Journalist, Photographer, and HTML Developer—ensuring a smooth workflow.
Review the outputs at each stage to maintain quality and coherence.
After your analysis:
Assign the Journalist to create imaginative and plausible headlines and stories.
Once the stories are ready, pass them to the Photographer to generate realistic images that complement the content.
Finally, direct the HTML Developer to integrate the content and images into a professional, responsive HTML layout.
Your goal: Create a cohesive, compelling edition of "Tomorrow News" that provides a realistic glimpse into the future across various domains like politics, economy, and technology.
The final output must be pure HTML!
        """
    )
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    ai_msg = llm_with_tools.bind_tools([transfer_to_journalist, transfer_to_photographer, transfer_to_html_developer, newstool]).invoke(messages)
    # If there are tool calls, the LLM needs to hand off to another agent
    if len(ai_msg.tool_calls) > 0:
        tool_call_id = ai_msg.tool_calls[-1]["id"]
        tool_name = ai_msg.tool_calls[-1]["name"]
        if tool_name != "get_todays_news_feed":
            # NOTE: it's important to insert a tool message here because LLM providers are expecting
            # all AI messages to be followed by a corresponding tool result message 
            tool_msg = {
                "role": "tool",
                "content": "Successfully transferred",
                "tool_call_id": tool_call_id,
            }
            return Command(goto=tool_name.replace("transfer_to_", ""), update={"messages": [ai_msg, tool_msg]})

    # If the expert has an answer, return it directly to the user
    return {"messages": [ai_msg]}

tool_node = ToolNode(tools=tools)

graph_builder.add_node("editor", editor)

graph_builder.add_node("tools", tool_node)

graph_builder.add_node("journalist", journalist)
graph_builder.add_node("photographer", photographer)
graph_builder.add_node("html_developer", html_developer)

graph_builder.add_conditional_edges(
    "editor",
    tools_condition
)

graph_builder.add_conditional_edges(
    "photographer",
    tools_condition
)

graph_builder.add_edge("journalist", "editor")
#graph_builder.add_edge("photographer", "editor")
graph_builder.add_edge("html_developer", "editor")

def next_condition(state: State)-> Literal["editor", "photographer"]:
  #print(f"state is here: {state["messages"]}")
  if state["messages"][-1].name == "get_todays_news_feed":
    return "editor"
  else:
    return "photographer"

graph_builder.add_conditional_edges("tools", next_condition)

graph_builder.set_entry_point("editor")

# Name the compiled graph so LangSmith traces show the project ("Tomorrow News")
# instead of the default "LangGraph" root run name.
ma_graph = graph_builder.compile(name="Tomorrow News (Supervisor)")

#ma_graph.get_graph().draw_mermaid_png(output_file_path="TomorrowNews/NewMultiAgent.png")