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


@tool
def handoff_to_storyteller():
    """Pass the baton to the Storyteller to write today's script and panel list."""
    return


@tool
def handoff_to_cartoonist():
    """Pass the baton to the Cartoonist to turn panels into illustrated prompts."""
    return


@tool
def handoff_to_layout():
    """Send everything to the Layout Artist to produce the final HTML page."""
    return


def _director_prompt(state: ComicState) -> str:
    return f"""
Role: Director of the ComicBook daily series.
Arc: {state['arc_title']}
Logline: {state['arc_logline']}
Day {state['arc_day']} of {state['arc_target']}.
Recent beats and callbacks:
{state['previous_beats']}

Responsibilities:
- Plot today's episode so it advances the arc while feeling self-contained.
- Lock tone (whimsical, noir, cosmic, etc.) and visual motifs for this day.
- Outline 4-6 panels with short beats and stakes.
- Hand off to the Storyteller using the transfer tool when beats feel ready.
"""


def director(state: ComicState) -> Command[Literal["storyteller"]]:
    messages = [{"role": "system", "content": _director_prompt(state)}] + state["messages"]
    ai_msg = llm.bind_tools([handoff_to_storyteller]).invoke(messages)

    if ai_msg.tool_calls:
        tool_call_id = ai_msg.tool_calls[-1]["id"]
        tool_msg = {
            "role": "tool",
            "content": "Direction locked. Moving to Storyteller.",
            "tool_call_id": tool_call_id,
        }
        return Command(goto="storyteller", update={"messages": [ai_msg, tool_msg]})

    # Default to progress even if handoff tool not called.
    return Command(goto="storyteller", update={"messages": [ai_msg]})


def _storyteller_prompt(state: ComicState) -> str:
    return f"""
Role: Storyteller.
You write the script for today's comic inside the ongoing arc "{state['arc_title']}" (day {state['arc_day']} of {state['arc_target']}).
Honor continuity while making this strip satisfying on its own.
Produce:
- A concise recap (2-3 lines) that today's strip will implicitly cover.
- Panel-by-panel script with dialogue, captions, and sound effects.
- Explicit visual anchors (setting, lighting, mood) that the Cartoonist can illustrate.
When the script is ready, call the transfer tool to hand off to the Cartoonist.
"""


def storyteller(state: ComicState) -> Command[Literal["cartoonist"]]:
    messages = [{"role": "system", "content": _storyteller_prompt(state)}] + state["messages"]
    ai_msg = llm.bind_tools([handoff_to_cartoonist]).invoke(messages)

    if ai_msg.tool_calls:
        tool_call_id = ai_msg.tool_calls[-1]["id"]
        tool_msg = {
            "role": "tool",
            "content": "Script drafted. Passing to Cartoonist.",
            "tool_call_id": tool_call_id,
        }
        return Command(goto="cartoonist", update={"messages": [ai_msg, tool_msg]})

    # Continue even without explicit handoff.
    return Command(goto="cartoonist", update={"messages": [ai_msg]})


def _cartoonist_prompt(state: ComicState) -> str:
    return f"""
Role: Cartoonist and storyboard artist.
Translate the script into image prompts for each panel of "{state['arc_title']}" (day {state['arc_day']}/{state['arc_target']}).
Guidelines:
- Keep characters and visual style consistent with the last published image: {state['last_image_url'] or 'no prior image'}.
- Always call get_image_by_text_with_reference when a reference image is available; otherwise call get_image_by_text.
- Do NOT fabricate image URLs—use the tool outputs only (blob URLs).
- Return a short list of panels with prompt + resulting image URL when tool runs.
- When panels are ready, use the transfer tool to send everything to the Layout Artist.
"""


cartoonist_llm = llm.bind_tools(
    [
        handoff_to_layout,
        get_image_by_text,
        get_image_by_text_with_reference,
    ]
)


def cartoonist(state: ComicState) -> Command[Literal["layout"]]:
    messages = [{"role": "system", "content": _cartoonist_prompt(state)}] + state["messages"]
    ai_msg = cartoonist_llm.invoke(messages)

    if ai_msg.tool_calls:
        tool_call = ai_msg.tool_calls[-1]
        tool_name = tool_call["name"]
        tool_call_id = tool_call["id"]
        if tool_name == "handoff_to_layout":
            tool_msg = {
                "role": "tool",
                "content": "Panels prepared. Moving to layout.",
                "tool_call_id": tool_call_id,
            }
            return Command(goto="layout", update={"messages": [ai_msg, tool_msg]})
        # Let the ToolNode execute pending image generations before layout.
        return {"messages": [ai_msg]}

    # If there are no tool calls and no handoff, still advance to layout with current outputs.
    return Command(goto="layout", update={"messages": [ai_msg]})


def _layout_prompt(state: ComicState) -> str:
    return f"""
Role: Layout Artist and final renderer.
Assemble today's ComicBook strip (day {state['arc_day']} of {state['arc_target']}) into a responsive HTML page.
Expectations:
- Keep panels readable on both mobile and desktop (stacked on small screens).
- Use one color paper style and real comicbook styling.
- Include the arc title, today's date, and a one-line recap.
- Close with a teaser for the next day without spoiling twists.
Return pure HTML only (no markdown fences).
"""


def layout(state: ComicState):
    messages = [{"role": "system", "content": _layout_prompt(state)}] + state["messages"]
    ai_msg = llm.invoke(messages)
    return {"messages": [ai_msg]}


graph_builder = StateGraph(ComicState)

tool_node = ToolNode(tools=[get_image_by_text, get_image_by_text_with_reference, handoff_to_layout])

graph_builder.add_node("director", director)
graph_builder.add_node("storyteller", storyteller)
graph_builder.add_node("cartoonist", cartoonist)
graph_builder.add_node("layout", layout)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge("storyteller", "cartoonist")
graph_builder.add_edge("tools", "cartoonist")

graph_builder.add_conditional_edges("cartoonist", tools_condition)

graph_builder.set_entry_point("director")
graph_builder.set_finish_point("layout")

comic_graph = graph_builder.compile()
#comic_graph.get_graph().draw_mermaid_png(output_file_path="comic_graph.png")
