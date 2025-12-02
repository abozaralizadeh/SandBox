import os
import re
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, List
from AIBlog.azurestorage import get_row, upsert_history, get_last_n_rows
from AIBlog.graph import *
from utils import get_flat_date, get_flat_date_hour, parse_flat_date_hour, strtobool


_PRIVATE_USE_RE = re.compile(r"[\ue000-\uf8ff]")


def _flatten_message_content(payload: Any) -> str:
    """Recursively extract plain text from LangChain/LangGraph message payloads."""
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        # Azure responses store text blocks under {'type': 'text', 'text': '...'}
        if "text" in payload and payload.get("type") in (None, "text", "output_text"):
            return str(payload["text"])
        # Skip annotation metadata; just flatten actual values.
        pieces = []
        for key, value in payload.items():
            if key == "annotations":
                continue
            pieces.append(_flatten_message_content(value))
        return "".join(pieces)
    if isinstance(payload, Iterable) and not isinstance(payload, (bytes, bytearray)):
        return "".join(_flatten_message_content(item) for item in payload)
    if hasattr(payload, "content"):
        return _flatten_message_content(payload.content)
    if hasattr(payload, "text"):
        return _flatten_message_content(payload.text)
    return str(payload)


def _extract_text_from_last_message(last_message: Any) -> str:
    """Extract text content from the last message, handling different formats."""
    # Fall back to direct attributes if flattening failed.
    try:
        if isinstance(last_message.content, str):
            raw = last_message.content
        elif hasattr(last_message, "text"):
            raw = str(last_message.text)
    except Exception:
        raw = str(last_message)
    return _PRIVATE_USE_RE.sub(" ", raw)


async def getaiblog(parsed_date):
    timestamp = datetime.utcnow()
    flat_date_hour = get_flat_date(parsed_date) + "_00"

    try:
        if not strtobool(os.environ.get("DEBUG", False)):
            if lastdayblog := get_row(flat_date_hour):
                return lastdayblog["html_content"], parse_flat_date_hour(flat_date_hour)
            flat_date_hour = get_flat_date() + "_00"
            if parsed_date is not None and (lastdayblog := get_row(flat_date_hour)):
                return lastdayblog["html_content"], parse_flat_date_hour(flat_date_hour)
    except Exception as e:
        print("Error fetching from storage:", e)
        if isinstance(e, KeyError) and e.args[0] == 'html_content':
            if ldbtimestamp := lastdayblog.metadata["timestamp"]:                 # datetime with timezone
                now = datetime.now(timezone.utc)
                age = now - ldbtimestamp

                if age > timedelta(minutes=30):
                    print("Entity is older than 30 minutes.")
                    pass
                else:
                    print("Entity is fresh.")
                    if not strtobool(os.environ.get("DEBUG", False)):
                        raise e
        else:
            raise e
        
    lastdayblogs = get_last_n_rows(30)
    lastdayblogstitles = [row.get("title", "") for row in lastdayblogs]

    react_agent = await get_react_agent()
    prompt = f"""
    Search and Identify a Narrow Topic:

    - Search for: "AI {timestamp.strftime('%Y-%b-%d')} site:arxiv.org OR site:nature.com OR site:openai.com/blog OR site:deepmind.google/blog OR site:huggingface.co/posts".
    - Exclude topics covered or similar in the last 30 blog posts, here is the list to exclude: {str(lastdayblogstitles)}
    - Review the results, read summaries or abstracts, and select one recent, narrowly-focused advancement in AI or Generative AI (e.g., a new training method, algorithmic improvement, novel application, or architectural innovation such as transformers, attention mechanisms, diffusion models, or multimodal systems).
    - Ensure the topic is recent, technically detailed, and specific enough for an in-depth scientific discussion.

    Deep Research and Data Collection:

    - Conduct a second detailed search specifically about your chosen narrow topic using academic and credible resources, including but not limited to: arxiv.org, nature.com, science.org, deepmind.google/blog, openai.com/blog, ai.googleblog.com, and other reputable sources.
    - Carefully read and extract detailed information, including technical specifics, algorithms, experimental results, figures, and key insights from multiple authoritative sources. to navigating the results use the tools provided by the react agent: 
           (ClickTool,
            NavigateTool,
            NavigateBackTool,
            ExtractTextTool,
            ExtractHyperlinksTool,
            GetElementsTool,
            CurrentWebPageTool)
        Note: arxiv.org (or any other academic website) shows only the abstract of the paper, so you need to click on the "HTML (experimental)" under the "Access Paper:" (or any other link that shows the full paper) to read the full paper.
    - Recursively follow references and citations within these articles to gather deeper insights.
    - Collect the most relevant and significant findings, methods, and implications of the advancement.
    - Search for critical reviews or discussions about the advancement to understand its impact and limitations.
    - Reference the sources of information and avoid plagiarism.

    Content and Structure of the Blog:

    - **Title:** Craft a precise, scientifically appealing title that specifically names or describes the particular AI or GenAI advancement, avoiding generic terms like "2025" or broad phrases. Then use the set_title tool to save the title.
        - Good Example: "Advancing Transformer Efficiency: Sparse Attention Mechanisms for High-Resolution Image Generation"
        - Bad Example: "Latest AI Advancements of 2025"
    - **Banner Image:** Use the image tool to generate a visually appealing banner that matches the topic of the blog. Place this banner immediately after the title for aesthetic purposes only. Do not use the image tool for any other content.
    - **Abstract (Optional but Recommended):** Summarize briefly the focus, significance, method, and findings of the advancement.
    - **Introduction:** Provide the context, briefly explain why this specific advancement is significant in the AI/GenAI community.
    - **Detailed Technical Sections:** Clearly divided into sections with scientific depth (e.g., Methods, Algorithms, Experimental Results, Comparisons to Previous Methods, Limitations, etc.).
    - **Figures, Tables, and Charts:** Present all technical information, data, tables, graphs, and charts using HTML, CSS, and text only. Do not use the image tool for these; instead, use HTML tables, CSS-styled charts, ASCII diagrams, or descriptive text as appropriate.
    - **Code Snippets (if relevant):** Include clearly formatted and contextually explained code examples illustrating critical technical points or methods discussed in the article.
    - **Conclusion:** Discuss implications, potential applications, future directions, and open questions or limitations clearly and insightfully.

    Visual and Technical Formatting:

    - Use professional, readable typography, clear headings (H1-H4), coherent layout, and effective spacing for readability.
    - The banner image must be described precisely for the image tool and placed only after the title.
    - All other images, figures, tables, and charts must be created using HTML, CSS, and text.
    - Use a consistent color scheme, professional style, responsive design suitable for desktop and mobile viewing (similar to Medium-style articles).

    Output Requirements:

    - Provide only the complete HTML code for the fully formatted blog page.
    - Ensure the HTML is ready to render immediately in a browser, with the banner image embedded directly using its URL and without text behind it.
    - No extra markdown or explanations—just the pure HTML. (not even ```html at start and ``` at the end). Your response will be parsed directly in a browser, so it should render correctly as HTML.
    """

    async for event in react_agent.astream(
        {"messages": [("system", prompt)]},
        {"recursion_limit": 1000}
    ):
        print("event: ", event)
        for value in event.values():
            print("React Agent:", value["messages"][-1].content)

    last_message = value["messages"][-1]
    # When the web_search tool is active, responses arrive as structured chunks.
    content = _extract_text_from_last_message(last_message)
    if not strtobool(os.environ.get("DEBUG", False)) or strtobool(os.environ.get("DEBUG_SAVE", False)):
        upsert_history(rowkey=flat_date_hour, html_content=content)
    return content, timestamp
