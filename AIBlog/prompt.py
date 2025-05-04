import os
import asyncio
from datetime import datetime, timedelta
from AIBlog.azurestorage import get_row, insert_history
from AIBlog.graph import *
from utils import get_flat_date, get_flat_date_hour, parse_flat_date_hour


async def getaiblog(parsed_date):
    timestamp = datetime.utcnow()
    flat_date_hour = get_flat_date(parsed_date) + "_00"

    if not os.environ.get("DEBUG", False):
        if lastdayblog := get_row(flat_date_hour):
            return lastdayblog["html_content"], parse_flat_date_hour(flat_date_hour)
        flat_date_hour = get_flat_date() + "_00"
        if parsed_date is not None and (lastdayblog := get_row(flat_date_hour)):
            return lastdayblog["html_content"], parse_flat_date_hour(flat_date_hour)

    react_agent = await get_react_agent()
    prompt = f"""
    Search and Identify a Narrow Topic:

    - Perform a search for: "AI {timestamp.strftime('%Y-%b-%d')} site:arxiv.org OR site:nature.com OR site:openai.com/blog OR site:deepmind.google/blog".
    - Navigate to the results, carefully read summaries or abstracts, and identify one specific, narrowly-focused advancement in AI or Generative AI technology (e.g., a new training method, specific algorithm improvement, a novel application in a particular field, new architecture improvements like transformers, attention mechanisms, diffusion models, multimodal advancements, etc.).
    - Ensure the topic is recent, detailed, technically rich, and sufficiently narrow to allow an in-depth scientific discussion.

    Deep Research and Data Collection:

    - Conduct a second detailed search specifically about your chosen narrow topic using academic and credible resources, including but not limited to: arxiv.org, nature.com, science.org, deepmind.google/blog, openai.com/blog, ai.googleblog.com, and other reputable sources.
    - Carefully read and extract detailed information, including technical specifics, algorithms, experimental results, figures, and key insights from multiple authoritative sources.
    - Recursively follow references and citations within these articles to gather deeper insights.

    Content and Structure of the Blog:

    - **Title:** Craft a precise, scientifically appealing title that specifically names or describes the particular AI or GenAI advancement, avoiding generic terms like "2025" or broad phrases.
        - Good Example: "Advancing Transformer Efficiency: Sparse Attention Mechanisms for High-Resolution Image Generation"
        - Bad Example: "Latest AI Advancements of 2025"
    - **Abstract (Optional but Recommended):** Summarize briefly the focus, significance, method, and findings of the advancement.
    - **Introduction:** Provide the context, briefly explain why this specific advancement is significant in the AI/GenAI community.
    - **Detailed Technical Sections:** Clearly divided into sections with scientific depth (e.g., Methods, Algorithms, Experimental Results, Comparisons to Previous Methods, Limitations, etc.).
    - **Figures and Images:** Include or describe clear figures, tables, or images illustrating key concepts, methods, architectures, results, or comparisons. Clearly define descriptions of these images to ensure they are well-suited for image generation tools.
    - **Code Snippets (if relevant):** Include clearly formatted and contextually explained code examples illustrating critical technical points or methods discussed in the article.
    - **Conclusion:** Discuss implications, potential applications, future directions, and open questions or limitations clearly and insightfully.

    Visual and Technical Formatting:

    - Use professional, readable typography, clear headings (H1-H4), coherent layout, and effective spacing for readability.
    - Images must be described precisely to ensure accurate generation or embedding.
    - Use a consistent color scheme, professional style, responsive design suitable for desktop and mobile viewing (similar to Medium-style articles).

    Output Requirements:

    - Provide only the complete HTML code for the fully formatted blog page.
    - Ensure the HTML is ready to render immediately in a browser, with all images embedded directly using their URLs.
    - No extra markdown or explanations—just the HTML.
    """

    async for event in react_agent.astream(
        {"messages": [("system", prompt)]},
        {"recursion_limit": 1000}
    ):
        print("event: ", event)
        for value in event.values():
            print("React Agent:", value["messages"][-1].content)
    content = value["messages"][-1].content
    insert_history(rowkey=flat_date_hour, html_content=content)
    return content, timestamp
