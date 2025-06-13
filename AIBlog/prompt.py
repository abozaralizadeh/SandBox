import os
import asyncio
from datetime import datetime, timedelta
from AIBlog.azurestorage import get_row, upsert_history, get_last_n_rows
from AIBlog.graph import *
from utils import get_flat_date, get_flat_date_hour, parse_flat_date_hour, strtobool


async def getaiblog(parsed_date):
    timestamp = datetime.utcnow()
    flat_date_hour = get_flat_date(parsed_date) + "_00"

    if not strtobool(os.environ.get("DEBUG", False)):
        if lastdayblog := get_row(flat_date_hour):
            return lastdayblog["html_content"], parse_flat_date_hour(flat_date_hour)
        flat_date_hour = get_flat_date() + "_00"
        if parsed_date is not None and (lastdayblog := get_row(flat_date_hour)):
            return lastdayblog["html_content"], parse_flat_date_hour(flat_date_hour)
        
    lastdayblogs = get_last_n_rows(30)
    lastdayblogstitles = [row.get("title", "") for row in lastdayblogs]

    react_agent = await get_react_agent()
    prompt = f"""
    Search and Identify a Narrow Topic:

    - Search for: "AI {timestamp.strftime('%Y-%b-%d')} site:arxiv.org OR site:nature.com OR site:openai.com/blog OR site:deepmind.google/blog OR site:huggingface.co/posts".
    - Exclude topics covered in the last 30 blog posts: {str(lastdayblogstitles)}
    - Review the results, read summaries or abstracts, and select one recent, narrowly-focused advancement in AI or Generative AI (e.g., a new training method, algorithmic improvement, novel application, or architectural innovation such as transformers, attention mechanisms, diffusion models, or multimodal systems).
    - Ensure the topic is recent, technically detailed, and specific enough for an in-depth scientific discussion.

    Deep Research and Data Collection:

    - Conduct a second detailed search specifically about your chosen narrow topic using academic and credible resources, including but not limited to: arxiv.org, nature.com, science.org, deepmind.google/blog, openai.com/blog, ai.googleblog.com, and other reputable sources.
    - Carefully read and extract detailed information, including technical specifics, algorithms, experimental results, figures, and key insights from multiple authoritative sources.
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
    content = value["messages"][-1].content
    upsert_history(rowkey=flat_date_hour, html_content=content)
    return content, timestamp
