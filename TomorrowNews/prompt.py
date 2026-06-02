import os
from datetime import datetime, timedelta
from TomorrowNews.azurestorage import get_row, insert_history
from TomorrowNews.graph import news_graphs, news_graph
from TomorrowNews.ReAct import supervisor
from TomorrowNews.supervisor import ma_graph
from utils import get_flat_date, get_flat_date_hour, parse_flat_date_hour, strtobool

LANGUAGE_CONFIG = {
    "en": {
        "name": "English",
        "prompt_suffix": "",
    },
    "fa": {
        "name": "Persian",
        "prompt_suffix": (
            "\n\nIMPORTANT: Write the entire newspaper content in Persian (فارسی). "
            "The newspaper title should be 'Tomorrow News - اخبار فردا'. "
            "Use right-to-left (RTL) layout by adding dir=\"rtl\" to the HTML body tag "
            "and using appropriate RTL CSS (text-align: right, direction: rtl). "
            "All headlines, stories, and text must be in fluent, idiomatic Persian."
        ),
    },
    "it": {
        "name": "Italian",
        "prompt_suffix": (
            "\n\nIMPORTANT: Write the entire newspaper content in Italian (Italiano). "
            "The newspaper title should be 'Tomorrow News - Le Notizie di Domani'. "
            "All headlines, stories, and text must be in fluent, idiomatic Italian."
        ),
    },
}

GENERATION_ORDER = ["en", "fa", "it"]


def _get_rowkey_base(parsed_date):
    if parsed_date and parsed_date.date() >= datetime(2025, 1, 25).date():
        return get_flat_date(parsed_date) + "_00"
    elif parsed_date:
        return get_flat_date_hour(parsed_date)
    else:
        return get_flat_date(parsed_date) + "_00"


def _get_rowkey(parsed_date, lang="en"):
    return f"{_get_rowkey_base(parsed_date)}_{lang}"


def _build_system_prompt(timestamp, next_day, lang="en"):
    lang_config = LANGUAGE_CONFIG.get(lang, LANGUAGE_CONFIG["en"])
    base_prompt = f"""Using today's ({timestamp.strftime('%Y-%m-%d')}) actual newspaper as a foundation, \
apply reasoning and analysis to predict future events. \
Create the next day's ({next_day.strftime('%Y-%m-%d')}) edition of 'Tomorrow News,' \
complete with imaginative yet plausible headlines and stories. \
Avoid simply continuing or expanding on today's news—instead, \
focus on predicting the next events and news that could arise as consequences of current happenings or emerge unexpectedly. \
Make it feel like a genuine glimpse into the future of politics, geopolitics, economy, events, Culture, \
Environment, Technology, Health, Security, Education, Science, Energy, Trade, Human Rights, Diplomacy, Military, \
Infrastructure, Agriculture, Transportation, Media, Religion, Demographics, Finance, Law, Tourism, Sports, Migration and what come next!\

Next, design an HTML page for the newspaper. The layout should resemble a professional newspaper, optimized for both desktop and mobile screens. Ensure that the design includes:

A clear newspaper header with the title Tomorrow News.
A two-column layout on the left and right sides of the page, with a wide central column for the main content.
Visually appealing headlines in the central column with a proper headline style, including a large main headline at the top.
Use the content created before and fill all the columns.
A well-balanced font selection with readable sizes, appropriate contrast, and a cohesive color scheme.
A clean, minimalistic border around the entire page for a polished look.
Each story should be long enough to fill the column space like a real newspaper article—meaning substantial content for each story to resemble actual newspaper columns in length.
Images for news stories: Use the image tool to create realistic photos that complement the headlines and add them as appropriate (with image URLs integrated into the HTML).
to create the best photo, think as a newspaper photographer and describe the photo with details and tell the tool explicitly to create a realistic photo.
Be aware that the image tool has content filtering and cannot create any image, try to create images that are not going to filtered, an in case of an error raised by content filtering retry another photo instead of that.
The layout should avoid unnecessary gaps and ensure that content is well-aligned and fits seamlessly into the space.
Prioritize responsive design, so the layout adapts beautifully to both desktop and mobile screens.
After creating the visual design and content, ensure the HTML is well-structured and ready to be rendered correctly by a browser, making it appear as a genuine newspaper page, with functional columns, images, and headings.

Output: Pure HTML code without anything extra! (not even ```html at start and ``` at the end) that includes all of the above elements in a clean, readable, and responsive format. your response will be parsed directly in a browser, so should be rendered correctly as an HTML"""

    return base_prompt + lang_config["prompt_suffix"]


def _try_cache(parsed_date, lang="en"):
    """Check cache for a given date and language. Returns (html_content, timestamp) or None."""
    rowkey = _get_rowkey(parsed_date, lang)
    if cached := get_row(rowkey):
        base_key = rowkey.rsplit("_", 1)[0]
        return cached["html_content"], parse_flat_date_hour(base_key)

    if lang == "en":
        old_key = _get_rowkey_base(parsed_date)
        if cached := get_row(old_key):
            return cached["html_content"], parse_flat_date_hour(old_key)

    return None


def _generate_single(parsed_date, lang="en"):
    """Generate TomorrowNews for a single language."""
    timestamp = datetime.utcnow()
    next_day = timestamp + timedelta(days=1)
    graph = news_graphs.get(lang, news_graphs["en"])
    prompt = _build_system_prompt(timestamp, next_day, lang)

    for event in graph.stream({"messages": [("system", prompt)]}):
        print("event: ", event)
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)
    content = value["messages"][-1].content

    rowkey = _get_rowkey(parsed_date, lang)
    insert_history(rowkey=rowkey, html_content=content, language=lang)
    return content, timestamp


def gettomorrownews(parsed_date, lang="en"):
    if not strtobool(os.environ.get("DEBUG", False)):
        cached = _try_cache(parsed_date, lang)
        if cached:
            return cached

        if lang == "en":
            base = _get_rowkey_base(None)
            fallback_key = f"{base}_{lang}"
            if cached_fb := get_row(fallback_key):
                return cached_fb["html_content"], parse_flat_date_hour(base)
            if cached_fb := get_row(base):
                return cached_fb["html_content"], parse_flat_date_hour(base)

    results = _generate_all(parsed_date)
    return results[lang]


def _generate_all(parsed_date):
    """Generate TomorrowNews for all languages sequentially (en → fa → it)."""
    results = {}
    for lang in GENERATION_ORDER:
        if not strtobool(os.environ.get("DEBUG", False)):
            cached = _try_cache(parsed_date, lang)
            if cached:
                results[lang] = cached
                continue
        results[lang] = _generate_single(parsed_date, lang)
    return results


timestamp = datetime.utcnow()
next_day = timestamp + timedelta(days=1)
system_prompt = f"""
You are the Editor, the central figure in producing the next day's edition of "Tomorrow News" (dated {next_day.strftime('%Y-%m-%d')}), starting from today's newspaper ({timestamp.strftime('%Y-%m-%d')}).
Your role involves analyzing current news, predicting future events, and orchestrating the creation of content and design. You will:
Analyze today's news (using tool) to forecast future events.
Delegate tasks to the appropriate agents—Journalist, Photographer, and HTML Developer—ensuring a smooth workflow.
Review the outputs at each stage to maintain quality and coherence.
After your analysis:
Assign the Journalist to create imaginative and plausible headlines and stories.
Once the stories are ready, pass them to the Photographer to generate realistic images that complement the content.
Finally, direct the HTML Developer to integrate the content and images into a professional, responsive HTML layout.
Your goal: Create a cohesive, compelling edition of "Tomorrow News" that provides a realistic glimpse into the future across various domains like politics, economy, and technology.
The final output must be pure HTML!
"""

def gettomorrownews_multiagent(parsed_date):
    timestamp = datetime.utcnow()
    memory = []
    for event in ma_graph.stream({"messages": [("system", system_prompt)]}, subgraphs=True):
        print("event: ", event)
        memory.append(event)

    _, result = memory[-1]
    r = result['editor']['messages'][-1].content
    return r, timestamp
