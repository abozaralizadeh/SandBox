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
            "All headlines, stories, and text must be in fluent, idiomatic Persian. "
            "For typography, import and use the 'Vazirmatn' font from Google Fonts for body text "
            "and 'Noto Naskh Arabic' for headlines to give it an elegant, official newspaper look. "
            "Add this in the HTML <head>: "
            '<link href="https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;700;900&family=Noto+Naskh+Arabic:wght@400;700&display=swap" rel="stylesheet"> '
            "Then set body { font-family: 'Vazirmatn', sans-serif; } and headlines (h1, h2, h3) { font-family: 'Noto Naskh Arabic', serif; }."
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
    base_prompt = f"""You are a journalist writing the {next_day.strftime('%Y-%m-%d')} edition of 'Tomorrow News'. \
First, read today's ({timestamp.strftime('%Y-%m-%d')}) real newspaper using the tool to understand the current situation. \

CRITICAL: Every single story must report a NEW EVENT that has NOT happened yet—a development that occurs AFTER today. \
Do NOT describe, recap, or re-explain today's existing news. That is the single most important rule. \
If today's paper reports "negotiations are ongoing", your story is what the negotiations PRODUCED (a deal signed, a walkout, a new demand). \
If today reports "a storm is approaching", your story is the storm's AFTERMATH (the damage, the response, the death toll). \
If today reports "a company plans a launch", your story is the launch HAPPENING (the result, the reaction, the stock move). \
Every headline must be an EVENT THAT MOVED THINGS FORWARD, not a status update on the present. \

Method: for each major thread in today's news, ask "what is the most likely NEXT thing to happen?" and then write that next thing \
as a concrete, finished event. Take a reasonable hypothetical step forward in the chain of cause and effect. \
Stay grounded—real names, real places, real institutions, plausible outcomes. No science fiction, no sensationalism, no exaggeration. \
But you MUST commit to specific outcomes: who won, what number, which decision, what consequence. \

Write everything as if it has ALREADY HAPPENED—past or present tense, exactly like a real newspaper reporting completed events. \
NEVER use speculative language: no "most likely", "expected to", "probably", "could", "might", "forecast", "analysts predict", "is set to". \
Write with full certainty: "Parliament passed the bill", "The ceasefire collapsed after...", "Markets surged 4% as...". \
A reader landing on this page should believe they are reading real news from the future, full of fresh events—not a summary of today. \

Cover a wide range of domains: politics, geopolitics, economy, Culture, Environment, Technology, Health, Security, \
Education, Science, Energy, Trade, Human Rights, Diplomacy, Military, Infrastructure, Agriculture, Transportation, \
Media, Religion, Demographics, Finance, Law, Tourism, Sports, and Migration.\

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
CRITICAL LAYOUT RULE: All three columns (left, center, right) must be roughly equal in total content height. Do NOT leave any column short or empty at the bottom while another column is long. Distribute stories and images across all columns so they end at approximately the same vertical position. If one column is running short, add more story content or move a story into it. Every column must be filled from top to bottom—no blank space at the bottom of any column.
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
