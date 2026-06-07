# Tomorrow News Speaks Three Languages: How AI Reports the Future in English, Persian, and Italian

*A follow-up to "Tomorrow News: How AI Crafts Future's Headlines and Stories." Last time I built a machine that reads today's paper and writes tomorrow's. This time I taught it new languages — and learned that the hard part was never the languages.*

---

What if tomorrow's newspaper existed in more than one language — each one reading the world the way that language's readers actually do?

In the [first article](https://ai.gopubby.com/tomorrow-news-how-ai-crafts-futures-headlines-and-stories-8f2b37fd841e), I introduced **Tomorrow News**: an experimental project where a single AI agent wears multiple "masks" — Editor, Journalist, Photographer, HTML Developer — reads the day's real headlines, and produces a complete, professionally laid-out newspaper for *the next day*. Imaginative, plausible, and entirely AI-generated. I closed that piece with a promise: that the future of news might not be about reporting what has happened, but what could happen next.

This is that "next." And it turned into three separate challenges: making Tomorrow News **multilingual**, making it **less predictable in the boring sense and more predictive in the real sense**, and making it **harder to misuse**. The languages were the easy part. Convincing a language model to genuinely *predict* instead of politely *paraphrase* was the boss fight.

Let's start with the fun part.

---

## Three Newsrooms, Not One Translation

The lazy way to "add languages" is to generate the English paper and run it through a translator. I refused. A translated BBC front page isn't an *Italian* newspaper — it's an English newspaper in a costume. Italians and Iranians don't encounter the same events in the same order with the same weight. The *worldview* is part of the content.

So instead of one newsroom and a translator, Tomorrow News now runs **three independent newsrooms**, each reading its own national paper:

| Language | Source feed |
|----------|-------------|
| English  | BBC News |
| Persian  | BBC Persian (`feeds.bbci.co.uk/persian/rss.xml`) |
| Italian  | ANSA (`ansa.it/sito/ansait_rss.xml`) |

The same cast of masks — Editor, Journalist, Photographer, HTML Developer — but each reading a different world and writing in a different voice. The first refactor turned the original hard-coded news tool into a **factory** that binds a fresh tool to each feed:

```python
RSS_URLS = {
    "en": "https://feeds.bbci.co.uk/news/rss.xml",
    "fa": "https://feeds.bbci.co.uk/persian/rss.xml",
    "it": "https://www.ansa.it/sito/ansait_rss.xml",
}

def create_news_feed_tool(rss_url):
    @tool
    def get_todays_news_feed() -> list[dict]:
        """get todays news as a list of dicts containing title and description"""
        data = getRSS(rss_url)
        items = data.get("rss", {}).get("channel", {}).get("item", [])
        if isinstance(items, dict):      # a thin news day can yield ONE <item>, not a list
            items = [items]
        return [{"title": i.get("title", ""), "description": i.get("description", "")}
                for i in items]
    return get_todays_news_feed
```

That `isinstance(items, dict)` guard cost me a real debugging session. The original single-source code did `for item in data['rss']['channel']['item']`, which works beautifully — until a feed returns a *single* `<item>` that `xmltodict` parses as a dict instead of a list, and your loop quietly starts iterating over dictionary keys. **The moment you add a second data source, defensive parsing stops being optional.** Feeds disagree about everything, including whether "a list of one" is a list.

Each language then gets its **own compiled LangGraph**, and on a cache miss the editions are produced **one after another** — English, then Persian, then Italian:

```python
news_graphs = {
    lang: create_news_graph(create_news_feed_tool(url))
    for lang, url in RSS_URLS.items()
}

GENERATION_ORDER = ["en", "fa", "it"]
```

It's more compute than translation. It's also the honest version of the idea: three newsrooms, three worldviews, three papers.

---

## A New Column — and an Untouched Archive

Storage needed a new dimension. The original schema in **Azure Table Storage** keyed each edition purely by date (`YYYYMMDD_00`). Now every row carries a **`language`** column, and the row key is namespaced by language:

```
20260607_00_en
20260607_00_fa
20260607_00_it
```

The catch: there's a back-catalog of English editions saved under the **old, un-suffixed keys**, and I had no intention of migrating or orphaning them. So the cache lookup tries the new language-suffixed key first, and for English **falls back** to the legacy key:

```python
def _try_cache(parsed_date, lang="en"):
    if cached := get_row(_get_rowkey(parsed_date, lang)):     # new: ..._en / _fa / _it
        ...
    if lang == "en":
        if cached := get_row(_get_rowkey_base(parsed_date)):  # legacy: no suffix
            ...
```

Every old paper still loads; every new paper is multilingual. A schema change nobody has to notice — which is how a schema change should feel.

---

## Making Persian Look Persian

Here's a distinction I underestimated: *supporting* a language and *rendering it well* are two different jobs.

Persian is **right-to-left**, and a left-aligned RTL newspaper looks instantly, viscerally broken. So the Persian edition gets `dir="rtl"` and proper RTL CSS. But the bigger upgrade was **typography**. The browser's default Arabic-script fallback looks like an error dialog, not a broadsheet. So the Persian newsroom is instructed to embed two Google Fonts:

- **Vazirmatn** for body text — clean, modern, readable.
- **Noto Naskh Arabic** for headlines — calligraphic, weighty, *official*.

Each language carries its own masthead too — *"اخبار فردا"* for Persian, *"Le Notizie di Domani"* for Italian. Small touches, but they're the line between "a webpage that happens to be in Persian" and "a Persian newspaper."

---

## The Real Challenge: Predicting vs. Paraphrasing

Now the boss fight — and the one that taught me the most.

I'd open a freshly generated edition and feel… nothing. *This is just today's news, reworded.* "Negotiations continue." "The storm approaches." "Markets remain volatile." Technically future-tense, emotionally a recap. It read like the **present** wearing tomorrow's date.

So I did the obvious thing: I told the model to be **bold, surprising, creative.** And I got precisely what anyone who's overshot that dial gets — **aliens, rogue robots, miracle cures.** Every edition curdled into a sci-fi tabloid. Pure exaggeration, the opposite failure.

The fix wasn't a setting. It was reframing the agent's **job**, in two parts.

**1. Report the next *event*, not the current *situation*.** I gave the Journalist mask an explicit transformation rule, with worked examples baked into the prompt:

> If today's paper says "negotiations are ongoing," your story is what the negotiations *produced* — a deal signed, a walkout, a new demand. If today says "a storm is approaching," your story is the *aftermath*. If today says "a company plans a launch," your story is the launch *happening* — the result, the reaction, the stock move.

Every headline has to be an event that **moved things forward**, generated by asking *"what is the single most likely next thing to happen?"* and then committing to a concrete outcome — who won, what number, which decision — while staying anchored to real names, real places, real institutions. No fantasy. But no fence-sitting either.

**2. You're a journalist, not an analyst.** This correction came from a wonderfully specific bug. The Persian edition printed a headline that translated to:

> *"The most likely news for tomorrow: not a final deal, nor a total collapse, but a slow return to quiet diplomacy…"*

The model was **leaking its own reasoning onto the front page.** It hedged — "most likely," "could," "is expected to" — because I'd framed it as someone *making predictions*. But newspapers don't hedge. So I banned speculative language outright and rewrote the role:

> Write everything as if it has ALREADY HAPPENED — past or present tense, like a real newspaper reporting completed events. NEVER use "most likely," "expected to," "probably," "could," "might," "forecast," "analysts predict." Write with full certainty: "Parliament passed the bill," "The ceasefire collapsed after…"

That single shift — from *"predict what will happen"* to *"report what already happened, in a future you invented"* — was the most effective change in the entire project. Same underlying task, but the model's voice went from a cautious forecaster to a confident reporter.

And **temperature**? I wish I had a principled formula. I don't. I bounced from 1.0 (dull) to 1.3 (occasionally unhinged) to 1.1 (safe but flat) and back to **1.3** — but only *after* the prompt was strong enough to keep that energy on the rails. Temperature is the gas pedal; the prompt is the steering wheel. Floor the gas before you can steer and you get aliens.

---

## Challenges and Innovations

A few smaller battles worth logging:

**Ragged columns.** The signature three-column layout sometimes left one column empty while another ran long — the visual equivalent of a half-finished crossword. LLMs have no innate sense of *vertical balance*, so the HTML Developer mask now gets an explicit rule: all three columns must end at roughly the same height, and if one runs short, redistribute or add content. No blank gaps at the bottom.

**No predicting from a void.** You could originally navigate to a *future* date — at which point the backend would dutifully try to generate news from a newspaper that doesn't exist yet. That's nonsense: Tomorrow News predicts tomorrow *from today*. Future dates are now disabled in the calendar and clamped on load. Read any past edition you like; you simply can't ask the machine to predict the day after a day that hasn't happened.

**Shareable language state.** The selected language now lives in the URL as a query param (`?lang=fa`), right alongside the date. Send someone a link and they get the exact edition you saw — right language, right day. For the legacy archive, which only ever existed in English, the dropdown is disabled and pinned to English.

---

## What I'd Tell Past Me

- **Independent generation beats translation** whenever *worldview* is part of the content. Three feeds, three graphs, three newsrooms — pricier, far more authentic.
- **Defensive parsing is mandatory the instant you add a second data source.** Sources disagree about everything.
- **The model's *role* is a stronger lever than its instructions.** "Be a journalist reporting the past" outperformed paragraphs of "please be creative but realistic." Reframe before you over-specify.
- **Temperature is the gas, the prompt is the wheel.** Fix the prompt first.
- **Generative prediction lives on a knife-edge** between boring (paraphrase) and absurd (sci-fi). The whole craft is staying in the narrow band where it's *surprising but plausible.*

---

## Conclusion: Tomorrow, in Three Voices

When I wrote the first article, I said the future of news might be less about reporting what has happened and more about imagining what could happen next. A year on, that future also speaks Persian and Italian — and, just as importantly, it has learned to speak with *conviction* instead of hedging, and with *restraint* instead of inventing flying saucers.

Tomorrow News remains, proudly, a fiction machine. Every edition says so. But it's now a fiction machine that reasons in three languages, reports with a straight face, and knows it can't predict a future it has no present for.

Go read tomorrow's paper. Just don't bet on it.

*Explore the live project: **[Tomorrow News](https://github.com/abozaralizadeh/SandBox)** · Open-source code on **[GitHub](https://github.com/abozaralizadeh/SandBox)**. As always, every word of Tomorrow News is AI-generated and speculative — fiction for research and entertainment, not a forecast of real events.*
