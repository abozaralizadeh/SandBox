# The Comic That Draws Itself: Building a Daily AI Graphic-Novel Studio — in Three Languages

*A team of AI agents writes, draws, letters, and publishes a brand-new comic episode every single day — with storylines that continue for weeks, a house art style, and characters who actually stay themselves from panel to panel. Here's how the studio works, why visual consistency nearly broke me, and what happened when I let a language model design its own speech bubbles.*

---

What if a comic published a new episode every morning — not a three-panel strip, a full multi-panel page — drawn in a consistent style, starring characters who remember what they look like, inside a story that genuinely *continues*? And what if it told that same day's episode three times over, written natively in English, Italian, and Persian — not translated, but **retold**?

That's **ComicBook**: a daily, self-running graphic-novel machine. Every day is one **episode**. Episodes chain into **arcs** — multi-week serialized stories with a planned beginning, middle, and end. When an arc wraps, the machine invents a brand-new one: new genre, new cast, new art style, new color palette. It's an anthology that runs continuous serials, forever, with nobody at the desk.

The fun pitch is "a comic that draws itself." The honest pitch is harder: a comic that draws itself is mostly a fight against three things — *the same face twice*, *a story with a spine*, and *a language model that would rather paraphrase than perform.* Let's get into all of it.

---

## The Studio: a Crew, Not a Prompt

The naive version of this project is one giant prompt to an image model: *"draw a six-panel comic about…"*. That produces a single muddy image with garbled text and characters who mutate between panels. Useless.

So ComicBook isn't a prompt. It's a small **studio of specialist agents**, each a real `Agent` in the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python), each with its own tools, temperature, and job description, run one after another by the SDK's `Runner`:

| Mask | Job | Tools | Heat |
|------|-----|-------|------|
| **Director** | Owns the arc lifecycle: invents arcs, writes the story outline, plans today's episode (panel count, sizes, beats, cliffhanger) | WebSearch, arc status, start/end arc, save outline | 1.2 (hot — it's the imagination) |
| **Storyteller** | Turns the Director's plan into a panel-by-panel script: setting, characters, poses, dialogue, captions, SFX, camera angle | — | default |
| **Cartoonist** | Draws every panel and assembles the page | character sheet, panel image, key-panel marker, layout | default |
| **Reteller** | Retells the finished episode *natively* in Italian & Persian | — | 0.9 |
| **OutlineAdapter** | Adapts the arc outline into each language once per arc | — | 0.7 |

```python
director = Agent(
    name="Director",
    instructions=DIRECTOR_INSTRUCTIONS,
    tools=[WebSearchTool(search_context_size="high"),
           get_arc_status, start_new_arc, end_current_arc, save_story_outline],
    model_settings=ModelSettings(temperature=1.2),   # runs hot on purpose
)

cartoonist = Agent(
    name="Cartoonist",
    instructions=CARTOONIST_INSTRUCTIONS,
    tools=[generate_character_sheet, generate_panel_image, mark_key_panel, assemble_layout],
)
```

The pipeline is deliberately **sequential** — Director → Storyteller → Cartoonist → Reteller — and each stage is wrapped in a LangSmith `trace()` so I can watch the whole studio think. It's more compute than a one-shot prompt. It's also the only version that actually works, because each specialist hands the next one something structured instead of vibes.

---

## The Hardest Problem: Drawing the Same Face Twice

Here is the dirty secret of AI comics, and the thing that ate most of my time: **image models are stateless and stochastic.** Ask for "Juniper, a girl with copper hair and a storm-diver's green coat" twice and you get two different girls. Now ask for her *six times on one page*, then *again tomorrow*, and *again for three weeks*. A comic whose protagonist's face reshuffles every panel isn't a comic. It's a ransom note.

There is no temperature knob for "be the same person." Consistency had to be *engineered*, in three layers.

**Layer 1 — the character sheet (the visual bible).** Once per arc, before any panel is drawn, the Cartoonist generates a single high-quality reference image containing *every* character in the arc — including ones who don't appear until episode 8 — full-body, faces visible, each labeled, in the arc's chosen art style. It's cached on the arc and reused for the entire run.

```python
prompt = (
    f"Character and environment reference sheet, {style} art style. "
    f"Show ALL characters listed below clearly with full body, distinct visual "
    f"features, face clearly visible, each labeled by name. "
    f"No text overlays, no speech bubbles. Characters: {arc_chars}"
)
url = await create_image(prompt, size="wide", quality="high")   # generated ONCE, then cached
```

**Layer 2 — sequential generation with reference chaining.** Panels are *never* drawn in parallel. They're drawn one at a time, and every panel is conditioned on a **stack of reference images** in strict priority order: the character sheet first (always), then any "key panels" of characters introduced mid-arc, then the last couple of panels from *this* episode, then an anchor or two from *yesterday's* episode. `gpt-image-1` accepts up to 16 reference images in a single edit call, so each new drawing literally looks at what came before:

```python
_add(reference_url)                           # the character sheet — always first
for u in state["key_panel_urls"][-3:]:        # characters introduced later in the arc
    _add(u)
for u in state["generated_panel_urls"][-2:]:  # the last panels from THIS episode
    _add(u)
for u in state["prev_episode_images"][-2:]:   # an anchor from yesterday

if len(image_urls) > 1:
    url = await create_image_with_references(prompt, image_urls, size)   # up to 16 refs
```

**Layer 3 — key panels for newcomers.** When a character walks on who wasn't on the original sheet, the Cartoonist calls `mark_key_panel`, which permanently pins that panel as a reference for every future drawing in the arc. The cast can grow without losing its memory.

The result is a comic where Juniper is *recognizably Juniper* across six panels and twenty-one days. That, far more than any single pretty image, is the engineering.

---

## Words Are Not Drawn

Closely related, and one of the best decisions in the project: **the image model never draws a single letter.** Every panel prompt gets this suffix bolted on:

```python
no_text_suffix = ". No text, no speech bubbles, no captions, no letters, no words, no writing."
if "no text" not in prompt.lower():
    prompt = prompt.rstrip(".") + no_text_suffix
```

Why fight the model to spell when it can't? The panels come back as **pure art**. The words — dialogue, captions, sound effects — are layered on afterward as positioned HTML/CSS: a caption box at the top, speech bubbles stacked at the bottom, comic-style SFX. The bubbles even zoom on hover.

This one architectural choice pays for itself three times:

1. **Crisp, real typography** instead of the melted pseudo-letters image models produce.
2. **Editable text** — fix a typo without re-rolling a whole generated panel.
3. **Multilingual for free** — the same drawing can carry English, Italian, *and* Persian text, because the words live in a separate layer. Hold that thought; it's the whole back half of this post.

Panels also carry a **size** — `wide`, `tall`, or `square` — and a little layout engine composes them into a real comic grid (wide panels span the row; tall panels claim two rows; squares pair up). The Storyteller varies sizes for *pacing* — a wide establishing shot slows time down, a run of squares speeds it up — which is a thing human comic artists do and which the model learned to do on request.

---

## The Arc System: a Story With a Spine (and a Director Who Keeps Trying to End It)

A daily comic that resets every morning is a gag strip. I wanted *serialization* — stories that build for weeks. So episodes live inside **arcs**. The Director, on arc creation, commits up front to a title, a logline, a genre, a full cast (with visual descriptions), an art style, a color theme, a planned episode count, and a **story outline** that every later episode must honor. Each day advances the arc by one episode. When the planned finale ships, the arc closes and a fresh one is born.

Which surfaced my favorite bug in the project.

The Director runs *hot* — temperature 1.2, because arc invention needs imagination. But a hot model is a dramatic one, and a dramatic model loves a finale. Mine kept reaching for `end_current_arc` two or three episodes *early* — guillotining a planned twelve-episode saga at episode nine because it felt a satisfying ending in the air. Stories died mid-sentence.

The fix is a **three-layer guard**, and the lesson is the layering:

1. **In the prompt** — explicit CASE A / CASE B logic: *if today's episode number ≤ planned episodes, you are mid-arc; you may NOT close.*
2. **In the input** — I pre-compute the booleans (`is_today_the_finale`, `is_arc_complete_before_today`) and hand them to the model, so it never has to do the arithmetic that it kept getting wrong.
3. **In the tool itself** — the backstop. Even if the first two fail, `end_current_arc` simply *refuses*:

```python
if planned_eps > 0 and episodes_done < planned_eps:
    return {
        "error": "arc_not_complete",
        "message": (
            f"Cannot close arc yet: only {episodes_done} of {planned_eps} planned "
            f"episodes have been published. Today's episode (#{state['episode_number']}) "
            f"belongs to THIS arc. Plan it now and do NOT close the arc."
        ),
    }
```

The takeaway has outlived this bug: **don't ask a creative model to enforce an invariant in prose.** If something must be true, make the *tool* enforce it. Prompts persuade; tools guarantee.

---

## The Boss Fight: Retelling, Not Translating

Now the part that taught me the most — and, fittingly, the part about language.

Adding Italian and Persian *looked* easy, because the text already lived in its own layer. The lazy path: take the English script and translate it. I built that first. It was wrong, and it was wrong in an instructive way.

The problem wasn't word choice. It was **structure**. The translator mapped the English fragment-for-fragment — bubble 1 → bubble 1, caption 1 → caption 1 — which quietly forced every language into *English's* text architecture: the same number of speech bubbles, the same caption breaks, the same rhythm. Persian and Italian came out as English wearing a costume. Technically translated; emotionally *decoded English.*

And sometimes it was just plain wrong on tone. The single example that reframed the whole problem — a line where a character snaps:

> **"Don't you dare tell me that was just wind."**

The literal Persian rendering came back as `جرئت نکن بگی فقط صدای باد بود` — which reads like a parent scolding a toddler. But "don't you dare" isn't a prohibition; it's a *dare*. A native speaker throws it back as a challenge: `اگه جرئت داری بگو فقط صدای باد بود` — *"if you dare, say it was just the wind."* Word-for-word is a failure even when it's "correct."

**Fix, part one: fire the Translator, hire a Reteller.** The new agent isn't given fragments to convert. It's given the finished art's description, the adapted outline, the cast, and the English script *as intent reference only* — and told to **retell** the episode as if it had been written in that language from the first draft. Full freedom to split one line into a back-and-forth, merge two into one punchy beat, drop a line the picture already makes obvious, re-voice everything — under exactly two hard rules: stay true to what each panel literally *shows*, and never lose a plot beat. English remains the master that drives the *drawings*; Italian and Persian are independent retellings of the same pictures. (A per-arc, per-language **glossary** rides along so names and places stay consistent across episodes — the Reteller emits an `updated_glossary` every run that's fed back next time.)

**Fix, part two — the overcorrection, and the real lesson.** If the Reteller can rewrite freely, I reasoned, why not let it *letter* freely too? Native text is a different length, and Persian reads right-to-left, so surely the boxes should move. I gave the agent a 9-zone placement grid and let it art-direct every speech bubble per panel. It sounded principled.

It looked *terrible.* Bubbles parked over faces. Captions stranded in dead corners. Of course it did — **the agent can't see the pixels.** It was positioning boxes from a text description of a picture it had never looked at. So I tore it out and went back to fixed positions — caption on top, bubbles on the bottom, automatically mirrored for right-to-left — and kept *only* the retelling of the words.

That ruined-then-reverted detour is the lesson I'd tattoo on past-me:

> **Give the model freedom over what it's good at, and hard structure for what it can't see.** Words: free. Layout: fixed.

It rhymes with a thing I learned on a sibling project, [Tomorrow News](https://ai.gopubby.com/tomorrow-news-how-ai-crafts-futures-headlines-and-stories-8f2b37fd841e): *temperature is the gas, the prompt is the wheel.* Same shape of mistake — flooring creative freedom before the structure exists to steer it.

---

## Making Persian Look Persian

Supporting a language and *rendering it beautifully* are two different jobs, and the gap is widest for Persian.

Persian is **right-to-left**, and an RTL story laid out left-to-right looks instantly, viscerally broken. So the Persian edition gets `dir="rtl"`, the speech bubbles flip their growth direction, and the whole page mirrors:

```python
rtl_css = (
    ".comic-page[dir=rtl] { direction: rtl; font-family: 'Vazirmatn', sans-serif; }"
    ".comic-page[dir=rtl] .speech-bubble { transform-origin: bottom right; }"
    ".comic-page[dir=rtl] .panel-overlay { align-items: flex-end; }"
) if is_rtl else ""
```

Typography matters as much as direction. English and Italian get **Bangers** — the punchy, all-caps font that *is* comic-book lettering. Persian gets **Vazirmatn**, a clean modern Persian typeface, because the browser's default Arabic-script fallback looks like a system error, not a broadsheet. Even the furniture is localized: the page chrome says *Episode* in English, *Episodio* in Italian, and *قسمت* in Persian. Small touches — but they're the line between "a webpage that happens to be in Persian" and "a Persian comic."

---

## Challenges and Innovations

A few smaller battles worth logging:

**Color identity per arc.** Every arc carries a `color_theme` — page background, caption colors, speech borders, SFX color, fonts — that the Director picks at creation and the whole arc inherits. A noir detective arc and a sunny folktale arc don't just have different art; they have different *skins*.

**The character sheet is the cheapest insurance you'll ever buy.** One high-quality reference image per arc, generated before anything else, prevents a hundred downstream "why does she look different now" problems. Spend the tokens early.

**Storage that hides its own complexity.** Arcs and episodes live in Azure Table Storage; images, character sheets, and any HTML too big for a table row spill over to Blob Storage automatically. Each episode stores three HTML columns — `html_content`, `html_content_it`, `html_content_fa` — so a language switch is a column read, not a regeneration.

**Failures stay contained.** Each language's retelling is wrapped so that if Italian falls over, English and Persian still save. The expensive part — the *art* — is generated once and shared; only the cheap text layer varies, so a hiccup in one language never costs you a redraw.

**A lock, so you don't draw the same day twice.** Episodes are cached by date and language, and a generation lock keeps two simultaneous visitors from both kicking off the same costly pipeline.

---

## What I'd Tell Past Me

- **Consistency is the entire game in AI comics.** A character sheet plus sequential reference-chaining beats any single clever prompt. Engineer the memory; don't pray for it.
- **Never draw text with an image model.** Letter in HTML. You get crisp type, editable words, and multilingual support for free.
- **Enforce invariants in the tool, not the prompt.** A hot, creative model *will* break a rule you only wrote in prose. The function is the only thing that can actually say no.
- **Retell, don't translate, when voice and worldview are the content.** Fragment-for-fragment mapping smuggles the source language's skeleton into every translation.
- **Freedom where it can see; structure where it can't.** Let the model own the words. Don't let it place boxes on a picture it never looked at.
- **Episodic memory needs an explicit spine.** Arcs, outlines, glossaries, key panels — the model remembers nothing between runs, so the *system* has to remember for it.

---

## Conclusion: an Anthology That Never Sleeps

When I started, "a comic that draws itself" sounded like a single magic prompt. It turned out to be a small studio with a memory problem, a discipline problem, and a translation problem — and the interesting part of the project was solving each one honestly. A character who stays herself across weeks. A story with a planned spine and a Director who isn't allowed to cut it short. And three editions that don't translate each other so much as *re-perform* the same silent pictures, each in its own voice.

It's still, proudly, a fiction machine. Nothing in it is real; every panel is invented. But it's a fiction machine with a house style, a continuing cast, and a straight face — one that wakes up every morning, draws the next page, and tells you the story three different ways.

Go read today's episode. Tomorrow there'll be a new one. There always will be.

*Explore the project: **[SandBox on GitHub](https://github.com/abozaralizadeh/SandBox)** — open-source code for ComicBook and its sibling experiments. Every panel and every line is AI-generated: fiction for research and entertainment, not a real publication.*
