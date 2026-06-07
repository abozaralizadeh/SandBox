# Giving the AI Government a Face and a Voice: Building GenBox's Self-Producing Newsroom

*A while ago I built GenBox — a small virtual world where an AI makes a high-level decision every single day and types it onto a retro TV. This is the follow-up: the TV now broadcasts an actual news segment — anchor, field reporter, interview, and a spoken narration — all generated, stitched, and synced automatically. Here's how it works, and everything that fought back.*

---

## Where we left off

If you read the [first post](https://abozar-alizadeh.medium.com/exploring-ai-driven-governance-building-a-virtual-world-where-ai-rules-22419690a409), you'll remember the premise: GenBox is a contained little world that an autonomous AI governs one decision at a time — economy, society, environment, global politics. Each day the model produces a JSON object with three fields — `output` (the decision to announce), `prompt` (a seed for tomorrow's thinking), and `context` (what's changed) — and the `output` scrolls up a green CRT screen on a retro television. Flask, Azure OpenAI, an Azure Table for memory. Simple, and weirdly hypnotic.

But one thing always nagged at me. A government that rules the entire world... and all it gets is monospaced text drifting up a screen? If an AI is making the calls, it should at least have to face the evening news.

So I gave it one.

The TV's default mode is now a **video news bulletin**. The lowest knob flips back to the classic scrolling text whenever you want it. The other knobs still move through the timeline (yesterday, tomorrow) and pause — except now they pause the video too. And in text mode, a voice reads the decision aloud, like a government spokesperson explaining a new program.

It looks like a small feature. It was not a small feature. It was a month of the API politely — and sometimes impolitely — telling me no.

## What you see now

When you open a recent day, the TV plays a short, self-produced news segment:

1. **Anchor lead** — *"Good evening. The AI World Government today decided to…"* — one or two sentences that summarize the decision.
2. **Field report** — a correspondent on location, intercut with b-roll (solar farms, ports, markets, maps).
3. **Interview** — a quick exchange with an official, an expert, or an affected citizen.
4. **Sign-off** — the anchor closes.

The whole thing is generated from that day's `output` text. No human edits a frame. And when it's not playing video, the scrolling text gets a voiceover — the same decision, narrated. The two modes loop into each other: the video ends, the text scrolls and the narration plays, the narration ends, the video comes back.

Here's the honest engineering story of getting there.

## Challenge 1: Sora 2 is a job, not a request

The video is generated with **Sora 2 on Azure OpenAI**, and the first mental adjustment is that you don't "call" it and get a video back. You *create a job*, *poll* it until it's done, then *download* the MP4. It's asynchronous by design.

The second adjustment: clips are capped at **4, 8, or 12 seconds**. A news bulletin is a minute-plus. So a segment isn't one generation — it's many short clips that have to be produced, then merged into one continuous video. That single constraint shapes everything downstream.

## Challenge 2: the create call that hung forever

My first create requests just… hung. For minutes. No error, no job, nothing. Meanwhile a read-only *list* call to the very same endpoint returned `200 OK` instantly. Auth was fine. The endpoint was fine. But `POST` to create hung every time.

Two culprits, found the hard way:

- **It's multipart, not JSON.** Azure's Sora 2 create mirrors OpenAI's v1 `/videos` surface, which is `multipart/form-data` (it carries an optional reference image file). I was sending a tidy JSON body. The fix was to send the fields the way `curl -F` does — as multipart parts — even when there's no file.
- **There is no `seed`.** I'd been passing a `seed` for reproducibility, the way you would with images. Sora 2 has no such parameter, and the gateway kept rejecting it. I ripped it out entirely (which, as you'll see, made the consistency problem much more interesting).

## Challenge 3: the load balancer that broke everything

This one cost me the most sleep, and it's the most generalizable lesson.

My Sora endpoints sit behind an API Management gateway that **round-robins across several Azure OpenAI resources** — I'd distributed credits across them on purpose. Sensible for stateless calls. Quietly fatal for Sora.

Because Sora is **job-scoped**: the video `id` you get from *create* only exists on the resource that served that create. The follow-up *poll* and *download* must hit that *same* resource. But a round-robin gateway cheerfully sends your poll to a *different* backend — one that has never heard of your job id. So it hangs or 404s, and you can't tell whether generation failed or your request just got misrouted.

A round-robin balancer fundamentally can't preserve per-job affinity, because the job id doesn't encode which backend made it.

The fix was to move the round-robin *into the app*: keep a pool of the resources, pick one per job, and pin that job's entire `create → poll → download` lifecycle to it. You still spread load across all your resources (and use those distributed credits) — you just distribute by **job**, not by **HTTP call**. As a bonus, when a clip fails, the per-clip retry naturally rolls to the next resource, so a single resource running out of credits just fails that clip over.

## Challenge 4: consistency, or why the anchor kept becoming a different person

Here's the one that genuinely surprised me. Every clip of "the anchor" looked like a *different* anchor. Different face, different desk, different studio. A newsroom staffed by a rotating cast of strangers.

The obvious levers to fix this all failed in sequence:

- **A seed?** Doesn't exist in Sora 2 (see above).
- **A reference image?** Sora's `input_reference` (it becomes the first frame) **rejects images containing human faces** — a deliberate deepfake-prevention guardrail. So I can't take a frame of my anchor and feed it back in.
- **The Characters / Cameos feature** that's *designed* for reusable consistent people? Gated behind enterprise human-likeness access. Not available by default.

So how do you keep a face consistent when you're not allowed to *use* the face? The answer, buried in the docs, is **remix**: you reference a previously completed video, and Sora reuses its "framework, scene transitions, and visual layout" while you change only what you describe.

That reframed the whole pipeline:

> The **first** clip of each speaker is a fresh generation. **Every later clip of that same speaker is a remix of their first clip** — same person, same set — changing only the new line they speak.

The studio anchor, the field reporter, the interviewee each get their own "base" clip, and their subsequent lines remix from it. Face-free b-roll keeps the older trick — chaining the last frame of one shot into the first frame of the next — because for b-roll, `input_reference` is allowed. No faces, no gated features, consistent newsroom.

## Challenge 5: stitching clips with no video pipeline

Merging a half-dozen clips into one MP4 means `ffmpeg`. But the app deploys to an Azure App Service that has no `apt`, no Docker step — just `pip install`. The save here is **`imageio-ffmpeg`**, whose Linux wheel ships a self-contained static `ffmpeg` binary. Pip installs it, and you're merging video on a host that has never heard of ffmpeg.

A couple of footnotes that ate an afternoon each: I extract the last frame of a b-roll clip to seed the next one, and the concat filter refuses to cooperate unless every clip has an audio stream — so silent b-roll gets a synthetic silent track injected before merging.

## Challenge 6: the anchor kept getting cut off mid-word

Sora cuts at *exactly* N seconds. If the script fills the whole clip, the last syllable dies at the hard cut. The fix is two-sided: tell the producer to write shorter lines (budget words to finish early), and add an instruction to *every* clip prompt — *finish the sentence about half a second before the clip ends, then hold a calm closing expression; don't start another sentence.* The character now lands the line and waits for the cut.

## Challenge 7: no job queue, but generation takes minutes

The app is plain Flask on gunicorn. No Celery, no queue, no workers. And a full segment takes minutes to render. I'm not going to block an HTTP request for that.

So generation runs in a **background thread**, kicked off lazily the first time the page polls for status. Single-flight is guaranteed by an **Azure Table lock** — only one of gunicorn's worker processes generates a given day, and the status is written back to the table so *any* worker can answer the polling. The page shows the scrolling text instantly and switches to video the moment it's ready. It's the humblest possible "job system," and it's held up.

## Challenge 8: the producer is an agent

The creative decisions — how to turn a dense policy paragraph into an anchor lead, a report, and an interview — are made by a small **producer agent** built on the OpenAI Agents SDK (the same pattern I use elsewhere in the project). It reads the day's decision and returns a structured JSON shot list: each shot's type (anchor / reporter / interview / b-roll), its duration, who's speaking, and the exact words.

Crucially, the agent *only* produces the plan. Plain Python *executes* it — enforcing the clip caps, the remix-for-consistency logic, the retries, the merging. The model does the storytelling; the code does the mechanics. That separation made the whole thing testable and kept a creative model from improvising its way into a runaway bill.

## Challenge 9: then I gave it a voice

Generating a face is dramatic; giving it a voice turned out to be the most *useful* addition. I added a TTS deployment on the **same** Azure resources and call the v1 `/audio/speech` endpoint with a deep, authoritative voice and a tone instruction — *"speak as a calm, authoritative government spokesperson formally addressing the public."*

The narration renders in **seconds**, long before the video finishes, so text mode gets sound almost immediately. It plays over the scrolling text, and — because timing matters — the scroll slows down to match the length of the speech, so the text finishes scrolling exactly as the spokesperson finishes talking. It's stored in the same blob, referenced in the same table, gated by the same "new dates only" rule as the video.

## Challenge 10: the browser had opinions too

The backend wasn't the only adversary. A grab-bag of front-end fights:

- **Private blobs.** The merged MP4 lives in a private container, so a raw `<video src=blobUrl>` just 403s. I serve it through a tiny **same-origin proxy with HTTP Range support**, so the video plays and seeks regardless of the container's access settings.
- **Autoplay with sound.** No website can autoplay audio without a prior user gesture — full stop. So I play *with sound first*, and only fall back to muted (with a "🔊 click for sound" nudge) if the browser blocks it; the first click anywhere turns it on for good.
- **The scroll arriving late.** A subtle one: long decisions started scrolling *way* below the screen and took forever to appear. The CSS offsets were percentages relative to the *text's own height*, so the taller the text, the further off-screen it began. I switched to a **container-relative** start and a duration that scales with length — and, when narration exists, matches the audio exactly. Long or short, the text now enters on time and finishes with the voice.

## Why bother?

The original GenBox asked a quiet question: *what does it feel like when an AI makes the calls?* Adding a face and a voice doesn't change the answer, but it changes how the question lands. There's something genuinely uncanny about a polished anchor, in a slick studio, confidently framing each decision as *the AI World Government, governing efficiently* — optimistic, fluent, reassuring.

That's the point, and it's deliberately theatrical. The presentation layer of power is its own kind of argument. When governance gets a broadcast voice — calm, authoritative, always on message — it's worth noticing how easily we'd believe it. GenBox is still a sandbox, still a safe space to play out these scenarios. But now it also rehearses the *aesthetics* of automated governance, which might be the part we're least prepared for.

## Come watch the news

You can watch the AI government's daily broadcast at **[sandboxes.live/genbox](https://sandboxes.live/genbox)** — let it load, and the anchor will take it from there (give it a click for sound). The lowest knob flips back to the classic scrolling text with its new narration; the others walk the timeline and pause.

It's all open source, and I'd love company — on the engineering, on the prompts, or on the harder question underneath all of it. We started by letting an AI *make* the decisions. This chapter let it *announce* them. The next one is up to us.

Let's keep building futures we can examine — where AI doesn't just assist humanity, but helps us see, and question, the worlds we might be walking into.
