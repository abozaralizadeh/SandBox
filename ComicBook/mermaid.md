# ComicBook Pipeline — Agent Flow

`run_comic_pipeline` is a **single `Runner.run(Director)`** whose control flows by OpenAI
Agents-SDK **handoffs**: Director → Storyteller → Cartoonist → Reteller. A deterministic recovery
re-runs any stage whose output is missing, so a missed handoff never strands the comic.

```mermaid
flowchart TD

    %% ── Initialization ──────────────────────────────────────────────────
    START(["🗓️  run_comic_pipeline(target_date)"])

    subgraph INIT ["🔧  Pipeline Initialization"]
        direction LR
        I1["Load active arc\n+ episode number\n(arc or arc_debug partition)"]
        I2["Hydrate last episode HTML\n→ prev_episode_images 0–3 panels"]
        I3["Hydrate first episode HTML\n→ character-anchor panels\n(skipped on ep. 1)"]
        I4["Load key_panels\n(mid-arc character refs)"]
        I1 --> I2 --> I3 --> I4
    end

    START --> INIT

    %% ── Director ─────────────────────────────────────────────────────────
    subgraph DIR ["🎭  Director   temp=1.2"]
        D_resp["Responsibility\n─────────────────────────\n• Owns the full arc lifecycle\n• NEW arc: web-search inspiration →\n  form candidate → check_arc_originality →\n  retry until distinct → start_new_arc\n• Writes the story outline on arc start\n• Plans today's episode: panel count,\n  sizes, beats, tone, cliffhanger\n• Never skips / never closes early\n• Hands off to the Storyteller"]
        D_tools["Tools\n─────────────────────────\n🔍 WebSearch — research & inspiration\n📋 get_arc_status — active arc + history\n🔎 check_arc_originality — OriginalityCritic\n    (as_tool): ok / too_similar + guidance\n🆕 start_new_arc — title, logline, genre,\n    characters (ALL with full visuals),\n    art_style, color_theme, planned_eps\n    (refuses a recently-used art style)\n✅ end_current_arc — close finished arc\n📝 save_story_outline — full arc plan"]
    end

    INIT -->|"arc state\nepisode number\nrecent summaries"| DIR

    %% ── Originality Critic (as_tool) ─────────────────────────────────────
    subgraph CRIT ["🔎  OriginalityCritic   temp=0.2   (as_tool, not in the chain)"]
        CR_resp["Responsibility\n─────────────────────────\n• Calls get_recent_arcs and compares a\n  candidate premise vs recent arcs\n• Looks past surface theme — plot_shape,\n  conflict, archetypes, setting, art_style\n• Returns ok / too_similar + the most\n  similar arc + concrete retry guidance"]
    end

    DIR -. "as_tool\ncheck_arc_originality" .-> CRIT

    %% ── Storyteller ──────────────────────────────────────────────────────
    subgraph STORY ["✍️  Storyteller   temp=0.5"]
        S_resp["Responsibility\n─────────────────────────\n• Transforms Director's plan into a\n  full panel-by-panel script\n• Per panel: setting, characters, poses,\n  expressions, camera angle, lighting\n• Dialogue — distinct voice per character\n• Captions — clarity first (what/why/changed)\n• SFX — sparingly; 3–4 line recap + teaser\n• No tools — writes the script as a message,\n  then hands off to the Cartoonist"]
    end

    DIR ==>|"transfer_to_Storyteller\n(plan in conversation)"| STORY

    %% ── Cartoonist ───────────────────────────────────────────────────────
    subgraph TOON ["🎨  Cartoonist"]
        C_resp["Responsibility\n─────────────────────────\n• Pulls the full arc roster + art style\n  via get_cartoonist_brief\n• Generates the character reference sheet\n  (once per arc, cached; quality=high)\n• Generates each panel SEQUENTIALLY so\n  each finished panel feeds the next\n• MUST finish assemble_layout before it may\n  hand off (hard gate)"]

        subgraph C_TOOLS ["Tools (called in order)"]
            direction TB
            CT0["⓪ get_cartoonist_brief\n— full roster · art_style · key panels"]
            CT1["① generate_character_sheet\n— ALL arc characters (incl. future eps)\n— cached on arc after ep. 1"]
            CT2["② generate_panel_image  ×N\n— SEQUENTIAL (panel 1 → 2 → … → N)\n— refs: sheet + key panels + session + anchors"]
            CT3["③ mark_key_panel\n— after any panel with a NEW mid-arc char"]
            CT4["④ assemble_layout\n— writes html_en to state"]
            CT0 --> CT1 --> CT2 --> CT3 --> CT4
        end
    end

    STORY ==>|"transfer_to_Cartoonist\n(script in conversation)"| TOON

    %% ── Reteller ─────────────────────────────────────────────────────────
    subgraph TRANS ["🗣️  Reteller   temp=0.9   (one run, both languages: IT then FA)"]
        T_resp["Responsibility\n─────────────────────────\n• RETELLS the episode natively over the\n  shared images — not a translation\n• Full freedom over WORDS & pacing; faithful\n  to the fixed art + every plot beat\n• MAIN title = the ARC title (consistent);\n  episode title becomes a SUBTITLE\n• ep.1: adapts + saves the localized outline\n  (no separate OutlineAdapter agent)\n• Maintains the per-language glossary\n• Last stage — does NOT hand off"]

        subgraph T_TOOLS ["Tools (per language)"]
            direction TB
            RT1["get_localization_brief\n— manifest + outlines + glossary\n+ arc_title_local / arc_title_en"]
            RT2["save_local_outline\n— ep.1 only, when none exists"]
            RT3["assemble_localized\n— title + subtitle + native panels\n→ writes html_it / html_fa to state"]
            RT1 --> RT2 --> RT3
        end
    end

    TOON ==>|"transfer_to_Reteller\n(plan + script in conversation)"| TRANS

    %% ── Recovery ─────────────────────────────────────────────────────────
    RECOV["🛟  Deterministic recovery\n─────────────\nIf html_en / html_it / html_fa is missing\n(an agent did not hand off), run that\nstage directly with a clean input"]
    TRANS -. "missing artifact" .-> RECOV

    %% ── Storage ──────────────────────────────────────────────────────────
    subgraph STORAGE ["☁️  Azure Storage"]
        direction LR
        ARC_TBL[("Arcs Table  (PK arc or arc_debug)\n─────────────\ntitle · logline · genre\nplanned_episodes\nart_style · color_theme\ncharacters (full visuals)\nstory_outline (en/it/fa)\ntitle_it · title_fa\ncharacter_sheet_url\nkey_panels (JSON list)\nglossary_it · glossary_fa\nstatus · episodes_count")]
        EP_TBL[("Episodes Table\n─────────────\nRowKey = date\nPartitionKey = arc_id\nepisode_number · story_summary\nhtml_content (en/it/fa)\nhtml_blob_name_*\n+ generation_lock(_debug)")]
        BLOB[("Blob Storage\n─────────────\npanel images (.png)\ncharacter sheets (.png)\nHTML overflow (.html)\nstory outlines (.txt)\nglossaries (.json)")]
    end

    %% ── Storage writes ───────────────────────────────────────────────────
    DIR    -->|"start/end arc\nsave_story_outline"| ARC_TBL
    CT1    -->|"character_sheet_url"| ARC_TBL
    CT2    -->|"panel images"| BLOB
    CT1    -->|"character sheet image"| BLOB
    CT3    -->|"key_panels entry"| ARC_TBL
    RT2    -->|"localized outline (it/fa)"| ARC_TBL
    RT3    -->|"updated glossary + title_{lang}"| ARC_TBL

    subgraph SAVE ["💾  save_episode (post-pipeline · prompt.py)"]
        SV["Writes episode row with HTML (en/it/fa),\nstory_summary, episode_number\nIncrements arc episodes_count\n(skipped entirely in DEBUG_SAVE=false)"]
    end

    TRANS --> SAVE
    RECOV --> SAVE
    SAVE  --> EP_TBL

    %% ── Output ───────────────────────────────────────────────────────────
    RESULT(["📦  Pipeline result\n─────────────────\nhtml (en) · html_it · html_fa\nsummary (Director plan)\npanel_notes (Storyteller script)\narc · episode_number"])

    SAVE --> RESULT

    %% ── Styling ──────────────────────────────────────────────────────────
    classDef agent   fill:#1e1e3a,stroke:#6366f1,color:#e0e0ff,rx:8
    classDef tools   fill:#0f1a2e,stroke:#334466,color:#a0b4d0,rx:6
    classDef storage fill:#1a1a10,stroke:#b8860b,color:#f0e0a0,rx:6
    classDef io      fill:#0f2a1a,stroke:#4ade80,color:#d0ffe0,rx:20
    classDef init    fill:#1a1020,stroke:#8b5cf6,color:#ddd0ff,rx:6

    class DIR,STORY,TOON,TRANS,CRIT agent
    class C_TOOLS,CT0,CT1,CT2,CT3,CT4,T_TOOLS,RT1,RT2,RT3 tools
    class ARC_TBL,EP_TBL,BLOB storage
    class START,RESULT io
    class INIT,I1,I2,I3,I4,RECOV init
```

> **Handoffs:** each `==>` edge is an SDK handoff (`transfer_to_<next>`) with
> `input_filter=remove_all_tools`, so the next agent inherits the plan/script **messages** but not
> the prior stage's tool calls. Every chained agent is wrapped with
> `prompt_with_handoff_instructions(...)` for reliable transfers; the Cartoonist is additionally
> hard-gated to finish `assemble_layout` before it may transfer.

## Agent Summary

| Agent | Role | Temp | Tools | In handoff chain? |
|---|---|---|---|---|
| **Director** | Arc lifecycle + originality + episode planner | 1.2 | WebSearch, get_arc_status, **check_arc_originality** (as_tool), start_new_arc, end_current_arc, save_story_outline | entry → Storyteller |
| **OriginalityCritic** | Judges a candidate arc vs recent arcs | 0.2 | get_recent_arcs | no — invoked via `as_tool` |
| **Storyteller** | Panel-by-panel script writer | 0.5 | — | → Cartoonist |
| **Cartoonist** | Image generation + HTML assembly (en) | 1.0 | get_cartoonist_brief, generate_character_sheet, generate_panel_image, mark_key_panel, assemble_layout | → Reteller |
| **Reteller** | Native retelling IT + FA (one run) | 0.9 | get_localization_brief, save_local_outline, assemble_localized | terminus |

## Key Design Decisions

| Decision | Reason |
|---|---|
| **Handoff chain + deterministic recovery** | Agents collaborate via SDK handoffs (Director→Storyteller→Cartoonist→Reteller); if a model fails to call its transfer tool, the pipeline runs the missing stage directly so a comic always ships |
| **No LLM calls inside a tool** | A `@function_tool` does only deterministic work; model-reasoning steps are Agents reached via `as_tool` (OriginalityCritic) or handoffs |
| **Three-layer originality guard** | Prompt mandates search→check→retry; the OriginalityCritic (as_tool) judges core-story similarity; `start_new_arc` refuses a recently-used art style |
| **Temperature split** | Director 1.2 (creative engine) and OriginalityCritic 0.2 (judge); Storyteller 0.5 faithfully executes the plan; Reteller 0.9 |
| Panels generated **sequentially** | Each finished panel URL feeds as a reference into the next call, maintaining visual consistency |
| Character sheet uses **ALL arc characters** at `quality=high` | Generated once on ep. 1 and cached — must cover every character who ever appears |
| **key_panels** list on the arc | Mid-arc characters get a dedicated reference panel persisted across future episodes |
| **Reteller does both languages in one run** | Retells IT then FA via tools; adapts + saves the localized outline on ep.1 itself (the separate OutlineAdapter agent was removed) |
| **Reteller retells natively** | Fragment translation forced English's text architecture onto every language; retelling lets each language restructure dialogue/captions. Box POSITIONS stay fixed (caption top, bubbles bottom, RTL-aware) |
| **Localized title from the ARC + episode subtitle** | The main title is the arc title (stored as `title_{lang}`, consistent every episode); the episode's native title is a subtitle under it |
| **Readability guard** | `_assemble_html` flips any low-contrast text color to near-black/near-white so a box is never light-on-light or dark-on-dark |
| **DEBUG / DEBUG_SAVE** | `DEBUG` isolates to an `arc_debug` partition (`debugarc_*` ids, `generation_lock_debug`) so local tests never touch production; `DEBUG_SAVE=false` is a pure dry run |
| **No generation time limit** | Chat + image clients use a 1-hour timeout (`COMICBOOK_LLM_TIMEOUT` / `COMICBOOK_IMAGE_TIMEOUT`) so slow generations are not cut off |
