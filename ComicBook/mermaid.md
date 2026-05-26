# ComicBook Pipeline — Agent Flow

```mermaid
flowchart TD

    %% ── Initialization ──────────────────────────────────────────────────
    START(["🗓️  run_comic_pipeline(target_date)"])

    subgraph INIT ["🔧  Pipeline Initialization"]
        direction LR
        I1["Load active arc\n+ episode number"]
        I2["Hydrate last episode HTML\n→ prev_episode_images 0–3 panels"]
        I3["Hydrate first episode HTML\n→ character-anchor panels\n(skipped on ep. 1)"]
        I4["Load key_panels\n(mid-arc character refs)"]
        I1 --> I2 --> I3 --> I4
    end

    START --> INIT

    %% ── Director ─────────────────────────────────────────────────────────
    subgraph DIR ["🎭  Director   temp=1.2"]
        D_resp["Responsibility\n─────────────────────────\n• Owns the full arc lifecycle\n• Creates new arcs with fresh genre,\n  characters & art style each time\n• Writes the story outline on arc start\n• Plans today's episode: panel count,\n  sizes, beats, tone, cliffhanger\n• Enforces episode-sequence integrity\n  (never skips, never closes early)\n• Alternates Western / non-Western\n  settings every other day"]
        D_tools["Tools\n─────────────────────────\n🔍 WebSearch — research & inspiration\n📋 get_arc_status — active arc + history\n🆕 start_new_arc — title, logline, genre,\n    characters (ALL with full visuals),\n    art_style, color_theme, planned_eps\n✅ end_current_arc — close finished arc\n📝 save_story_outline — full arc plan\n    with per-episode breakdown"]
    end

    INIT -->|"arc state\nepisode number\nrecent summaries"| DIR

    %% ── Outline Adapter (ep 1 only) ──────────────────────────────────────
    subgraph OA ["🌐  OutlineAdapter   ×2 languages   (episode 1 only)"]
        OA_resp["Responsibility\n─────────────────────────\n• Adapts the story outline to Italian\n  and Persian — not a literal translation\n• Rewrites as if originally authored in\n  the target language & culture\n• Preserves all plot points, character arcs\n  and episode-by-episode breakdown\n• Saved once; reused by Translator\n  every episode for consistent context"]
    end

    DIR -->|"story outline\n(ep. 1 only)"| OA

    %% ── Storyteller ──────────────────────────────────────────────────────
    subgraph STORY ["✍️  Storyteller"]
        S_resp["Responsibility\n─────────────────────────\n• Transforms Director's plan into a\n  full panel-by-panel script\n• Per panel: setting, characters, poses,\n  expressions, camera angle, lighting\n• Dialogue — distinct voice per character\n• Captions — narrative clarity first,\n  then artistry; answers what/why/what changed\n• SFX — sparingly, only strong audible events\n• Opens with 3–4 line recap (any reader can\n  catch up); closes with 1-line teaser\n• No tools — pure creative text output"]
    end

    DIR -->|"Episode plan\n+ story outline"| STORY

    %% ── Cartoonist ───────────────────────────────────────────────────────
    subgraph TOON ["🎨  Cartoonist"]
        C_resp["Responsibility\n─────────────────────────\n• Generates the character reference sheet\n  (once per arc, cached; quality=high)\n• Generates each panel image SEQUENTIALLY\n  so each completed panel feeds the next\n• Marks panels introducing new mid-arc\n  characters as key panels\n• Assembles the final HTML comic page\n  with arc color theme + RTL support"]

        subgraph C_TOOLS ["Tools (called in order)"]
            direction TB
            CT1["① generate_character_sheet\n— ALL arc characters (incl. future eps)\n— quality=high, size=wide\n— cached on arc after ep. 1"]
            CT2["② generate_panel_image  ×N\n— SEQUENTIAL (panel 1 → 2 → … → N)\n— refs: char sheet + key panels (≤3)\n  + session panels (≤2) + arc anchors (≤2)"]
            CT3["③ mark_key_panel\n— called right after any panel\n  showing a new mid-arc character\n— persists URL to arc key_panels list"]
            CT4["④ assemble_layout\n— arc_title, episode_number, date\n— recap, teaser, panels_json\n— returns final HTML"]
            CT1 --> CT2 --> CT3 --> CT4
        end
    end

    STORY -->|"Panel-by-panel script\n+ FULL ARC ROSTER\n+ mid-arc char refs"| TOON

    %% ── Translator ───────────────────────────────────────────────────────
    subgraph TRANS ["🗣️  Translator   ×2 languages   (sequential: IT → FA)"]
        T_resp["Responsibility\n─────────────────────────\n• Translates title, recap, teaser,\n  all dialogue, captions and SFX\n• Idiomatic — rewrites from intent,\n  not word-for-word from English\n• Dialogue: spoken/colloquial register\n  tuned to each character's personality\n• Captions/narration: literary register\n• Adapts idioms & cultural references\n• Maintains arc glossary for consistency\n  of names, places & coined terms\n• Outputs updated_glossary every run\n• No tools — pure text-in / JSON-out"]
    end

    TOON -->|"en HTML + panels\n+ Director plan\n+ Storyteller script\n+ arc glossary\n+ lang story outline"| TRANS

    %% ── Storage ──────────────────────────────────────────────────────────
    subgraph STORAGE ["☁️  Azure Storage"]
        direction LR
        ARC_TBL[("Arcs Table\n─────────────\ntitle · logline · genre\nplanned_episodes\nart_style · color_theme\ncharacters (full visuals)\nstory_outline (en/it/fa)\ncharacter_sheet_url\nkey_panels (JSON list)\nglossary_it · glossary_fa\nstatus · episodes_count")]
        EP_TBL[("Episodes Table\n─────────────\nRowKey = date\nPartitionKey = arc_id\nepisode_number\nstory_summary\nhtml_content (en)\nhtml_content_it\nhtml_content_fa\nhtml_blob_name_*")]
        BLOB[("Blob Storage\n─────────────\npanel images (.png)\ncharacter sheets (.png)\nHTML overflow (.html)\nstory outlines (.txt)\nglossaries (.json)")]
    end

    %% ── Storage writes ───────────────────────────────────────────────────
    DIR       -->|"start_new_arc\nend_current_arc\nsave_story_outline"| ARC_TBL
    OA        -->|"adapted outlines\n(it + fa)"| ARC_TBL
    CT1       -->|"character_sheet_url"| ARC_TBL
    CT2       -->|"panel images"| BLOB
    CT1       -->|"character sheet image"| BLOB
    CT3       -->|"key_panels entry"| ARC_TBL
    TRANS     -->|"updated glossary\n(it + fa)"| ARC_TBL

    subgraph SAVE ["💾  save_episode (post-pipeline)"]
        SV["Writes episode row\nto Episodes Table\nwith HTML (en/it/fa),\nstory_summary,\nepisode_number\nIncrements arc episodes_count"]
    end

    TOON  -->|"en HTML"| SAVE
    TRANS -->|"it + fa HTML"| SAVE
    SAVE  --> EP_TBL

    %% ── Output ───────────────────────────────────────────────────────────
    RESULT(["📦  Pipeline result\n─────────────────\nhtml (en)\nhtml_it · html_fa\nsummary (Director plan)\npanel_notes (Storyteller script)\narc · episode_number"])

    SAVE --> RESULT

    %% ── Styling ──────────────────────────────────────────────────────────
    classDef agent   fill:#1e1e3a,stroke:#6366f1,color:#e0e0ff,rx:8
    classDef tools   fill:#0f1a2e,stroke:#334466,color:#a0b4d0,rx:6
    classDef storage fill:#1a1a10,stroke:#b8860b,color:#f0e0a0,rx:6
    classDef io      fill:#0f2a1a,stroke:#4ade80,color:#d0ffe0,rx:20
    classDef init    fill:#1a1020,stroke:#8b5cf6,color:#ddd0ff,rx:6

    class DIR,STORY,TOON,TRANS,OA agent
    class C_TOOLS,CT1,CT2,CT3,CT4 tools
    class ARC_TBL,EP_TBL,BLOB storage
    class START,RESULT io
    class INIT,I1,I2,I3,I4 init
```

## Agent Summary

| Agent | Role | Tools | Runs |
|---|---|---|---|
| **Director** | Arc lifecycle + episode planner | WebSearch, get_arc_status, start_new_arc, end_current_arc, save_story_outline | Every episode |
| **Storyteller** | Panel-by-panel script writer | — | Every episode |
| **Cartoonist** | Image generation + HTML assembly | generate_character_sheet, generate_panel_image, mark_key_panel, assemble_layout | Every episode |
| **Translator** | Literary translation (IT + FA) | — | Every episode × 2 |
| **OutlineAdapter** | Adapts story outline to target language | — | Episode 1 only × 2 |

## Key Design Decisions

| Decision | Reason |
|---|---|
| Panels generated **sequentially** | Each completed panel URL feeds as a reference into the next call, maintaining visual consistency |
| Character sheet uses **ALL arc characters** (incl. future eps) at `quality=high` | Generated once on ep. 1 and cached — must cover every character who ever appears |
| **key_panels** list on arc entity | Mid-arc characters (not on original sheet) get a dedicated reference panel persisted across all future episodes |
| **prev_episode_images** = last ep. panels + first ep. panels | Last episode for immediate continuity; first episode as the character-introduction visual anchor |
| **OutlineAdapter** runs on ep. 1 only | Adapts the outline once per arc; Translator reads it every episode for consistent story context |
| Translations run **sequentially** (IT → FA) | Avoids race conditions on glossary writes to the Arcs Table |
