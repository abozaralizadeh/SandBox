# ComicBook — Technical Flow

End-to-end architecture of ComicBook: a daily, multi-agent AI comic strip with dynamic
story arcs, character consistency, and en/it/fa editions. Built on the **OpenAI Agents SDK**.
(Renders on GitHub, Mermaid Live, and most Markdown viewers.)

## System flow

```mermaid
flowchart TB
  subgraph BROWSER["Comic Viewer — templates/comicbook.html"]
    direction TB
    CV["Comic page<br/>language select · date picker · prev / next / arc nav"]
    CFETCH["fetch /comicbookcontent?dt and lang"]
    CIDX["fetch /comicbookindex (episodes + arcs)"]
  end

  subgraph API["Flask API — main.py"]
    direction TB
    RC["GET /comicbook → render page"]
    RCC["GET /comicbookcontent<br/>headers: Timestamp, Arc-Id"]
    RCI["GET /comicbookindex → JSON"]
  end

  subgraph ORCH["Orchestrator — ComicBook/prompt.py"]
    direction TB
    GC["get_comicbook(date, lang)"]
    CACHE{"episode cached?<br/>get_episode_by_date"}
    LOCK{"acquire generation_lock?"}
    SAVE["save_episode<br/>(blob offload if > 32K)"]
  end

  subgraph PIPE["Agent Pipeline — ComicBook/agents.py · OpenAI Agents SDK · Runner.run"]
    direction TB
    DIR["Director (temp 1.2)<br/>create/continue arc + episode plan"]
    OA["OutlineAdapter (episode 1 only)<br/>adapt outline → it, fa"]
    ST["Storyteller<br/>panel-by-panel script"]
    CART["Cartoonist (max 30 turns)<br/>character sheet + panels + HTML"]
    RET["Reteller (temp 0.9)<br/>rewrite panels → it, fa + glossary"]
  end

  subgraph TOOLS["Function Tools"]
    direction TB
    WS["WebSearchTool"]
    ARCT["Arc tools<br/>get_arc_status · start/end_arc · save_story_outline"]
    IMGT["Image tools<br/>generate_character_sheet · generate_panel_image<br/>mark_key_panel · assemble_layout"]
  end

  subgraph STORE["Azure Storage — ComicBook/azurestorage.py"]
    TEP[("Table comicbook<br/>episodes (PK arc_id · RK date)<br/>+ generation_lock")]
    TARC[("Table comicbookarcs<br/>arc meta · outline · glossary<br/>key_panels · character_sheet_url")]
    BLOB[("Blob comicbook-html<br/>HTML · outlines · glossaries · panel images")]
  end

  subgraph AZ["Azure OpenAI + LangSmith"]
    CHAT["Chat model (gpt-4o)<br/>all 5 agents"]
    IMG["Image model (gpt-image)<br/>AZURE_OPENAI_*_DALLE"]
    LSM["LangSmith<br/>wrap_openai + @traceable"]
  end

  CV --> RC
  CFETCH --> RCC
  CIDX --> RCI
  RCI --> TEP
  RCI --> TARC

  RCC --> GC --> CACHE
  CACHE -->|hit| RCC
  CACHE -->|miss| LOCK
  LOCK -->|busy| RCC
  LOCK -->|got it| DIR

  DIR --> ST --> CART --> RET --> SAVE
  DIR -.->|episode 1| OA
  DIR --> WS
  DIR --> ARCT
  CART --> IMGT

  DIR --> CHAT
  ST --> CHAT
  CART --> CHAT
  RET --> CHAT
  OA --> CHAT
  IMGT --> IMG

  ARCT <--> TARC
  IMGT --> BLOB
  SAVE --> TEP
  SAVE --> TARC
  SAVE -->|"HTML > 32K"| BLOB
  GC -->|"html(lang) + Timestamp + Arc-Id"| RCC

  DIR -.-> LSM
  CART -.-> LSM
  IMGT -.-> LSM
```

## Agent pipeline + consistency

```mermaid
flowchart LR
  D["Director<br/>new or continue arc<br/>(arc ends when episodes ≥ planned)"] --> S["Storyteller<br/>panel script"]
  S --> C["Cartoonist"]
  C --> CS["character sheet<br/>once per arc (cached on arc)"]
  CS --> PI["panel images<br/>references: sheet → key panels<br/>→ prior-episode anchors"]
  PI --> L["assemble HTML layout (en)"]
  L --> R["Reteller → it, fa<br/>reuse fixed panel images + glossary"]
```

## Runtime sequence

```mermaid
sequenceDiagram
    actor U as Reader
    participant CV as Comic viewer
    participant API as Flask (main.py)
    participant P as get_comicbook
    participant AG as Agents (agents.py)
    participant T as Azure Table
    participant B as Azure Blob
    participant AOAI as Azure OpenAI

    U->>CV: open /comicbook (date + lang)
    CV->>API: GET /comicbookcontent?dt and lang
    API->>P: get_comicbook(date, lang)
    P->>T: get_episode_by_date (cache)
    alt cached
        T-->>P: episode HTML (hydrate from blob if large)
    else not cached
        P->>T: acquire generation_lock
        P->>AG: run_comic_pipeline(date)
        AG->>AOAI: Director (arc + episode plan, web search)
        AG->>AOAI: Storyteller (panel script)
        AG->>AOAI: Cartoonist → image model (character sheet + panels)
        AOAI-->>B: upload panel images
        AG->>AOAI: Reteller (it + fa) + glossary
        AG-->>P: html, html_it, html_fa, arc
        P->>T: save_episode (+ arc meta)
        P->>B: offload HTML if > 32K
        P->>T: release lock
    end
    P-->>API: html(lang) + Timestamp + Arc-Id
    API-->>CV: comic HTML
```

### Notes
- **Dynamic arcs:** the Director invents and ends story arcs organically (an arc runs as
  many episodes as it needs), tracked in the `comicbookarcs` table.
- **Character consistency:** the Cartoonist generates one **character reference sheet** per
  arc, then draws every panel with references (sheet → mid-arc key panels → prior-episode
  anchors) via Azure OpenAI image editing.
- **Multi-language:** English is native; **OutlineAdapter** (episode 1) localizes the story
  outline and **Reteller** rewrites each episode's panels into it/fa, with a per-language
  **glossary** for consistent names/terms. The same panel images are reused across languages.
- **Caching + single-flight lock:** one episode per date is cached; `generation_lock`
  (a partition in the episodes table, TTL-guarded) prevents concurrent regeneration.
- **Blob offload:** HTML / outlines / glossaries over 32K chars are stored in blob, with the
  name kept in the table; panel images live in the same `comicbook-html` container.
- **Separate deployments:** chat (`gpt-4o`) for the agents, image (`gpt-image`) via the
  `AZURE_OPENAI_*_DALLE` resource. LangSmith traces the run (`wrap_openai` + `@traceable`).
