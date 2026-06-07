# AI Blog — Technical Flow

End-to-end architecture of AIBlog: an autonomous researcher that discovers new ML/GenAI work,
browses sources, and publishes a daily HTML blog post with a generated banner. Built on a
**LangGraph ReAct agent** with token-aware context management. (Renders on GitHub, Mermaid
Live, and most Markdown viewers.)

## System flow

```mermaid
flowchart TB
  subgraph BROWSER["Blog UI — templates/aiblog.html"]
    direction TB
    BUI["iframe blog post<br/>date picker · spinner"]
    BF["fetch /aiblogcontent?dt"]
  end

  subgraph API["Flask API — main.py (async route)"]
    direction TB
    RA["GET /aiblog → render"]
    RAC["GET /aiblogcontent<br/>header: Timestamp"]
  end

  subgraph ORCH["Orchestrator — AIBlog/prompt.py"]
    direction TB
    GAB["getaiblog(date) — async"]
    BCACHE{"cached and fresh?<br/>rowkey YYYYMMDD_00 · < 30 min"}
    EXTR["extract HTML · strip private-use chars"]
    CLOSE["close Playwright browser (finally)"]
  end

  subgraph AGENT["ReAct Agent — AIBlog/graph.py"]
    direction TB
    LLM["TokenAwareAzureChatOpenAI<br/>compress tool msgs · summarize · trim (~270K budget)"]
    REACT["create_react_agent loop<br/>agent ↔ tools (recursion 1000)"]
  end

  subgraph TOOLS["Tools"]
    direction TB
    WS["web_search (DuckDuckGo + Tavily)"]
    BROWSE["Playwright browse tools<br/>navigate · click · extract text/links"]
    IMG["get_image_by_text → DALL-E 3"]
    TITLE["set_title → table"]
  end

  subgraph STORE["Azure Storage — AIBlog/azurestorage.py"]
    TBL[("Table aiblog<br/>PK getimagetool · RK YYYYMMDD_00<br/>html_content · title")]
    BLB[("Blob aiblog<br/>images + HTML > 32K")]
  end

  subgraph EXT["External + Azure OpenAI"]
    CHAT["Chat model"]
    DALLE["Image model (dall-e-3)"]
    SRCH["DuckDuckGo · Tavily · arxiv / blog pages"]
  end

  BUI --> RA
  BF --> RAC
  RAC --> GAB --> BCACHE
  BCACHE -->|fresh hit| RAC
  BCACHE -->|miss / stale| REACT
  REACT --> LLM --> CHAT
  REACT --> WS --> SRCH
  REACT --> BROWSE --> SRCH
  REACT --> IMG --> DALLE
  IMG -->|upload| BLB
  REACT --> TITLE --> TBL
  REACT --> EXTR
  EXTR -->|upsert_history| TBL
  EXTR -->|"HTML > 32K"| BLB
  GAB --> CLOSE
  GAB -->|html + Timestamp| RAC
```

## Token-aware context control

```mermaid
flowchart LR
  IN["agent step:<br/>messages + tool outputs"] --> CHK{"over token budget?"}
  CHK -->|no| SEND["send to Azure OpenAI"]
  CHK -->|yes| C1["1. compress oversized tool<br/>messages (map-reduce)"]
  C1 --> C2["2. summarize old history<br/>(keep last ~6 turns)"]
  C2 --> C3["3. trim oldest non-system<br/>messages (fallback)"]
  C3 --> SEND
```

## Runtime sequence

```mermaid
sequenceDiagram
    actor U as Reader
    participant UI as Blog UI
    participant API as Flask (async)
    participant P as getaiblog
    participant A as ReAct agent
    participant T as Azure Table
    participant B as Azure Blob
    participant X as Search + Azure OpenAI

    U->>UI: pick date
    UI->>API: GET /aiblogcontent?dt
    API->>P: await getaiblog(date)
    P->>T: cache lookup (YYYYMMDD_00)
    alt fresh (< 30 min)
        T-->>P: html_content
    else research + write
        loop reason ↔ act (token-managed)
            P->>A: astream
            A->>X: web_search / browse pages
            A->>X: get_image_by_text → DALL-E
            X-->>B: upload banner image
            A->>T: set_title
        end
        A-->>P: final blog HTML
        P->>T: upsert_history (+ blob if > 32K)
        P->>P: close Playwright browser
    end
    P-->>API: html + Timestamp
    API-->>UI: write into iframe
```

### Notes
- **Async route + Playwright:** `/aiblogcontent` is an async Flask route; the agent browses
  real pages with Playwright, and the browser is closed in a `finally` block to avoid
  "event loop is closed" errors when Flask tears down.
- **Token-aware LLM:** `TokenAwareAzureChatOpenAI` keeps the context under budget by
  compressing large tool outputs, summarizing old turns, then trimming — so long research
  sessions don't overflow the context window.
- **Topic dedup:** the last ~30 published titles are passed in so the agent picks something new.
- **Research stack:** DuckDuckGo + Tavily search (`TAVILY_API_KEY`) plus Playwright browsing
  of arxiv / vendor blogs; a DALL-E 3 banner is generated and stored in blob.
- **Storage:** one post per day (`YYYYMMDD_00`) in the `aiblog` table; HTML over 32K chars and
  all images live in the `aiblog` blob container. `DEBUG` / `DEBUG_SAVE` flags control the
  cache and whether output is persisted.
