# Tomorrow News — Technical Flow

End-to-end architecture of TomorrowNews: a speculative "tomorrow's newspaper" generated daily
from real RSS headlines, with AI-written articles and images, in en/fa/it. Built on
**LangChain + LangGraph**. (Renders on GitHub, Mermaid Live, and most Markdown viewers.)

## System flow

```mermaid
flowchart TB
  subgraph BROWSER["Newspaper UI — templates/tomorrownews.html"]
    direction TB
    NUI["iframe newspaper<br/>date controls · language select · spinner"]
    NF["fetch /tomorrownewscontent?dt and lang"]
  end

  subgraph API["Flask API — main.py"]
    direction TB
    RN["GET /tomorrownews → render"]
    RNC["GET /tomorrownewscontent<br/>header: Timestamp"]
  end

  subgraph ORCH["Orchestrator — TomorrowNews/prompt.py"]
    direction TB
    GTN["gettomorrownews(date, lang)"]
    NCACHE{"cached?<br/>rowkey date_lang"}
    GENALL["_generate_all → en, fa, it<br/>_generate_single per language"]
  end

  subgraph GRAPH["LangGraph — TomorrowNews/graph.py"]
    direction TB
    AGENT["agent node<br/>AzureChatOpenAI (temp 1.3) + tools"]
    COND{"tool calls?"}
    TOOLSN["tools node (ToolNode)"]
  end

  subgraph TOOLS["Tools (@tool)"]
    NEWS["get_todays_news_feed (per language)<br/>BBC / ANSA RSS via xmltodict"]
    IMG["get_image_by_text → DALL-E 3"]
  end

  subgraph STORE["Azure Storage — TomorrowNews/azurestorage.py"]
    TBL[("Table tomorrownews<br/>PK getimagetool · RK date_lang<br/>html_content / html_blob_name")]
    BLB[("Blob tomorrownews<br/>images + HTML > 32K")]
  end

  subgraph EXT["External + Azure OpenAI"]
    RSS["RSS feeds<br/>BBC (en/fa) · ANSA (it)"]
    CHAT["Chat model"]
    DALLE["Image model (dall-e-3)"]
  end

  NUI --> RN
  NF --> RNC
  RNC --> GTN --> NCACHE
  NCACHE -->|hit| RNC
  NCACHE -->|miss| GENALL --> AGENT
  AGENT --> COND
  COND -->|yes| TOOLSN --> AGENT
  COND -->|"no: final HTML"| GENALL
  TOOLSN --> NEWS
  TOOLSN --> IMG
  AGENT --> CHAT
  NEWS --> RSS
  IMG --> DALLE
  IMG -->|upload| BLB
  GENALL -->|insert_history per lang| TBL
  GENALL -->|"HTML > 32K"| BLB
  GENALL -->|html + Timestamp| RNC
```

## Runtime sequence

```mermaid
sequenceDiagram
    actor U as Reader
    participant UI as Newspaper UI
    participant API as Flask
    participant P as gettomorrownews
    participant G as LangGraph (agent ↔ tools)
    participant T as Azure Table
    participant B as Azure Blob
    participant X as RSS + Azure OpenAI

    U->>UI: pick date + language
    UI->>API: GET /tomorrownewscontent?dt and lang
    API->>P: gettomorrownews(date, lang)
    P->>T: cache lookup (date_lang)
    alt cached
        T-->>P: html_content
    else generate (en, fa, it)
        loop agent reasons, calls tools
            P->>G: graph.stream
            G->>X: get_todays_news_feed (RSS)
            G->>X: chat (write articles)
            G->>X: get_image_by_text → DALL-E
            X-->>B: upload image
        end
        G-->>P: final newspaper HTML
        P->>T: insert_history (per language)
        P->>B: offload HTML if > 32K
    end
    P-->>API: html + Timestamp
    API-->>UI: write into iframe
```

### Notes
- **Per-language generation:** `_generate_all` produces en → fa → it, each cached separately
  by a `date_lang` row key (PartitionKey `getimagetool`). Persian adds RTL + font directives.
- **Row-key granularity:** hourly before 2025-01-25, daily after; language UI is enabled from
  a later cutoff.
- **ReAct-style graph:** a single `agent ↔ tools` LangGraph loop — fetch RSS headlines, write
  the speculative articles, generate images — until the agent emits the final HTML newspaper.
- **Real source headlines:** BBC RSS (en/fa) and ANSA (it) parsed with `xmltodict`.
- **Images:** DALL-E 3 images are downloaded and re-uploaded to blob, embedded as `<img>`;
  HTML over 32K chars is offloaded to blob with the name kept in the table.
- A richer multi-agent `supervisor.py` graph (Editor → Journalist → Photographer → HTML) exists
  but is **not** wired into the default route.
