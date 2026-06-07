# GenBox — Technical Flow

End-to-end architecture of GenBox: the daily AI decision, the self-producing news video,
the TTS narration, storage, and the retro-TV playback. (Renders on GitHub, Mermaid Live,
and most Markdown viewers.)

## System flow

```mermaid
flowchart TB
  subgraph BROWSER["Retro TV — templates/tv.html"]
    direction TB
    TVUI["CRT screen<br/>scrolling text · video · narration"]
    TVLOOP["Loop and knobs<br/>prev/next day · pause both · text↔video toggle"]
    TVTXT["fetch decision text"]
    TVPOLL["poll status every 4s"]
    TVPV["play video (default mode)"]
    TVPA["play narration over scroll<br/>playbackRate 1.3x, scroll synced"]
  end

  subgraph API["Flask API — main.py"]
    direction TB
    RP["GET /genbox → render TV"]
    RS["GET /get-string"]
    RST["GET /genbox-video-status"]
    RV["GET /genbox-video<br/>Range proxy"]
    RA["GET /genbox-audio<br/>Range proxy"]
  end

  subgraph DEC["Daily Decision — GenBox/prompt.py"]
    GLLM["get_llm_response(date)"]
  end

  subgraph ORCH["Orchestrator — GenBox/video.py"]
    direction TB
    ENS["ensure_generation_started(date)<br/>non-blocking, idempotent"]
    GATE{"enabled and date ≥ cutoff?"}
    VT["video thread<br/>_run_video_generation"]
    AT["audio thread<br/>_run_audio_generation"]
  end

  subgraph PIPE["News Pipeline — GenBox/newsvideo"]
    direction TB
    PROD["Producer Agent — producer_agent.py<br/>OpenAI Agents SDK → JSON shot list"]
    BV["build_news_video — pipeline.py<br/>anchor lead · report · interview · sign-off"]
    SC["Sora client — sora_client.py<br/>round-robin pool · per-job affinity<br/>create / remix / poll / download"]
    MX["ffmpeg — mux.py (imageio-ffmpeg)<br/>b-roll last-frame chain + concat"]
    BA["build_news_audio — tts_client.py"]
    TC["synthesize_speech<br/>government-spokesperson voice"]
  end

  subgraph STORE["Azure Storage — GenBox/azurestorage.py"]
    TBL[("Table pocstvhistory<br/>assistant: daily decisions<br/>video: status + urls<br/>video_lock / audio_lock")]
    BLB[("Blob genbox-video<br/>merged MP4 + narration MP3")]
  end

  subgraph AZ["Azure OpenAI + LangSmith"]
    CHAT["Chat model<br/>decisions + producer"]
    SORAEP["Sora 2 deployments<br/>resource pool"]
    TTSEP["TTS deployment<br/>same pool"]
    LSM["LangSmith tracing<br/>wrap_openai + @traceable"]
  end

  %% frontend <-> api
  TVUI --> RP
  TVTXT --> RS
  TVPOLL --> RST
  TVPV --> RV
  TVPA --> RA

  %% decision
  RS --> GLLM
  GLLM <-->|"read history / save decision"| TBL
  GLLM -->|"chat completion (if not cached)"| CHAT
  GLLM -->|decision text| RS

  %% orchestration
  RST --> ENS --> GATE
  GATE -->|no| RST
  GATE -->|"yes: video needed"| VT
  GATE -->|"yes: tts + not ready"| AT
  ENS <-->|"status + single-flight locks"| TBL
  ENS -->|status JSON| RST

  %% video thread (slow)
  VT --> BV
  BV --> PROD -->|LLM| CHAT
  BV -->|per shot| SC -->|REST jobs| SORAEP
  BV -->|chain + merge| MX
  BV -->|upload MP4| BLB
  VT -->|"status=ready, video_url"| TBL

  %% audio thread (fast)
  AT --> BA --> TC -->|/audio/speech| TTSEP
  BA -->|upload MP3| BLB
  AT -->|"audio_status=ready, audio_url"| TBL

  %% proxies read blob
  RV -->|stream bytes| BLB
  RA -->|stream bytes| BLB

  %% tracing
  GLLM -.-> LSM
  PROD -.-> LSM
  BV -.-> LSM
  SC -.-> LSM
  BA -.-> LSM
  TC -.-> LSM
```

## Runtime sequence

```mermaid
sequenceDiagram
    actor U as Viewer
    participant TV as TV (tv.html)
    participant API as Flask (main.py)
    participant GEN as Orchestrator (video.py)
    participant T as Azure Table
    participant B as Azure Blob
    participant AOAI as Azure OpenAI

    U->>TV: open /genbox
    TV->>API: GET /get-string?date
    API->>T: read history
    API->>AOAI: chat → decision (if not cached)
    API->>T: save decision
    API-->>TV: decision text → scrolling text

    loop poll every 4s
        TV->>API: GET /genbox-video-status?date
        API->>GEN: ensure_generation_started
        GEN->>T: read video meta
        alt eligible and not yet generated
            GEN->>T: acquire locks, set generating
            GEN-)AOAI: TTS narration (background, fast)
            AOAI--)B: upload narration MP3
            GEN->>T: audio_status=ready, audio_url
            GEN-)AOAI: Producer + Sora clips (background, slow)
            AOAI--)B: upload merged MP4
            GEN->>T: status=ready, video_url
        end
        API-->>TV: status, video_url, audio_status, audio_url
    end

    Note over TV: narration ready → play over scroll (synced)
    TV->>API: GET /genbox-audio?date (Range)
    API->>B: stream MP3
    Note over TV: video ready → switch to bulletin
    TV->>API: GET /genbox-video?date (Range)
    API->>B: stream MP4
    Note over TV: loop — video ends → text+narration → video
```

## Per-clip consistency (Sora)

```mermaid
flowchart LR
  SL["Shot list"] --> Q{"shot type?"}
  Q -->|"anchor / reporter / interview<br/>first time"| C["create_clip<br/>(fresh, round-robin resource)"]
  Q -->|"same speaker again"| RX["remix_clip<br/>(reuse that speaker's base clip,<br/>same resource = job affinity)"]
  Q -->|"b-roll"| BR["create_clip + frame chain<br/>(last frame → next first frame)"]
  C --> KEEP["remember base job id<br/>per speaker"]
  KEEP --> RX
  C --> MERGE["concat → MP4"]
  RX --> MERGE
  BR --> MERGE
```

### Notes
- **Decision vs. media gates:** text decisions are always generated; **video + narration are gated** to `date ≥ GENBOX_VIDEO_CUTOFF_DATE` and cached per date.
- **Non-blocking:** the page shows scrolling text immediately; video/narration are produced in **background threads** guarded by **Azure Table single-flight locks**, so any gunicorn worker can serve status.
- **Audio is decoupled from video** — narration backfills dates that already have (or are missing) a video.
- **Consistency:** Sora 2 has no seed and rejects faces in `input_reference`, so each speaker's first clip is a fresh **create** and later clips are **remixes** of it (b-roll uses last-frame chaining).
- **Same pool for everything:** decisions/producer use the chat deployment; video and TTS share the **Sora resource pool** (round-robin, per-job affinity).
