# trAIde Dashboard — Architecture

A **read-only** spectator dashboard for the *trAIde* autonomous crypto trading bot (a separate
repo). trAIde runs three agents — **Trading**, **Research**, and **Supervisor** — and continuously
publishes a *sanitized, public-safe* projection of their activity to Azure Storage. This SandBox
project reads that projection and renders it. **No private information** (account IDs, balances,
total equity, position sizes, API keys) is ever stored or shown. In the default `normalized`
disclosure mode, no dollar figure exists in the data at all — money is an indexed return curve
(starting at 100) plus percentages.

## Data flow

```mermaid
flowchart LR
  subgraph trAIde repo (producer)
    A[Trading / Research / Supervisor agents] --> M[MemoryStore<br/>.agent_memory.json<br/>~14 day retention]
    M --> P[DashboardPublisher<br/>sanitize + normalize<br/>throttled, idempotent]
  end

  P -- upsert --> T[(Azure Table<br/>equity / decision / trade / meta)]
  P -- overwrite --> B[(Azure Blob<br/>live.json + rollups/*.json)]

  subgraph SandBox repo (consumer, this project)
    AZ[TrAIde/azurestorage.py<br/>read-only] --> R[Flask routes /traide/*]
    R --> H[templates/traide.html<br/>ECharts neon dashboard]
  end

  T --> AZ
  B --> AZ
  H -- fetch every ~45s --> R
```

Azure is the **durable system of record**: trAIde's local memory is pruned to ~14 days, but these
rows are written before pruning and never deleted, so daily / weekly / monthly / all-time history
accumulates indefinitely. The equity curve is built incrementally — only *today's* row is rewritten
each publish; once a UTC day rolls over its row is immutable — so the all-time curve survives the
local prune.

## Storage layout (written by `trAIde/src/dashboard_publisher.py`)

| Store | Name / keys | Contents |
|---|---|---|
| Blob | `live.json` | Full current snapshot: KPIs, positions, decision feed, equity tail, notes, research |
| Blob | `rollups/{daily,weekly,monthly,alltime}.json` | Pre-bucketed equity series + KPIs |
| Table | PK `equity`, RK `{day:08d}` | `indexClose`, `drawdownPct`, optional `dayRealizedPnl` |
| Table | PK `decision`, RK `{ts:010d}-{symbol}` | `data` = JSON of one sanitized decision |
| Table | PK `trade`, RK `{day:08d}-{ts}-{symbol}-{action}` | `data` = JSON of one closed-trade outcome |
| Table | PK `meta`, RK `state` | `generatedTs`, `schema`, `disclosure`, `indexAnchor` |

All table writes are idempotent upserts with deterministic RowKeys, so repeated publishes never
duplicate. Tables are the durable accumulator; blobs are cheap, rebuildable projections.

## Routes (`main.py`)

| Route | Returns |
|---|---|
| `GET /traide` | The dashboard HTML shell |
| `GET /traide/live` | `live.json` snapshot (Referer-guarded) |
| `GET /traide/equity?period=day\|week\|month\|all` | Ascending equity points for the period |
| `GET /traide/feed?limit=` | Recent decisions, newest first |
| `GET /traide/trades?limit=` | Recent closed-trade outcomes, newest first |

## Configuration

Reads the shared `connection_string` (Azure account `pkrstr`) plus `traide_table_name` and
`traide_blob_name` — set to the same values trAIde publishes to (`traidedashboard`,
`traide-dashboard`). All reads are server-side, so the connection string never reaches the browser
(no CORS, no SAS needed). If unset, `azurestorage.py` degrades to empty results and the page shows
an "Agents warming up…" empty state.
