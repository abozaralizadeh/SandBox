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
| Blob | `live.json` | Full current snapshot: KPIs, **strategy edge**, **taker flow**, positions, pending orders, coin universe, recently-closed positions, decision feed, equity tail, notes, research |
| Blob | `rollups/{daily,weekly,monthly,alltime}.json` | Pre-bucketed equity series + KPIs |
| Table | PK `equity`, RK `{day:08d}` | `indexClose`, `drawdownPct`, optional `dayRealizedPnl` |
| Table | PK `decision`, RK `{ts:010d}-{symbol}` | `data` = JSON of one sanitized decision |
| Table | PK `trade`, RK `{day:08d}-{ts}-{symbol}-{action}` | `data` = JSON of one closed-trade outcome |
| Table | PK `meta`, RK `state` | `generatedTs`, `schema`, `disclosure`, `indexAnchor` |

All table writes are idempotent upserts with deterministic RowKeys, so repeated publishes never
duplicate. Tables are the durable accumulator; blobs are cheap, rebuildable projections.

## Strategy edge (`live.json` → `strategyEdge`, rendered by the "Signal quality" panel)

Win rate and PnL answer *whether* the bot is winning, but they conflate three different things — was
the direction call right, was the fill any good, was the exit managed well — so they cannot say
**why**. `strategyEdge` measures the signal alone: forward return from the market price at the moment
of the call, signed by the traded direction, against the round-trip cost it has to clear. Each *setup
family* is scored separately and the producer allocates risk toward whichever currently pays, so
`familyRiskFactor` explains where capital is going rather than only reporting the result.

```jsonc
"strategyEdge": {
  "verdict": "no edge",              // edge | no edge | insufficient data
  "n": 55, "costPct": 0.12, "bestHorizon": "60m",
  "byHorizon": { "60m": {"n": 55, "mean_pct": -0.135, "hit_rate": 0.35, "net_of_cost_pct": -0.255} },
  "byFamily": {
    "continuation": {"n": 34, "mean_pct": -0.226, "hit_rate": 0.28, "verdict": "no edge"},
    "fade_extreme": {"n": 15, "mean_pct":  0.191, "hit_rate": 0.57, "verdict": "insufficient data"}
  },
  "familyRiskFactor": {"continuation": 0.5, "fade_extreme": 1.0},
  "slippagePctPerSide": 0.01, "slippageSource": "measured"
}
```

Percentages, counts and verdicts only — no balance, equity, position size or account identifier is
involved, so it is safe under the default `normalized` disclosure mode. The renderer sorts families
worst-first (the one costing money leads) and degrades to an empty state when the key is absent, so
an older producer that does not publish it still renders fine.

## Taker flow (`live.json` → `takerFlow`, rendered by the "Taker flow" panel)

Every other panel here is derived from closed candles — what price *did*. `takerFlow` is the only one
that shows **who was pushing it**: the share of taker volume lifting the offer, sampled from KuCoin's
public trade tape each poll. It has two halves because they answer different questions.

`live` is the current state of the market and moves between agent runs — `buyShare` is
volume-weighted, `buyTradeShare` is one vote per trade, and the gap between them is the large-order /
small-order split. `ageSec` is published so a stalled sampler cannot be mistaken for a calm tape.

`byHorizon` is an **experiment in progress**. The tape reading is stamped onto every direction call
and scored forward; what is shown is the forward return of calls made *with* the flow minus those
made *against* it. The spread form is deliberate — "with-flow calls returned +0.1%" says nothing if
every call returned +0.1%, so subtracting the against group cancels the book's directional bias.
Verdicts are staged: `informative` = the spread clears its own standard error; `tradable` = the
with-flow group also clears the round trip. Only the second would be worth acting on, and at a ~0.2%
round trip against a few basis points of short-horizon drift it is expected to fail. Read either
against `coverage`, which is the fraction of scored calls that carried a reading at all.

```jsonc
"takerFlow": {
  "enabled": true,
  "verdict": "no information",       // tradable | informative | no information | insufficient data
  "n": 96, "coverage": 0.41, "costPct": 0.21, "neutralBand": 0.05,
  "byHorizon": {
    "5m": {
      "with":    {"n": 34, "mean_pct":  0.041, "hit_rate": 0.56, "stderr_pct": 0.033},
      "against": {"n": 28, "mean_pct": -0.012, "hit_rate": 0.46, "stderr_pct": 0.040},
      "spread_pct": 0.053, "spread_stderr_pct": 0.052, "verdict": "informative"
    }
  },
  "live": {
    "SOL-USDT": {"buyShare": 0.74, "buyShareEwma": 0.66, "buyTradeShare": 0.59,
                 "trades": 100, "spanSec": 311.2, "ageSec": 63, "samples": 214}
  }
}
```

Shares, counts, ages and percentages only — no balance, size or account identifier is involved, so it
is safe in every disclosure mode. **Nothing in the producer's trading path reads any of this**; it is
published so the experiment can be watched while it runs rather than graded once in private. Like
`strategyEdge`, the renderer degrades to an empty state when the key is absent.

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
