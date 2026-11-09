"""Read-only access to the trAIde public dashboard data in Azure Blob + Table.

The PRODUCER lives in the separate trAIde repo and writes a sanitized, public-safe projection
of its trading agents' activity here. This module only ever READS. It is intentionally guarded:
if the storage account / container / table is not configured, every function degrades to an
empty result instead of raising, so the rest of the SandBox Flask app keeps working.

Data contract (written by trAIde/src/dashboard_publisher.py):
  Blob  live.json                     -> full current snapshot (KPIs, positions, feed, notes, ...)
  Blob  rollups/{period}.json         -> pre-bucketed equity series for daily|weekly|monthly|alltime
  Table PK 'equity'   RK {day:08d}    -> {indexClose, drawdownPct, [dayRealizedPnl]}
  Table PK 'decision' RK {ts}-{sym}   -> {data: <json decision>}
  Table PK 'trade'    RK {day}-...    -> {data: <json closed trade>}
  Table PK 'meta'     RK 'state'      -> {generatedTs, schema, disclosure, indexAnchor}
"""

import functools
import json
import threading
import time
import os

from dotenv import load_dotenv

load_dotenv()

connection_string = os.getenv("connection_string")
container_name = os.getenv("traide_blob_name", "traide-dashboard")
table_name = os.getenv("traide_table_name", "traidedashboard")

PK_EQUITY = "equity"

# Equity-index sanity band, mirroring the publisher's own guard in trAIde's dashboard_publisher.
# The index compounds daily onto the previous close, so ONE bad point is permanent and every later
# point carries it. The writer already refuses to publish such points, but this reader queries the
# durable table DIRECTLY — and Azure history is never rewritten — so the corrupt rows are still there
# and were being served straight to the chart. Live on 2026-09-10: days 20693-20695 held ~7.27e7
# against a base of 100, which pinned the y-axis at [0, 8e7] and flattened four months of real
# history into a line along zero with one spike. Filtering belongs in BOTH the writer and the reader:
# a guard that only exists on the write path cannot clean data that predates it.
INDEX_BASE = 100.0
INDEX_SANITY_FACTOR = 1000.0     # a value outside base/1000 .. base*1000 is corruption, not return
PK_DECISION = "decision"
PK_TRADE = "trade"
PK_PLAN = "plan"
PK_META = "meta"

_VALID_PERIODS = {"daily", "weekly", "monthly", "alltime"}

# Lazy, guarded client init — a misconfigured env must not break the whole app on import.
_container_client = None
_table_client = None

try:
    if connection_string and container_name and table_name:
        from azure.storage.blob import BlobServiceClient
        from azure.data.tables import TableServiceClient

        _blob_service = BlobServiceClient.from_connection_string(connection_string)
        _container_client = _blob_service.get_container_client(container_name)

        _table_service = TableServiceClient.from_connection_string(conn_str=connection_string)
        _table_client = _table_service.get_table_client(table_name)
except Exception as exc:  # pragma: no cover - defensive
    print(f"[TrAIde] Azure init skipped: {exc}")
    _container_client = None
    _table_client = None


# ---------------------------------------------------------------------------
# Read-through cache
# ---------------------------------------------------------------------------
# The dashboard polls /traide/live every 30s, /traide/equity and /traide/trades every 60s and
# /traide/plans every 90s, and it does that PER OPEN TAB. The storage account is in another
# region from the web app, so every one of those reads is billed egress. Caching here rather
# than in the routes means any number of tabs (and any other caller) costs one read per TTL.
#
# Each of the 4 gunicorn workers keeps its own cache, so the real ceiling is 4 reads per TTL,
# not 1. That is still far below one read per tab per poll.
#
# The published data only changes when the bot republishes live.json, so a few tens of seconds
# of staleness is invisible: the page already shows a "generated at" timestamp from the payload.
_CACHE_TTL = int(os.getenv("TRAIDE_CACHE_TTL", "45"))
# The decision feed is not fetched by the page at all, so nothing is waiting on it being fresh.
_FEED_CACHE_TTL = int(os.getenv("TRAIDE_FEED_CACHE_TTL", "300"))

_cache: dict = {}
_cache_lock = threading.Lock()


def _cached(ttl_getter):
    """Memoize a reader on its arguments for `ttl_getter()` seconds.

    Results are handed out shared, not copied: every caller in this app passes them straight
    to jsonify and none of them mutate. Keep it that way, or copy at the call site.
    """
    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = (fn.__name__, args, tuple(sorted(kwargs.items())))
            now = time.time()
            with _cache_lock:
                hit = _cache.get(key)
                if hit is not None and hit[0] > now:
                    return hit[1]
            value = fn(*args, **kwargs)
            with _cache_lock:
                _cache[key] = (now + ttl_getter(), value)
                if len(_cache) > 256:          # bound it; keys vary with limit/start_day
                    for k, v in [(k, v) for k, v in _cache.items() if v[0] <= now]:
                        _cache.pop(k, None)
            return value
        wrapper.cache_clear = lambda: _cache.clear()
        return wrapper
    return decorate


def is_configured() -> bool:
    return _container_client is not None and _table_client is not None


def _download_json(blob_name: str):
    if _container_client is None:
        return None
    try:
        data = _container_client.get_blob_client(blob_name).download_blob().readall()
        return json.loads(data)
    except Exception:
        return None


@_cached(lambda: _CACHE_TTL)
def get_live_snapshot() -> dict:
    """The whole current snapshot (fast path). Returns {} when nothing has been published yet."""
    return _download_json("live.json") or {}


@_cached(lambda: _CACHE_TTL)
def get_rollup(period: str) -> dict:
    """Pre-bucketed series for one of daily|weekly|monthly|alltime. {} on miss."""
    if period not in _VALID_PERIODS:
        return {}
    return _download_json(f"rollups/{period}.json") or {}


@_cached(lambda: _CACHE_TTL)
def get_equity_series(start_day=None, end_day=None) -> list:
    """Durable daily index series, ascending by day. Range-bounded by zero-padded day RowKeys."""
    if _table_client is None:
        return []
    try:
        filt = f"PartitionKey eq '{PK_EQUITY}'"
        if start_day is not None:
            filt += f" and RowKey ge '{int(start_day):08d}'"
        if end_day is not None:
            filt += f" and RowKey le '{int(end_day):08d}'"
        rows = _table_client.query_entities(query_filter=filt, results_per_page=1000)
        out = []
        dropped = 0
        _lo, _hi = INDEX_BASE / INDEX_SANITY_FACTOR, INDEX_BASE * INDEX_SANITY_FACTOR
        for r in rows:
            try:
                day = int(r["RowKey"])
            except (KeyError, ValueError):
                continue
            close = r.get("indexClose")
            if close is not None:
                try:
                    close_f = float(close)
                except (TypeError, ValueError):
                    dropped += 1
                    continue
                if not (_lo <= close_f <= _hi):
                    dropped += 1
                    continue
            point = {
                "day": day,
                "indexClose": close,
                "drawdownPct": r.get("drawdownPct"),
            }
            if "dayRealizedPnl" in r:
                point["dayRealizedPnl"] = r.get("dayRealizedPnl")
            out.append(point)
        out.sort(key=lambda p: p["day"])
        if dropped:
            print(f"[TrAIde] equity series: hid {dropped} corrupt point(s) outside "
                  f"{_lo:g}..{_hi:g}; the underlying rows remain in the durable table.")
        return out
    except Exception as exc:
        print(f"[TrAIde] get_equity_series error: {exc}")
        return []


# These partitions only ever grow (44k decisions, 18k plans), and the page polls them every
# 60-90 seconds, so reading the whole partition to show the newest 30-40 rows was the second
# largest reader in the app after AIOPS. RowKeys are time-ordered and fixed width, so a
# `RowKey ge` bound is a key range the service can satisfy without handing us the history.
# If the window turns out too narrow the query is simply repeated unbounded, so what the page
# shows never changes, only what it costs to fetch.
# Windows to try for the decision feed, narrowest first. Measured on the live table: 3 days
# holds ~870 decisions, comfortably more than the route's max limit of 100, while the whole
# partition is 44k rows / 24 MB. The wider steps exist because activity is bursty: if the bot
# goes quiet the narrow window would come up short, and widening by a step is far cheaper than
# falling straight back to the whole partition.
FEED_WINDOW_DAYS = [int(d) for d in
                    os.getenv("TRAIDE_FEED_WINDOW_DAYS", "3,21,90").split(",") if d.strip()]


def _query_data_partition(partition_key: str, limit: int, rowkey_floor: str = None,
                          wider_floors: list = None) -> list:
    if _table_client is None:
        return []
    try:
        filt = f"PartitionKey eq '{partition_key}'"
        if rowkey_floor:
            filt += f" and RowKey ge '{rowkey_floor}'"
        rows = _table_client.query_entities(
            query_filter=filt, results_per_page=1000
        )
        items = []
        for r in rows:
            raw = r.get("data")
            if not raw:
                continue
            try:
                items.append(json.loads(raw))
            except (TypeError, ValueError):
                continue
        items.sort(key=lambda d: d.get("ts", 0), reverse=True)
        if rowkey_floor and len(items) < max(1, int(limit)):
            # The window held fewer rows than asked for. Widen a step, and only read the whole
            # partition once the wider windows are exhausted. The reader never sees a shorter
            # list than before, it just costs less to find it.
            if wider_floors:
                return _query_data_partition(partition_key, limit,
                                             rowkey_floor=wider_floors[0],
                                             wider_floors=wider_floors[1:])
            return _query_data_partition(partition_key, limit)
        return items[: max(1, int(limit))]
    except Exception as exc:
        print(f"[TrAIde] query '{partition_key}' error: {exc}")
        return []


def _epoch_floor(days: int) -> str:
    """RowKey floor for partitions keyed `{epoch_seconds}-...` (decision, plan)."""
    return str(int(time.time()) - max(1, days) * 86400)


@_cached(lambda: _FEED_CACHE_TTL)
def get_decision_feed(limit: int = 30) -> list:
    """Most recent agent decisions, newest first."""
    floors = [_epoch_floor(d) for d in FEED_WINDOW_DAYS] or [None]
    return _query_data_partition(PK_DECISION, limit,
                                 rowkey_floor=floors[0], wider_floors=floors[1:])


@_cached(lambda: _CACHE_TTL)
def get_closed_trades(limit: int = 100) -> list:
    """Most recent closed-trade outcomes, newest first.

    Deliberately unwindowed: the whole partition is ~410 rows / 0.17 MB, and closed trades are
    sparse enough (about 200 days to accumulate 200) that any useful window would miss the
    route's limit and cost an extra round trip for nothing."""
    return _query_data_partition(PK_TRADE, limit)


@_cached(lambda: _CACHE_TTL)
def get_plans(limit: int = 40, start_day=None) -> list:
    """Durable research plans, newest first. Accumulates past the producer's local cap, so this
    spans several days. `start_day` (epoch day) bounds how far back to read."""
    if _table_client is None:
        return []
    try:
        filt = f"PartitionKey eq '{PK_PLAN}'"
        if start_day is not None:
            # `day` is a plain property, so that clause alone still walks the partition. The
            # RowKey bound is the one the service can seek on; both are kept so the result is
            # identical either way.
            filt += (f" and day ge {int(start_day)}"
                     f" and RowKey ge '{int(start_day) * 86400}'")
        rows = _table_client.query_entities(query_filter=filt, results_per_page=1000)
        items = []
        for r in rows:
            raw = r.get("data")
            if not raw:
                continue
            try:
                items.append(json.loads(raw))
            except (TypeError, ValueError):
                continue
        items.sort(key=lambda d: d.get("ts", 0), reverse=True)
        return items[: max(1, int(limit))]
    except Exception as exc:
        print(f"[TrAIde] get_plans error: {exc}")
        return []


@_cached(lambda: _CACHE_TTL)
def get_meta() -> dict:
    """Singleton publisher state (generatedTs, disclosure, schema). {} on miss."""
    if _table_client is None:
        return {}
    try:
        from azure.core.exceptions import ResourceNotFoundError

        try:
            entity = _table_client.get_entity(partition_key=PK_META, row_key="state")
        except ResourceNotFoundError:
            return {}
        return dict(entity)
    except Exception:
        return {}
