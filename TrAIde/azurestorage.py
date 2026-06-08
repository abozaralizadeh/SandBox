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

import json
import os

from dotenv import load_dotenv

load_dotenv()

connection_string = os.getenv("connection_string")
container_name = os.getenv("traide_blob_name", "traide-dashboard")
table_name = os.getenv("traide_table_name", "traidedashboard")

PK_EQUITY = "equity"
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


def get_live_snapshot() -> dict:
    """The whole current snapshot (fast path). Returns {} when nothing has been published yet."""
    return _download_json("live.json") or {}


def get_rollup(period: str) -> dict:
    """Pre-bucketed series for one of daily|weekly|monthly|alltime. {} on miss."""
    if period not in _VALID_PERIODS:
        return {}
    return _download_json(f"rollups/{period}.json") or {}


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
        for r in rows:
            try:
                day = int(r["RowKey"])
            except (KeyError, ValueError):
                continue
            point = {
                "day": day,
                "indexClose": r.get("indexClose"),
                "drawdownPct": r.get("drawdownPct"),
            }
            if "dayRealizedPnl" in r:
                point["dayRealizedPnl"] = r.get("dayRealizedPnl")
            out.append(point)
        out.sort(key=lambda p: p["day"])
        return out
    except Exception as exc:
        print(f"[TrAIde] get_equity_series error: {exc}")
        return []


def _query_data_partition(partition_key: str, limit: int) -> list:
    if _table_client is None:
        return []
    try:
        rows = _table_client.query_entities(
            query_filter=f"PartitionKey eq '{partition_key}'", results_per_page=1000
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
        return items[: max(1, int(limit))]
    except Exception as exc:
        print(f"[TrAIde] query '{partition_key}' error: {exc}")
        return []


def get_decision_feed(limit: int = 30) -> list:
    """Most recent agent decisions, newest first."""
    return _query_data_partition(PK_DECISION, limit)


def get_closed_trades(limit: int = 100) -> list:
    """Most recent closed-trade outcomes, newest first."""
    return _query_data_partition(PK_TRADE, limit)


def get_plans(limit: int = 40, start_day=None) -> list:
    """Durable research plans, newest first. Accumulates past the producer's local cap, so this
    spans several days. `start_day` (epoch day) bounds how far back to read."""
    if _table_client is None:
        return []
    try:
        filt = f"PartitionKey eq '{PK_PLAN}'"
        if start_day is not None:
            filt += f" and day ge {int(start_day)}"
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
