import hashlib
import json
import os
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.data.tables import TableServiceClient, UpdateMode
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

from utils import get_flat_date_full

load_dotenv()

connection_string = os.getenv("connection_string")
container_name = os.getenv("aiops_blob_name")
table_name = os.getenv("aiops_table_name")
problem_table_name = os.getenv("aiops_problem_table_name")

if not connection_string:
    raise ValueError("Azure Storage connection string (connection_string) is not configured.")

if not container_name:
    raise ValueError("Azure Storage container name (aiops_blob_name) is not configured.")

if not table_name:
    raise ValueError("Azure Table name (aiops_table_name) is not configured.")

if not problem_table_name:
    raise ValueError("Azure problem catalog table name (aiops_problem_table_name) is not configured.")

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
table_service_client = TableServiceClient.from_connection_string(conn_str=connection_string)
try:
    table_service_client.create_table(table_name=table_name)
except Exception:
    pass
table_client = table_service_client.get_table_client(table_name)

try:
    table_service_client.create_table(table_name=problem_table_name)
except Exception:
    pass
problems_table_client = table_service_client.get_table_client(problem_table_name)
container_client = blob_service_client.get_container_client(container_name)


def _ensure_container() -> None:
    try:
        container_client.create_container()
    except Exception:
        # Container likely exists already
        pass


def _problem_partition(problem: str) -> str:
    normalized = " ".join(problem.strip().lower().split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _serialize_metadata(metadata: Optional[Dict[str, Any]]) -> str:
    if not metadata:
        return "{}"
    try:
        return json.dumps(metadata)
    except (TypeError, ValueError):
        return "{}"


def _deserialize_metadata(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
    return {}


def upload_bytes_to_blob(data: bytes, suffix: str = ".json") -> str:
    _ensure_container()
    blob_name = f"{get_flat_date_full()}_{hashlib.sha1(data).hexdigest()}{suffix}"
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(BytesIO(data), overwrite=True)
    return blob_client.url


def save_iteration(
    *,
    problem: str,
    rowkey: str,
    html_content: str,
    summary: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    partition_key = _problem_partition(problem)
    entity = {
        "PartitionKey": partition_key,
        "RowKey": rowkey,
        "problem": problem,
        "summary": summary,
        "html_content": html_content,
        "metadata": _serialize_metadata(metadata),
        "created_at": datetime.utcnow().isoformat(),
    }
    table_client.upsert_entity(entity=entity, mode=UpdateMode.MERGE)
    #register_problem(problem)


def parse_iteration_rowkey(rowkey: str) -> Optional[datetime]:
    """The datetime encoded in an iteration RowKey, across every format ever written
    (`YYYYMMDD_HHMMSS` today, plus the older `YYYYMMDD_HH` / `YYYYMMDD`)."""
    for fmt in ("%Y%m%d_%H%M%S", "%Y%m%d_%H", "%Y%m%d"):
        try:
            return datetime.strptime(rowkey or "", fmt)
        except ValueError:
            continue
    return None


def _iteration_sort_key(rowkey: str) -> Tuple[datetime, str]:
    return (parse_iteration_rowkey(rowkey) or datetime.min, rowkey or "")


def iteration_date_key(rowkey: str) -> str:
    """The calendar day an iteration belongs to (`YYYYMMDD`) — the notebook shows one
    entry per day, so this is what iterations are deduplicated on."""
    parsed = parse_iteration_rowkey(rowkey)
    return parsed.strftime("%Y%m%d") if parsed else (rowkey or "")


def get_iterations(problem: str) -> List[Dict[str, Any]]:
    """All iterations for a problem, newest first, **at most one per calendar day**.

    One day = one lab-notebook entry. Runs before the single-flight lock existed could
    race and write several rows for the same date (the frontend triggers generation from
    a blocking request, and there are 4 gunicorn workers), so dates with more than one
    row survive in storage. Collapsing them here — keeping the latest run of that day,
    which is the one that saw all the earlier context — repairs the timeline for reads
    without destroying history."""
    partition_key = _problem_partition(problem)
    query_filter = f"PartitionKey eq '{partition_key}'"
    entities = table_client.query_entities(query_filter=query_filter, results_per_page=1000)

    by_day: Dict[str, Dict[str, Any]] = {}
    for entity in entities:
        row = dict(entity)
        day = iteration_date_key(row.get("RowKey", ""))
        current = by_day.get(day)
        if current is None or _iteration_sort_key(row.get("RowKey", "")) > _iteration_sort_key(current.get("RowKey", "")):
            by_day[day] = row

    materialized = list(by_day.values())
    materialized.sort(key=lambda row: _iteration_sort_key(row.get("RowKey", "")), reverse=True)
    return materialized


def rowkey_for_date(problem: str, flat_date: str) -> Optional[str]:
    """The RowKey already used by this problem's iteration on `flat_date` (YYYYMMDD), or
    None. Saving under it keeps a re-run from adding a second entry for the same day."""
    partition_key = _problem_partition(problem)
    try:
        entities = table_client.query_entities(
            query_filter=f"PartitionKey eq '{partition_key}' and RowKey ge '{flat_date}' and RowKey lt '{flat_date}~'",
            select=["RowKey"],
            results_per_page=100,
        )
        keys = [e["RowKey"] for e in entities
                if e.get("RowKey") and iteration_date_key(e["RowKey"]) == flat_date]
    except Exception as exc:
        print(f"Error looking up today's iteration rowkey: {exc}")
        return None
    return max(keys, key=_iteration_sort_key) if keys else None


def get_iteration_slice(problem: str, offset: int, limit: int) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    history = get_iterations(problem)
    slice_end = offset + limit
    window = history[offset:slice_end]
    next_offset = slice_end if slice_end < len(history) else None
    return window, next_offset


def latest_iteration(problem: str) -> Optional[Dict[str, Any]]:
    entries = get_iterations(problem)
    return entries[0] if entries else None


# ---------------------------------------------------------------------------
# Single-flight lock + research memory.
# Both live in the iterations table under dedicated PartitionKeys, so they are
# invisible to `get_iterations` (which filters on the problem's own partition).
# ---------------------------------------------------------------------------

# An iteration is a multi-minute agent run, so the lock has to outlive it; a lock older
# than this is treated as abandoned (its worker died) and may be reclaimed.
ITERATION_LOCK_TTL = int(os.getenv("AIOPS_ITERATION_LOCK_TTL", "5400"))


def try_acquire_iteration_lock(problem: str) -> bool:
    """Cross-worker single flight for `_run_iteration`. Without it, every concurrent
    visitor (4 gunicorn workers, and the frontend triggers generation from a blocking
    request that takes minutes) starts its own run of the same day."""
    partition_key, row_key = "iteration_lock", _problem_partition(problem)
    entity = {
        "PartitionKey": partition_key,
        "RowKey": row_key,
        "problem": problem,
        "started_at": datetime.utcnow().isoformat(),
    }
    try:
        table_client.create_entity(entity=entity)
        return True
    except ResourceExistsError:
        try:
            existing = table_client.get_entity(partition_key, row_key)
            age = (datetime.utcnow() - datetime.fromisoformat(existing["started_at"])).total_seconds()
        except Exception:
            return False
        if age > ITERATION_LOCK_TTL:
            try:
                table_client.delete_entity(partition_key, row_key)
                table_client.create_entity(entity=entity)
                return True
            except Exception:
                return False
        return False


def release_iteration_lock(problem: str) -> None:
    try:
        table_client.delete_entity("iteration_lock", _problem_partition(problem))
    except Exception:
        pass


def load_memory_record(problem: str) -> Dict[str, Any]:
    """The problem's accumulated research memory (see AIOpenProblemSolver/memory.py)."""
    try:
        entity = table_client.get_entity("memory", _problem_partition(problem))
    except Exception:
        return {}
    return _deserialize_metadata(entity.get("payload"))


def save_memory_record(problem: str, payload: Dict[str, Any]) -> None:
    table_client.upsert_entity(
        entity={
            "PartitionKey": "memory",
            "RowKey": _problem_partition(problem),
            "problem": problem,
            "payload": _serialize_metadata(payload),
            "updated_at": datetime.utcnow().isoformat(),
        },
        mode=UpdateMode.REPLACE,
    )


def list_problems(max_entities: int = 2000) -> List[Dict[str, Optional[str]]]:
    problems: Dict[str, Dict[str, Optional[str]]] = {}
    try:
        count = 0
        for entity in problems_table_client.list_entities(results_per_page=1000):
            problem = entity.get("problem")
            if not problem:
                continue
            normalized = problem.strip()
            if normalized not in problems:
                problems[normalized] = {
                    "name": normalized,
                    "description": entity.get("description"),
                }
            count += 1
            if max_entities and count >= max_entities:
                break
    except Exception as exc:
        print(f"Error retrieving problems: {exc}")
        return []
    return sorted(problems.values(), key=lambda entry: entry["name"].lower())


def get_problem_details(problem: str) -> Optional[Dict[str, Any]]:
    if not problem:
        return None
    partition_key = "catalog"
    row_key = _problem_partition(problem)
    try:
        entity = problems_table_client.get_entity(partition_key=partition_key, row_key=row_key)
    except ResourceNotFoundError:
        return None
    return dict(entity)


def get_problem_progress(problem: str) -> Dict[str, Optional[Any]]:
    history = get_iterations(problem)
    if not history:
        return {"progress_percent": None, "progress_comment": ""}
    entry = history[0]   # newest — `get_iterations` returns newest first

    metadata = _deserialize_metadata(entry.get("metadata"))
    progress_percent = metadata.get("progress_percent")
    try:
        progress_percent = float(progress_percent)
    except (TypeError, ValueError):
        progress_percent = None
    if progress_percent is not None:
        progress_percent = max(0.0, min(100.0, progress_percent))

    progress_comment = metadata.get("progress_comment", "")
    if not isinstance(progress_comment, str):
        progress_comment = str(progress_comment or "")
    progress_comment = progress_comment.strip()

    return {
        "progress_percent": progress_percent,
        "progress_comment": progress_comment,
    }


def register_problem(problem: str, description: Optional[str] = None) -> None:
    raise NotImplementedError("Problem registration must be handled manually.")
