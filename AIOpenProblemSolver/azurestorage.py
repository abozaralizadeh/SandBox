import hashlib
import json
import os
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from azure.core.exceptions import ResourceNotFoundError
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
    register_problem(problem)


def get_iterations(problem: str) -> List[Dict[str, Any]]:
    partition_key = _problem_partition(problem)
    query_filter = f"PartitionKey eq '{partition_key}'"
    entities = table_client.query_entities(query_filter=query_filter, results_per_page=1000)

    materialized: List[Dict[str, Any]] = []
    for entity in entities:
        materialized.append(dict(entity))

    materialized.sort(key=lambda row: row["RowKey"], reverse=True)
    return materialized


def get_iteration_slice(problem: str, offset: int, limit: int) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    history = get_iterations(problem)
    slice_end = offset + limit
    window = history[offset:slice_end]
    next_offset = slice_end if slice_end < len(history) else None
    return window, next_offset


def latest_iteration(problem: str) -> Optional[Dict[str, Any]]:
    entries = get_iterations(problem)
    return entries[0] if entries else None


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


def register_problem(problem: str, description: Optional[str] = None) -> None:
    raise NotImplementedError("Problem registration must be handled manually.")
