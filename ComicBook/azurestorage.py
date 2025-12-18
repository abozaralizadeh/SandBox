import os
import uuid
from datetime import datetime
from typing import List, Optional
from io import BytesIO

import requests

from azure.data.tables import TableServiceClient, UpdateMode
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

from utils import get_flat_date, get_flat_date_full

load_dotenv()

connection_string = os.getenv("connection_string")
container_name = os.getenv("comicbook_blob_name", "comicbook-html")
episodes_table_name = os.getenv("comicbook_table_name", "comicbook")
arcs_table_name = os.getenv("comicbook_arc_table_name", "comicbookarcs")

if not connection_string:
    raise EnvironmentError("Missing 'connection_string' in environment for ComicBook storage.")

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

table_service_client = TableServiceClient.from_connection_string(conn_str=connection_string)
episodes_table = table_service_client.get_table_client(episodes_table_name)
arcs_table = table_service_client.get_table_client(arcs_table_name)

MAX_TABLE_PROPERTY_CHARS = 32000


def _ensure_container():
    try:
        container_client.create_container()
    except Exception:
        pass


def _ensure_table(client):
    try:
        client.create_table()
    except Exception:
        pass


_ensure_container()
_ensure_table(episodes_table)
_ensure_table(arcs_table)


def save_photo_to_blob(photo_url: str) -> str:
    """Download an image and upload it to the ComicBook blob container."""
    response = requests.get(photo_url)
    response.raise_for_status()
    _ensure_container()
    blob_name = f"{get_flat_date_full()}_{uuid.uuid4()}.png"
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(BytesIO(response.content), overwrite=True)
    return blob_client.url


def upload_image_bytes_to_blob(image_bytes: bytes) -> str:
    """Upload raw image bytes to blob storage and return the blob URL."""
    _ensure_container()
    blob_name = f"{get_flat_date_full()}_{uuid.uuid4()}.png"
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(BytesIO(image_bytes), overwrite=True)
    return blob_client.url


def _should_offload_to_blob(html_content: str) -> bool:
    return bool(html_content) and len(html_content) > MAX_TABLE_PROPERTY_CHARS


def _attach_html_payload(entity: dict, html_content: str) -> dict:
    if html_content is None:
        html_content = ""
    try:
        if _should_offload_to_blob(html_content):
            blob_name = upload_html_to_blob(html_content)
            entity["html_content"] = ""
            entity["html_blob_name"] = blob_name
        else:
            entity["html_content"] = html_content
            entity["html_blob_name"] = ""
    except Exception as blob_error:
        print(f"[ComicBook] Failed to upload HTML to blob, trimming payload: {blob_error}")
        entity["html_content"] = html_content[:MAX_TABLE_PROPERTY_CHARS]
        entity["html_blob_name"] = ""
    return entity


def upload_html_to_blob(html_content: str) -> str:
    _ensure_container()
    blob_name = f"{get_flat_date_full()}_{uuid.uuid4()}.html"
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(html_content.encode("utf-8"), overwrite=True)
    return blob_name


def download_html_from_blob(identifier: str) -> str:
    blob_client = container_client.get_blob_client(identifier)
    downloader = blob_client.download_blob()
    return downloader.readall().decode("utf-8")


def _hydrate_html_content(entity: dict) -> dict:
    blob_name = entity.get("html_blob_name")
    if blob_name:
        try:
            entity["html_content"] = download_html_from_blob(blob_name)
        except Exception as exc:
            print(f"[ComicBook] Unable to fetch blob '{blob_name}': {exc}")
    return entity


def _sort_by_rowkey(entities: list) -> list:
    return sorted(entities, key=lambda x: x.get("RowKey", ""), reverse=True)


def start_new_arc(title: str, logline: str, target_days: int, start_date: Optional[datetime] = None) -> dict:
    start_date = start_date or datetime.utcnow()
    arc_id = f"arc_{get_flat_date(start_date)}_{uuid.uuid4().hex[:6]}"
    entity = {
        "PartitionKey": "arc",
        "RowKey": arc_id,
        "title": title,
        "logline": logline,
        "start_date": start_date.isoformat(),
        "status": "active",
        "target_days": target_days,
        "episodes_count": 0,
        "last_episode_date": "",
    }
    arcs_table.create_entity(entity=entity)
    return entity


def close_arc(arc_id: str, end_date: Optional[datetime] = None):
    end_date = end_date or datetime.utcnow()
    entity = {
        "PartitionKey": "arc",
        "RowKey": arc_id,
        "status": "closed",
        "end_date": end_date.isoformat(),
    }
    arcs_table.upsert_entity(entity=entity, mode=UpdateMode.MERGE)


def get_latest_arc() -> Optional[dict]:
    arcs = list(arcs_table.query_entities("PartitionKey eq 'arc'", results_per_page=100))
    if not arcs:
        return None
    arcs = sorted(arcs, key=lambda x: x.get("start_date", ""), reverse=True)
    return arcs[0]


def ensure_active_arc(target_date: Optional[datetime] = None, min_days: int = 7, max_days: int = 10) -> dict:
    target_date = target_date or datetime.utcnow()
    arc = get_latest_arc()
    if arc and arc.get("status") == "active":
        started = datetime.fromisoformat(arc["start_date"])
        episode_count = int(arc.get("episodes_count", 0))
        target_span = int(arc.get("target_days", max_days))
        elapsed_days = (target_date.date() - started.date()).days
        if elapsed_days < target_span and episode_count < target_span:
            return arc
        close_arc(arc["RowKey"], end_date=target_date)
    import random

    target = random.randint(min_days, max_days)
    logline = "A fresh comic adventure led by AI characters exploring daily twists."
    title = f"Arc starting {target_date.strftime('%Y-%m-%d')}"
    return start_new_arc(title=title, logline=logline, target_days=target, start_date=target_date)


def _count_episodes(arc_id: str) -> int:
    entities = list(episodes_table.query_entities(f"PartitionKey eq '{arc_id}'", results_per_page=200))
    return len(entities)


def get_recent_episodes(arc_id: str, limit: int = 3) -> List[dict]:
    episodes = list(episodes_table.query_entities(f"PartitionKey eq '{arc_id}'", results_per_page=200))
    episodes = _sort_by_rowkey(episodes)
    hydrated = [_hydrate_html_content(e) for e in episodes[:limit]]
    return hydrated


def get_episode_by_date(date_key: str) -> Optional[dict]:
    try:
        entities = list(episodes_table.query_entities(f"RowKey eq '{date_key}'", results_per_page=5))
        if not entities:
            return None
        entity = _hydrate_html_content(_sort_by_rowkey(entities)[0])
        return entity
    except Exception:
        return None


def save_episode(
    arc: dict,
    episode_date: datetime,
    html_content: str,
    storyboard_summary: str,
    panel_notes: str,
) -> dict:
    arc_id = arc["RowKey"]
    episode_number = _count_episodes(arc_id) + 1
    row_key = get_flat_date(episode_date)
    entity = {
        "PartitionKey": arc_id,
        "RowKey": row_key,
        "arc_title": arc.get("title", ""),
        "episode_number": episode_number,
        "story_summary": storyboard_summary,
        "panel_notes": panel_notes,
    }
    entity = _attach_html_payload(entity, html_content)
    episodes_table.upsert_entity(entity=entity, mode=UpdateMode.REPLACE)
    arcs_table.upsert_entity(
        entity={
            "PartitionKey": "arc",
            "RowKey": arc_id,
            "episodes_count": episode_number,
            "last_episode_date": episode_date.isoformat(),
            "last_story_summary": storyboard_summary[:MAX_TABLE_PROPERTY_CHARS],
        },
        mode=UpdateMode.MERGE,
    )
    return entity
