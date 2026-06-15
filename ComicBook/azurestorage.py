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

# ---------------------------------------------------------------------------
# Debug mode — isolate local test runs from production comics.
#   DEBUG=true        → read/write a separate "arc_debug" partition (debug arcs get
#                       "debugarc_*" ids), so production comics are never read or touched.
#   DEBUG_SAVE=false  → skip ALL table/HTML persistence (pure dry run). The pipeline still
#                       runs fully in-memory and returns HTML; nothing is written. Only
#                       meaningful when DEBUG=true — production ALWAYS persists.
# Generated panel-image blobs are NOT gated, so a debug comic is still viewable; only
# arc/episode/outline/glossary records are isolated and (optionally) skipped.
# ---------------------------------------------------------------------------
DEBUG = os.getenv("DEBUG", "false").strip().lower() in ("1", "true", "yes")
DEBUG_SAVE = os.getenv("DEBUG_SAVE", "true").strip().lower() in ("1", "true", "yes")
_ARC_PARTITION = "arc_debug" if DEBUG else "arc"
_ARC_ID_PREFIX = "debugarc" if DEBUG else "arc"
_PERSIST = DEBUG_SAVE if DEBUG else True
if DEBUG:
    print(
        f"[ComicBook] DEBUG mode ON — arc partition='{_ARC_PARTITION}', "
        f"persistence={'ENABLED' if _PERSIST else 'DISABLED (dry run, nothing written)'}"
    )

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


def get_blob_container_url() -> str:
    """Public base URL of the ComicBook blob container (no trailing slash)."""
    return container_client.url.rstrip("/")


def blob_exists(blob_name: str) -> bool:
    try:
        return container_client.get_blob_client(blob_name).exists()
    except Exception:
        return False


def download_blob_bytes(blob_name: str) -> bytes:
    return container_client.get_blob_client(blob_name).download_blob().readall()


def upload_blob_bytes(blob_name: str, data: bytes, content_type: str) -> str:
    """Upload raw bytes with an explicit content-type; return the blob's public URL."""
    from azure.storage.blob import ContentSettings
    _ensure_container()
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(
        BytesIO(data),
        overwrite=True,
        content_settings=ContentSettings(content_type=content_type),
    )
    return blob_client.url


def _should_offload_to_blob(html_content: str) -> bool:
    return bool(html_content) and len(html_content) > MAX_TABLE_PROPERTY_CHARS


def _attach_html_payload(entity: dict, html_content: str, suffix: str = "") -> dict:
    if html_content is None:
        html_content = ""
    content_key = f"html_content{suffix}"
    blob_key = f"html_blob_name{suffix}"
    try:
        if _should_offload_to_blob(html_content):
            blob_name = upload_html_to_blob(html_content)
            entity[content_key] = ""
            entity[blob_key] = blob_name
        else:
            entity[content_key] = html_content
            entity[blob_key] = ""
    except Exception as blob_error:
        print(f"[ComicBook] Failed to upload HTML to blob ({suffix or 'en'}), trimming: {blob_error}")
        entity[content_key] = html_content[:MAX_TABLE_PROPERTY_CHARS]
        entity[blob_key] = ""
    return entity


def upload_html_to_blob(html_content: str) -> str:
    _ensure_container()
    blob_name = f"{get_flat_date_full()}_{uuid.uuid4()}.html"
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(html_content.encode("utf-8"), overwrite=True)
    return blob_name


def upload_text_to_blob(content: str, extension: str = ".txt") -> str:
    _ensure_container()
    blob_name = f"{get_flat_date_full()}_{uuid.uuid4()}{extension}"
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(content.encode("utf-8"), overwrite=True)
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


def _hydrate_html_for_lang(entity: dict, lang: str = "en") -> dict:
    suffix = "" if lang == "en" else f"_{lang}"
    blob_key = f"html_blob_name{suffix}"
    content_key = f"html_content{suffix}"
    blob_name = entity.get(blob_key)
    if blob_name:
        try:
            entity[content_key] = download_html_from_blob(blob_name)
        except Exception as exc:
            print(f"[ComicBook] Unable to fetch blob '{blob_name}': {exc}")
    return entity


def _sort_by_rowkey(entities: list) -> list:
    return sorted(entities, key=lambda x: x.get("RowKey", ""), reverse=True)


def start_new_arc(title: str, logline: str, target_days: int, start_date: Optional[datetime] = None) -> dict:
    start_date = start_date or datetime.utcnow()
    arc_id = f"{_ARC_ID_PREFIX}_{get_flat_date(start_date)}_{uuid.uuid4().hex[:6]}"
    entity = {
        "PartitionKey": _ARC_PARTITION,
        "RowKey": arc_id,
        "title": title,
        "logline": logline,
        "start_date": start_date.isoformat(),
        "status": "active",
        "target_days": target_days,
        "episodes_count": 0,
        "last_episode_date": "",
    }
    if _PERSIST:
        arcs_table.create_entity(entity=entity)
    else:
        print(f"[ComicBook][dry-run] skip create arc {arc_id}")
    return entity


def close_arc(arc_id: str, end_date: Optional[datetime] = None):
    if not _PERSIST:
        print(f"[ComicBook][dry-run] skip close arc {arc_id}")
        return
    end_date = end_date or datetime.utcnow()
    entity = {
        "PartitionKey": _ARC_PARTITION,
        "RowKey": arc_id,
        "status": "closed",
        "end_date": end_date.isoformat(),
    }
    arcs_table.upsert_entity(entity=entity, mode=UpdateMode.MERGE)


def get_latest_arc() -> Optional[dict]:
    arcs = list(arcs_table.query_entities(f"PartitionKey eq '{_ARC_PARTITION}'", results_per_page=100))
    if not arcs:
        return None
    arcs = sorted(arcs, key=lambda x: x.get("start_date", ""), reverse=True)
    return arcs[0]


def get_active_arc() -> Optional[dict]:
    arc = get_latest_arc()
    if arc and arc.get("status") == "active":
        return arc
    return None


def update_arc_metadata(arc_id: str, **kwargs):
    if not _PERSIST:
        return
    entity = {"PartitionKey": _ARC_PARTITION, "RowKey": arc_id}
    entity.update(kwargs)
    arcs_table.upsert_entity(entity=entity, mode=UpdateMode.MERGE)


def save_arc_story_outline(arc_id: str, story_outline: str, lang: str = "en"):
    if not _PERSIST:
        return
    suffix = "" if lang == "en" else f"_{lang}"
    key = f"story_outline{suffix}"
    blob_key = f"story_outline_blob_name{suffix}"
    entity = {"PartitionKey": _ARC_PARTITION, "RowKey": arc_id}
    if len(story_outline) > MAX_TABLE_PROPERTY_CHARS:
        blob_name = upload_text_to_blob(story_outline, extension=".txt")
        entity[key] = ""
        entity[blob_key] = blob_name
    else:
        entity[key] = story_outline
        entity[blob_key] = ""
    arcs_table.upsert_entity(entity=entity, mode=UpdateMode.MERGE)


def get_arc_story_outline(arc: dict, lang: str = "en") -> str:
    suffix = "" if lang == "en" else f"_{lang}"
    key = f"story_outline{suffix}"
    blob_key = f"story_outline_blob_name{suffix}"
    blob_name = arc.get(blob_key, "")
    if blob_name:
        try:
            return download_html_from_blob(blob_name)
        except Exception as exc:
            print(f"[ComicBook] Unable to fetch story outline blob '{blob_name}': {exc}")
            return arc.get(key, "")
    return arc.get(key, "")


def save_arc_glossary(arc_id: str, lang: str, glossary: dict):
    if not _PERSIST:
        return
    import json
    key = f"glossary_{lang}"
    content = json.dumps(glossary, ensure_ascii=False)
    entity = {"PartitionKey": _ARC_PARTITION, "RowKey": arc_id}
    if len(content) > MAX_TABLE_PROPERTY_CHARS:
        blob_name = upload_text_to_blob(content, extension=".json")
        entity[key] = ""
        entity[f"{key}_blob_name"] = blob_name
    else:
        entity[key] = content
        entity[f"{key}_blob_name"] = ""
    arcs_table.upsert_entity(entity=entity, mode=UpdateMode.MERGE)


def get_arc_glossary(arc: dict, lang: str) -> dict:
    import json
    key = f"glossary_{lang}"
    blob_name = arc.get(f"{key}_blob_name", "")
    if blob_name:
        try:
            return json.loads(download_html_from_blob(blob_name))
        except Exception:
            pass
    raw = arc.get(key, "")
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            pass
    return {}


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


def save_key_panel(arc_id: str, panel_url: str, character_name: str, episode_number: int) -> None:
    """Append a key panel to the arc's key_panels list (capped at 20 most recent)."""
    if not _PERSIST:
        return
    import json
    try:
        arc = arcs_table.get_entity(partition_key=_ARC_PARTITION, row_key=arc_id)
    except Exception:
        arc = {}
    raw = arc.get("key_panels", "[]")
    try:
        panels = json.loads(raw) if raw else []
    except Exception:
        panels = []
    panels.append({"url": panel_url, "character": character_name, "episode": episode_number})
    panels = panels[-20:]  # keep most recent 20
    content = json.dumps(panels, ensure_ascii=False)
    arcs_table.upsert_entity(
        entity={"PartitionKey": _ARC_PARTITION, "RowKey": arc_id, "key_panels": content},
        mode=UpdateMode.MERGE,
    )


def get_key_panels(arc: dict) -> list:
    """Return the list of key panels stored for this arc."""
    import json
    raw = arc.get("key_panels", "")
    if not raw:
        return []
    try:
        return json.loads(raw)
    except Exception:
        return []


def get_first_episode(arc_id: str) -> Optional[dict]:
    """Return the oldest (first) episode of an arc with HTML hydrated — used as a character-anchor reference."""
    episodes = list(episodes_table.query_entities(f"PartitionKey eq '{arc_id}'", results_per_page=200))
    if not episodes:
        return None
    first = sorted(episodes, key=lambda x: x.get("RowKey", ""))[0]
    return _hydrate_html_content(first)


def get_recent_episodes(arc_id: str, limit: int = 3, hydrate_html: bool = True) -> List[dict]:
    episodes = list(episodes_table.query_entities(f"PartitionKey eq '{arc_id}'", results_per_page=200))
    episodes = _sort_by_rowkey(episodes)
    result = episodes[:limit]
    if hydrate_html:
        result = [_hydrate_html_content(e) for e in result]
    return result


def get_episode_by_date(date_key: str, lang: str = "en") -> Optional[dict]:
    try:
        entities = list(episodes_table.query_entities(f"RowKey eq '{date_key}'", results_per_page=5))
        if not entities:
            return None
        entity = _sort_by_rowkey(entities)[0]
        entity = _hydrate_html_for_lang(entity, lang)
        return entity
    except Exception:
        return None


def get_episode_index() -> List[dict]:
    """Return a lightweight index of all episodes across all arcs, sorted by date."""
    all_eps = list(episodes_table.query_entities(
        "PartitionKey ne ''",
        select=["PartitionKey", "RowKey", "arc_title", "episode_number"],
        results_per_page=500,
    ))
    # Keep debug episodes ("debugarc_*" partitions) out of the production index and vice versa.
    all_eps = [e for e in all_eps if str(e.get("PartitionKey", "")).startswith("debugarc") == DEBUG]
    all_eps.sort(key=lambda x: x.get("RowKey", ""))
    return [
        {
            "date": e["RowKey"],
            "arc_id": e["PartitionKey"],
            "arc_title": e.get("arc_title", ""),
            "episode_number": int(e.get("episode_number", 0)),
        }
        for e in all_eps
    ]


def get_arc_list() -> List[dict]:
    """Return all arcs sorted by start_date."""
    arcs = list(arcs_table.query_entities(
        f"PartitionKey eq '{_ARC_PARTITION}'",
        select=["RowKey", "title", "status", "start_date", "episodes_count", "character_sheet_url"],
        results_per_page=100,
    ))
    arcs.sort(key=lambda x: x.get("start_date", ""))
    return [
        {
            "arc_id": a["RowKey"],
            "title": a.get("title", ""),
            "status": a.get("status", ""),
            "start_date": a.get("start_date", ""),
            "episodes_count": int(a.get("episodes_count", 0)),
            "character_sheet_url": a.get("character_sheet_url", ""),
        }
        for a in arcs
    ]


def get_recent_arc_summaries(limit: int = 10) -> List[dict]:
    """Return the most recent arcs with title, logline, genre, and art_style (newest first)."""
    arcs = list(arcs_table.query_entities(
        f"PartitionKey eq '{_ARC_PARTITION}'",
        select=["RowKey", "title", "logline", "genre", "art_style", "start_date", "status"],
        results_per_page=100,
    ))
    arcs.sort(key=lambda x: x.get("start_date", ""), reverse=True)
    return [
        {
            "title": a.get("title", ""),
            "logline": a.get("logline", ""),
            "genre": a.get("genre", ""),
            "art_style": a.get("art_style", ""),
        }
        for a in arcs[:limit]
    ]


def save_episode(
    arc: dict,
    episode_date: datetime,
    html_content: str,
    storyboard_summary: str,
    panel_notes: str,
    html_content_it: str = "",
    html_content_fa: str = "",
) -> dict:
    arc_id = arc["RowKey"]
    row_key = get_flat_date(episode_date)
    if not _PERSIST:
        episode_number = int(arc.get("episodes_count", 0)) + 1
        print(f"[ComicBook][dry-run] skip save_episode {arc_id}/{row_key} (ep {episode_number})")
        return {
            "PartitionKey": arc_id,
            "RowKey": row_key,
            "arc_title": arc.get("title", ""),
            "episode_number": episode_number,
        }
    episode_number = _count_episodes(arc_id) + 1
    entity = {
        "PartitionKey": arc_id,
        "RowKey": row_key,
        "arc_title": arc.get("title", ""),
        "episode_number": episode_number,
        "story_summary": storyboard_summary,
        "panel_notes": panel_notes,
    }
    entity = _attach_html_payload(entity, html_content)
    entity = _attach_html_payload(entity, html_content_it, "_it")
    entity = _attach_html_payload(entity, html_content_fa, "_fa")
    episodes_table.upsert_entity(entity=entity, mode=UpdateMode.REPLACE)
    arcs_table.upsert_entity(
        entity={
            "PartitionKey": _ARC_PARTITION,
            "RowKey": arc_id,
            "episodes_count": episode_number,
            "last_episode_date": episode_date.isoformat(),
            "last_story_summary": storyboard_summary[:MAX_TABLE_PROPERTY_CHARS],
        },
        mode=UpdateMode.MERGE,
    )
    return entity
