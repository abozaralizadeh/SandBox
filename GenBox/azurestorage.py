from azure.data.tables import TableServiceClient, TableEntity, UpdateMode
from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient, ContentSettings
import os
import uuid
from io import BytesIO
from datetime import datetime, timezone
from dotenv import load_dotenv
from utils import get_flat_date

load_dotenv()
# Configuration
connection_string = os.getenv('connection_string')
table_name = os.getenv('genbox_table_name')
video_container_name = os.getenv('genbox_video_blob_name', 'genbox-video')

# Create a TableServiceClient
service_client = TableServiceClient.from_connection_string(conn_str=connection_string)

# Get a reference to the table client
table_client = service_client.get_table_client(table_name)

# Blob storage for generated news videos (the merged MP4 per date).
_blob_service_client = BlobServiceClient.from_connection_string(connection_string)
_video_container_client = _blob_service_client.get_container_client(video_container_name)


def _ensure_video_container():
    try:
        _video_container_client.create_container()
    except Exception:
        pass


_ensure_video_container()

def insert_history(role, content):

    rowkey = get_flat_date()
    # Define the entity (row) to insert
    entity = {
        "PartitionKey": role,  # Logical grouping for entities
        "RowKey": rowkey,                # Unique identifier within the partition
        "role": role,
        "content": content
    }
    # Insert the entity
    try:
        table_client.create_entity(entity=entity)
        print("Row inserted successfully!")
    except Exception as e:
        print(f"Error inserting row: {e}")
        # if "EntityAlreadyExists" in str(e):
        #     entity["RowKey"] = "000" + entity["RowKey"]
        #     table_client.create_entity(entity=entity)
        #     print("Row inserted successfully!")


def get_last_n_rows(n=10):
    try:
        # Query all entities
        role = "assistant"
        entities = table_client.query_entities(query_filter=f"PartitionKey eq '{role}'", results_per_page=1000)

        # Convert the entities to a sorted list by RowKey (descending order)
        sorted_entities = sorted(
            entities,
            key=lambda x: x.metadata["timestamp"],  # Sort by RowKey
            reverse=False               # Ascending order
        )

        last_n_rows_complete = sorted_entities[-n:]

        last_n_rows = [
            {key: value for key, value in row.items() if key not in ["PartitionKey", "RowKey"]}
            for row in last_n_rows_complete
        ]

        return last_n_rows

    except Exception as e:
        print(f"Error retrieving rows: {e}")
        return None
    

def get_row(partitionkey, rowkey):
    try:
        # Retrieve the entity (row) using the PartitionKey and RowKey
        entity = table_client.get_entity(partition_key=partitionkey, row_key=rowkey)
        print(f"Row retrieved successfully: {entity}")
        return entity
    except Exception as e:
        print(f"Error retrieving row: {e}")
        return None


# ---------------------------------------------------------------------------
# News-video storage (blob MP4 + per-date metadata + single-flight lock)
# Metadata and lock live in the existing GenBox table under dedicated
# PartitionKeys ("video" and "video_lock") so no new table is needed.
# ---------------------------------------------------------------------------

_VIDEO_LOCK_TTL = 1800  # seconds; stale locks are reclaimable after this


def upload_video_bytes_to_blob(video_bytes: bytes, flat_date: str) -> str:
    """Upload a merged MP4 and return its blob URL."""
    _ensure_video_container()
    blob_name = f"{flat_date}_{uuid.uuid4().hex[:8]}.mp4"
    blob_client = _video_container_client.get_blob_client(blob_name)
    blob_client.upload_blob(
        BytesIO(video_bytes),
        overwrite=True,
        content_settings=ContentSettings(content_type="video/mp4"),
    )
    return blob_client.url


_AUDIO_CONTENT_TYPES = {
    "mp3": "audio/mpeg", "wav": "audio/wav", "opus": "audio/ogg",
    "aac": "audio/aac", "flac": "audio/flac", "pcm": "audio/L16",
}


def upload_audio_bytes_to_blob(audio_bytes: bytes, flat_date: str, fmt: str = "mp3") -> str:
    """Upload TTS narration audio to the SAME video blob container; return its blob URL."""
    _ensure_video_container()
    fmt = (fmt or "mp3").lower()
    blob_name = f"{flat_date}_{uuid.uuid4().hex[:8]}.{fmt}"
    blob_client = _video_container_client.get_blob_client(blob_name)
    blob_client.upload_blob(
        BytesIO(audio_bytes),
        overwrite=True,
        content_settings=ContentSettings(content_type=_AUDIO_CONTENT_TYPES.get(fmt, "audio/mpeg")),
    )
    return blob_client.url


def get_video_meta(flat_date: str):
    """Return the video metadata entity for a date, or None."""
    try:
        return table_client.get_entity(partition_key="video", row_key=flat_date)
    except Exception:
        return None


def video_blob_name_from_url(video_url: str) -> str:
    """Extract the blob name (last path segment) from a stored video URL."""
    return (video_url or "").rstrip("/").rsplit("/", 1)[-1].split("?", 1)[0]


def get_video_blob_size(blob_name: str) -> int:
    """Size in bytes of a video blob (for HTTP Range responses)."""
    return _video_container_client.get_blob_client(blob_name).get_blob_properties().size


def download_video_blob_bytes(blob_name: str, offset: int = None, length: int = None) -> bytes:
    """Download all of a video blob, or a byte range when offset/length are given."""
    blob_client = _video_container_client.get_blob_client(blob_name)
    return blob_client.download_blob(offset=offset, length=length).readall()


def set_video_meta(flat_date: str, **fields):
    """Upsert (MERGE) video metadata; status transitions never clobber other fields."""
    now = datetime.now(timezone.utc).isoformat()
    entity = {"PartitionKey": "video", "RowKey": flat_date, "updated_at": now}
    entity.update(fields)
    if "created_at" not in entity and get_video_meta(flat_date) is None:
        entity["created_at"] = now
    table_client.upsert_entity(entity=entity, mode=UpdateMode.MERGE)


def _try_acquire_lock(partition: str, flat_date: str) -> bool:
    """Single-flight lock (per partition) so only one worker generates a given date."""
    entity = {
        "PartitionKey": partition,
        "RowKey": flat_date,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        table_client.create_entity(entity=entity)
        return True
    except ResourceExistsError:
        try:
            existing = table_client.get_entity(partition, flat_date)
            started = datetime.fromisoformat(existing["started_at"])
            age = (datetime.now(timezone.utc) - started).total_seconds()
        except Exception:
            return False
        if age > _VIDEO_LOCK_TTL:
            try:
                table_client.delete_entity(partition, flat_date)
                table_client.create_entity(entity=entity)
                return True
            except Exception:
                return False
        return False


def _release_lock(partition: str, flat_date: str):
    try:
        table_client.delete_entity(partition, flat_date)
    except Exception:
        pass


def try_acquire_video_lock(flat_date: str) -> bool:
    return _try_acquire_lock("video_lock", flat_date)


def release_video_lock(flat_date: str):
    _release_lock("video_lock", flat_date)


def try_acquire_audio_lock(flat_date: str) -> bool:
    return _try_acquire_lock("audio_lock", flat_date)


def release_audio_lock(flat_date: str):
    _release_lock("audio_lock", flat_date)