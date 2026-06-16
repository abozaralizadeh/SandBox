from datetime import datetime, timezone
from typing import Optional

from azure.core.exceptions import ResourceExistsError

from ComicBook.azurestorage import DEBUG, _PERSIST, episodes_table, get_episode_by_date, save_episode
from utils import get_flat_date

_LOCK_TTL_SECONDS = 3600
# Debug runs use an isolated lock partition so a local test never blocks production for the same
# date (and dry runs take no lock at all). Production keeps the plain "generation_lock".
_LOCK_PARTITION = "generation_lock_debug" if DEBUG else "generation_lock"


def _try_acquire_lock(flat_date: str) -> bool:
    if not _PERSIST:
        return True  # dry-run debug: nothing is persisted, so no shared lock is needed
    entity = {
        "PartitionKey": _LOCK_PARTITION,
        "RowKey": flat_date,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        episodes_table.create_entity(entity=entity)
        return True
    except ResourceExistsError:
        existing = episodes_table.get_entity(_LOCK_PARTITION, flat_date)
        started = datetime.fromisoformat(existing["started_at"])
        age = (datetime.now(timezone.utc) - started).total_seconds()
        if age > _LOCK_TTL_SECONDS:
            episodes_table.delete_entity(_LOCK_PARTITION, flat_date)
            try:
                episodes_table.create_entity(entity=entity)
                return True
            except ResourceExistsError:
                return False
        return False


def _release_lock(flat_date: str):
    if not _PERSIST:
        return
    try:
        episodes_table.delete_entity(_LOCK_PARTITION, flat_date)
    except Exception:
        pass


def get_comicbook(parsed_date: Optional[datetime] = None, lang: str = "en"):
    target_date = parsed_date or datetime.now(timezone.utc)
    flat_date = get_flat_date(target_date)

    if cached := get_episode_by_date(flat_date, lang=lang):
        content_key = "html_content" if lang == "en" else f"html_content_{lang}"
        html = cached.get(content_key, "") or cached.get("html_content", "")
        return html, target_date, cached.get("PartitionKey")

    if not _try_acquire_lock(flat_date):
        return "<p>This episode is already being generated. Please try again shortly.</p>", target_date, ""

    try:
        from ComicBook.agents import run_comic_pipeline
        result = run_comic_pipeline(target_date)
    except Exception:
        _release_lock(flat_date)
        raise

    _release_lock(flat_date)

    html = result["html"]
    html_it = result.get("html_it", "")
    html_fa = result.get("html_fa", "")
    arc = result.get("arc")
    summary = result.get("summary", "")
    panel_notes = result.get("panel_notes", "")

    if arc:
        saved = save_episode(
            arc=arc,
            episode_date=target_date,
            html_content=html,
            storyboard_summary=summary[:32000],
            panel_notes=panel_notes[:32000],
            html_content_it=html_it,
            html_content_fa=html_fa,
        )
        lang_html = {"en": html, "it": html_it, "fa": html_fa}
        return lang_html.get(lang, html), target_date, saved.get("PartitionKey")

    lang_html = {"en": html, "it": html_it, "fa": html_fa}
    return lang_html.get(lang, html), target_date, ""
