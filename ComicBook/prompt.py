import threading
import time
from datetime import datetime, timezone
from typing import Optional

from ComicBook.azurestorage import get_episode_by_date, save_episode
from utils import get_flat_date

_generation_lock = threading.Lock()
_dates_in_progress: dict[str, float] = {}
_LOCK_TTL_SECONDS = 3600


def get_comicbook(parsed_date: Optional[datetime] = None, lang: str = "en"):
    target_date = parsed_date or datetime.now(timezone.utc)
    flat_date = get_flat_date(target_date)

    if cached := get_episode_by_date(flat_date, lang=lang):
        content_key = "html_content" if lang == "en" else f"html_content_{lang}"
        html = cached.get(content_key, "") or cached.get("html_content", "")
        return html, target_date, cached.get("PartitionKey")

    with _generation_lock:
        started_at = _dates_in_progress.get(flat_date)
        if started_at is not None and (time.monotonic() - started_at) < _LOCK_TTL_SECONDS:
            return "<p>This episode is already being generated. Please try again shortly.</p>", target_date, ""
        _dates_in_progress[flat_date] = time.monotonic()

    try:
        from ComicBook.agents import run_comic_pipeline
        result = run_comic_pipeline(target_date)
    except Exception:
        with _generation_lock:
            _dates_in_progress.pop(flat_date, None)
        raise

    with _generation_lock:
        _dates_in_progress.pop(flat_date, None)

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
