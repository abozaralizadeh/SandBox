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


_NO_EPISODE_HTML = (
    "<p style='color:#9aa;padding:24px 0;text-align:center'>"
    "No episode was published on this date.</p>"
)

_GENERATION_FAILED_HTML = (
    "<p style='color:#9aa;padding:24px 0;text-align:center'>"
    "Today's episode could not be completed. Please check back shortly.</p>"
)

# A failed run is NOT persisted (see below), so the date stays generatable — but each retry is
# a full pipeline run with image generation, so cap how many times one date may be attempted.
_FAILURE_PARTITION = "generation_failure_debug" if DEBUG else "generation_failure"
_MAX_GENERATION_ATTEMPTS = 3


def _failed_attempts(flat_date: str) -> int:
    if not _PERSIST:
        return 0
    try:
        return int(episodes_table.get_entity(_FAILURE_PARTITION, flat_date).get("attempts", 0) or 0)
    except Exception:
        return 0


def _record_failed_attempt(flat_date: str) -> int:
    attempts = _failed_attempts(flat_date) + 1
    if not _PERSIST:
        return attempts
    try:
        episodes_table.upsert_entity(entity={
            "PartitionKey": _FAILURE_PARTITION,
            "RowKey": flat_date,
            "attempts": attempts,
            "last_attempt_at": datetime.now(timezone.utc).isoformat(),
        })
    except Exception:
        pass
    return attempts


def get_comicbook(parsed_date: Optional[datetime] = None, lang: str = "en"):
    target_date = parsed_date or datetime.now(timezone.utc)
    flat_date = get_flat_date(target_date)

    if cached := get_episode_by_date(flat_date, lang=lang):
        content_key = "html_content" if lang == "en" else f"html_content_{lang}"
        html = cached.get(content_key, "") or cached.get("html_content", "")
        return html, target_date, cached.get("PartitionKey")

    if flat_date != get_flat_date():
        # An older date with no episode is a gap (the app was down that day), and it must NOT
        # be generated now. Episodes are chapters: `save_episode` numbers them and advances
        # the arc's last_episode_date/last_story_summary, so writing a back-dated episode
        # today would leave the running arc describing an episode that is not its newest —
        # and the next real episode would continue from the wrong state. Reachable from a
        # shared link (?date=…) even though the UI's picker already refuses gap dates.
        return _NO_EPISODE_HTML, target_date, ""

    if _failed_attempts(flat_date) >= _MAX_GENERATION_ATTEMPTS:
        # Generation has already failed repeatedly for this date. Stop burning a full pipeline
        # run (and its image spend) on every page view; the date simply stays a gap, which the
        # archive and navigation already handle.
        return _GENERATION_FAILED_HTML, target_date, ""

    if not _try_acquire_lock(flat_date):
        return "<p>This episode is already being generated. Please try again shortly.</p>", target_date, ""

    try:
        from ComicBook.agents import run_comic_pipeline
        result = run_comic_pipeline(target_date)
    except Exception:
        # A crash counts against the same per-date budget as a soft failure: without this, an
        # error that reproduces every run (a max-turns overrun, a dead image endpoint) would
        # restart the whole pipeline on every single page view for the rest of the day.
        _release_lock(flat_date)
        _record_failed_attempt(flat_date)
        raise

    _release_lock(flat_date)

    if result.get("failed"):
        # NEVER persist a failed run. save_episode would store the blank fallback page, and
        # get_episode_by_date serves any stored row from cache — so the broken page would be
        # returned for that date forever and never regenerate. It would also consume an episode
        # slot (episodes_count) and blank the arc's last_story_summary, which is the "where the
        # story stands" note the next day's Director reads. Leaving the date unwritten keeps it
        # a normal gap that a later visitor can still fill.
        attempts = _record_failed_attempt(flat_date)
        print(f"[ComicBook] generation FAILED for {flat_date} "
              f"(attempt {attempts}/{_MAX_GENERATION_ATTEMPTS}) — nothing persisted")
        return _GENERATION_FAILED_HTML, target_date, ""

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
