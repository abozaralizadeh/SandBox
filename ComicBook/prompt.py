from datetime import datetime, timezone
from typing import Optional

from ComicBook.agents import run_comic_pipeline
from ComicBook.azurestorage import get_episode_by_date, save_episode
from utils import get_flat_date


def get_comicbook(parsed_date: Optional[datetime] = None):
    target_date = parsed_date or datetime.now(timezone.utc)
    flat_date = get_flat_date(target_date)

    if cached := get_episode_by_date(flat_date):
        return cached.get("html_content", ""), target_date, cached.get("PartitionKey")

    result = run_comic_pipeline(target_date)

    html = result["html"]
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
        )
        return html, target_date, saved.get("PartitionKey")

    return html, target_date, ""
