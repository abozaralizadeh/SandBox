"""Publication cadence for the GenBox channel.

`GENBOX_GENERATION_INTERVAL_DAYS` (default 1 = every day) says how often a new decision
is produced. Slots sit on a fixed grid anchored at `GENBOX_SCHEDULE_ANCHOR_DATE`
(default 1970-01-01), so the grid is derived purely from the calendar — it does not
depend on when the app was deployed or restarted, and every gunicorn worker agrees on it
without shared state.

**The interval is only ever applied forward.** It decides which date is the *live* channel
(the one allowed to generate); it is never used to compute which past channels exist,
because a past channel's existence is a fact of storage, not of today's configuration.
Changing the interval therefore re-times future publications and leaves the archive — and
the TV's back/next navigation over it — completely intact (see `channel_index`).
"""
import os
import re
from datetime import date as _date, datetime, timedelta

from dotenv import load_dotenv

from utils import get_flat_date

load_dotenv()

_DEFAULT_ANCHOR = _date(1970, 1, 1)
_FLAT_DATE_RE = re.compile(r"^\d{8}$")


def _as_date(value=None):
    """Coerce a date/datetime/ISO-string/None into a `date` (None -> today)."""
    if value is None:
        return _date.today()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, _date):
        return value
    return datetime.fromisoformat(str(value)).date()


def generation_interval_days() -> int:
    """How many days between two publications. Always >= 1; a malformed value means 1."""
    raw = (os.getenv("GENBOX_GENERATION_INTERVAL_DAYS") or "").strip()
    if not raw:
        return 1
    try:
        return max(1, int(float(raw)))
    except ValueError:
        print(f"GenBox: invalid GENBOX_GENERATION_INTERVAL_DAYS={raw!r}; falling back to 1")
        return 1


def _anchor_date() -> _date:
    raw = (os.getenv("GENBOX_SCHEDULE_ANCHOR_DATE") or "").strip()
    if not raw:
        return _DEFAULT_ANCHOR
    try:
        return _as_date(raw)
    except ValueError:
        print(f"GenBox: invalid GENBOX_SCHEDULE_ANCHOR_DATE={raw!r}; falling back to {_DEFAULT_ANCHOR}")
        return _DEFAULT_ANCHOR


def is_generation_date(value=None) -> bool:
    """True when `value` lands exactly on a publication slot of the current grid."""
    return (_as_date(value) - _anchor_date()).days % generation_interval_days() == 0


def current_slot_date(value=None) -> _date:
    """The publication slot in effect on `value` (default today): the latest slot on or
    before that date. With the default interval of 1 this is simply the date itself."""
    d = _as_date(value)
    offset = (d - _anchor_date()).days % generation_interval_days()
    return d - timedelta(days=offset)


def next_slot_date(value=None) -> _date:
    """The first publication slot strictly after `value`."""
    return current_slot_date(value) + timedelta(days=generation_interval_days())


def channel_index() -> dict:
    """The TV's channel list: every date that actually has a saved decision, plus the live
    slot, ascending.

    Navigation is driven by this list rather than by stepping `interval` days at a time, so
    turning the dial keeps working across an interval change: the archive was published on
    whatever cadence was configured at the time, and those dates are read back from storage
    as-is. Only the trailing `live` entry comes from the current interval.
    """
    from GenBox.azurestorage import list_decision_dates  # local import: avoids a cycle

    live = current_slot_date()
    flat_dates = set(list_decision_dates())
    flat_dates.add(get_flat_date(live))
    dates = sorted(
        datetime.strptime(f, "%Y%m%d").date()
        for f in flat_dates
        if _FLAT_DATE_RE.match(f or "")
    )
    return {
        "dates": [d.isoformat() for d in dates],
        "live": live.isoformat(),
        "interval_days": generation_interval_days(),
    }
