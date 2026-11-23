"""Plan a run of the Generate workflow: which projects have work to do right now?

Runs before anything is installed (stdlib only), so an hourly schedule with nothing to do
costs a few seconds instead of a full dependency install. It maps the cron that fired to its
project, then reads the public indexes that sitepub writes into the `sandbox-site` container
and drops every project whose content for today is already published.

    python3 jobs/needed.py --schedule "7 * * * *"           # from a scheduled run
    python3 jobs/needed.py --project all [--force]          # from a manual run

Prints `projects=[...]` (JSON) for $GITHUB_OUTPUT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from datetime import date, datetime, timedelta, timezone

BASE = os.getenv("SITE_BASE_URL", "https://pkrstr.blob.core.windows.net/sandbox-site").rstrip("/")

PROJECTS = ("tomorrownews", "aiblog", "comicbook", "genbox", "aiops")

# Must match the `cron:` lines in .github/workflows/generate.yml character for character.
# Each one stands in for one of the Azure Logic Apps that used to ping the App Service.
CRONS = {
    "7 * * * *": "tomorrownews",     # Scheduler01: hourly
    "37 */3 * * *": "aiblog",        # Scheduler02: every 3 h
    "23 1-23/4 * * *": "comicbook",  # Scheduler04: every 4 h
    "52 */3 * * *": "genbox",        # Scheduler00: every 3 h
    "47 * * * *": "aiops",           # Scheduler03: hourly
}

# A GenBox status that will not change by running again.
_TERMINAL = {"ready", "failed", "disabled"}


def _get(path: str):
    try:
        req = urllib.request.Request(f"{BASE}/{path}", headers={"Cache-Control": "no-cache"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.load(resp)
    except Exception as exc:  # noqa: BLE001 - unknown means "go and look"
        print(f"[needed] {path}: {exc}", file=sys.stderr)
        return None


def _slug(problem: str) -> str:
    """Same as sitepub's and the Worker's: sha1 of the lower-cased, space-collapsed name."""
    return hashlib.sha1(" ".join(problem.lower().split()).encode()).hexdigest()


def _genbox_slot(today: date, interval: int) -> date:
    raw = (os.getenv("GENBOX_SCHEDULE_ANCHOR_DATE") or "").strip()
    anchor = date.fromisoformat(raw[:10]) if raw else date(1970, 1, 1)
    return today - timedelta(days=(today - anchor).days % max(1, interval))


def needed(project: str, today: date) -> bool:
    iso, flat = today.isoformat(), today.strftime("%Y%m%d")

    if project == "tomorrownews":
        latest = (_get("tomorrownews/index.json") or {}).get("latest") or {}
        return any(not str(latest.get(lang) or "").startswith(iso) for lang in ("en", "fa", "it"))

    if project == "aiblog":
        return not str((_get("aiblog/index.json") or {}).get("latest") or "").startswith(iso)

    if project == "comicbook":
        episodes = (_get("comicbook/index.json") or {}).get("episodes") or []
        return not any(e.get("date") == flat for e in episodes)

    if project == "genbox":
        channels = _get("genbox/channels.json") or {}
        interval = int(os.getenv("GENBOX_GENERATION_INTERVAL_DAYS") or channels.get("interval_days") or 1)
        slot = _genbox_slot(today, interval)
        if slot.isoformat() not in (channels.get("dates") or []):
            return True
        status = _get(f"genbox/status/{slot.strftime('%Y%m%d')}.json") or {}
        return any(status.get(k) not in _TERMINAL for k in ("status", "audio_status"))

    if project == "aiops":
        problems = (_get("aiops/problems.json") or {}).get("problems")
        if not problems:
            return True
        for p in problems:
            keys = _get(f"aiops/{_slug(p.get('name', ''))}/keys.json") or []
            if not keys or not str(keys[0]).startswith(flat):
                return True
        return False

    raise ValueError(project)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", default="")
    parser.add_argument("--project", default="")
    parser.add_argument("--force", default="false")
    args = parser.parse_args(argv)

    if args.schedule:
        wanted = [CRONS[args.schedule]] if args.schedule in CRONS else []
    elif args.project in ("", "all"):
        wanted = list(PROJECTS)
    else:
        wanted = [args.project]

    force = args.force.lower() == "true"
    today = datetime.now(timezone.utc).date()
    chosen = [p for p in wanted if force or needed(p, today)]
    for p in wanted:
        print(f"[needed] {p}: {'run' if p in chosen else 'already published for ' + today.isoformat()}",
              file=sys.stderr)
    print(f"projects={json.dumps(chosen)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
