"""Publish what the pages read as plain files in a public blob container.

The site is static now (GitHub Pages) and has no Python behind it at request time. The
Cloudflare Worker in `worker/` answers the old data URLs (`/tomorrownewscontent`,
`/aiblogindex`, `/ai-open-problem-solver/history`, ...) by reading the files written here,
so every page keeps calling exactly the URLs it always called.

Tables stay the source of truth. This module only READS them with the same helpers the
Flask routes used and writes a copy that the Worker can serve without credentials:

    sandbox-site/
      tomorrownews/index.json                 {"editions": {lang: [iso]}, "latest": {lang: iso}}
      tomorrownews/{YYYYMMDD_HH}_{lang}.html  one edition (HH is 00 from 2025-01-25 on)
      aiblog/index.json                       {"editions": [iso], "latest": iso}
      aiblog/{YYYYMMDD}.html
      comicbook/index.json                    {"episodes": [...], "arcs": [...]}
      comicbook/{YYYYMMDD}_{lang}.html        panel <img> already pointed at /cbimg
      genbox/channels.json                    {"dates": [iso], "live": iso, "interval_days": n}
      genbox/text/{YYYYMMDD}.txt              what /get-string returned for that date
      genbox/status/{YYYYMMDD}.json           what /genbox-video-status returned
      aiops/problems.json                     {"problems": [...]}
      aiops/{sha1}/details.json               what /problem-details returned
      aiops/{sha1}/keys.json                  every iteration RowKey, newest first
      aiops/{sha1}/e/{RowKey}.json            one formatted iteration

Unlike the Flask indexes, nothing here adds "today" before today exists. Today used to be
listed because the first visit generated it; now a scheduled job does, and lists it once
it is really there.

Usage:
    python sitepub.py all [--full]      # --full rewrites every past file too (backfill)
    python sitepub.py tomorrownews|aiblog|comicbook|genbox|aiops [--full]
    python sitepub.py ... --out DIR     # dry run: write to a local folder, touch no storage
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

load_dotenv()

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError  # noqa: E402
from azure.storage.blob import BlobServiceClient, ContentSettings, PublicAccess  # noqa: E402
from flask import Flask  # noqa: E402

from utils import get_flat_date, parse_flat_date_hour  # noqa: E402

SITE_CONTAINER = os.getenv("site_blob_name", "sandbox-site")
# How many recent days an incremental run re-publishes. Two covers a run that crossed
# midnight UTC; older files never change once written.
RECENT_DAYS = int(os.getenv("SITEPUB_RECENT_DAYS", "2"))

# The Flask routes answered through `jsonify`; serialise the same way (datetimes become
# HTTP dates, keys sorted) so the Worker hands the pages the bytes they always got.
_json = Flask(__name__).json


def _dumps(obj) -> bytes:
    return _json.dumps(obj).encode("utf-8")


class _DropRowDumps(io.TextIOBase):
    """The storage helpers print every row they read, HTML included ("Row retrieved
    successfully: {...}"). In a backfill that is megabytes of log, and in a public Actions log
    it is noise, so those lines are dropped while a publish runs. A stream filter rather than
    `redirect_stdout`, because the reads happen on worker threads and swapping sys.stdout
    from several threads at once leaks lines."""

    def __init__(self, inner):
        self.inner = inner
        self._dropping = False

    def write(self, text):
        if text.startswith(("Row retrieved successfully", "Error retrieving row")):
            self._dropping = True
            return len(text)
        if self._dropping and text == "\n":
            self._dropping = False
            return len(text)
        self._dropping = False
        return self.inner.write(text)

    def flush(self):
        self.inner.flush()


class Sink:
    """Where published files go: the public container, or a local folder for a dry run."""

    def __init__(self, out_dir: str | None = None):
        self.out_dir = out_dir
        self.written = 0
        self.skipped = 0
        self._existing: set[str] | None = None
        if out_dir:
            self.container = None
            return
        service = BlobServiceClient.from_connection_string(os.environ["connection_string"])
        self.container = service.get_container_client(SITE_CONTAINER)
        try:
            self.container.create_container(public_access=PublicAccess.BLOB)
            print(f"[sitepub] created container {SITE_CONTAINER}")
        except ResourceExistsError:
            pass

    def existing(self, prefix: str) -> set[str]:
        """Names already published under `prefix` (one listing call, used to skip re-uploads
        of files that never change, like an old AIOPS iteration)."""
        if self.out_dir:
            root = os.path.join(self.out_dir, prefix)
            found = set()
            for base, _, files in os.walk(root):
                for f in files:
                    found.add(os.path.relpath(os.path.join(base, f), self.out_dir))
            return found
        return {b.name for b in self.container.list_blobs(name_starts_with=prefix)}

    def put(self, name: str, data: bytes | str, content_type: str, max_age: int = 60):
        if isinstance(data, str):
            data = data.encode("utf-8")
        if self.out_dir:
            path = os.path.join(self.out_dir, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as fh:
                fh.write(data)
        else:
            self.container.upload_blob(
                name,
                data,
                overwrite=True,
                content_settings=ContentSettings(
                    content_type=content_type,
                    cache_control=f"public, max-age={max_age}",
                ),
            )
        self.written += 1


HTML = "text/html; charset=utf-8"
JSON = "application/json"
TEXT = "text/plain; charset=utf-8"


def _recent_flat_dates(days: int = RECENT_DAYS) -> set[str]:
    today = datetime.now(timezone.utc).date()
    return {(today - timedelta(days=i)).strftime("%Y%m%d") for i in range(days)}


def _parallel(fn, items, workers: int = 8):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))


# ---------------------------------------------------------------------------
# TomorrowNews
# ---------------------------------------------------------------------------

TN_LANGS = ("en", "fa", "it")


def publish_tomorrownews(sink: Sink, full: bool = False):
    from TomorrowNews.azurestorage import get_row, list_edition_keys

    # Same grouping as TomorrowNews.prompt.edition_index, minus the "today" it adds.
    # Keys: YYYYMMDD_HH (early hourly, English), YYYYMMDD_00 (daily, English), and
    # YYYYMMDD_00_{lang}. For English, `_try_cache` prefers `{base}_en` over the bare `{base}`.
    rows: dict[tuple[str, str], str] = {}   # (base, lang) -> RowKey to read
    for key in list_edition_keys():
        parts = key.split("_")
        base = "_".join(parts[:2])
        lang = parts[2] if len(parts) > 2 else "en"
        if lang not in TN_LANGS:
            continue
        try:
            parse_flat_date_hour(base)
        except ValueError:
            continue
        if (base, lang) in rows and len(parts) == 2:
            continue                       # a bare legacy key never beats `{base}_en`
        rows[(base, lang)] = key

    editions = {lang: sorted(parse_flat_date_hour(b).isoformat() for (b, l) in rows if l == lang)
                for lang in TN_LANGS}
    index = {"editions": editions,
             "latest": {lang: dates[-1] for lang, dates in editions.items() if dates}}

    recent = _recent_flat_dates()
    todo = [(b, l, k) for (b, l), k in sorted(rows.items()) if full or b[:8] in recent]

    def one(item):
        base, lang, key = item
        row = get_row(key)
        html = (row or {}).get("html_content")
        if not html:
            return False
        sink.put(f"tomorrownews/{base}_{lang}.html", html, HTML, max_age=300)
        return True

    done = sum(_parallel(one, todo))
    # The index goes last: it must never list an edition whose file is not there yet.
    sink.put("tomorrownews/index.json", _dumps(index), JSON)
    print(f"[sitepub] tomorrownews: {done}/{len(todo)} editions, "
          + ", ".join(f"{l}={len(editions[l])}" for l in TN_LANGS))


# ---------------------------------------------------------------------------
# AIBlog
# ---------------------------------------------------------------------------

def publish_aiblog(sink: Sink, full: bool = False):
    from AIBlog.azurestorage import get_row, list_edition_keys

    keys = []
    for key in list_edition_keys():
        try:
            parse_flat_date_hour(key)
        except ValueError:
            continue
        keys.append(key)

    recent = _recent_flat_dates()
    todo = [k for k in keys if full or k[:8] in recent]

    def one(key):
        row = get_row(key)
        html = (row or {}).get("html_content")
        if not html:
            return None
        sink.put(f"aiblog/{key[:8]}.html", html, HTML, max_age=300)
        return key

    published = {k for k in _parallel(one, todo) if k}
    # A row that has html_blob_name but whose HTML cannot be read stays out of the index,
    # exactly like Flask, which reported it as a gap.
    listed = [k for k in keys if k in published or k not in todo]
    editions = sorted(parse_flat_date_hour(k).isoformat() for k in listed)
    index = {"editions": editions, "latest": editions[-1] if editions else None}
    sink.put("aiblog/index.json", _dumps(index), JSON)
    print(f"[sitepub] aiblog: {len(published)}/{len(todo)} posts, {len(editions)} in index")


# ---------------------------------------------------------------------------
# ComicBook
# ---------------------------------------------------------------------------

CB_LANGS = ("en", "it", "fa")


def publish_comicbook(sink: Sink, full: bool = False):
    from ComicBook.azurestorage import get_arc_list, get_episode_by_date, get_episode_index
    from ComicBook.imageproxy import _IMG_RE, ensure_webp_variant, rewrite_comic_images

    episodes = get_episode_index()
    arcs = get_arc_list()
    recent = _recent_flat_dates()
    dates = sorted({e["date"] for e in episodes if full or e["date"] in recent})

    def one(flat_date):
        written = 0
        for lang in CB_LANGS:
            cached = get_episode_by_date(flat_date, lang=lang)
            if not cached:
                continue
            # Same pick as ComicBook.prompt.get_comicbook.
            content_key = "html_content" if lang == "en" else f"html_content_{lang}"
            html = cached.get(content_key, "") or cached.get("html_content", "")
            # Make sure every panel's WebP exists now, so /cbimg in the Worker is a plain
            # redirect and never has to transcode anything.
            # (Skipped in a dry run, which must not write to storage.)
            for match in _IMG_RE.finditer(html or "") if not sink.out_dir else ():
                ensure_webp_variant(match.group(1))
            body = rewrite_comic_images(html)
            # The Arc-Id header travels as the first line of the file, so the Worker needs one
            # read per request. `<!--arc:...-->` is harmless if anything ever renders it raw.
            sink.put(f"comicbook/{flat_date}_{lang}.html",
                     f"<!--arc:{cached.get('PartitionKey') or ''}-->\n{body}", HTML, max_age=300)
            written += 1
        return written

    done = sum(_parallel(one, dates, workers=4))
    sink.put("comicbook/index.json", _dumps({"episodes": episodes, "arcs": arcs}), JSON)
    print(f"[sitepub] comicbook: {done} pages over {len(dates)} dates, "
          f"{len(episodes)} episodes in index")


# ---------------------------------------------------------------------------
# GenBox
# ---------------------------------------------------------------------------

def publish_genbox(sink: Sink, full: bool = False):
    from GenBox.azurestorage import get_row, list_decision_dates
    from GenBox.prompt import _extract_output
    from GenBox.schedule import current_slot_date, generation_interval_days
    from GenBox.video import video_status
    from utils import get_readable_date

    flat_dates = list_decision_dates()
    recent = _recent_flat_dates()
    # The live slot is the newest slot of the grid; it can sit a few days back with a
    # weekly cadence, so it is always refreshed too.
    live_slot = get_flat_date(current_slot_date())
    todo = [d for d in flat_dates if full or d in recent or d == live_slot]

    def one(flat):
        row = get_row("assistant", flat)
        day = datetime.strptime(flat, "%Y%m%d").date()
        output = _extract_output(row)
        # What GenBox.prompt.get_llm_response returned for a stored date.
        text = f"{output}\n\n{get_readable_date(day)}" if output else ""
        if text:
            sink.put(f"genbox/text/{flat}.txt", text, TEXT, max_age=300)
        status = video_status(day)
        sink.put(f"genbox/status/{flat}.json", _dumps(status), JSON, max_age=60)
        return flat if text else None

    with_text = {d for d in _parallel(one, todo) if d}
    listed = [d for d in flat_dates if d in with_text or d not in todo]
    dates = [datetime.strptime(d, "%Y%m%d").date().isoformat() for d in listed]
    live_iso = datetime.strptime(live_slot, "%Y%m%d").date().isoformat()
    if live_iso not in dates and dates:
        live_iso = dates[-1]      # the live slot has no bulletin yet: keep the newest on air
    channels = {"dates": dates, "live": live_iso, "interval_days": generation_interval_days()}
    sink.put("genbox/channels.json", _dumps(channels), JSON)
    print(f"[sitepub] genbox: {len(with_text)}/{len(todo)} bulletins, live {live_iso}, "
          f"{len(dates)} channels")


# ---------------------------------------------------------------------------
# AI Open Problem Solver
# ---------------------------------------------------------------------------

def publish_aiops(sink: Sink, full: bool = False):
    from AIOpenProblemSolver.azurestorage import (
        _fetch_iterations,
        _problem_partition,
        get_problem_details,
        get_problem_progress,
        iteration_rowkeys,
        list_problems,
    )
    from AIOpenProblemSolver.prompt import _format_entity

    problems = list_problems()
    sink.put("aiops/problems.json", _dumps({"problems": problems}), JSON, max_age=300)

    total = 0
    for entry in problems:
        problem = entry["name"]
        slug = _problem_partition(problem)
        prefix = f"aiops/{slug}/"
        keys = iteration_rowkeys(problem)

        have = set() if full else sink.existing(prefix + "e/")
        # The newest few are always re-written: a re-run on the same day replaces that day's
        # entry under the same RowKey.
        fresh = set(keys[:3])
        todo = [k for k in keys if f"{prefix}e/{k}.json" not in have or k in fresh]

        def one(key, problem=problem, prefix=prefix):
            rows = _fetch_iterations(problem, [key])
            if not rows:
                return 0
            sink.put(f"{prefix}e/{key}.json", _dumps(_format_entity(rows[0])), JSON, max_age=300)
            return 1

        total += sum(_parallel(one, todo))

        details = get_problem_details(problem) or {}
        progress = get_problem_progress(problem)
        sink.put(f"{prefix}details.json", _dumps({
            "problem": problem,
            "description": details.get("description", "") if details else "",
            "progress_percent": progress.get("progress_percent"),
            "progress_comment": progress.get("progress_comment"),
        }), JSON)
        # Keys last, so a page never asks for an entry that is not uploaded yet.
        sink.put(f"{prefix}keys.json", _dumps(keys), JSON)
    print(f"[sitepub] aiops: {len(problems)} problems, {total} entries written")


PUBLISHERS = {
    "tomorrownews": publish_tomorrownews,
    "aiblog": publish_aiblog,
    "comicbook": publish_comicbook,
    "genbox": publish_genbox,
    "aiops": publish_aiops,
}


def publish(projects, full: bool = False, out_dir: str | None = None) -> bool:
    """Publish the given projects. Returns False if any of them failed (the others still run)."""
    sink = Sink(out_dir)
    ok = True
    if not isinstance(sys.stdout, _DropRowDumps):
        sys.stdout = _DropRowDumps(sys.stdout)
    for name in projects:
        try:
            PUBLISHERS[name](sink, full=full)
        except Exception as exc:  # noqa: BLE001 - one project must not stop the others
            ok = False
            print(f"[sitepub] {name} FAILED: {type(exc).__name__}: {exc}")
    print(f"[sitepub] wrote {sink.written} files")
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("project", choices=["all", *PUBLISHERS])
    parser.add_argument("--full", action="store_true", help="re-publish the whole archive")
    parser.add_argument("--out", help="dry run: write into this folder instead of blob storage")
    args = parser.parse_args(argv)
    projects = list(PUBLISHERS) if args.project == "all" else [args.project]
    return 0 if publish(projects, full=args.full, out_dir=args.out) else 1


if __name__ == "__main__":
    sys.exit(main())
