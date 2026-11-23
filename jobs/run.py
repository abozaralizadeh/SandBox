"""One scheduled job of the Generate workflow (.github/workflows/generate.yml).

These replace the five Azure Logic Apps that pinged the App Service on a timer
(Scheduler00..04). A job makes today's content if it is missing, waits for any background
thread the generator hands the work to, then publishes the static copy (sitepub.py) that the
Cloudflare Worker serves.

    python -m jobs.run tomorrownews|aiblog|comicbook|genbox|aiops [--publish-only]

`--publish-only` skips generation. The workflow passes it until the `GENERATE_ENABLED` repo
variable is `true`, so the App Service and the Actions jobs never generate side by side.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import threading
import time
import traceback
from contextlib import contextmanager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(ROOT, ".env"))

# How long to wait for the background work a generator started. GenBox's Sora video and an
# AIOPS iteration are the long ones; the workflow's timeout-minutes sits above this.
JOIN_TIMEOUT = int(os.getenv("JOB_THREAD_TIMEOUT", "6000"))
_OUR_PACKAGES = {"GenBox", "AIOpenProblemSolver", "TomorrowNews", "AIBlog", "ComicBook"}


@contextmanager
def background_threads():
    """Collect the threads our own generators start while inside the block.

    The generators were written for a web server: GenBox and AIOPS hand the slow part to a
    daemon thread and return at once. A job that returned then would exit and kill that work,
    so this notes threads whose target lives in one of our packages and the job joins them.
    Library threads (LangSmith's uploader, SDK pools) are left alone: some never end."""
    started: list[threading.Thread] = []
    original = threading.Thread.start

    def start(self, *args, **kwargs):
        target = getattr(self, "_target", None)
        if (getattr(target, "__module__", "") or "").split(".")[0] in _OUR_PACKAGES:
            started.append(self)
        return original(self, *args, **kwargs)

    threading.Thread.start = start
    try:
        yield started
    finally:
        threading.Thread.start = original


def wait_for(threads, timeout: int = JOIN_TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    for t in threads:
        t.join(max(0.0, deadline - time.monotonic()))
    alive = [t.name for t in threads if t.is_alive()]
    if alive:
        print(f"[jobs] still running after {timeout}s, giving up on: {alive}")
    return not alive


# ---------------------------------------------------------------------------
# One function per project. Each raises when today's content could not be made.
# ---------------------------------------------------------------------------

def gen_tomorrownews():
    # _generate_all, not gettomorrownews: the latter returns as soon as the English edition
    # is cached, so a Persian or Italian edition that failed would never be retried.
    from TomorrowNews.prompt import _generate_all

    results = _generate_all(None)
    missing = [lang for lang in ("en", "fa", "it") if not results.get(lang)]
    print(f"[jobs] tomorrownews: ready {sorted(k for k, v in results.items() if v)}")
    if missing:
        raise RuntimeError(f"no edition for {missing}")


def gen_aiblog():
    from AIBlog.prompt import getaiblog

    html, _ = asyncio.run(getaiblog(None))
    if not html:
        raise RuntimeError("AIBlog produced no post")
    print("[jobs] aiblog: today's post is stored")


def gen_comicbook():
    from ComicBook.prompt import get_comicbook

    html, _, arc_id = get_comicbook(None, lang="en")
    if arc_id:
        print(f"[jobs] comicbook: today's episode is stored ({arc_id})")
    elif "already being generated" in (html or ""):
        print("[jobs] comicbook: another run holds the lock, leaving it to that one")
    else:
        raise RuntimeError("ComicBook produced no episode")


def gen_genbox():
    from GenBox.prompt import get_llm_response
    from GenBox.schedule import current_slot_date
    from GenBox.video import ensure_generation_started, video_status

    slot = current_slot_date()
    # 1. The decision text. Before, the first visitor of the slot day started it through
    #    /get-string; with no visitor there was no bulletin.
    with background_threads() as started:
        text = get_llm_response(None)
    if not text and started:
        wait_for(started)
    # 2. Video and narration (what Scheduler00 did through /genbox-video-status).
    with background_threads() as started:
        ensure_generation_started(slot)
    wait_for(started)
    status = video_status(slot)
    print(f"[jobs] genbox {slot}: video {status.get('status')}, audio {status.get('audio_status')}")
    if not get_llm_response(None):
        raise RuntimeError(f"no GenBox decision for {slot}")


def gen_aiops():
    from AIOpenProblemSolver.azurestorage import list_problems
    from AIOpenProblemSolver.prompt import ensure_latest_iteration

    concurrency = max(1, int(os.getenv("AIOPS_JOB_CONCURRENCY", "3")))
    names = [p["name"] for p in list_problems() if p.get("name")]
    running: list[threading.Thread] = []
    for name in names:
        while sum(t.is_alive() for t in running) >= concurrency:
            time.sleep(5)
        with background_threads() as started:
            asyncio.run(ensure_latest_iteration(name))
        running += started
        if started:
            print(f"[jobs] aiops: started today's iteration for {name!r}")
            time.sleep(20)   # stagger the browsers, like the Logic App's 1-minute delay
    wait_for(running)
    print(f"[jobs] aiops: {len(running)} iterations run for {len(names)} problems")


JOBS = {
    "tomorrownews": gen_tomorrownews,
    "aiblog": gen_aiblog,
    "comicbook": gen_comicbook,
    "genbox": gen_genbox,
    "aiops": gen_aiops,
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("project", choices=list(JOBS))
    parser.add_argument("--publish-only", action="store_true")
    parser.add_argument("--full", action="store_true", help="republish the whole archive")
    args = parser.parse_args(argv)

    ok = True
    if args.publish_only:
        print(f"[jobs] {args.project}: publish only (generation is off)")
    else:
        try:
            JOBS[args.project]()
        except BaseException as exc:  # noqa: BLE001 - publish whatever exists anyway
            traceback.print_exc()
            print(f"[jobs] {args.project} generation FAILED: {type(exc).__name__}: {exc}")
            ok = False

    import sitepub

    ok = sitepub.publish([args.project], full=args.full) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
