"""Render the seven pages to static files for GitHub Pages.

The pages were Flask templates, but only two use Jinja for anything: the landing page hides
the trAIde card behind HIDE_TRAIDE_PAGE, and the AIOPS page is seeded with the default
problem. Everything else they show comes from fetch() calls to the data paths, which the
Cloudflare Worker answers on the same origin. So the build renders each template once and
copies static/ next to them.

GitHub Pages serves `/tomorrownews` from `tomorrownews.html`, so every page keeps its URL.

    python scripts/build_site.py [OUT_DIR]      # default _site
"""
from __future__ import annotations

import os
import shutil
import sys

from flask import Flask, render_template

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def pages() -> dict[str, tuple[str, dict]]:
    """Output file -> (template, context). Mirrors the page routes in main.py."""
    problem = os.getenv("AIOPS_DEFAULT_PROBLEM", "Riemann Hypothesis")
    return {
        "index.html": ("index.html", {"hide_traide": _flag("HIDE_TRAIDE_PAGE", True)}),
        "tomorrownews.html": ("tomorrownews.html", {}),
        "comicbook.html": ("comicbook.html", {}),
        "aiblog.html": ("aiblog.html", {}),
        "genbox.html": ("tv.html", {}),
        "traide.html": ("traide.html", {}),
        # The description and progress were read from storage at request time. The page's
        # own script already loads both (problem list, then history), so they start empty.
        "ai-open-problem-solver.html": ("aiopenproblemsolver.html", {
            "default_problem": problem,
            "default_problem_description": "",
            "default_problem_progress_percent": None,
            "default_problem_progress_comment": "",
        }),
    }


def build(out_dir: str) -> None:
    app = Flask(__name__, template_folder=os.path.join(ROOT, "templates"))
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    with app.app_context():
        for name, (template, context) in pages().items():
            with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
                fh.write(render_template(template, **context))
            print(f"[build_site] {name} <- {template}")
    shutil.copytree(os.path.join(ROOT, "static"), os.path.join(out_dir, "static"))
    print(f"[build_site] done: {out_dir}")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "_site"))
