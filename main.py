# Remove Azure App Service's legacy helpers that shadow modern libraries
import sys
sys.path = [p for p in sys.path if not p.startswith("/agents/python")]

import os

from dotenv import load_dotenv
load_dotenv()
from flask import Flask, jsonify, make_response, request, render_template, Response, redirect
from AIBlog.prompt import *
from TomorrowNews.prompt import *
from ComicBook.prompt import get_comicbook
from ComicBook.azurestorage import get_episode_index, get_arc_list
from ComicBook.imageproxy import ensure_webp_variant, blob_name_if_ours, rewrite_comic_images
from GenBox.prompt import get_llm_response
from GenBox.video import ensure_generation_started, video_status
from GenBox.azurestorage import (
    video_blob_name_from_url,
    get_video_blob_size,
    download_video_blob_bytes,
)
from AIOpenProblemSolver.prompt import get_problem_history
from AIOpenProblemSolver.azurestorage import (
    get_problem_details,
    get_problem_progress,
    list_problems,
)
from TrAIde.azurestorage import (
    get_live_snapshot as traide_get_live_snapshot,
    get_equity_series as traide_get_equity_series,
    get_decision_feed as traide_get_decision_feed,
    get_closed_trades as traide_get_closed_trades,
    get_plans as traide_get_plans,
)

app = Flask(__name__)

cache = {}


def _env_flag(name, default=False):
    """Parse a boolean-ish environment variable; falls back to `default` when unset."""
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")

@app.route('/get-string', methods=['GET'])
def get_string():
    # Retrieve the 'date' query parameter from the request
    date_param = request.args.get('date')

    if date_param:
        try:
            # Parse the date parameter to a Python datetime object
            from datetime import datetime
            parsed_date = datetime.fromisoformat(date_param)
            parsed_date = parsed_date.date()
            if cache.get(str(parsed_date), False):
                return cache[str(parsed_date)]
        except ValueError:
            return jsonify({"error": "Invalid date format. Use ISO 8601 format, e.g., YYYY-MM-DDTHH:MM:SS"}), 400
    else:
        parsed_date = None

    # Pass the parsed date to the get_llm_response function (if applicable)
    response = get_llm_response(date=parsed_date)

    if len(response) > 20 and parsed_date:
        cache[str(parsed_date)] = response

    return response
    #return jsonify({"output": "Hello, this is a string from the backend!"})

@app.route('/tomorrownews', methods=['GET'])
def tomorrownews():
    return render_template('tomorrownews.html')

@app.route('/tomorrownewscontent', methods=['GET'])
def tomorrownewscontent():
    referer = request.headers.get('Referer', '')
    if referer:
        parsed_date = None
        date_param = request.args.get('dt')
        lang = request.args.get('lang', 'en')
        if lang not in ('en', 'fa', 'it'):
            lang = 'en'
        if date_param:
            try:
                from datetime import datetime
                parsed_date = datetime.fromisoformat(date_param)
            except:
                parsed_date = None
        tomorrownews, dt = gettomorrownews(parsed_date, lang=lang)
        response = make_response(tomorrownews)
        response.headers['Timestamp'] = dt
        return response
    else:
        return "404 Not Found", 404

@app.route('/comicbook', methods=['GET'])
def comicbook():
    return render_template('comicbook.html')

@app.route('/comicbookcontent', methods=['GET'])
def comicbookcontent():
    referer = request.headers.get('Referer', '')
    if referer:
        parsed_date = None
        date_param = request.args.get('dt')
        lang = request.args.get('lang', 'en')
        if lang not in ('en', 'it', 'fa'):
            lang = 'en'
        if date_param:
            try:
                from datetime import datetime
                parsed_date = datetime.fromisoformat(date_param)
            except Exception:
                parsed_date = None
        comic_html, dt, arc_id = get_comicbook(parsed_date, lang=lang)
        response = make_response(rewrite_comic_images(comic_html))
        response.headers['Timestamp'] = dt
        response.headers['Arc-Id'] = arc_id or ""
        return response
    else:
        return "404 Not Found", 404


@app.route('/cbimg', methods=['GET'])
def cbimg():
    """Resized-WebP proxy for ComicBook panel images. Lazily transcodes + caches a
    derivative in blob storage, then redirects the browser to it. Only serves images
    from our own container (open-relay guard)."""
    u = request.args.get('u', '')
    try:
        w = int(request.args.get('w', '768'))
    except ValueError:
        w = 768

    target = ensure_webp_variant(u, w)
    if not target:
        # Transcode failed (or unknown width) but the source is still ours: fall back
        # to the original PNG so the panel renders. Reject anything outside our container.
        target = u if blob_name_if_ours(u) else None
    if not target:
        return "Not found", 404

    response = redirect(target, code=302)
    response.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
    return response

@app.route('/comicbookindex', methods=['GET'])
def comicbookindex():
    return jsonify({"episodes": get_episode_index(), "arcs": get_arc_list()})
    
# @app.route('/tomorrownewsreact', methods=['GET'])
# def tomorrownewsreact():
#     parsed_date = None
#     date_param = request.args.get('dt')
#     if date_param:
#         try:
#             # Parse the date parameter to a Python datetime object
#             from datetime import datetime
#             parsed_date = datetime.fromisoformat(date_param)
#         except:
#             parsed_date = None
#     tomorrownews, datetime = gettomorrownews_react(parsed_date)
#     # Create a response object and add a custom header
#     response = make_response(tomorrownews)
#     response.headers['Timestamp'] = datetime  # Replace 'Custom-Header' and 'CustomValue' with your desired values
#     return response

@app.route('/aiblog', methods=['GET'])
def aiblog():
    return render_template('aiblog.html')

@app.route('/aiblogcontent', methods=['GET'])
async def aiblogcontent():
    referer = request.headers.get('Referer', '')
    if referer:
        parsed_date = None
        date_param = request.args.get('dt')
        if date_param:
            try:
                # Parse the date parameter to a Python datetime object
                from datetime import datetime
                parsed_date = datetime.fromisoformat(date_param)
            except:
                parsed_date = None
        aiblogcontent, datetime = await getaiblog(parsed_date)
        # Create a response object and add a custom header
        response = make_response(aiblogcontent)
        response.headers['Timestamp'] = datetime  # Replace 'Custom-Header' and 'CustomValue' with your desired values
        return response
    else:
        return "404 Not Found", 404

@app.route('/genbox')
def genbox():
    return render_template('tv.html')

def _parse_date_arg(date_param):
    if not date_param:
        return None
    try:
        from datetime import datetime
        return datetime.fromisoformat(date_param).date()
    except ValueError:
        return None


def _parse_range(range_header, size):
    """Parse an HTTP Range header into (start, end) inclusive, or None."""
    try:
        units, rng = range_header.split("=", 1)
        if units.strip().lower() != "bytes":
            return None
        start_s, end_s = rng.split("-", 1)
        if start_s.strip() == "":
            suffix = int(end_s)
            if suffix <= 0:
                return None
            start, end = max(0, size - suffix), size - 1
        else:
            start = int(start_s)
            end = int(end_s) if end_s.strip() else size - 1
        if start > end or start >= size:
            return None
        return start, min(end, size - 1)
    except Exception:
        return None


@app.route('/genbox-video-status', methods=['GET'])
def genbox_video_status():
    # Lazily kicks off background generation on the first poll for an eligible date,
    # then reports {status, video_url}. Non-blocking.
    return jsonify(ensure_generation_started(_parse_date_arg(request.args.get('date'))))


_AUDIO_MIMES = {"mp3": "audio/mpeg", "wav": "audio/wav", "opus": "audio/ogg",
                "aac": "audio/aac", "flac": "audio/flac"}


def _serve_blob(blob_name, mimetype):
    """Stream a blob from the genbox-video container with HTTP Range support, so media
    plays/seeks regardless of the container's public-access setting (same-origin proxy)."""
    try:
        size = get_video_blob_size(blob_name)
    except Exception:
        return ("Not found", 404)
    rng = _parse_range(request.headers.get("Range"), size) if request.headers.get("Range") else None
    if rng:
        start, end = rng
        data = download_video_blob_bytes(blob_name, offset=start, length=end - start + 1)
        resp = Response(data, status=206, mimetype=mimetype)
        resp.headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    else:
        data = download_video_blob_bytes(blob_name)
        resp = Response(data, mimetype=mimetype)
    resp.headers["Accept-Ranges"] = "bytes"
    resp.headers["Content-Length"] = str(len(data))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route('/genbox-video', methods=['GET'])
def genbox_video():
    st = video_status(_parse_date_arg(request.args.get('date')))
    if st.get("status") != "ready" or not st.get("video_url"):
        return ("Video not ready", 404)
    return _serve_blob(video_blob_name_from_url(st["video_url"]), "video/mp4")


@app.route('/genbox-audio', methods=['GET'])
def genbox_audio():
    # Streams the TTS narration MP3 (stored in the same blob container as the video).
    st = video_status(_parse_date_arg(request.args.get('date')))
    if st.get("audio_status") != "ready" or not st.get("audio_url"):
        return ("Audio not ready", 404)
    blob_name = video_blob_name_from_url(st["audio_url"])
    ext = blob_name.rsplit(".", 1)[-1].lower()
    return _serve_blob(blob_name, _AUDIO_MIMES.get(ext, "audio/mpeg"))

@app.route('/ai-open-problem-solver', methods=['GET'])
def ai_open_problem_solver():
    default_problem = os.environ.get("AIOPS_DEFAULT_PROBLEM", "Riemann Hypothesis")
    description = ""
    progress_percent = None
    progress_comment = ""
    if default_problem:
        details = get_problem_details(default_problem)
        if details:
            description = details.get("description", "")
        status = get_problem_progress(default_problem)
        progress_percent = status.get("progress_percent")
        progress_comment = status.get("progress_comment", "")
    return render_template(
        'aiopenproblemsolver.html',
        default_problem=default_problem,
        default_problem_description=description,
        default_problem_progress_percent=progress_percent,
        default_problem_progress_comment=progress_comment,
    )

@app.route('/ai-open-problem-solver/history', methods=['GET'])
async def ai_open_problem_solver_history():
    problem = request.args.get("problem") or os.environ.get("AIOPS_DEFAULT_PROBLEM")
    if not problem:
        return jsonify({"error": "Missing 'problem' parameter."}), 400

    try:
        offset = int(request.args.get("offset", 0))
        limit = int(request.args.get("limit", 5))
    except ValueError:
        return jsonify({"error": "Offset and limit must be integers."}), 400

    ensure_latest = request.args.get("ensure_latest", "false").lower() == "true"
    limit = max(1, min(limit, 10))

    history = await get_problem_history(
        problem,
        offset=offset,
        limit=limit,
        ensure_latest=ensure_latest,
    )
    return jsonify({"problem": problem, **history})

@app.route('/ai-open-problem-solver/problems', methods=['GET'])
def ai_open_problem_solver_problems():
    problems = list_problems()
    return jsonify({"problems": problems})


@app.route('/ai-open-problem-solver/problem-details', methods=['GET'])
def ai_open_problem_solver_problem_details():
    problem = request.args.get("problem")
    if not problem:
        return jsonify({"error": "Missing 'problem' parameter."}), 400
    details = get_problem_details(problem)
    progress = get_problem_progress(problem)
    if not details:
        return jsonify(
            {
                "problem": problem,
                "description": "",
                "progress_percent": progress.get("progress_percent"),
                "progress_comment": progress.get("progress_comment"),
            }
        )
    return jsonify(
        {
            "problem": problem,
            "description": details.get("description", ""),
            "progress_percent": progress.get("progress_percent"),
            "progress_comment": progress.get("progress_comment"),
        }
    )

@app.route('/traide', methods=['GET'])
def traide():
    return render_template('traide.html')


def _traide_guard():
    """Block hot-linking of the data endpoints; the page itself sets a /traide Referer."""
    return '/traide' in request.headers.get('Referer', '')


@app.route('/traide/live', methods=['GET'])
def traide_live():
    if not _traide_guard():
        return jsonify({"error": "forbidden"}), 403
    return jsonify(traide_get_live_snapshot())


@app.route('/traide/equity', methods=['GET'])
def traide_equity():
    if not _traide_guard():
        return jsonify({"error": "forbidden"}), 403
    import time as _time
    period = (request.args.get('period') or 'all').lower()
    today = int(_time.time() // 86400)
    span = {"day": 1, "week": 7, "month": 31, "all": None}.get(period, None)
    start = (today - span) if span else None
    return jsonify({"period": period, "points": traide_get_equity_series(start_day=start)})


@app.route('/traide/feed', methods=['GET'])
def traide_feed():
    if not _traide_guard():
        return jsonify({"error": "forbidden"}), 403
    try:
        limit = max(1, min(int(request.args.get('limit', 30)), 100))
    except ValueError:
        limit = 30
    return jsonify({"items": traide_get_decision_feed(limit)})


@app.route('/traide/trades', methods=['GET'])
def traide_trades():
    if not _traide_guard():
        return jsonify({"error": "forbidden"}), 403
    try:
        limit = max(1, min(int(request.args.get('limit', 100)), 200))
    except ValueError:
        limit = 100
    return jsonify({"items": traide_get_closed_trades(limit)})


@app.route('/traide/plans', methods=['GET'])
def traide_plans():
    if not _traide_guard():
        return jsonify({"error": "forbidden"}), 403
    import time as _time
    try:
        limit = max(1, min(int(request.args.get('limit', 40)), 100))
    except ValueError:
        limit = 40
    try:
        days = max(1, min(int(request.args.get('days', 3)), 14))
    except ValueError:
        days = 3
    start = int(_time.time() // 86400) - days + 1   # inclusive window of the last `days` days
    return jsonify({"items": traide_get_plans(limit, start_day=start)})


@app.route('/')
def home():
    # The /traide page stays reachable by URL; this flag only hides its card on the landing page.
    return render_template('index.html', hide_traide=_env_flag("HIDE_TRAIDE_PAGE", True))

if __name__ == '__main__':
    app.run(debug=True, port=5050)
