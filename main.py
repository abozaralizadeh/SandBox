import os
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, jsonify, make_response, request, render_template
from AIBlog.prompt import *
from TomorrowNews.prompt import *
from GenBox.prompt import get_llm_response
from AIOpenProblemSolver.prompt import get_problem_history
from AIOpenProblemSolver.azurestorage import list_problems

app = Flask(__name__)

cache = {}

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
        if date_param:
            try:
                # Parse the date parameter to a Python datetime object
                from datetime import datetime
                parsed_date = datetime.fromisoformat(date_param)
            except:
                parsed_date = None
        tomorrownews, datetime = gettomorrownews(parsed_date)
        # Create a response object and add a custom header
        response = make_response(tomorrownews)
        response.headers['Timestamp'] = datetime  # Replace 'Custom-Header' and 'CustomValue' with your desired values
        return response
    else:
        return "404 Not Found", 404
    
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

@app.route('/ai-open-problem-solver', methods=['GET'])
def ai_open_problem_solver():
    default_problem = os.environ.get("AIOPS_DEFAULT_PROBLEM", "Riemann Hypothesis")
    return render_template('aiopenproblemsolver.html', default_problem=default_problem)

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

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5050)
