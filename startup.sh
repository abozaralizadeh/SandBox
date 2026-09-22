#!/bin/bash
# App Service startup. Three jobs: keep the App Insights agent out of Python, bootstrap
# Playwright's browser, then serve.
#
# With ApplicationInsightsAgent_EXTENSION_VERSION=~3, the platform PREPENDS /agents/python to
# PYTHONPATH, and the agent's bootstrap runs at interpreter startup and imports its OWN, older
# typing_extensions — before main.py exists. main.py's sys.path strip is therefore too late: the
# stale module is already in sys.modules, and everything after it inherits it. Since openai-agents
# pulls in mcp -> a recent anyio (`from typing_extensions import sentinel`), every worker then dies
# at import with exit code 3. This took the site down on 2026-09-22; it had also failed every
# agent-on start since at least 09-18 and only survived because Azure retried with the agent off.
# Removing it here, before any Python runs, is exactly the configuration of every successful
# start in that period.
PYTHONPATH=$(printf '%s' "${PYTHONPATH:-}" | tr ':' '\n' | grep -v '/agents/python' | paste -sd: -)
export PYTHONPATH
echo "[startup] PYTHONPATH=${PYTHONPATH}"
#
# Chromium needs system libraries (libglib2.0-0, libnss3, libatk…) that are NOT guaranteed to
# be in the App Service Python image, and the container filesystem is ephemeral — anything
# apt installs is gone on the next cold start, so this has to run every time. It broke on
# 2026-09-08 with `BrowserType.launch: Host system is missing dependencies to run browsers`,
# which took AIBlog down completely (the launch happens while building the agent, before any
# request work) and silently killed AIOPS's daily iteration.
#
# This runs in the BACKGROUND on purpose. The site failed to start twice on 2026-09-08 and was
# briefly blocked for "consecutive cold start failures", so gunicorn must come up immediately;
# a slow apt-get in front of it risks the container start-time limit and takes the whole site
# down, not just browsing. Both callers now degrade to no browse tools, so the worst case while
# this finishes is a run that searches without fetching pages.
#
# Every step prints a [playwright-setup] marker so the container log says exactly which path
# worked — `playwright install-deps` needs root, and if it is refused the apt fallback's output
# is the evidence for whether this is fixable from inside the container at all.
bootstrap_browser() {
  echo "starting; playwright=$(command -v playwright || echo MISSING) user=$(whoami)"

  if playwright install-deps chromium; then
    echo "install-deps OK"
  else
    echo "install-deps FAILED (rc=$?) — falling back to apt-get"
    if apt-get update -qq && apt-get install -y --no-install-recommends \
        libglib2.0-0 libnspr4 libnss3 libatk1.0-0 libatk-bridge2.0-0 libatspi2.0-0 \
        libx11-6 libxcomposite1 libxdamage1 libxext6 libxfixes3 libxrandr2 \
        libdrm2 libgbm1 libxcb1 libxkbcommon0 libpango-1.0-0 libcairo2 libasound2; then
      echo "apt-get OK"
    else
      echo "apt-get FAILED (rc=$?) — browsing stays disabled; AIBlog/AIOPS will run search-only"
    fi
  fi

  # Downloads into the container's own cache each cold start (~170MB). Pointing
  # PLAYWRIGHT_BROWSERS_PATH at /home would persist it across restarts, but /home is an Azure
  # Files share and running a browser binary off SMB is its own source of flakiness — left off
  # deliberately; the download is not the thing that was broken.
  if playwright install chromium; then
    echo "chromium binary ready at ${PLAYWRIGHT_BROWSERS_PATH:-default cache}"
  else
    echo "chromium download FAILED (rc=$?)"
  fi
  echo "done"
}

bootstrap_browser 2>&1 | sed -u 's/^/[playwright-setup] /' &

gunicorn --bind=0.0.0.0 --timeout 3600 --workers 4 --threads 2 main:app
