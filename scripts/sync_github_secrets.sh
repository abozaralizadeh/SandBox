#!/usr/bin/env bash
# Copy the App Service's settings into GitHub Actions secrets for the Generate workflow.
#
# Run it from the repo root, with `az` and `gh` signed in. It reads each value
# from Azure and hands it straight to `gh secret set` on stdin, so nothing is printed.
# Re-run it whenever you change a setting (or before you delete the App Service, since
# after that this is the only copy of the settings outside your .env).
#
#   bash scripts/sync_github_secrets.sh
set -euo pipefail

RG="${RG:-sandbox_rg}"
APP="${APP:-genbox}"
REPO="${REPO:-abozaralizadeh/SandBox}"

# Platform plumbing and leftovers from other apps that SandBox never reads.
SKIP='^(APPINSIGHTS_|APPLICATIONINSIGHTS_|ApplicationInsightsAgent_|DiagnosticServices_|InstrumentationEngine_|SnapshotDebugger_|XDT_|SCM_|WEBSITE|PORT$|ENV$|AZURE_COSMOSDB_|ELASTICSEARCH_|MONGODB_|AZURE_SEARCH_|NO_|AUTH_CLIENT_SECRET$|DATASOURCE_TYPE$)'

az webapp config appsettings list -g "$RG" -n "$APP" -o json | python3 -c '
import json, re, subprocess, sys
skip, repo = re.compile(sys.argv[1]), sys.argv[2]
names = []
for s in json.load(sys.stdin):
    name, value = s["name"], s.get("value") or ""
    if skip.match(name) or not value:
        continue
    subprocess.run(["gh", "secret", "set", name, "--repo", repo],
                   input=value.encode(), check=True, stdout=subprocess.DEVNULL)
    names.append(name)
    print(f"  secret {name}")
# GitHub upper-cases secret names; this list keeps the original spelling (names only).
subprocess.run(["gh", "variable", "set", "SANDBOX_ENV_NAMES", "--repo", repo,
                "--body", ",".join(sorted(names))], check=True, stdout=subprocess.DEVNULL)
print(f"{len(names)} secrets set on {repo}, plus SANDBOX_ENV_NAMES")
# The Generate workflow names every secret it passes on (never toJSON(secrets), which GitHub
# flags as malicious). A setting that is not listed there never reaches the jobs.
wf = open(".github/workflows/generate.yml").read()
missing = [n for n in names if f"secrets.{n} " not in wf and f"secrets.{n}}}" not in wf]
if missing:
    print("NOT passed on by .github/workflows/generate.yml, add a line for each:")
    for n in missing:
        print(f"          {n}: ${{{{ secrets.{n} }}}}")
' "$SKIP" "$REPO"
