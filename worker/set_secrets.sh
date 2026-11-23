#!/usr/bin/env bash
# Give the Worker read-only access to the two trAIde objects it serves live.
#
# Makes two SAS tokens with READ permission only, one for the `traidedashboard` table and
# one for the `traide-dashboard` container, and stores them as Worker secrets. Nothing is
# printed. Run it yourself from worker/, with `az` and `wrangler` signed in. Re-run it
# before the tokens expire (EXPIRY, two years by default).
#
#   cd worker && bash set_secrets.sh
set -euo pipefail

RG="${RG:-sandbox_rg}"
APP="${APP:-genbox}"
EXPIRY="${EXPIRY:-$(python3 -c 'import datetime as d; print((d.date.today()+d.timedelta(days=730)).isoformat())')}"

CS="$(az webapp config appsettings list -g "$RG" -n "$APP" --query "[?name=='connection_string'].value | [0]" -o tsv)"
if [ -z "$CS" ]; then
  echo "connection_string not found on $APP; set CS yourself and edit this script" >&2
  exit 1
fi

az storage table generate-sas --name traidedashboard --permissions r --https-only \
  --expiry "$EXPIRY" --connection-string "$CS" -o tsv \
  | tr -d '\n' | npx wrangler secret put TRAIDE_TABLE_SAS

az storage container generate-sas --name traide-dashboard --permissions r --https-only \
  --expiry "$EXPIRY" --connection-string "$CS" -o tsv \
  | tr -d '\n' | npx wrangler secret put TRAIDE_BLOB_SAS

echo "TRAIDE_TABLE_SAS and TRAIDE_BLOB_SAS set, valid until $EXPIRY"
