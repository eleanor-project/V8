#!/usr/bin/env bash
# Import the governance escalations dashboard into Grafana.
set -euo pipefail

GRAFANA_URL="${GRAFANA_URL:-http://localhost:3000}"
GRAFANA_API_TOKEN="${GRAFANA_API_TOKEN:-}"
GRAFANA_USER="${GRAFANA_USER:-}"
GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-}"
FOLDER_ID="${GRAFANA_FOLDER_ID:-0}"

DASHBOARD_PATH="$(pwd)/monitoring/grafana/governance-escalations-dashboard.json"

if [ ! -f "$DASHBOARD_PATH" ]; then
  echo "Dashboard file not found: $DASHBOARD_PATH"
  exit 1
fi

if [ -z "$GRAFANA_API_TOKEN" ] && { [ -z "$GRAFANA_USER" ] || [ -z "$GRAFANA_PASSWORD" ]; }; then
  echo "Set GRAFANA_API_TOKEN or GRAFANA_USER/GRAFANA_PASSWORD for Grafana auth."
  exit 1
fi

payload="$(python3 - <<'PY'
import json
from pathlib import Path

dashboard_path = Path("monitoring/grafana/governance-escalations-dashboard.json")
dashboard = json.loads(dashboard_path.read_text())
request = {
    "dashboard": dashboard,
    "overwrite": True,
    "folderId": int(__import__("os").getenv("GRAFANA_FOLDER_ID", "0")),
}
print(json.dumps(request))
PY
)"

auth_header=()
if [ -n "$GRAFANA_API_TOKEN" ]; then
  auth_header=(-H "Authorization: Bearer ${GRAFANA_API_TOKEN}")
fi

echo "Importing governance escalations dashboard into Grafana..."
if [ -n "$GRAFANA_API_TOKEN" ]; then
  curl -sS -X POST "${GRAFANA_URL}/api/dashboards/db" \
    "${auth_header[@]}" \
    -H "Content-Type: application/json" \
    -d "$payload"
else
  curl -sS -X POST "${GRAFANA_URL}/api/dashboards/db" \
    -u "${GRAFANA_USER}:${GRAFANA_PASSWORD}" \
    -H "Content-Type: application/json" \
    -d "$payload"
fi

echo ""
echo "Grafana dashboard imported."
