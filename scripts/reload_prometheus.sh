#!/usr/bin/env bash
# Reload Prometheus configuration via HTTP API.
set -euo pipefail

PROM_URL="${PROMETHEUS_URL:-http://localhost:9090}"

echo "Reloading Prometheus at ${PROM_URL}/-/reload..."
curl -sS -X POST "${PROM_URL}/-/reload"
echo ""
echo "Prometheus reload triggered."
