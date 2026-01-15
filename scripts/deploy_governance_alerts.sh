#!/usr/bin/env bash
# Deploy governance alert rules into a Prometheus rules directory and validate them.
set -euo pipefail

RULES_SRC="$(pwd)/monitoring/prometheus/governance-alerts.yml"
PROM_DIR="${PROMETHEUS_RULES_DIR:-/etc/prometheus/rules}"
DEST="$PROM_DIR/governance-escalations.yml"

if ! command -v promtool &> /dev/null; then
  echo "promtool not found; install it (https://prometheus.io/docs/prometheus/latest/getting_started/)"
  exit 1
fi

echo "Validating governance alert rules..."
promtool check rules "${RULES_SRC}"

mkdir -p "$PROM_DIR"
cp "$RULES_SRC" "$DEST"

echo "Governance alerts deployed to ${DEST}"
echo "Reload Prometheus (e.g., via 'kill -HUP <prometheus pid>' or API) to pick up the new rules."
