#!/usr/bin/env bash
# Deploy governance alert rules into a Prometheus rules directory and validate them.
set -euo pipefail

RULES_SRC="$(pwd)/monitoring/prometheus/governance-alerts.yml"
PROM_DIR="${PROMETHEUS_RULES_DIR:-$(pwd)/monitoring/prometheus}"
DEST="$PROM_DIR/governance-escalations.yml"
PROM_CONTAINER="${PROMETHEUS_CONTAINER:-prometheus}"

if ! command -v promtool &> /dev/null; then
  if command -v docker &> /dev/null && docker ps --format '{{.Names}}' | grep -q "^${PROM_CONTAINER}$"; then
    echo "promtool not found locally; falling back to docker exec ${PROM_CONTAINER} promtool"
    docker exec "${PROM_CONTAINER}" promtool check rules /etc/prometheus/rules/governance-alerts.yml
    exit 0
  fi
  echo "promtool not found; install it or ensure a running Prometheus container is available."
  exit 1
fi

echo "Validating governance alert rules..."
promtool check rules "${RULES_SRC}"

mkdir -p "$PROM_DIR"
cp "$RULES_SRC" "$DEST"

echo "Governance alerts deployed to ${DEST}"
echo "Reload Prometheus (e.g., via 'kill -HUP <prometheus pid>' or API) to pick up the new rules."
