#!/usr/bin/env bash
# End-to-end staging bootstrap for governance observability.
set -euo pipefail

echo "Deploying governance alert rules..."
scripts/deploy_governance_alerts.sh

echo "Reloading Prometheus..."
scripts/reload_prometheus.sh

echo "Importing Grafana dashboard..."
scripts/deploy_grafana_dashboard.sh

echo "Governance observability bootstrap complete."
