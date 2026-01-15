# Governance Observability Notes
_Last updated: January 2026_

This document covers the new visibility hooks introduced around constitutional governance (human review and escalation) so operators can monitor when the engine trips the governance gate.

## Structured Logging

- **Log key**: `governance_human_review_required`
- **Location**: `engine/runtime/governance.py` inside `apply_governance_flags_to_aggregation()`
- **Level**: `INFO`
- **Logged payload**:
  * `trace_id`: correlates with API/WebSocket traces (also captured by CorrelationContext)
  * `governance_flags`: the review triggers/metadata from the governance gate
  * `execution_gate_reason`: a human-friendly explanation of why human review was required
  * `escalation_tier`: the tier assigned (`TIER_2`, `TIER_3`, etc.)

Use these logs in Structured Logging dashboards (Splunk, Datadog, ELK) to count escalations, filter by trigger, and feed into alerts (e.g., rate of `TIER_2` escalations over 5 minutes).

## Metrics

- **Metric name**: `eleanor_escalations_total`
- **Location**: `engine/observability/business_metrics.py` (`record_escalation()`) and `engine/runtime/governance.py`
- **Labels**:
  * `tier`: escalation tier (`2`, `3`, etc.)
  * `critic`: the first review trigger (if configured) or `unknown`
  * `reason`: either the first governance trigger name or the execution gate reason string (e.g., `severity_threshold_exceeded`)

Prometheus/Grafana dashboards/readouts can count escalations per tier, detect spikes during high-risk windows, and correlate with `eleanor_decisions_total`. Alert on sustained elevated rates or unexpected tier breakdowns.

## Next Steps

1. **Grafana Dashboard**: Import `monitoring/grafana/governance-escalations-dashboard.json` into Grafana (templated Prometheus datasource). Panels already include:
   * Escalations by Tier (`sum by (tier) (rate(eleanor_escalations_total[5m]))`)
   * Escalation Reasons (`topk(10, sum by (reason) (rate(...)))`)
   * Critic Escalation Heatmap (`sum by (critic, tier) (rate(...))`)
2. **Alerts & automation**:
   * Load `monitoring/prometheus/governance-alerts.yml` into your Prometheus rules directory.
   * Use `scripts/deploy_governance_alerts.sh` (or similar) to validate via `promtool` and copy the file into your rules directory.
     - If `promtool` is missing locally, the script will fall back to `docker exec prometheus promtool` when a Prometheus container is running.
   * Trigger a Prometheus reload using `scripts/reload_prometheus.sh` (defaults to `http://localhost:9090`, override with `PROMETHEUS_URL`).
   * `Tier3EscalationSurge` (>=0.01/s for 5m)
   * `FundamentalRightsEscalation` (reason includes `fundamental_rights`)
   * `EscalationSpikevsHourlyAverage` (>2× hourly average)
2. **Escalation tracking**: Optionally expose `tier` breakdown by policy profile to ensure certain workloads don’t drive disproportionate human review.
3. **Alerting**: Raise alerts when `eleanor_escalations_total` increases unexpectedly or when `execution_gate_reason` matches critical flags (e.g., `fundamental_rights_implicated`). See `monitoring/prometheus/governance-alerts.yml` for templates.
4. **Grafana import**: Run `scripts/deploy_grafana_dashboard.sh` with either `GRAFANA_API_TOKEN` or `GRAFANA_USER`/`GRAFANA_PASSWORD` to import `monitoring/grafana/governance-escalations-dashboard.json` automatically.
   * Set `GRAFANA_URL` to your staging Grafana base URL and optionally `GRAFANA_FOLDER_ID` if you want it in a specific folder.
5. **One-shot bootstrap**: `scripts/bootstrap_governance_observability.sh` chains the alert deploy, Prometheus reload, and Grafana import into one command.
4. **Reporting Tool Evaluation**: Review Grafana/Prometheus dashboards, logging sinks, and alerting workflows for opportunities:
   * Do panels show escalation counts with trace IDs so on-call engineers can pivot directly to `governance_human_review_required` logs?
   * Are there gaps in labels (e.g., missing `policy_profile` label) that hinder slicing by workload?
   * Could we annotate dashboards with governance events (HTTP requests that triggered escalations) or surface them in existing reporting tools?
 Document findings in this doc so future iterations can prioritize those enhancements.

By combining the structured log entry and the Prometheus counter, teams get both the trace-level detail and the high-level trend needed for prod readiness.
