# Release Notes

## Unreleased – Dependency observability refresh
- Evidence recorder JSONL output now uses `model_dump_json()` so Pydantic 3 will stay on track and log consumers keep working.
- Dependency tracking now counts failures per dependency, records the last failure timestamp and error, exposes `status`, totals, and timestamps through `/admin/dependencies`, and also exports `eleanor_dependency_failures_total` / `eleanor_dependency_failure_last_timestamp_seconds` Prometheus metrics for alerting.
- Prometheus exporters can now read those metrics via `python scripts/dependency_prometheus_exporter.py` if you need a sidecar with different labels/ports, and Grafana alerts (e.g., `sum(eleanor_dependency_failures_total)>0 for 5m`) can be wired to your incident channel.
- The Metrics dashboard gained summary cards, per-dependency failure details, and an alert recommendation tied to `/admin/dependencies total_failures > 0`, and the updated UI bundle is committed so deployments get the richer experience immediately.
