# Release Notes

## Unreleased – Dependency observability refresh
- Evidence recorder JSONL output now uses `model_dump_json()` so Pydantic 3 will stay on track and log consumers keep working.
- Dependency tracking now counts failures per dependency, records the last failure timestamp and error, exposes `status`, totals, and timestamps through `/admin/dependencies`, and also exports `eleanor_dependency_failures_total` / `eleanor_dependency_failure_last_timestamp_seconds` Prometheus metrics for alerting.
- The Metrics dashboard gained summary cards, per-dependency failure details, and an alert recommendation tied to `/admin/dependencies total_failures > 0`, and the updated UI bundle is committed so deployments get the richer experience immediately.
