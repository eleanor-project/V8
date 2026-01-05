# Release Notes

## Unreleased – Dependency observability refresh
- Evidence recorder JSONL output now uses `model_dump_json()` so Pydantic 3 will stay on track and log consumers keep working.
- Dependency tracking now counts failures per dependency, records the last failure timestamp and error, and exposes `status`, totals, and timestamps through `/admin/dependencies`, which is now guarded by a stronger smoke test.
- The Metrics dashboard gained summary cards, per-dependency failure details, and an alert recommendation tied to `/admin/dependencies total_failures > 0`, and the updated UI bundle is committed so deployments get the richer experience immediately.
