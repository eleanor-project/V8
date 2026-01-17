# Test Coverage Baseline (95% Target)

Date: 2026-01-17

## Summary

- Command: `python3 scripts/coverage_report.py`
- Overall coverage: **65.35%**
- Test run status: **FAILED** (39 failed, 10 errors, 846 passed, 12 skipped; 222 warnings)

Because the test run failed, coverage totals may be skewed. Fixing the failing
integration/load tests is the first priority before measuring progress toward
95%.

## Critical files below target

- `engine/security/audit.py` – 46.0%
- `engine/resource_manager.py` – 55.8%
- `engine/engine.py` – 81.8%

## Zero coverage modules (top offenders)

- `engine/aggregator/adaptive_weighting.py`
- `engine/audit/immutable_log.py`
- `engine/database/pool.py`
- `engine/precedent/temporal_evolution.py`
- `engine/router/intelligent_selector.py`
- `engine/security/audit/alerts.py`
- `engine/security/audit/siem.py`
- `engine/security/input_validation.py`

## Immediate next steps

1. Fix failing integration/load tests (see `tests/chaos/` and `tests/integration/`).
2. Re-run coverage to get a clean baseline.
3. Prioritize tests for critical files listed above.
