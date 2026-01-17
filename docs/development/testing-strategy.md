# Testing Strategy

Testing is centered around pytest with unit, integration, and benchmark suites.

## Test layout

- `tests/` contains unit and integration tests.
- `tests/benchmarks/` contains performance benchmarks.

## Core tools

- `pytest` for test execution.
- `pytest-asyncio` for async tests.
- `hypothesis` for property-based coverage.
- `pytest-benchmark` for performance baselines.

## Common commands

```bash
pytest
pytest tests/test_engine_validation.py -q
pytest tests/benchmarks --benchmark-only
```

## What to validate

- Governance decisions and escalation logic.
- Critic outputs and aggregation behavior.
- Resilience fallbacks and degraded-mode behavior.
- Configuration loading and environment overrides.

See `docs/guides/testing.md` for practical setup steps.
