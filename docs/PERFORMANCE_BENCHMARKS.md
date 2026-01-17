# Performance Benchmarks

This repository uses `pytest-benchmark` to track baseline performance for core validation
and engine execution paths.

## Benchmarks

Benchmarks live in `tests/benchmarks/`:

- `test_benchmark_validation.py` — input validation throughput
- `test_benchmark_engine.py` — end-to-end engine runs (detail levels 1 and 3)

## Running Benchmarks

```bash
pytest tests/benchmarks --benchmark-only --benchmark-json=benchmark.json
```

## Baseline

A baseline snapshot is stored at:

```
tests/benchmarks/baseline.json
```

To update the baseline:

```bash
pytest tests/benchmarks --benchmark-only --benchmark-json=tests/benchmarks/baseline.json
```

## Regression Checks

Use `scripts/compare_benchmarks.py` to compare current results to the baseline:

```bash
python scripts/compare_benchmarks.py benchmark.json tests/benchmarks/baseline.json --fail-on-regression 10
```

The command exits non-zero if any benchmark regresses beyond the configured threshold.
