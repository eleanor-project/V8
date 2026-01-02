#!/usr/bin/env python3
"""
Lightweight performance regression gate for pytest-benchmark JSON output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _load_payload(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid benchmark JSON: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Check benchmark results against thresholds.")
    parser.add_argument(
        "--input",
        default="benchmark.json",
        help="Path to pytest-benchmark JSON output (default: benchmark.json).",
    )
    parser.add_argument(
        "--threshold-ms",
        type=float,
        default=float(os.getenv("ELEANOR_BENCHMARK_THRESHOLD_MS", "1000")),
        help="Mean runtime threshold in milliseconds (default: 1000).",
    )
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"No benchmark output found at {path}", file=sys.stderr)
        return 1

    payload = _load_payload(path)
    benchmarks = payload.get("benchmarks", [])
    if not benchmarks:
        print("No benchmarks found in JSON output.", file=sys.stderr)
        return 1

    failures: list[str] = []
    for bench in benchmarks:
        name = bench.get("name", "unknown")
        stats = bench.get("stats", {})
        mean = stats.get("mean")
        if mean is None:
            continue
        mean_ms = float(mean) * 1000.0
        if mean_ms > args.threshold_ms:
            failures.append(f"{name}: {mean_ms:.2f}ms > {args.threshold_ms:.2f}ms")

    if failures:
        print("Performance regression detected:", file=sys.stderr)
        for item in failures:
            print(f"- {item}", file=sys.stderr)
        return 1

    print(f"Performance checks passed (threshold={args.threshold_ms:.2f}ms).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
