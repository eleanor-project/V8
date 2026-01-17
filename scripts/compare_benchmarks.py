"""Compare pytest-benchmark JSON outputs and fail on regressions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple


def _load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _index_by_name(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {bench["name"]: bench for bench in data.get("benchmarks", [])}


def _compare_metric(current: float, baseline: float) -> Tuple[float, float]:
    if baseline == 0:
        return 0.0, 0.0
    delta = current - baseline
    pct = (delta / baseline) * 100.0
    return delta, pct


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare benchmark JSON results.")
    parser.add_argument("current", type=Path, help="Current benchmark JSON")
    parser.add_argument("baseline", type=Path, help="Baseline benchmark JSON")
    parser.add_argument(
        "--fail-on-regression",
        type=float,
        default=10.0,
        help="Fail if mean latency regresses by this percent",
    )
    args = parser.parse_args()

    if not args.baseline.exists():
        print(f"Baseline not found: {args.baseline}. Skipping comparison.")
        return 0

    current = _load(args.current)
    baseline = _load(args.baseline)

    current_map = _index_by_name(current)
    baseline_map = _index_by_name(baseline)

    regressions = []
    for name, bench in current_map.items():
        if name not in baseline_map:
            print(f"No baseline for {name}; skipping.")
            continue
        current_mean = bench.get("stats", {}).get("mean")
        baseline_mean = baseline_map[name].get("stats", {}).get("mean")
        if current_mean is None or baseline_mean is None:
            continue
        _, pct = _compare_metric(current_mean, baseline_mean)
        print(f"{name}: {pct:+.2f}% vs baseline")
        if pct > args.fail_on_regression:
            regressions.append((name, pct))

    if regressions:
        print("\nRegressions detected:")
        for name, pct in regressions:
            print(f"- {name}: {pct:+.2f}%")
        return 1

    print("\nNo regressions detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
