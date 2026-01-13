#!/usr/bin/env python3
"""Generate detailed coverage report with actionable items."""
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any


def run_coverage() -> Dict[str, Any]:
    """Run pytest with coverage and return data."""
    print("Running coverage analysis...")
    result = subprocess.run(
        [
            "pytest",
            "--cov=engine",
            "--cov=governance",
            "--cov=api",
            "--cov-report=json:coverage.json",
            "--cov-report=term-missing",
            "-v"
        ],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    
    try:
        with open("coverage.json") as f:
            return json.load(f)
    except FileNotFoundError:
        print("ERROR: coverage.json not found. Tests may have failed.")
        sys.exit(1)


def analyze_coverage(data: Dict[str, Any], threshold: float = 95.0) -> List[Dict[str, Any]]:
    """Analyze coverage data and identify gaps."""
    below_threshold = []
    
    for filepath, metrics in data.get("files", {}).items():
        summary = metrics.get("summary", {})
        coverage_pct = summary.get("percent_covered", 0)
        
        if coverage_pct < threshold:
            below_threshold.append({
                "file": filepath,
                "coverage": coverage_pct,
                "missing_lines": len(metrics.get("missing_lines", [])),
                "excluded_lines": len(metrics.get("excluded_lines", [])),
                "num_statements": summary.get("num_statements", 0),
                "num_covered": summary.get("covered_lines", 0)
            })
    
    return sorted(below_threshold, key=lambda x: x["coverage"])


def generate_report(below_threshold: List[Dict[str, Any]], total_coverage: float):
    """Generate formatted coverage report."""
    print("\n" + "="*80)
    print(f"COVERAGE REPORT - Target: 95%")
    print("="*80)
    print(f"\nOverall Coverage: {total_coverage:.2f}%")
    
    if not below_threshold:
        print("\n✅ All files meet the 95% coverage threshold!")
        return True
    
    print(f"\n❌ {len(below_threshold)} files below 95% coverage threshold:\n")
    
    for item in below_threshold:
        print(f"📄 {item['file']}")
        print(f"   Coverage: {item['coverage']:.1f}%")
        print(f"   Missing Lines: {item['missing_lines']}")
        print(f"   Statements: {item['num_covered']}/{item['num_statements']}")
        print()
    
    # Priority files that need immediate attention
    critical_files = [
        "engine/engine.py",
        "engine/validation.py",
        "engine/security/audit.py",
        "engine/resource_manager.py"
    ]
    
    critical_gaps = [item for item in below_threshold 
                     if any(cf in item['file'] for cf in critical_files)]
    
    if critical_gaps:
        print("\n⚠️  CRITICAL FILES BELOW THRESHOLD:")
        for item in critical_gaps:
            print(f"   - {item['file']}: {item['coverage']:.1f}%")
    
    return False


def main():
    """Main execution."""
    print("ELEANOR V8 - Test Coverage Analysis")
    print("="*80 + "\n")
    
    # Run coverage
    data = run_coverage()
    
    # Analyze results
    total_coverage = data.get("totals", {}).get("percent_covered", 0)
    below_threshold = analyze_coverage(data, threshold=95.0)
    
    # Generate report
    success = generate_report(below_threshold, total_coverage)
    
    # Exit with appropriate code
    if success or total_coverage >= 95.0:
        print("\n✅ Coverage target achieved!")
        sys.exit(0)
    else:
        print(f"\n❌ Coverage ({total_coverage:.2f}%) below 95% target")
        sys.exit(1)


if __name__ == "__main__":
    main()
