#!/usr/bin/env python3
"""Generate detailed coverage report with actionable items."""
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any


def run_coverage_analysis() -> Dict[str, Any]:
    """Run pytest with coverage and return results."""
    print("Running test suite with coverage analysis...")
    
    result = subprocess.run(
        [
            "pytest",
            "--cov=engine",
            "--cov=governance",
            "--cov=api",
            "--cov-report=json:coverage.json",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "-v"
        ],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    # Load coverage data
    with open("coverage.json") as f:
        return json.load(f)


def analyze_coverage_gaps(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Identify files below 95% coverage threshold."""
    threshold = 95.0
    below_threshold = []
    
    for file_path, metrics in data["files"].items():
        # Skip test files and __init__ files
        if "test_" in file_path or "__init__.py" in file_path:
            continue
        
        coverage_pct = metrics["summary"]["percent_covered"]
        if coverage_pct < threshold:
            below_threshold.append({
                "file": file_path,
                "coverage": coverage_pct,
                "missing_lines": len(metrics["missing_lines"]),
                "missing_branches": metrics["summary"].get("missing_branches", 0),
                "total_statements": metrics["summary"]["num_statements"]
            })
    
    return sorted(below_threshold, key=lambda x: x["coverage"])


def generate_report(data: Dict[str, Any], gaps: List[Dict[str, Any]]) -> bool:
    """Generate detailed coverage report."""
    total_coverage = data["totals"]["percent_covered"]
    
    print("\n" + "="*70)
    print("ELEANOR V8 TEST COVERAGE REPORT")
    print("="*70)
    print(f"\nOverall Coverage: {total_coverage:.2f}%")
    print(f"Target: 95.00%")
    print(f"Gap: {max(0, 95.0 - total_coverage):.2f}%")
    
    if total_coverage >= 95.0:
        print("\n✅ Coverage target achieved!")
    else:
        print("\n❌ Coverage below target")
    
    if gaps:
        print(f"\n{len(gaps)} files below 95% coverage:\n")
        
        # Group by severity
        critical = [g for g in gaps if g["coverage"] < 70]
        high = [g for g in gaps if 70 <= g["coverage"] < 85]
        medium = [g for g in gaps if 85 <= g["coverage"] < 95]
        
        if critical:
            print("🔴 CRITICAL (<70%):")
            for item in critical:
                print(f"  {item['file']}: {item['coverage']:.1f}% "
                      f"({item['missing_lines']} lines, {item['missing_branches']} branches)")
        
        if high:
            print("\n🟠 HIGH (70-85%):")
            for item in high:
                print(f"  {item['file']}: {item['coverage']:.1f}% "
                      f"({item['missing_lines']} lines, {item['missing_branches']} branches)")
        
        if medium:
            print("\n🟡 MEDIUM (85-95%):")
            for item in medium:
                print(f"  {item['file']}: {item['coverage']:.1f}% "
                      f"({item['missing_lines']} lines, {item['missing_branches']} branches)")
    else:
        print("\n✅ All files above 95% coverage!")
    
    print("\n" + "="*70)
    print(f"HTML report generated: htmlcov/index.html")
    print("="*70 + "\n")
    
    return total_coverage >= 95.0


def main():
    """Main entry point."""
    try:
        data = run_coverage_analysis()
        gaps = analyze_coverage_gaps(data)
        success = generate_report(data, gaps)
        
        sys.exit(0 if success else 1)
    
    except FileNotFoundError:
        print("Error: coverage.json not found. Did pytest run successfully?")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
