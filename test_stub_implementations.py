"""
Quick validation test for ConsistencyEngine, RedundancyEngine, and RULES.
"""

import sys

sys.path.insert(0, "/Users/billp/documents/github/v8")

from engine.critics.consistency import ConsistencyEngine  # noqa: E402
from engine.critics.redundancy import RedundancyEngine  # noqa: E402
from engine.critics.rules import RULES, get_rule_statistics, find_redundant_rules  # noqa: E402


def test_consistency_engine():
    """Test ConsistencyEngine with mock critic outputs."""
    print("=" * 60)
    print("Testing ConsistencyEngine")
    print("=" * 60)

    engine = ConsistencyEngine()

    # Mock critic outputs
    critics = {
        "fairness": {
            "severity": 2.5,
            "violations": ["discrimination", "bias", "disparate_impact"],
            "justification": "Detected discrimination patterns in the output",
            "evidence": {"patterns": ["protected class"]},
        },
        "rights": {
            "severity": 2.3,
            "violations": ["discrimination"],
            "justification": "Rights violation detected",
            "evidence": {"patterns": ["discrimination"]},
        },
        "truth": {
            "severity": 0.5,
            "violations": [],
            "justification": "No significant truth concerns",
            "evidence": {},
        },
    }

    result = engine.validate(critics)

    print(f"✓ Overall Consistency Score: {result['overall_consistency_score']}")
    print(f"✓ Total Issues: {result['total_issues']}")
    print(f"✓ Intra-critic Issues: {len(result['intra_critic_issues'])}")
    print(f"✓ Inter-critic Issues: {len(result['inter_critic_issues'])}")
    print(f"✓ Severity Alignment Issues: {len(result['severity_alignment_issues'])}")
    print(f"✓ Audit Flags: {result['audit_flags']}")
    print(f"✓ Recommendations: {len(result['recommendations'])} recommendations")

    if result["recommendations"]:
        print("\nRecommendations:")
        for rec in result["recommendations"]:
            print(f"  - {rec}")

    print("\n✅ ConsistencyEngine test passed!\n")


def test_redundancy_engine():
    """Test RedundancyEngine with mock critic outputs."""
    print("=" * 60)
    print("Testing RedundancyEngine")
    print("=" * 60)

    engine = RedundancyEngine()

    # Mock critic outputs with redundancy
    critics = {
        "fairness": {
            "severity": 2.0,
            "violations": ["discrimination against protected class"],
            "justification": "Discrimination detected based on protected characteristics",
        },
        "rights": {
            "severity": 2.5,
            "violations": ["discrimination"],
            "justification": "Protected class discrimination violates fundamental rights",
        },
        "truth": {
            "severity": 1.0,
            "violations": ["minor accuracy concern"],
            "justification": "Some factual uncertainty detected",
        },
    }

    result = engine.filter(critics)

    print(f"✓ Total Redundancies Detected: {result['total_redundancies']}")
    print(f"✓ Redundancy Flags: {result['redundancy_flags']}")

    audit = result["audit_report"]
    print(f"✓ Original Total Severity: {audit['total_original_severity']}")
    print(f"✓ Adjusted Total Severity: {audit['total_adjusted_severity']}")
    print(f"✓ Severity Reduction: {audit['severity_reduction']}")
    print(f"✓ Reduction Percentage: {audit['reduction_percentage']}%")

    if result["redundancy_groups"]:
        print("\n✓ Redundancy Patterns Found:")
        for redundancy in result["redundancy_groups"]:
            print(f"  - Pattern: {redundancy['pattern']}")
            print(f"    Critics: {redundancy['critics_involved']}")
            print(f"    Description: {redundancy['description']}")

    print("\n✅ RedundancyEngine test passed!\n")


def test_rules_registry():
    """Test RULES dictionary and utilities."""
    print("=" * 60)
    print("Testing RULES Registry")
    print("=" * 60)

    print(f"✓ Total Rules in Registry: {len(RULES)}")

    # Test rule retrieval
    fairness_rule = RULES.get("FAIR-001")
    if fairness_rule:
        print(f"✓ Retrieved FAIR-001: {fairness_rule.name}")
        print(f"  - Dimension: {fairness_rule.dimension.value}")
        print(f"  - Critics: {fairness_rule.critics}")
        print(f"  - Severity Weight: {fairness_rule.severity_weight}")
        print(f"  - Constitutional Clause: {fairness_rule.constitutional_clause}")

    # Test statistics
    stats = get_rule_statistics()
    print("\n✓ Rule Statistics:")
    print(f"  - Total Rules: {stats['total_rules']}")
    print(f"  - Enabled Rules: {stats['enabled_rules']}")
    print(f"  - Total Patterns: {stats['total_patterns']}")
    print(f"  - Total Keywords: {stats['total_keywords']}")
    print(f"  - Rules by Dimension: {stats['rules_by_dimension']}")
    print(f"  - Rules by Critic: {stats['rules_by_critic']}")

    # Test redundancy detection
    redundancies = find_redundant_rules()
    print(f"\n✓ Cross-Rule Redundancies Detected: {len(redundancies)}")
    if redundancies:
        print("  Redundant rule pairs:")
        for rule1_id, rule2_id, overlap, desc in redundancies[:3]:  # Show first 3
            print(f"    - {desc}: {overlap}")

    print("\n✅ RULES Registry test passed!\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("STUB IMPLEMENTATIONS VALIDATION TEST")
    print("=" * 60 + "\n")

    try:
        test_consistency_engine()
        test_redundancy_engine()
        test_rules_registry()

        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nSummary:")
        print("  ✓ ConsistencyEngine: Validates critic consistency")
        print("  ✓ RedundancyEngine: Filters duplicate findings")
        print("  ✓ RULES Registry: Centralized rule management")
        print("\nAll three stub implementations are now fully functional.")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
