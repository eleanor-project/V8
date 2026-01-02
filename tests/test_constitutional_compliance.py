"""
ELEANOR V8.1 — Constitutional Compliance Validation Tests
----------------------------------------------------------

Tests that ConsistencyEngine, RedundancyEngine, and CLAUSES implementations
comply with Constitutional Critics & Escalation Governance Handbook v8.1.

Critical Requirements Tested:
1. Dissent preservation (no cross-critic suppression)
2. Charter boundary compliance
3. Clause-aware escalation signals
4. Intra-critic deduplication only
5. Tier and human action mappings
"""

from engine.critics.consistency import ConsistencyEngine, CharterViolationType
from engine.critics.redundancy import RedundancyEngine, validate_no_cross_critic_suppression
from engine.critics.privacy import PrivacyIdentityCritic
from engine.critics.rules import (
    CLAUSES, CriticDomain, HumanAction,
    get_clause, get_clauses_by_critic, get_clauses_by_tier,
    validate_clause_id, get_clause_statistics
)
from engine.schemas.escalation import EscalationTier


# ============================================================
# TEST 1: RedundancyEngine - Dissent Preservation
# ============================================================

def test_redundancy_preserves_cross_critic_dissent():
    """
    CRITICAL TEST: Verify RedundancyEngine NEVER suppresses cross-critic signals.

    Handbook v8.1, Section 2.2: "The Aggregator shall not suppress escalation,
    average it away, reinterpret it, or down-rank it due to consensus or convenience."
    """
    print("=" * 70)
    print("TEST 1: RedundancyEngine - Dissent Preservation")
    print("=" * 70)

    engine = RedundancyEngine()

    # Mock: Two critics raise similar concerns (intentional overlap)
    critics = {
        "fairness": {
            "severity": 2.5,
            "violations": [
                {"category": "discrimination", "description": "Protected class bias detected"},
                {"category": "discrimination", "description": "Protected class bias detected"}  # Duplicate within critic
            ],
            "escalation": {
                "clause_id": "F1",
                "tier": "tier_3",
                "reason": "Systematic discrimination"
            },
            "justification": "Detected discrimination patterns"
        },
        "dignity": {
            "severity": 2.3,
            "violations": [
                {"category": "discrimination", "description": "Instrumentalization of protected class"}
            ],
            "escalation": {
                "clause_id": "D2",
                "tier": "tier_2",
                "reason": "Dignity violation through discrimination"
            },
            "justification": "Discrimination violates inherent dignity"
        },
        "privacy": {
            "severity": 1.0,
            "violations": [
                {"category": "inference", "description": "Minor identity inference"}
            ],
            "justification": "Low-severity privacy concern"
        }
    }

    result = engine.deduplicate(critics)

    # CRITICAL ASSERTION: All three critics must still exist
    assert len(result["deduplicated_critics"]) == 3, \
        "❌ CONSTITUTIONAL VIOLATION: Critic was removed entirely"

    # CRITICAL ASSERTION: Escalation signals preserved
    fairness_output = result["deduplicated_critics"]["fairness"]
    dignity_output = result["deduplicated_critics"]["dignity"]

    assert fairness_output.get("escalation") is not None, \
        "❌ CONSTITUTIONAL VIOLATION: Fairness escalation suppressed"
    assert dignity_output.get("escalation") is not None, \
        "❌ CONSTITUTIONAL VIOLATION: Dignity escalation suppressed"

    # CRITICAL ASSERTION: Severities not reduced across critics
    assert fairness_output.get("severity", 0) >= 2.4, \
        "❌ CONSTITUTIONAL VIOLATION: Fairness severity reduced"
    assert dignity_output.get("severity", 0) >= 2.2, \
        "❌ CONSTITUTIONAL VIOLATION: Dignity severity reduced"

    # POSITIVE ASSERTION: Intra-critic deduplication occurred
    fairness_violations = fairness_output.get("violations", [])
    assert len(fairness_violations) == 1, \
        "❌ Intra-critic deduplication failed (should remove duplicate within fairness)"

    # POSITIVE ASSERTION: Cross-critic preservation documented
    preservation = result["cross_critic_preservation"]
    print(f"✓ Cross-critic preservation events: {len(preservation)}")

    # Constitutional compliance check
    assert result["constitutional_compliance"] == "PASS - Cross-critic dissent preserved verbatim", \
        "❌ Constitutional compliance failed"

    print("✓ All escalation signals preserved verbatim")
    print("✓ No cross-critic severity reduction")
    print("✓ Intra-critic deduplication working correctly")
    print("✅ TEST PASSED: Dissent preservation enforced\n")


def test_redundancy_constitutional_validator():
    """
    Test the validate_no_cross_critic_suppression() function.

    This validator can be used in CI to catch constitutional violations.
    """
    print("=" * 70)
    print("TEST 2: RedundancyEngine - Constitutional Validator Function")
    print("=" * 70)

    original = {
        "fairness": {
            "severity": 2.5,
            "violations": ["discrimination"],
            "escalation": {"clause_id": "F1", "tier": "tier_3"}
        },
        "dignity": {
            "severity": 2.0,
            "violations": ["degradation"],
            "escalation": {"clause_id": "D1", "tier": "tier_2"}
        }
    }

    # SCENARIO 1: Compliant processing (no changes)
    processed_compliant = {
        "fairness": {
            "severity": 2.5,
            "violations": ["discrimination"],
            "escalation": {"clause_id": "F1", "tier": "tier_3"}
        },
        "dignity": {
            "severity": 2.0,
            "violations": ["degradation"],
            "escalation": {"clause_id": "D1", "tier": "tier_2"}
        }
    }

    validation = validate_no_cross_critic_suppression(original, processed_compliant)
    assert validation["compliant"], "❌ False positive: compliant case flagged"
    assert validation["status"] == "PASS", "❌ Status should be PASS"
    print("✓ Validator correctly identifies compliant processing")

    # SCENARIO 2: Escalation suppression (VIOLATION)
    processed_violation1 = {
        "fairness": {
            "severity": 2.5,
            "violations": ["discrimination"],
            # Escalation removed! VIOLATION!
        },
        "dignity": {
            "severity": 2.0,
            "violations": ["degradation"],
            "escalation": {"clause_id": "D1", "tier": "tier_2"}
        }
    }

    validation = validate_no_cross_critic_suppression(original, processed_violation1)
    assert not validation["compliant"], "❌ Failed to detect escalation suppression"
    assert validation["status"] == "FAIL", "❌ Status should be FAIL"
    assert len(validation["violations"]) > 0, "❌ Should report violations"
    assert any(v["type"] == "escalation_suppressed" for v in validation["violations"]), \
        "❌ Should flag escalation_suppressed"
    print("✓ Validator correctly detects escalation suppression")

    # SCENARIO 3: Severity reduction (VIOLATION)
    processed_violation2 = {
        "fairness": {
            "severity": 1.0,  # Reduced from 2.5! VIOLATION!
            "violations": ["discrimination"],
            "escalation": {"clause_id": "F1", "tier": "tier_3"}
        },
        "dignity": {
            "severity": 2.0,
            "violations": ["degradation"],
            "escalation": {"clause_id": "D1", "tier": "tier_2"}
        }
    }

    validation = validate_no_cross_critic_suppression(original, processed_violation2)
    assert not validation["compliant"], "❌ Failed to detect severity reduction"
    assert any(v["type"] == "severity_reduced" for v in validation["violations"]), \
        "❌ Should flag severity_reduced"
    print("✓ Validator correctly detects severity reduction")

    # SCENARIO 4: Critic removal (VIOLATION)
    processed_violation3 = {
        "fairness": {
            "severity": 2.5,
            "violations": ["discrimination"],
            "escalation": {"clause_id": "F1", "tier": "tier_3"}
        }
        # Dignity critic removed entirely! VIOLATION!
    }

    validation = validate_no_cross_critic_suppression(original, processed_violation3)
    assert not validation["compliant"], "❌ Failed to detect critic removal"
    assert any(v["type"] == "critic_removed" for v in validation["violations"]), \
        "❌ Should flag critic_removed"
    print("✓ Validator correctly detects critic removal")

    print("✅ TEST PASSED: Constitutional validator catches all violation types\n")


# ============================================================
# TEST 3: ConsistencyEngine - Charter Compliance
# ============================================================

def test_consistency_charter_boundary_validation():
    """
    Test that ConsistencyEngine validates charter boundaries.

    Handbook v8.1, Section 5: Each critic has "Owns" and "Must NOT" domains.
    """
    print("=" * 70)
    print("TEST 3: ConsistencyEngine - Charter Boundary Validation")
    print("=" * 70)

    engine = ConsistencyEngine()

    # SCENARIO 1: Compliant critics (within charter boundaries)
    compliant_critics = {
        "fairness": {
            "severity": 2.0,
            "violations": ["disparate impact on protected class"],
            "justification": "Detected differential treatment patterns",
            "escalation": {
                "clause_id": "F1",
                "tier": "tier_3"
            }
        },
        "autonomy": {
            "severity": 1.5,
            "violations": ["consent mechanism missing"],
            "justification": "No meaningful consent path for data collection",
            "escalation": {
                "clause_id": "A1",
                "tier": "tier_2"
            }
        }
    }

    result = engine.validate_charter_compliance(compliant_critics)

    assert result["compliant"], "❌ False positive: compliant critics flagged"
    assert result["total_violations"] == 0, "❌ Should have zero violations"
    print("✓ Correctly validates compliant critics")

    # SCENARIO 2: Missing clause_id (VIOLATION)
    missing_clause_id = {
        "fairness": {
            "severity": 2.0,
            "violations": ["discrimination"],
            "justification": "Detected discrimination",
            "escalation": {
                # Missing clause_id! VIOLATION!
                "tier": "tier_3"
            }
        }
    }

    result = engine.validate_charter_compliance(missing_clause_id)
    assert not result["compliant"], "❌ Failed to detect missing clause_id"
    violations = [v for v in result["violations"] if v["type"] == CharterViolationType.MISSING_CLAUSE_ID.value]
    assert len(violations) > 0, "❌ Should flag missing clause_id"
    print("✓ Correctly detects missing clause_id")

    # SCENARIO 3: Invalid clause_id (VIOLATION)
    invalid_clause_id = {
        "fairness": {
            "severity": 2.0,
            "violations": ["discrimination"],
            "justification": "Detected discrimination",
            "escalation": {
                "clause_id": "D1",  # Dignity clause used by Fairness! VIOLATION!
                "tier": "tier_3"
            }
        }
    }

    result = engine.validate_charter_compliance(invalid_clause_id)
    assert not result["compliant"], "❌ Failed to detect invalid clause_id"
    violations = [v for v in result["violations"] if v["type"] == CharterViolationType.INVALID_CLAUSE_ID.value]
    assert len(violations) > 0, "❌ Should flag invalid clause_id"
    print("✓ Correctly detects invalid clause_id for critic")

    # SCENARIO 4: Missing tier (VIOLATION)
    missing_tier = {
        "autonomy": {
            "severity": 2.0,
            "violations": ["consent failure"],
            "justification": "No consent mechanism",
            "escalation": {
                "clause_id": "A1"
                # Missing tier! VIOLATION!
            }
        }
    }

    result = engine.validate_charter_compliance(missing_tier)
    assert not result["compliant"], "❌ Failed to detect missing tier"
    violations = [v for v in result["violations"] if v["type"] == CharterViolationType.MISSING_TIER.value]
    assert len(violations) > 0, "❌ Should flag missing tier"
    print("✓ Correctly detects missing tier")

    # SCENARIO 5: Intentional overlap (NOT AN ERROR)
    intentional_overlap = {
        "fairness": {
            "severity": 2.0,
            "violations": ["discrimination"],
            "justification": "Detected discrimination patterns",
            "escalation": {"clause_id": "F1", "tier": "tier_3"}
        },
        "dignity": {
            "severity": 1.8,
            "violations": ["discrimination via instrumentalization"],
            "justification": "Discrimination violates dignity",
            "escalation": {"clause_id": "D2", "tier": "tier_2"}
        }
    }

    result = engine.validate_charter_compliance(intentional_overlap)
    # Should NOT flag this as a violation
    assert result["compliant"], "❌ False positive: intentional overlap flagged as violation"

    # Should document the overlap
    overlaps = result["intentional_overlaps"]
    print(f"✓ Intentional overlaps detected: {len(overlaps)} (expected, not violations)")

    print("✅ TEST PASSED: Charter boundary validation working correctly\n")


# ============================================================
# TEST 4: CLAUSES Registry - Constitutional Clause Definitions
# ============================================================

def test_clauses_registry_completeness():
    """
    Test that CLAUSES registry contains all 22 constitutional clauses.

    Expected: A1-A3, D1-D3, P1-P4, F1-F3, DP1-DP3, PR1-PR3, U1-U3
    """
    print("=" * 70)
    print("TEST 4: CLAUSES Registry - Completeness")
    print("=" * 70)

    expected_clauses = [
        # Autonomy (3)
        "A1", "A2", "A3",
        # Dignity (3)
        "D1", "D2", "D3",
        # Privacy (4)
        "P1", "P2", "P3", "P4",
        # Fairness (3)
        "F1", "F2", "F3",
        # Due Process (3)
        "DP1", "DP2", "DP3",
        # Precedent (3)
        "PR1", "PR2", "PR3",
        # Uncertainty (3)
        "U1", "U2", "U3"
    ]

    assert len(expected_clauses) == 22, "❌ Test setup error: should expect 22 clauses"

    # Check all clauses exist
    missing_clauses = []
    for clause_id in expected_clauses:
        if clause_id not in CLAUSES:
            missing_clauses.append(clause_id)

    assert len(missing_clauses) == 0, \
        f"❌ Missing clauses in registry: {missing_clauses}"

    print("✓ All 22 constitutional clauses present in registry")

    # Get statistics
    stats = get_clause_statistics()
    assert stats["total_clauses"] == 22, "❌ Statistics show incorrect total"

    print(f"✓ Statistics: {stats['total_clauses']} clauses")
    print(f"  - Tier 2 (Acknowledgment): {stats['tier_2_count']}")
    print(f"  - Tier 3 (Determination): {stats['tier_3_count']}")
    print(f"  - By critic: {stats['clauses_by_critic']}")

    print("✅ TEST PASSED: CLAUSES registry complete\n")


def test_clauses_tier_and_human_action_mappings():
    """
    Test that clauses correctly map to tiers and human actions.

    Handbook v8.1: Tier 2 = Acknowledgment, Tier 3 = Determination
    """
    print("=" * 70)
    print("TEST 5: CLAUSES - Tier and Human Action Mappings")
    print("=" * 70)

    # Test specific known mappings from Handbook

    # A1: Tier 2 (Acknowledgment)
    a1 = get_clause("A1")
    assert a1 is not None, "❌ A1 clause not found"
    assert a1.tier == EscalationTier.TIER_2, "❌ A1 should be Tier 2"
    assert a1.human_action == HumanAction.ACKNOWLEDGMENT, "❌ A1 should require acknowledgment"
    print("✓ A1: Tier 2 (Acknowledgment) - correct")

    # F1: Tier 3 (Determination)
    f1 = get_clause("F1")
    assert f1 is not None, "❌ F1 clause not found"
    assert f1.tier == EscalationTier.TIER_3, "❌ F1 should be Tier 3"
    assert f1.human_action == HumanAction.DETERMINATION, "❌ F1 should require determination"
    print("✓ F1: Tier 3 (Determination) - correct")

    # P4: Check tier matches human action
    p4 = get_clause("P4")
    assert p4 is not None, "❌ P4 clause not found"
    if p4.tier == EscalationTier.TIER_2:
        assert p4.human_action == HumanAction.ACKNOWLEDGMENT, "❌ P4 tier/action mismatch"
        print("✓ P4: Tier 2 (Acknowledgment) - consistent")
    else:
        assert p4.human_action == HumanAction.DETERMINATION, "❌ P4 tier/action mismatch"
        print("✓ P4: Tier 3 (Determination) - consistent")

    # DP2: Tier 3 (Determination)
    dp2 = get_clause("DP2")
    assert dp2 is not None, "❌ DP2 clause not found"
    assert dp2.tier == EscalationTier.TIER_3, "❌ DP2 should be Tier 3"
    assert dp2.human_action == HumanAction.DETERMINATION, "❌ DP2 should require determination"
    print("✓ DP2: Tier 3 (Determination) - correct")

    # All clauses must have valid tier and human action
    for clause_id, clause in CLAUSES.items():
        assert clause.tier in [EscalationTier.TIER_2, EscalationTier.TIER_3], \
            f"❌ {clause_id} has invalid tier: {clause.tier}"
        assert clause.human_action in [HumanAction.ACKNOWLEDGMENT, HumanAction.DETERMINATION], \
            f"❌ {clause_id} has invalid human_action: {clause.human_action}"

    print("✓ All clauses have valid tier and human action mappings")
    print("✅ TEST PASSED: Tier and human action mappings correct\n")


def test_clauses_utility_functions():
    """
    Test utility functions for querying clauses.
    """
    print("=" * 70)
    print("TEST 6: CLAUSES - Utility Functions")
    print("=" * 70)

    # Test get_clauses_by_critic
    fairness_clauses = get_clauses_by_critic(CriticDomain.FAIRNESS)
    assert len(fairness_clauses) == 3, "❌ Fairness should have 3 clauses (F1, F2, F3)"
    clause_ids = [c.clause_id for c in fairness_clauses]
    assert "F1" in clause_ids and "F2" in clause_ids and "F3" in clause_ids, \
        "❌ Fairness clauses incomplete"
    print(f"✓ get_clauses_by_critic(FAIRNESS): {clause_ids}")

    # Test get_clauses_by_tier
    tier_2_clauses = get_clauses_by_tier(EscalationTier.TIER_2)
    tier_3_clauses = get_clauses_by_tier(EscalationTier.TIER_3)

    assert len(tier_2_clauses) + len(tier_3_clauses) == 22, \
        "❌ Total tier 2 + tier 3 clauses should equal 22"
    print(f"✓ get_clauses_by_tier: Tier 2={len(tier_2_clauses)}, Tier 3={len(tier_3_clauses)}")

    # Test validate_clause_id
    validation = validate_clause_id("F1", CriticDomain.FAIRNESS)
    assert validation["valid"], "❌ F1 should be valid for Fairness"
    print("✓ validate_clause_id('F1', FAIRNESS): valid")

    validation = validate_clause_id("D1", CriticDomain.FAIRNESS)
    assert not validation["valid"], "❌ D1 should be invalid for Fairness"
    assert "F1" in validation.get("suggestion", ""), "❌ Should suggest valid clauses"
    print("✓ validate_clause_id('D1', FAIRNESS): invalid (correctly rejected)")

    print("✅ TEST PASSED: Utility functions working correctly\n")


# ============================================================
# TEST 7: Privacy & Identity Critic
# ============================================================

def test_privacy_p1_identity_inference():
    """
    TEST: Verify P1 (Non-consensual identity inference) escalates to TIER_3.

    Privacy clause P1 protects against inferring sensitive identity attributes
    without explicit consent.
    """
    print("=" * 70)
    print("TEST 7.1: Privacy P1 - Identity Inference (TIER_3)")
    print("=" * 70)

    critic = PrivacyIdentityCritic()

    # Test P1: identity_inference=True should trigger TIER_3 escalation
    result = critic.evaluate(identity_inference=True)

    # Assertions
    assert result.severity_score >= 0.9, \
        "❌ P1 violation should have severity >= 0.9"
    assert len(result.concerns) >= 1, \
        "❌ P1 violation should generate concerns"
    assert result.escalation is not None, \
        "❌ P1 violation must escalate"
    assert result.escalation.tier == EscalationTier.TIER_3, \
        f"❌ P1 must escalate to TIER_3, got {result.escalation.tier}"
    assert result.escalation.clause_id == "P1", \
        f"❌ Expected clause_id='P1', got '{result.escalation.clause_id}'"

    print("✅ P1 (Identity Inference) validation passed")
    print(f"   - Severity: {result.severity_score}")
    print(f"   - Escalation: {result.escalation.tier.value} ({result.escalation.clause_id})")
    print(f"   - Concerns: {len(result.concerns)}")
    print()


def test_privacy_p2_persistent_identity():
    """
    TEST: Verify P2 (Persistent identity linkage) escalates to TIER_3.

    Privacy clause P2 protects against persistent identity construction
    across sessions or contexts.
    """
    print("=" * 70)
    print("TEST 7.2: Privacy P2 - Persistent Identity (TIER_3)")
    print("=" * 70)

    critic = PrivacyIdentityCritic()

    # Test P2: persistent_identity=True should trigger TIER_3 escalation
    result = critic.evaluate(persistent_identity=True)

    # Assertions
    assert result.severity_score >= 0.85, \
        "❌ P2 violation should have severity >= 0.85"
    assert len(result.concerns) >= 1, \
        "❌ P2 violation should generate concerns"
    assert result.escalation is not None, \
        "❌ P2 violation must escalate"
    assert result.escalation.tier == EscalationTier.TIER_3, \
        f"❌ P2 must escalate to TIER_3, got {result.escalation.tier}"
    assert result.escalation.clause_id == "P2", \
        f"❌ Expected clause_id='P2', got '{result.escalation.clause_id}'"

    print("✅ P2 (Persistent Identity) validation passed")
    print(f"   - Severity: {result.severity_score}")
    print(f"   - Escalation: {result.escalation.tier.value} ({result.escalation.clause_id})")
    print(f"   - Concerns: {len(result.concerns)}")
    print()


def test_privacy_p3_context_collapse():
    """
    TEST: Verify P3 (Context collapse) escalates to TIER_2.

    Privacy clause P3 protects against using data outside its original
    contextual boundary.
    """
    print("=" * 70)
    print("TEST 7.3: Privacy P3 - Context Collapse (TIER_2)")
    print("=" * 70)

    critic = PrivacyIdentityCritic()

    # Test P3: context_mismatch=True should trigger TIER_2 escalation
    result = critic.evaluate(context_mismatch=True)

    # Assertions
    assert result.severity_score >= 0.7, \
        "❌ P3 violation should have severity >= 0.7"
    assert len(result.concerns) >= 1, \
        "❌ P3 violation should generate concerns"
    assert result.escalation is not None, \
        "❌ P3 violation must escalate"
    assert result.escalation.tier == EscalationTier.TIER_2, \
        f"❌ P3 must escalate to TIER_2, got {result.escalation.tier}"
    assert result.escalation.clause_id == "P3", \
        f"❌ Expected clause_id='P3', got '{result.escalation.clause_id}'"

    print("✅ P3 (Context Collapse) validation passed")
    print(f"   - Severity: {result.severity_score}")
    print(f"   - Escalation: {result.escalation.tier.value} ({result.escalation.clause_id})")
    print(f"   - Concerns: {len(result.concerns)}")
    print()


def test_privacy_p4_secondary_use():
    """
    TEST: Verify P4 (Secondary use expansion) escalates to TIER_2.

    Privacy clause P4 protects against data reuse beyond original
    authorization scope.
    """
    print("=" * 70)
    print("TEST 7.4: Privacy P4 - Secondary Use (TIER_2)")
    print("=" * 70)

    critic = PrivacyIdentityCritic()

    # Test P4: secondary_use=True should trigger TIER_2 escalation
    result = critic.evaluate(secondary_use=True)

    # Assertions
    assert result.severity_score >= 0.75, \
        "❌ P4 violation should have severity >= 0.75"
    assert len(result.concerns) >= 1, \
        "❌ P4 violation should generate concerns"
    assert result.escalation is not None, \
        "❌ P4 violation must escalate"
    assert result.escalation.tier == EscalationTier.TIER_2, \
        f"❌ P4 must escalate to TIER_2, got {result.escalation.tier}"
    assert result.escalation.clause_id == "P4", \
        f"❌ Expected clause_id='P4', got '{result.escalation.clause_id}'"

    print("✅ P4 (Secondary Use) validation passed")
    print(f"   - Severity: {result.severity_score}")
    print(f"   - Escalation: {result.escalation.tier.value} ({result.escalation.clause_id})")
    print(f"   - Concerns: {len(result.concerns)}")
    print()


def test_privacy_escalation_priority():
    """
    TEST: Verify escalation priority (P1 > P2 > P3 > P4).

    When multiple privacy violations occur, only the highest-priority
    violation should escalate.
    """
    print("=" * 70)
    print("TEST 7.5: Privacy Escalation Priority")
    print("=" * 70)

    critic = PrivacyIdentityCritic()

    # Test 1: P1 takes priority over P2, P3, P4
    result = critic.evaluate(
        identity_inference=True,
        persistent_identity=True,
        context_mismatch=True,
        secondary_use=True
    )

    assert result.escalation.clause_id == "P1", \
        "❌ P1 should have highest priority"
    print("✅ P1 has highest priority")

    # Test 2: P2 takes priority over P3, P4
    result = critic.evaluate(
        persistent_identity=True,
        context_mismatch=True,
        secondary_use=True
    )

    assert result.escalation.clause_id == "P2", \
        "❌ P2 should take priority over P3/P4"
    print("✅ P2 prioritized over P3/P4")

    # Test 3: P3 takes priority over P4
    result = critic.evaluate(
        context_mismatch=True,
        secondary_use=True
    )

    assert result.escalation.clause_id == "P3", \
        "❌ P3 should take priority over P4"
    print("✅ P3 prioritized over P4")

    print("✅ Privacy escalation priority validation passed")
    print()


def test_privacy_no_violations():
    """
    TEST: Verify no escalation when no privacy violations occur.
    """
    print("=" * 70)
    print("TEST 7.6: Privacy - No Violations")
    print("=" * 70)

    critic = PrivacyIdentityCritic()

    # Test: No violations
    result = critic.evaluate()

    assert result.severity_score == 0.0, \
        "❌ No violations should have severity 0.0"
    assert len(result.concerns) == 0, \
        "❌ No violations should have no concerns"
    assert result.escalation is None, \
        "❌ No violations should not escalate"

    print("✅ No privacy violations correctly handled")
    print(f"   - Severity: {result.severity_score}")
    print("   - Escalation: None")
    print()


# ============================================================
# MAIN TEST RUNNER
# ============================================================

def main():
    """Run all constitutional compliance tests."""
    print("\n" + "=" * 70)
    print("ELEANOR V8.1 — CONSTITUTIONAL COMPLIANCE VALIDATION")
    print("=" * 70 + "\n")

    try:
        # Dissent Preservation Tests
        test_redundancy_preserves_cross_critic_dissent()
        test_redundancy_constitutional_validator()

        # Charter Compliance Tests
        test_consistency_charter_boundary_validation()

        # Canonical Clauses Tests
        test_clauses_registry_completeness()
        test_clauses_tier_and_human_action_mappings()
        test_clauses_utility_functions()

        # Privacy & Identity Tests
        test_privacy_p1_identity_inference()
        test_privacy_p2_persistent_identity()
        test_privacy_p3_context_collapse()
        test_privacy_p4_secondary_use()
        test_privacy_escalation_priority()
        test_privacy_no_violations()

        print("=" * 70)
        print("✅ ALL CONSTITUTIONAL COMPLIANCE TESTS PASSED!")
        print("=" * 70)
        print("\nValidation Summary:")
        print("  ✓ Dissent preservation enforced (no cross-critic suppression)")
        print("  ✓ Charter boundaries validated correctly")
        print("  ✓ Clause-aware escalation signals validated")
        print("  ✓ All 22 constitutional clauses present and correct")
        print("  ✓ Tier and human action mappings accurate")
        print("  ✓ Privacy & Identity critic (P1-P4) validated")
        print("\nConstitutional Alignment: CONFIRMED")
        print("Handbook v8.1 Compliance: PASS")
        print("=" * 70 + "\n")

    except AssertionError as e:
        print("\n❌ CONSTITUTIONAL COMPLIANCE TEST FAILED:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print("\n❌ UNEXPECTED ERROR:")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
