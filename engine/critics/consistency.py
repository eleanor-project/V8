"""
ELEANOR V8 — ConsistencyEngine
-------------------------------

Validates mutual consistency across critic evaluations to ensure:
1. Intra-critic consistency: Violations within a critic don't contradict
2. Inter-critic consistency: Critics don't make contradictory claims
3. Severity-violation alignment: Severity scores match violation counts
4. Evidence consistency: Rationales support reported severities

Returns consistency metrics that feed into uncertainty quantification
without modifying or rejecting critic outputs.
"""

from typing import Dict, Any, List
import statistics


class ConsistencyEngine:
    """
    ConsistencyEngine validates critic evaluation consistency.

    Design principles:
    - Non-destructive: Never modifies critic outputs
    - Advisory: Provides metrics, doesn't reject evaluations
    - Transparent: Returns detailed consistency analysis for audit
    - Integrated: Feeds into UncertaintyEngine via consistency scores
    """

    def __init__(self):
        # Severity-violation alignment thresholds
        self.severity_thresholds = {
            0.0: (0, 0),      # No violations expected
            1.0: (1, 2),      # Minor: 1-2 violations
            2.0: (2, 5),      # Moderate: 2-5 violations
            3.0: (3, 100),    # Severe: 3+ violations
        }

        # Inter-critic contradiction patterns
        self.contradiction_patterns = [
            {
                "critics": ["truth", "risk"],
                "pattern": "high_confidence_with_high_risk",
                "check": self._check_truth_risk_contradiction
            },
            {
                "critics": ["fairness", "rights"],
                "pattern": "equity_with_rights_violation",
                "check": self._check_fairness_rights_contradiction
            },
            {
                "critics": ["pragmatics", "risk"],
                "pattern": "feasible_with_irreversible_harm",
                "check": self._check_pragmatics_risk_contradiction
            }
        ]

    def validate(self, critics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main entry point for consistency validation.

        Args:
            critics: Dictionary of critic_name -> critic_output

        Returns:
            Consistency analysis with:
            - overall_consistency_score: 0-1 (1 = fully consistent)
            - intra_critic_issues: List of within-critic inconsistencies
            - inter_critic_issues: List of cross-critic contradictions
            - severity_alignment_issues: List of severity-violation mismatches
            - recommendations: Suggested actions for inconsistencies
        """

        # Validate each critic internally
        intra_issues = self._validate_intra_critic_consistency(critics)

        # Check for inter-critic contradictions
        inter_issues = self._validate_inter_critic_consistency(critics)

        # Check severity-violation alignment
        severity_issues = self._validate_severity_alignment(critics)

        # Compute overall consistency score
        consistency_score = self._compute_consistency_score(
            intra_issues, inter_issues, severity_issues
        )

        return {
            "overall_consistency_score": consistency_score,
            "intra_critic_issues": intra_issues,
            "inter_critic_issues": inter_issues,
            "severity_alignment_issues": severity_issues,
            "total_issues": len(intra_issues) + len(inter_issues) + len(severity_issues),
            "recommendations": self._generate_recommendations(
                intra_issues, inter_issues, severity_issues
            ),
            "audit_flags": self._generate_audit_flags(consistency_score)
        }

    # ----------------------------------------------------------
    # Intra-critic consistency validation
    # ----------------------------------------------------------
    def _validate_intra_critic_consistency(
        self, critics: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Check for contradictions within each critic's evaluation.

        Examples:
        - Severity 0.0 but multiple violations listed
        - Contradictory violation categories
        - Rationale doesn't support severity
        """
        issues = []

        for critic_name, evaluation in critics.items():
            severity = evaluation.get("severity", 0.0)
            violations = evaluation.get("violations", [])
            justification = evaluation.get("justification", "")

            # Check 1: Severity-violation count mismatch
            violation_count = len(violations)
            expected_range = self._get_expected_violation_range(severity)

            if not (expected_range[0] <= violation_count <= expected_range[1]):
                issues.append({
                    "critic": critic_name,
                    "type": "severity_violation_mismatch",
                    "severity": severity,
                    "violation_count": violation_count,
                    "expected_range": expected_range,
                    "description": f"{critic_name}: Severity {severity} with {violation_count} violations (expected {expected_range})"
                })

            # Check 2: Zero severity with violations
            if severity < 0.1 and violation_count > 0:
                issues.append({
                    "critic": critic_name,
                    "type": "zero_severity_with_violations",
                    "severity": severity,
                    "violation_count": violation_count,
                    "description": f"{critic_name}: Zero severity but {violation_count} violations listed"
                })

            # Check 3: High severity with no justification
            if severity >= 2.0 and len(justification.strip()) < 10:
                issues.append({
                    "critic": critic_name,
                    "type": "high_severity_without_justification",
                    "severity": severity,
                    "justification_length": len(justification),
                    "description": f"{critic_name}: High severity ({severity}) with insufficient justification"
                })

        return issues

    def _get_expected_violation_range(self, severity: float) -> tuple:
        """Map severity score to expected violation count range."""
        if severity < 0.5:
            return (0, 0)
        elif severity < 1.5:
            return (1, 2)
        elif severity < 2.5:
            return (2, 5)
        else:
            return (3, 100)

    # ----------------------------------------------------------
    # Inter-critic consistency validation
    # ----------------------------------------------------------
    def _validate_inter_critic_consistency(
        self, critics: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Check for contradictions between different critics.

        Examples:
        - Truth critic says "high confidence" while Risk critic says "irreversible harm"
        - Fairness critic says "equitable" while Rights critic says "discrimination"
        """
        issues = []

        for pattern in self.contradiction_patterns:
            # Check if all required critics are present
            required_critics = pattern["critics"]
            if all(c in critics for c in required_critics):
                contradiction = pattern["check"](critics)
                if contradiction:
                    issues.append({
                        "type": "inter_critic_contradiction",
                        "pattern": pattern["pattern"],
                        "critics_involved": required_critics,
                        "description": contradiction,
                    })

        return issues

    def _check_truth_risk_contradiction(self, critics: Dict[str, Dict[str, Any]]) -> str:
        """Check for truth-risk contradictions."""
        truth = critics.get("truth", {})
        risk = critics.get("risk", {})

        truth_severity = truth.get("severity", 0.0)
        risk_severity = risk.get("severity", 0.0)

        # Contradiction: High truth confidence (low severity) + high risk
        if truth_severity < 1.0 and risk_severity >= 2.5:
            return f"Truth critic shows high confidence (severity {truth_severity}) while Risk critic flags severe harm (severity {risk_severity})"

        return ""

    def _check_fairness_rights_contradiction(self, critics: Dict[str, Dict[str, Any]]) -> str:
        """Check for fairness-rights contradictions."""
        fairness = critics.get("fairness", {})
        rights = critics.get("rights", {})

        fairness_severity = fairness.get("severity", 0.0)
        rights_severity = rights.get("severity", 0.0)

        # Contradiction: Low fairness concern + high rights violation
        # (Both should flag discrimination)
        if fairness_severity < 1.0 and rights_severity >= 2.5:
            rights_violations = rights.get("violations", [])
            if any("discrimination" in str(v).lower() for v in rights_violations):
                return f"Rights critic flags discrimination (severity {rights_severity}) while Fairness critic shows no concern (severity {fairness_severity})"

        return ""

    def _check_pragmatics_risk_contradiction(self, critics: Dict[str, Dict[str, Any]]) -> str:
        """Check for pragmatics-risk contradictions."""
        pragmatics = critics.get("pragmatics", {})
        risk = critics.get("risk", {})

        pragmatics_severity = pragmatics.get("severity", 0.0)
        risk_severity = risk.get("severity", 0.0)

        # Contradiction: Highly feasible + irreversible harm
        if pragmatics_severity < 1.0 and risk_severity >= 2.5:
            risk_violations = risk.get("violations", [])
            if any("irreversible" in str(v).lower() for v in risk_violations):
                return f"Pragmatics critic shows high feasibility (severity {pragmatics_severity}) while Risk critic flags irreversible harm (severity {risk_severity})"

        return ""

    # ----------------------------------------------------------
    # Severity alignment validation
    # ----------------------------------------------------------
    def _validate_severity_alignment(
        self, critics: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Validate that severity scores align with violation evidence.
        """
        issues = []

        for critic_name, evaluation in critics.items():
            severity = evaluation.get("severity", 0.0)
            violations = evaluation.get("violations", [])
            evidence = evaluation.get("evidence", {})

            # Check if evidence supports the severity
            if severity >= 2.0:  # Constitutional threshold
                if not violations:
                    issues.append({
                        "critic": critic_name,
                        "type": "high_severity_no_violations",
                        "severity": severity,
                        "description": f"{critic_name}: Constitutional severity ({severity}) with no violations"
                    })

                if not evidence or len(evidence) == 0:
                    issues.append({
                        "critic": critic_name,
                        "type": "high_severity_no_evidence",
                        "severity": severity,
                        "description": f"{critic_name}: Constitutional severity ({severity}) with no evidence"
                    })

        return issues

    # ----------------------------------------------------------
    # Consistency score computation
    # ----------------------------------------------------------
    def _compute_consistency_score(
        self,
        intra_issues: List[Dict[str, Any]],
        inter_issues: List[Dict[str, Any]],
        severity_issues: List[Dict[str, Any]]
    ) -> float:
        """
        Compute overall consistency score (0-1, where 1 = fully consistent).

        Weighting:
        - Intra-critic issues: 0.4 weight (most critical)
        - Inter-critic issues: 0.3 weight
        - Severity alignment: 0.3 weight
        """
        total_issues = len(intra_issues) + len(inter_issues) + len(severity_issues)

        if total_issues == 0:
            return 1.0

        # Weight different issue types
        weighted_score = (
            len(intra_issues) * 0.4 +
            len(inter_issues) * 0.3 +
            len(severity_issues) * 0.3
        )

        # Normalize to 0-1 range (assume max 10 issues is complete inconsistency)
        consistency = max(0.0, 1.0 - (weighted_score / 10.0))

        return round(consistency, 3)

    # ----------------------------------------------------------
    # Recommendations and audit flags
    # ----------------------------------------------------------
    def _generate_recommendations(
        self,
        intra_issues: List[Dict[str, Any]],
        inter_issues: List[Dict[str, Any]],
        severity_issues: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate actionable recommendations based on consistency issues."""
        recommendations = []

        if intra_issues:
            recommendations.append(
                f"Review {len(intra_issues)} intra-critic inconsistencies for potential critic logic errors"
            )

        if inter_issues:
            recommendations.append(
                f"Investigate {len(inter_issues)} inter-critic contradictions - may indicate complex edge case"
            )

        if severity_issues:
            recommendations.append(
                f"Verify {len(severity_issues)} severity-evidence misalignments for potential calibration issues"
            )

        # High-priority recommendation
        total_issues = len(intra_issues) + len(inter_issues) + len(severity_issues)
        if total_issues >= 5:
            recommendations.insert(0, "ESCALATE: High inconsistency detected - consider human review")

        return recommendations

    def _generate_audit_flags(self, consistency_score: float) -> List[str]:
        """Generate audit flags based on consistency score."""
        flags = []

        if consistency_score < 0.5:
            flags.append("CRITICAL_INCONSISTENCY")
        elif consistency_score < 0.7:
            flags.append("MODERATE_INCONSISTENCY")
        elif consistency_score < 0.9:
            flags.append("MINOR_INCONSISTENCY")
        else:
            flags.append("CONSISTENT")

        return flags
