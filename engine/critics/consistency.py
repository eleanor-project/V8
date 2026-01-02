"""
ELEANOR V8.1 — ConsistencyEngine (Constitutional Charter Compliance Validator)
-------------------------------------------------------------------------------

PURPOSE: Validates that critics follow their constitutional charters and emit
proper clause-aware escalation signals.

CRITICAL: This is a DIAGNOSTIC and VALIDATION tool for critic implementation.
It does NOT filter, suppress, or modify critic outputs at runtime.

Constitutional Principles (from Handbook v8.1):
1. Critics must stay within their charter boundaries
2. Dissent is expected and valued - contradictions are often intentional
3. Escalation signals must be clause-aware (A1, D2, P4, etc.)
4. Cross-critic overlap is intentional, not an error

Usage:
- During critic development: Validate charter compliance
- In CI/testing: Ensure critics don't violate boundaries
- For audit: Detect implementation drift from charters
- NEVER for runtime filtering or dissent suppression
"""

from typing import Dict, Any, List
from enum import Enum


class CriticDomain(Enum):
    """Canonical critic domains from Handbook v8.1."""
    AUTONOMY = "autonomy"
    DIGNITY = "dignity"
    PRIVACY = "privacy"
    FAIRNESS = "fairness"
    DUE_PROCESS = "due_process"
    PRECEDENT = "precedent"
    UNCERTAINTY = "uncertainty"


class CharterViolationType(Enum):
    """Types of charter boundary violations."""
    DOMAIN_OVERSTEP = "domain_overstep"
    MISSING_CLAUSE_ID = "missing_clause_id"
    INVALID_CLAUSE_ID = "invalid_clause_id"
    MISSING_TIER = "missing_tier"
    INVALID_TIER = "invalid_tier"


class ConsistencyEngine:
    """
    Constitutional Charter Compliance Validator.

    Validates that critic implementations follow their charters as defined
    in the Constitutional Critics & Escalation Governance Handbook v8.1.

    Design Principles:
    - DIAGNOSTIC ONLY: Never modifies critic outputs
    - PRESERVES DISSENT: Intentional overlap is not an error
    - CHARTER-FOCUSED: Validates boundaries, not outcomes
    - CI/AUDIT TOOL: For implementation validation, not runtime filtering
    """

    def __init__(self):
        # Charter boundaries from Handbook Section 5
        self.charter_boundaries = {
            "autonomy": {
                "owns": ["consent", "agency", "coercion", "reversibility"],
                "must_not": ["dignity_judgment", "fairness_judgment", "outcome_scoring"],
                "valid_clauses": ["A1", "A2", "A3"],
                "intentional_overlap": ["privacy"]  # Only in consent framing
            },
            "dignity": {
                "owns": ["intrinsic_worth", "non_degradation", "moral_presence", "instrumentalization"],
                "must_not": ["fairness_math", "consent_checks"],
                "valid_clauses": ["D1", "D2", "D3"],
                "intentional_overlap": ["due_process"]  # Voice vs appeal
            },
            "privacy": {
                "owns": ["identity_inference", "persistence", "linkage", "contextual_integrity", "secondary_use"],
                "must_not": ["fairness_scoring", "dignity_consequence_scoring"],
                "valid_clauses": ["P1", "P2", "P3", "P4"],
                "intentional_overlap": ["fairness"]  # When identity inference causes disparate impact
            },
            "fairness": {
                "owns": ["disparate_impact", "protected_class_outcomes", "bias_amplification", "differential_treatment"],
                "must_not": ["intent_judgment", "virtue_judgment", "dignity_framing"],
                "valid_clauses": ["F1", "F2", "F3"],
                "intentional_overlap": ["due_process", "privacy"]  # Explainability vs contestability
            },
            "due_process": {
                "owns": ["contestability", "attribution", "reviewability", "accountability"],
                "must_not": ["fairness_reargument", "dignity_reargument"],
                "valid_clauses": ["DP1", "DP2", "DP3"],
                "intentional_overlap": ["uncertainty", "dignity"]  # Auditability vs epistemic insufficiency
            },
            "precedent": {
                "owns": ["norm_creation", "precedent_voids", "precedent_conflicts", "legitimacy"],
                "must_not": ["present_harm_scoring", "consent_analysis"],
                "valid_clauses": ["PR1", "PR2", "PR3"],
                "intentional_overlap": ["due_process"]  # Legitimacy documentation
            },
            "uncertainty": {
                "owns": ["epistemic_insufficiency", "competence_bounds", "high_impact_unknowns", "missing_context"],
                "must_not": ["moral_judgment"],
                "valid_clauses": ["U1", "U2", "U3"],
                "intentional_overlap": ["due_process"]  # "Can we know" vs "can we review"
            }
        }

        # Valid tiers from Handbook Section 3
        self.valid_tiers = ["tier_2", "tier_3", 2, 3]

    def validate_charter_compliance(
        self,
        critics: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate that critic outputs comply with constitutional charters.

        Args:
            critics: Dictionary of critic_name -> critic_output

        Returns:
            Charter compliance report with:
            - compliant: Boolean indicating full compliance
            - violations: List of charter boundary violations
            - warnings: List of potential concerns (not violations)
            - intentional_overlaps: Detected overlaps that are expected
            - recommendations: Suggestions for critic implementation
        """

        violations = []
        warnings = []
        intentional_overlaps = []

        for critic_name, evaluation in critics.items():
            # Normalize critic name
            normalized_name = self._normalize_critic_name(critic_name)

            if normalized_name not in self.charter_boundaries:
                warnings.append({
                    "critic": critic_name,
                    "type": "unknown_critic",
                    "description": f"Critic '{critic_name}' not in canonical charter (v8.1)"
                })
                continue

            # Validate escalation signals are clause-aware
            escalation_violations = self._validate_escalation_signals(
                critic_name, normalized_name, evaluation
            )
            violations.extend(escalation_violations)

            # Detect potential charter boundary violations
            boundary_violations = self._detect_boundary_violations(
                critic_name, normalized_name, evaluation
            )
            violations.extend(boundary_violations)

            # Detect intentional overlaps (these are NOT errors)
            overlaps = self._detect_intentional_overlaps(
                critic_name, normalized_name, evaluation, critics
            )
            intentional_overlaps.extend(overlaps)

        return {
            "compliant": len(violations) == 0,
            "violations": violations,
            "warnings": warnings,
            "intentional_overlaps": intentional_overlaps,
            "total_violations": len(violations),
            "total_warnings": len(warnings),
            "recommendations": self._generate_recommendations(violations, warnings),
            "audit_note": "This is a charter compliance check. Dissent is preserved. Overlaps may be intentional."
        }

    def _normalize_critic_name(self, name: str) -> str:
        """Normalize critic name to canonical form."""
        name_lower = name.lower().replace("_", " ").replace("-", " ")

        # Map variations to canonical names
        if "autonomy" in name_lower or "agency" in name_lower:
            return "autonomy"
        elif "dignity" in name_lower or "rights" in name_lower:
            return "dignity"
        elif "privacy" in name_lower or "identity" in name_lower:
            return "privacy"
        elif "fairness" in name_lower or "discrimination" in name_lower:
            return "fairness"
        elif "due process" in name_lower or "accountability" in name_lower:
            return "due_process"
        elif "precedent" in name_lower or "legitimacy" in name_lower:
            return "precedent"
        elif "uncertainty" in name_lower:
            return "uncertainty"

        return name_lower

    def _validate_escalation_signals(
        self,
        critic_name: str,
        normalized_name: str,
        evaluation: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Validate that escalation signals are clause-aware.

        From Handbook 8.1: "Critics SHALL emit clause-aware escalation signals
        directly (e.g., P4, DP2). Escalation SHALL NOT be inferred at aggregation time."
        """
        violations = []
        charter = self.charter_boundaries[normalized_name]
        valid_clauses = charter["valid_clauses"]

        # Check for escalation signal
        escalation = evaluation.get("escalation")

        if escalation:
            # Should have clause_id
            clause_id = escalation.get("clause_id") if isinstance(escalation, dict) else None

            if not clause_id:
                violations.append({
                    "critic": critic_name,
                    "type": CharterViolationType.MISSING_CLAUSE_ID.value,
                    "severity": "high",
                    "description": f"{critic_name} has escalation without clause_id (violates Handbook 8.1)",
                    "fix": f"Emit clause-aware signal: {valid_clauses}"
                })
            elif clause_id not in valid_clauses:
                violations.append({
                    "critic": critic_name,
                    "type": CharterViolationType.INVALID_CLAUSE_ID.value,
                    "severity": "high",
                    "description": f"{critic_name} emitted invalid clause '{clause_id}' (valid: {valid_clauses})",
                    "fix": f"Use only chartered clauses: {valid_clauses}"
                })

            # Should have tier
            tier = escalation.get("tier") if isinstance(escalation, dict) else None

            if not tier:
                violations.append({
                    "critic": critic_name,
                    "type": CharterViolationType.MISSING_TIER.value,
                    "severity": "high",
                    "description": f"{critic_name} escalation missing tier",
                    "fix": "Specify tier_2 or tier_3"
                })
            elif tier not in self.valid_tiers:
                violations.append({
                    "critic": critic_name,
                    "type": CharterViolationType.INVALID_TIER.value,
                    "severity": "medium",
                    "description": f"{critic_name} has invalid tier '{tier}' (expected tier_2 or tier_3)",
                    "fix": "Use tier_2 or tier_3"
                })

        return violations

    def _detect_boundary_violations(
        self,
        critic_name: str,
        normalized_name: str,
        evaluation: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Detect potential charter boundary violations.

        From Handbook Section 5: Each critic has "Owns" and "Must NOT" boundaries.
        """
        violations = []
        # Check justification and violations for boundary violations
        justification = evaluation.get("justification", "").lower()
        violations_text = " ".join(str(v).lower() for v in evaluation.get("violations", []))
        combined_text = justification + " " + violations_text

        # Check for forbidden domains
        if normalized_name == "autonomy":
            if "dignity" in combined_text and "violation" in combined_text:
                violations.append({
                    "critic": critic_name,
                    "type": CharterViolationType.DOMAIN_OVERSTEP.value,
                    "severity": "medium",
                    "description": "Autonomy critic may be judging dignity (charter: must not judge dignity)",
                    "text_sample": combined_text[:100]
                })
            if "fairness" in combined_text or "equitable" in combined_text:
                violations.append({
                    "critic": critic_name,
                    "type": CharterViolationType.DOMAIN_OVERSTEP.value,
                    "severity": "medium",
                    "description": "Autonomy critic may be judging fairness (charter: must not judge fairness)",
                    "text_sample": combined_text[:100]
                })

        elif normalized_name == "dignity":
            if "consent" in combined_text or "authorization" in combined_text:
                violations.append({
                    "critic": critic_name,
                    "type": CharterViolationType.DOMAIN_OVERSTEP.value,
                    "severity": "medium",
                    "description": "Dignity critic may be checking consent (charter: must not do consent checks)",
                    "text_sample": combined_text[:100]
                })

        elif normalized_name == "fairness":
            if "intent" in combined_text or "intention" in combined_text:
                violations.append({
                    "critic": critic_name,
                    "type": CharterViolationType.DOMAIN_OVERSTEP.value,
                    "severity": "low",
                    "description": "Fairness critic may be judging intent (charter: must not judge intent)",
                    "text_sample": combined_text[:100]
                })

        return violations

    def _detect_intentional_overlaps(
        self,
        critic_name: str,
        normalized_name: str,
        evaluation: Dict[str, Any],
        all_critics: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect intentional overlaps between critics.

        From Handbook Section 5: "Intentional Overlap" is EXPECTED, not an error.
        """
        overlaps = []
        charter = self.charter_boundaries[normalized_name]
        intentional_overlap_critics = charter.get("intentional_overlap", [])

        for overlap_critic in intentional_overlap_critics:
            if overlap_critic in [self._normalize_critic_name(c) for c in all_critics.keys()]:
                overlaps.append({
                    "critics": [critic_name, overlap_critic],
                    "type": "intentional_overlap",
                    "description": f"{critic_name} and {overlap_critic} may overlap (this is intentional per charter)",
                    "note": "NOT AN ERROR - Handbook Section 5 explicitly allows this"
                })

        return overlaps

    def _generate_recommendations(
        self,
        violations: List[Dict[str, Any]],
        warnings: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations for fixing charter violations."""
        recommendations = []

        if not violations and not warnings:
            recommendations.append("✅ All critics comply with constitutional charters")
            return recommendations

        # High severity violations
        high_severity = [v for v in violations if v.get("severity") == "high"]
        if high_severity:
            recommendations.append(
                f"🔴 CRITICAL: Fix {len(high_severity)} high-severity charter violations immediately"
            )
            for v in high_severity[:3]:  # Show first 3
                recommendations.append(f"  - {v['description']}")

        # Medium severity violations
        medium_severity = [v for v in violations if v.get("severity") == "medium"]
        if medium_severity:
            recommendations.append(
                f"🟡 Review {len(medium_severity)} potential charter boundary violations"
            )

        # Warnings
        if warnings:
            recommendations.append(
                f"ℹ️  {len(warnings)} warnings - review for best practices"
            )

        recommendations.append(
            "📖 Reference: Constitutional Critics & Escalation Governance Handbook v8.1"
        )

        return recommendations
