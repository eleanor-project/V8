"""
ELEANOR V8.1 — RedundancyEngine (Intra-Critic Deduplication Only)
------------------------------------------------------------------

PURPOSE: Deduplicates violations WITHIN a single critic only.

CRITICAL CONSTITUTIONAL PRINCIPLE:
- NEVER consolidate or discount findings across different critics
- Cross-critic "redundancy" may be intentional overlap (see Handbook Section 5)
- Dissent preservation is MANDATORY - all cross-critic signals preserved verbatim

From Handbook v8.1, Section 2.2:
"The Aggregator shall not suppress escalation, average it away, reinterpret it,
or down-rank it due to consensus or convenience."

From Handbook v8.1, Section 4:
"Unilateral escalation authority: Any single critic may trigger escalation.
Escalation cannot be vetoed and is binding on execution."

Usage:
- Within-critic deduplication: If Fairness emits F1 twice for identical reasons -> deduplicate
- Cross-critic preservation: If Fairness emits F1 AND Rights emits discrimination -> PRESERVE BOTH
- Audit trail: Track what was deduplicated for transparency
"""

from typing import Dict, Any, List
import hashlib


class RedundancyEngine:
    """
    Intra-Critic Deduplication Engine (Constitutional Compliance Mode).

    Deduplicates violations WITHIN each critic while preserving ALL cross-critic
    signals verbatim per Handbook v8.1 dissent preservation requirements.

    Design Principles:
    - INTRA-CRITIC ONLY: Deduplicates within a single critic's output
    - PRESERVES CROSS-CRITIC DISSENT: Never consolidates across critics
    - NON-DESTRUCTIVE: Original outputs preserved for audit
    - TRANSPARENT: Full audit trail of deduplication decisions
    """

    def __init__(self):
        pass

    def deduplicate(self, critics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Deduplicate violations WITHIN each critic only.

        Args:
            critics: Dictionary of critic_name -> critic_output

        Returns:
            Deduplication report with:
            - deduplicated_critics: Cleaned critic outputs (intra-critic only)
            - intra_critic_deduplication: What was deduplicated within each critic
            - cross_critic_preservation: Confirmation that cross-critic signals preserved
            - audit_report: Full transparency log
        """

        deduplicated_critics = {}
        intra_critic_deduplication = {}
        cross_critic_preservation = []

        for critic_name, evaluation in critics.items():
            # Deduplicate violations within this critic
            dedup_result = self._deduplicate_within_critic(critic_name, evaluation)

            deduplicated_critics[critic_name] = dedup_result["cleaned_output"]
            intra_critic_deduplication[critic_name] = dedup_result["deduplication_log"]

        # Document that cross-critic signals were preserved
        cross_critic_preservation = self._document_cross_critic_preservation(critics)

        return {
            "deduplicated_critics": deduplicated_critics,
            "intra_critic_deduplication": intra_critic_deduplication,
            "cross_critic_preservation": cross_critic_preservation,
            "audit_report": self._generate_audit_report(
                intra_critic_deduplication, cross_critic_preservation
            ),
            "constitutional_compliance": "PASS - Cross-critic dissent preserved verbatim"
        }

    def _deduplicate_within_critic(
        self,
        critic_name: str,
        evaluation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deduplicate violations within a single critic's output.

        This removes duplicate violations that arose from multiple detection
        strategies (regex + keywords) flagging the same issue.
        """
        violations = evaluation.get("violations", [])

        if not violations:
            return {
                "cleaned_output": evaluation,
                "deduplication_log": {
                    "duplicates_removed": 0,
                    "details": []
                }
            }

        # Track unique violations
        unique_violations = []
        seen_hashes = set()
        duplicates_found = []

        for violation in violations:
            # Normalize violation for comparison
            normalized = self._normalize_violation(violation)
            violation_hash = hashlib.sha256(normalized.encode()).hexdigest()

            if violation_hash not in seen_hashes:
                unique_violations.append(violation)
                seen_hashes.add(violation_hash)
            else:
                duplicates_found.append({
                    "violation": str(violation)[:100],
                    "reason": "Duplicate within same critic"
                })

        # Build cleaned output
        cleaned_output = {
            **evaluation,
            "violations": unique_violations,
            "deduplication_metadata": {
                "original_count": len(violations),
                "deduplicated_count": len(unique_violations),
                "duplicates_removed": len(violations) - len(unique_violations)
            }
        }

        return {
            "cleaned_output": cleaned_output,
            "deduplication_log": {
                "duplicates_removed": len(duplicates_found),
                "details": duplicates_found
            }
        }

    def _normalize_violation(self, violation: Any) -> str:
        """Normalize violation for comparison within same critic."""
        if isinstance(violation, dict):
            # Use category + description for dict violations
            text = f"{violation.get('category', '')}:{violation.get('description', '')}"
        else:
            text = str(violation)

        # Normalize: lowercase, collapse whitespace
        normalized = text.lower().strip()
        normalized = " ".join(normalized.split())

        return normalized

    def _document_cross_critic_preservation(
        self, critics: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Document that cross-critic signals were preserved (not consolidated).

        This provides audit evidence that dissent preservation occurred.
        """
        preservation_log = []

        # Check for similar violations across critics (these are PRESERVED, not deduplicated)
        critic_names = list(critics.keys())

        for i, critic1_name in enumerate(critic_names):
            for critic2_name in critic_names[i+1:]:
                critic1 = critics[critic1_name]
                critic2 = critics[critic2_name]

                violations1 = critic1.get("violations", [])
                violations2 = critic2.get("violations", [])

                # Check for keyword overlap (similar concerns)
                text1 = " ".join(str(v).lower() for v in violations1)
                text2 = " ".join(str(v).lower() for v in violations2)

                # Common keywords that might indicate overlap
                overlap_keywords = ["discrimination", "privacy", "dignity", "fairness", "consent"]
                shared_keywords = [kw for kw in overlap_keywords if kw in text1 and kw in text2]

                if shared_keywords:
                    preservation_log.append({
                        "critics": [critic1_name, critic2_name],
                        "shared_concerns": shared_keywords,
                        "status": "PRESERVED - Both critic signals maintained verbatim",
                        "constitutional_principle": "Dissent preservation (Handbook Section 4)",
                        "note": "Overlap may be intentional per Handbook Section 5"
                    })

        return preservation_log

    def _generate_audit_report(
        self,
        intra_critic_deduplication: Dict[str, Dict[str, Any]],
        cross_critic_preservation: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate audit report for transparency."""

        total_intra_critic_duplicates = sum(
            log["duplicates_removed"] for log in intra_critic_deduplication.values()
        )

        critics_with_duplicates = [
            name for name, log in intra_critic_deduplication.items()
            if log["duplicates_removed"] > 0
        ]

        return {
            "summary": f"Removed {total_intra_critic_duplicates} intra-critic duplicates",
            "critics_affected": critics_with_duplicates,
            "cross_critic_signals_preserved": len(cross_critic_preservation),
            "constitutional_compliance": {
                "dissent_preservation": "ENFORCED",
                "cross_critic_consolidation": "PROHIBITED",
                "intra_critic_deduplication": "ALLOWED"
            },
            "handbook_reference": "Constitutional Critics & Escalation Governance Handbook v8.1, Section 4",
            "audit_note": "All cross-critic signals preserved verbatim per constitutional requirements"
        }


# ============================================================
# CONSTITUTIONAL COMPLIANCE VALIDATOR
# ============================================================

def validate_no_cross_critic_suppression(
    original_critics: Dict[str, Dict[str, Any]],
    processed_critics: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Validate that no cross-critic dissent was suppressed.

    This function can be used in CI/testing to ensure RedundancyEngine
    or any other component doesn't violate dissent preservation.

    Returns:
        Validation result with pass/fail and details
    """
    violations = []

    for critic_name in original_critics.keys():
        if critic_name not in processed_critics:
            violations.append({
                "type": "critic_removed",
                "critic": critic_name,
                "severity": "CRITICAL",
                "description": f"Critic '{critic_name}' was removed entirely"
            })
            continue

        original = original_critics[critic_name]
        processed = processed_critics[critic_name]

        # Check escalation signals preserved
        original_escalation = original.get("escalation")
        processed_escalation = processed.get("escalation")

        if original_escalation and not processed_escalation:
            violations.append({
                "type": "escalation_suppressed",
                "critic": critic_name,
                "severity": "CRITICAL",
                "description": f"Escalation signal from '{critic_name}' was suppressed"
            })

        # Check severity wasn't reduced (except for intra-critic deduplication metadata)
        original_severity = original.get("severity", 0.0)
        processed_severity = processed.get("severity", 0.0)

        if processed_severity < original_severity * 0.99:  # Allow for float rounding
            violations.append({
                "type": "severity_reduced",
                "critic": critic_name,
                "severity": "HIGH",
                "description": f"Severity reduced from {original_severity} to {processed_severity}",
                "note": "Severity reduction across critics violates dissent preservation"
            })

    return {
        "compliant": len(violations) == 0,
        "violations": violations,
        "status": "PASS" if len(violations) == 0 else "FAIL",
        "constitutional_principle": "Dissent preservation must be maintained",
        "handbook_reference": "Section 2.2, Section 4"
    }
