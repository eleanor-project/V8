"""
ELEANOR V8 — RedundancyEngine
------------------------------

Filters duplicate and overlapping findings across critics to avoid:
1. Double-counting the same violation across multiple critics
2. Redundant pattern matches
3. Cumulative severity inflation from duplicate issues
4. Evidence bloat from repeated findings

Returns redundancy analysis and adjusted severities for aggregation
without modifying original critic outputs.
"""

from typing import Dict, Any, List, Set, Tuple
from collections import defaultdict
import hashlib


class RedundancyEngine:
    """
    RedundancyEngine identifies and filters redundant findings across critics.

    Design principles:
    - Non-destructive: Preserves original critic outputs for audit
    - Advisory: Provides deduplication recommendations
    - Transparent: Full redundancy traceability
    - Integrated: Feeds adjusted severities to aggregator
    """

    def __init__(self):
        # Cross-critic violation equivalence patterns
        self.equivalence_patterns = {
            "discrimination": {
                "critics": ["fairness", "rights"],
                "keywords": ["discrimination", "discriminate", "protected class", "bias", "disparate"],
                "description": "Both fairness and rights critics flag discrimination"
            },
            "misinformation": {
                "critics": ["truth", "risk"],
                "keywords": ["misinformation", "false", "fabricated", "hallucination", "harm"],
                "description": "Both truth and risk critics flag harmful misinformation"
            },
            "coercion": {
                "critics": ["autonomy", "rights"],
                "keywords": ["coercion", "coercive", "manipulation", "manipulative", "consent"],
                "description": "Both autonomy and rights critics flag coercive practices"
            },
            "privacy_violation": {
                "critics": ["autonomy", "rights"],
                "keywords": ["privacy", "data", "personal information", "surveillance"],
                "description": "Both autonomy and rights critics flag privacy violations"
            },
            "irreversible_harm": {
                "critics": ["risk", "fairness"],
                "keywords": ["irreversible", "permanent", "catastrophic", "vulnerable population"],
                "description": "Both risk and fairness critics flag irreversible harm to vulnerable groups"
            }
        }

        # Redundancy group tracking
        self.redundancy_groups = defaultdict(list)

    def filter(self, critics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main entry point for redundancy filtering.

        Args:
            critics: Dictionary of critic_name -> critic_output

        Returns:
            Redundancy analysis with:
            - deduplicated_critics: Adjusted critic outputs
            - redundancy_groups: Detected redundancies
            - severity_adjustments: Recommended severity adjustments
            - audit_report: Full redundancy traceability
        """

        # Detect cross-critic redundancies
        redundancy_groups = self._detect_redundancies(critics)

        # Generate severity adjustments
        severity_adjustments = self._compute_severity_adjustments(
            critics, redundancy_groups
        )

        # Build deduplicated critic outputs
        deduplicated_critics = self._build_deduplicated_critics(
            critics, severity_adjustments
        )

        # Generate audit report
        audit_report = self._generate_audit_report(
            critics, redundancy_groups, severity_adjustments
        )

        return {
            "deduplicated_critics": deduplicated_critics,
            "redundancy_groups": redundancy_groups,
            "severity_adjustments": severity_adjustments,
            "total_redundancies": len(redundancy_groups),
            "audit_report": audit_report,
            "redundancy_flags": self._generate_redundancy_flags(redundancy_groups)
        }

    # ----------------------------------------------------------
    # Redundancy detection
    # ----------------------------------------------------------
    def _detect_redundancies(
        self, critics: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect redundant violations across critics.

        Strategy:
        1. Match violation patterns using equivalence mappings
        2. Compare violation text similarity
        3. Identify shared evidence
        """
        redundancies = []

        # Check each equivalence pattern
        for pattern_name, pattern_config in self.equivalence_patterns.items():
            involved_critics = pattern_config["critics"]
            keywords = pattern_config["keywords"]

            # Get critics that are involved in this pattern
            active_critics = {
                name: critics[name]
                for name in involved_critics
                if name in critics
            }

            if len(active_critics) < 2:
                continue  # Need at least 2 critics for redundancy

            # Check if both critics flag this issue
            flagged_critics = []
            for critic_name, evaluation in active_critics.items():
                violations = evaluation.get("violations", [])
                justification = evaluation.get("justification", "")

                # Check for keyword matches in violations or justification
                combined_text = " ".join(str(v) for v in violations) + " " + justification
                combined_text_lower = combined_text.lower()

                if any(kw in combined_text_lower for kw in keywords):
                    flagged_critics.append({
                        "critic": critic_name,
                        "severity": evaluation.get("severity", 0.0),
                        "violations": violations,
                        "matched_keywords": [kw for kw in keywords if kw in combined_text_lower]
                    })

            # If multiple critics flagged the same pattern
            if len(flagged_critics) >= 2:
                redundancies.append({
                    "pattern": pattern_name,
                    "description": pattern_config["description"],
                    "critics_involved": [c["critic"] for c in flagged_critics],
                    "severities": {c["critic"]: c["severity"] for c in flagged_critics},
                    "matched_keywords": {c["critic"]: c["matched_keywords"] for c in flagged_critics},
                    "recommendation": "Use highest severity, discount others"
                })

        # Also check for exact violation text matches
        text_redundancies = self._detect_text_redundancies(critics)
        redundancies.extend(text_redundancies)

        return redundancies

    def _detect_text_redundancies(
        self, critics: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect violations with similar or identical text across critics.
        """
        redundancies = []

        # Build violation text -> critics mapping
        violation_texts = defaultdict(list)

        for critic_name, evaluation in critics.items():
            violations = evaluation.get("violations", [])
            for violation in violations:
                # Normalize violation text for comparison
                normalized = self._normalize_violation_text(violation)
                violation_texts[normalized].append({
                    "critic": critic_name,
                    "original": violation,
                    "severity": evaluation.get("severity", 0.0)
                })

        # Find duplicates
        for normalized_text, instances in violation_texts.items():
            if len(instances) >= 2:
                redundancies.append({
                    "pattern": "text_match",
                    "description": f"Multiple critics flagged similar violation: '{normalized_text[:50]}...'",
                    "critics_involved": [inst["critic"] for inst in instances],
                    "severities": {inst["critic"]: inst["severity"] for inst in instances},
                    "matched_text": normalized_text,
                    "recommendation": "Consolidate duplicate violations"
                })

        return redundancies

    def _normalize_violation_text(self, violation: Any) -> str:
        """Normalize violation text for comparison."""
        if isinstance(violation, dict):
            # If violation is a dict, use description or category
            text = violation.get("description", violation.get("category", str(violation)))
        else:
            text = str(violation)

        # Normalize: lowercase, remove extra whitespace, remove punctuation
        normalized = text.lower().strip()
        normalized = " ".join(normalized.split())  # Collapse whitespace

        # Create hash for long texts (for efficient comparison)
        if len(normalized) > 100:
            return hashlib.md5(normalized.encode()).hexdigest()

        return normalized[:100]  # Truncate to first 100 chars

    # ----------------------------------------------------------
    # Severity adjustment computation
    # ----------------------------------------------------------
    def _compute_severity_adjustments(
        self,
        critics: Dict[str, Dict[str, Any]],
        redundancies: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute recommended severity adjustments to avoid double-counting.

        Strategy:
        - For each redundancy group, keep the highest severity
        - Discount redundant critics by 50% to prevent inflation
        - Maintain audit trail of original vs adjusted severities
        """
        adjustments = {}

        # Initialize with original severities
        for critic_name, evaluation in critics.items():
            adjustments[critic_name] = {
                "original_severity": evaluation.get("severity", 0.0),
                "adjusted_severity": evaluation.get("severity", 0.0),
                "adjustment_factor": 1.0,
                "redundancy_groups": [],
                "adjustment_reason": "No redundancy detected"
            }

        # Apply adjustments for each redundancy
        for redundancy in redundancies:
            critics_involved = redundancy["critics_involved"]
            severities = redundancy["severities"]

            if not severities:
                continue

            # Find highest severity critic
            highest_critic = max(severities.items(), key=lambda x: x[1])[0]
            highest_severity = severities[highest_critic]

            # Discount others by 50%
            for critic_name in critics_involved:
                if critic_name != highest_critic:
                    current_adjustment = adjustments[critic_name]["adjustment_factor"]
                    adjustments[critic_name]["adjustment_factor"] = current_adjustment * 0.5
                    adjustments[critic_name]["redundancy_groups"].append(redundancy["pattern"])
                    adjustments[critic_name]["adjustment_reason"] = f"Redundant with {highest_critic}"

        # Apply final adjustments
        for critic_name, adj in adjustments.items():
            adj["adjusted_severity"] = round(
                adj["original_severity"] * adj["adjustment_factor"], 2
            )

        return adjustments

    # ----------------------------------------------------------
    # Deduplicated critic outputs
    # ----------------------------------------------------------
    def _build_deduplicated_critics(
        self,
        critics: Dict[str, Dict[str, Any]],
        severity_adjustments: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build deduplicated critic outputs with adjusted severities.

        NOTE: Original outputs are preserved; adjustments are metadata.
        """
        deduplicated = {}

        for critic_name, evaluation in critics.items():
            adjustment = severity_adjustments.get(critic_name, {})

            deduplicated[critic_name] = {
                **evaluation,  # Preserve original
                "redundancy_metadata": {
                    "original_severity": adjustment.get("original_severity", evaluation.get("severity", 0.0)),
                    "adjusted_severity": adjustment.get("adjusted_severity", evaluation.get("severity", 0.0)),
                    "adjustment_factor": adjustment.get("adjustment_factor", 1.0),
                    "redundancy_groups": adjustment.get("redundancy_groups", []),
                    "adjustment_reason": adjustment.get("adjustment_reason", "No redundancy")
                }
            }

        return deduplicated

    # ----------------------------------------------------------
    # Audit reporting
    # ----------------------------------------------------------
    def _generate_audit_report(
        self,
        critics: Dict[str, Dict[str, Any]],
        redundancies: List[Dict[str, Any]],
        severity_adjustments: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate comprehensive audit report for redundancy filtering."""

        total_original_severity = sum(
            c.get("severity", 0.0) for c in critics.values()
        )

        total_adjusted_severity = sum(
            adj.get("adjusted_severity", 0.0) for adj in severity_adjustments.values()
        )

        severity_reduction = total_original_severity - total_adjusted_severity
        reduction_percentage = (
            (severity_reduction / total_original_severity * 100)
            if total_original_severity > 0 else 0.0
        )

        return {
            "total_critics": len(critics),
            "total_redundancies_detected": len(redundancies),
            "total_original_severity": round(total_original_severity, 2),
            "total_adjusted_severity": round(total_adjusted_severity, 2),
            "severity_reduction": round(severity_reduction, 2),
            "reduction_percentage": round(reduction_percentage, 1),
            "redundancy_patterns_found": list(set(r["pattern"] for r in redundancies)),
            "critics_with_adjustments": [
                name for name, adj in severity_adjustments.items()
                if adj["adjustment_factor"] < 1.0
            ],
            "audit_timestamp": None  # Could add timestamp if needed
        }

    def _generate_redundancy_flags(self, redundancies: List[Dict[str, Any]]) -> List[str]:
        """Generate flags based on redundancy detection."""
        flags = []

        redundancy_count = len(redundancies)

        if redundancy_count == 0:
            flags.append("NO_REDUNDANCY")
        elif redundancy_count <= 2:
            flags.append("LOW_REDUNDANCY")
        elif redundancy_count <= 5:
            flags.append("MODERATE_REDUNDANCY")
        else:
            flags.append("HIGH_REDUNDANCY")

        # Add pattern-specific flags
        pattern_types = set(r["pattern"] for r in redundancies)
        if "discrimination" in pattern_types:
            flags.append("DISCRIMINATION_REDUNDANCY")
        if "misinformation" in pattern_types:
            flags.append("MISINFORMATION_REDUNDANCY")

        return flags
