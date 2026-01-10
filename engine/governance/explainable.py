"""
ELEANOR V8 — Explainable Governance with Causal Reasoning
----------------------------------------------------------

Provides deep explanations of governance decisions using causal reasoning
and counterfactual analysis. This enables transparency, trust-building,
and regulatory compliance.
"""

import logging
from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class DetailLevel(str, Enum):
    """Level of explanation detail."""
    SUMMARY = "summary"
    DETAILED = "detailed"
    INTERACTIVE = "interactive"


@dataclass
class CausalFactor:
    """A causal factor that influenced the decision."""
    factor_name: str
    influence_score: float  # 0.0 to 1.0
    description: str
    source: str  # "critic", "precedent", "uncertainty", "aggregation"
    evidence: Optional[str] = None


@dataclass
class Counterfactual:
    """A counterfactual scenario showing what would change the decision."""
    scenario: str
    required_changes: List[str]
    hypothetical_decision: str
    confidence: float  # 0.0 to 1.0


@dataclass
class CriticContribution:
    """Contribution of a specific critic to the decision."""
    critic_name: str
    contribution_score: float  # 0.0 to 1.0
    severity_weight: float
    violations_count: int
    rationale: str
    was_decisive: bool


@dataclass
class Explanation:
    """Complete explanation of a governance decision."""
    decision: str  # GREEN, AMBER, RED
    primary_factors: List[CausalFactor]
    counterfactuals: List[Counterfactual]
    critic_contributions: List[CriticContribution]
    precedent_influence: Optional[str] = None
    uncertainty_impact: Optional[str] = None
    human_readable: str = ""
    interactive_data: Optional[Dict[str, Any]] = None


class ExplainableGovernance:
    """
    Generate explanations for governance decisions using causal reasoning.
    
    Features:
    - Causal decision trees showing decision factors
    - Counterfactual analysis ("what would need to change")
    - Critic contribution attribution
    - Human-readable explanations
    - Interactive explanation data for UI
    """
    
    def __init__(self):
        """Initialize explainable governance engine."""
        pass
    
    def explain_decision(
        self,
        result: Dict[str, Any],
        detail_level: DetailLevel = DetailLevel.SUMMARY
    ) -> Explanation:
        """
        Generate explanation for governance decision.
        
        Args:
            result: Engine result dictionary (AggregationOutput or EngineResult)
            detail_level: Level of detail for explanation
            
        Returns:
            Explanation object with full explanation
        """
        # Extract decision from result
        decision = self._extract_decision(result)
        
        # Extract causal factors
        causal_factors = self._extract_causal_factors(result)
        
        # Generate counterfactuals
        counterfactuals = self._generate_counterfactuals(result, decision)
        
        # Attribute critic contributions
        critic_contributions = self._attribute_critic_contributions(result)
        
        # Build human-readable explanation
        human_readable = self._format_explanation(
            decision,
            causal_factors,
            counterfactuals,
            critic_contributions,
            result,
            detail_level
        )
        
        # Build interactive data for UI (if requested)
        interactive_data = None
        if detail_level == DetailLevel.INTERACTIVE:
            interactive_data = self._build_interactive_data(
                decision,
                causal_factors,
                counterfactuals,
                critic_contributions,
                result
            )
        
        return Explanation(
            decision=decision,
            primary_factors=causal_factors,
            counterfactuals=counterfactuals,
            critic_contributions=critic_contributions,
            precedent_influence=self._extract_precedent_influence(result),
            uncertainty_impact=self._extract_uncertainty_impact(result),
            human_readable=human_readable,
            interactive_data=interactive_data
        )
    
    def _extract_decision(self, result: Dict[str, Any]) -> str:
        """Extract decision from result."""
        # Try multiple paths for decision
        decision = (
            result.get("decision") or
            result.get("aggregated", {}).get("decision") or
            result.get("route") or
            "UNKNOWN"
        )
        
        # Normalize decision values
        decision_upper = str(decision).upper()
        if decision_upper in ("GREEN", "ALLOW", "APPROVE"):
            return "GREEN"
        elif decision_upper in ("AMBER", "CONSTRAINED", "CONSTRAINED_ALLOW"):
            return "AMBER"
        elif decision_upper in ("RED", "DENY", "BLOCK", "REJECT"):
            return "RED"
        elif decision_upper in ("ESCALATE", "ABSTAIN"):
            return "AMBER"  # Escalation is treated as cautionary
        
        return decision_upper
    
    def _extract_causal_factors(self, result: Dict[str, Any]) -> List[CausalFactor]:
        """Extract causal factors that led to the decision."""
        factors = []
        
        # Extract from aggregated result if available
        aggregated = result.get("aggregated") or result
        critics = aggregated.get("critics") or result.get("critic_findings") or {}
        
        # Factor 1: Highest severity critic
        if critics:
            max_severity = 0.0
            max_critic_name = None
            max_critic_data = None
            
            for critic_name, critic_data in critics.items():
                severity = critic_data.get("severity", 0.0) if isinstance(critic_data, dict) else 0.0
                if severity > max_severity:
                    max_severity = severity
                    max_critic_name = critic_name
                    max_critic_data = critic_data
            
            if max_critic_name and max_severity > 0:
                violations = max_critic_data.get("violations", []) if isinstance(max_critic_data, dict) else []
                rationale = max_critic_data.get("rationale") or max_critic_data.get("justification", "") if isinstance(max_critic_data, dict) else ""
                
                factors.append(CausalFactor(
                    factor_name=f"Critical Violation: {max_critic_name}",
                    influence_score=min(1.0, max_severity / 3.0),  # Normalize severity
                    description=f"{max_critic_name} identified {len(violations)} violation(s) with severity {max_severity:.2f}",
                    source="critic",
                    evidence=rationale[:200] if rationale else None
                ))
        
        # Factor 2: Precedent alignment
        precedent = aggregated.get("precedent") or result.get("precedent_alignment")
        if precedent:
            alignment_score = precedent.get("alignment_score", 0.0) if isinstance(precedent, dict) else 0.0
            conflict_level = precedent.get("conflict_level", 0.0) if isinstance(precedent, dict) else 0.0
            
            if abs(alignment_score) > 0.1 or conflict_level > 0.3:
                factors.append(CausalFactor(
                    factor_name="Precedent Alignment",
                    influence_score=abs(alignment_score) * 0.7 + conflict_level * 0.3,
                    description=f"Historical precedents show {'strong alignment' if alignment_score > 0.5 else 'conflict' if conflict_level > 0.5 else 'mixed signals'}",
                    source="precedent",
                    evidence=precedent.get("analysis", "")[:200] if isinstance(precedent, dict) else None
                ))
        
        # Factor 3: Uncertainty
        uncertainty = aggregated.get("uncertainty") or result.get("uncertainty")
        if uncertainty:
            overall_uncertainty = uncertainty.get("overall_uncertainty", 0.0) if isinstance(uncertainty, dict) else 0.0
            
            if overall_uncertainty > 0.4:
                factors.append(CausalFactor(
                    factor_name="High Uncertainty",
                    influence_score=overall_uncertainty,
                    description=f"System uncertainty level {overall_uncertainty:.2f} indicates lack of confidence",
                    source="uncertainty",
                    evidence=uncertainty.get("explanation", "")[:200] if isinstance(uncertainty, dict) else None
                ))
        
        # Factor 4: Escalation signals
        escalations = result.get("escalations") or aggregated.get("escalation_summary", {}).get("escalations", [])
        if escalations and len(escalations) > 0:
            factors.append(CausalFactor(
                factor_name="Escalation Required",
                influence_score=0.8,
                description=f"{len(escalations)} escalation(s) triggered requiring human review",
                source="escalation",
                evidence=f"Escalations: {', '.join(str(e) for e in escalations[:3])}"
            ))
        
        # Sort by influence score
        factors.sort(key=lambda f: f.influence_score, reverse=True)
        
        return factors[:5]  # Top 5 factors
    
    def _generate_counterfactuals(
        self,
        result: Dict[str, Any],
        current_decision: str
    ) -> List[Counterfactual]:
        """Generate counterfactual scenarios."""
        counterfactuals = []
        aggregated = result.get("aggregated") or result
        critics = aggregated.get("critics") or {}
        
        # Counterfactual 1: If highest severity critic had lower severity
        if critics:
            max_severity_critic = None
            max_severity = 0.0
            
            for critic_name, critic_data in critics.items():
                severity = critic_data.get("severity", 0.0) if isinstance(critic_data, dict) else 0.0
                if severity > max_severity:
                    max_severity = severity
                    max_severity_critic = (critic_name, critic_data)
            
            if max_severity_critic and max_severity > 2.0:
                critic_name, critic_data = max_severity_critic
                required_severity = 1.5  # Target for GREEN
                
                counterfactuals.append(Counterfactual(
                    scenario=f"If {critic_name} severity was reduced from {max_severity:.2f} to {required_severity:.2f}",
                    required_changes=[
                        f"Address violations identified by {critic_name}",
                        f"Reduce severity to below {required_severity:.2f}"
                    ],
                    hypothetical_decision="GREEN" if current_decision != "GREEN" else current_decision,
                    confidence=0.7 if max_severity > required_severity else 0.3
                ))
        
        # Counterfactual 2: If precedent alignment improved
        precedent = aggregated.get("precedent") or result.get("precedent_alignment")
        if precedent and isinstance(precedent, dict):
            alignment_score = precedent.get("alignment_score", 0.0)
            if alignment_score < 0.5:
                counterfactuals.append(Counterfactual(
                    scenario="If precedent alignment improved to strong support",
                    required_changes=[
                        "Provide evidence supporting similar historical decisions",
                        "Align with established precedent patterns"
                    ],
                    hypothetical_decision="GREEN" if current_decision != "GREEN" else current_decision,
                    confidence=0.6
                ))
        
        # Counterfactual 3: If uncertainty was reduced
        uncertainty = aggregated.get("uncertainty") or result.get("uncertainty")
        if uncertainty and isinstance(uncertainty, dict):
            overall_uncertainty = uncertainty.get("overall_uncertainty", 0.0)
            if overall_uncertainty > 0.5:
                counterfactuals.append(Counterfactual(
                    scenario="If uncertainty was reduced to low levels",
                    required_changes=[
                        "Provide more complete information",
                        "Resolve conflicting signals from critics"
                    ],
                    hypothetical_decision="GREEN" if current_decision == "AMBER" else current_decision,
                    confidence=0.5
                ))
        
        return counterfactuals[:3]  # Top 3 counterfactuals
    
    def _attribute_critic_contributions(
        self,
        result: Dict[str, Any]
    ) -> List[CriticContribution]:
        """Attribute decision contributions to individual critics."""
        contributions = []
        aggregated = result.get("aggregated") or result
        critics = aggregated.get("critics") or result.get("critic_findings") or {}
        
        # Calculate total severity for normalization
        total_severity = sum(
            (c.get("severity", 0.0) if isinstance(c, dict) else 0.0)
            for c in critics.values()
        )
        
        # Find highest severity (decisive critic)
        max_severity = max(
            (c.get("severity", 0.0) if isinstance(c, dict) else 0.0)
            for c in critics.values()
        ) if critics else 0.0
        
        for critic_name, critic_data in critics.items():
            if not isinstance(critic_data, dict):
                continue
            
            severity = critic_data.get("severity", 0.0)
            violations = critic_data.get("violations", [])
            rationale = critic_data.get("rationale") or critic_data.get("justification", "")
            
            # Calculate contribution score
            contribution_score = severity / total_severity if total_severity > 0 else 0.0
            was_decisive = severity == max_severity and severity > 2.0
            
            contributions.append(CriticContribution(
                critic_name=critic_name,
                contribution_score=contribution_score,
                severity_weight=severity / 3.0,  # Normalize to 0-1
                violations_count=len(violations) if isinstance(violations, list) else 0,
                rationale=rationale[:300] if rationale else "No rationale provided",
                was_decisive=was_decisive
            ))
        
        # Sort by contribution score
        contributions.sort(key=lambda c: c.contribution_score, reverse=True)
        
        return contributions
    
    def _extract_precedent_influence(self, result: Dict[str, Any]) -> Optional[str]:
        """Extract precedent influence summary."""
        precedent = (
            result.get("aggregated", {}).get("precedent") or
            result.get("precedent_alignment")
        )
        
        if not precedent or not isinstance(precedent, dict):
            return None
        
        alignment_score = precedent.get("alignment_score", 0.0)
        support_strength = precedent.get("support_strength", 0.0)
        conflict_level = precedent.get("conflict_level", 0.0)
        
        if alignment_score > 0.6:
            return f"Strong precedent support (alignment: {alignment_score:.2f}, strength: {support_strength:.2f})"
        elif conflict_level > 0.5:
            return f"Precedent conflict detected (conflict: {conflict_level:.2f})"
        elif alignment_score < -0.3:
            return f"Precedent contradicts decision (alignment: {alignment_score:.2f})"
        else:
            return f"Neutral precedent alignment (score: {alignment_score:.2f})"
    
    def _extract_uncertainty_impact(self, result: Dict[str, Any]) -> Optional[str]:
        """Extract uncertainty impact summary."""
        uncertainty = (
            result.get("aggregated", {}).get("uncertainty") or
            result.get("uncertainty")
        )
        
        if not uncertainty or not isinstance(uncertainty, dict):
            return None
        
        overall_uncertainty = uncertainty.get("overall_uncertainty", 0.0)
        needs_escalation = uncertainty.get("needs_escalation", False)
        
        if overall_uncertainty > 0.7:
            return f"Very high uncertainty ({overall_uncertainty:.2f}) - decision confidence is low"
        elif overall_uncertainty > 0.4:
            return f"Moderate uncertainty ({overall_uncertainty:.2f}) - some confidence concerns"
        elif needs_escalation:
            return "Uncertainty triggered escalation requirement"
        else:
            return f"Low uncertainty ({overall_uncertainty:.2f}) - high confidence"
    
    def _format_explanation(
        self,
        decision: str,
        causal_factors: List[CausalFactor],
        counterfactuals: List[Counterfactual],
        critic_contributions: List[CriticContribution],
        result: Dict[str, Any],
        detail_level: DetailLevel
    ) -> str:
        """Format human-readable explanation."""
        lines = []
        
        # Decision summary
        lines.append(f"## Decision: {decision}")
        lines.append("")
        
        # Primary factors
        if causal_factors:
            lines.append("### Primary Factors")
            for i, factor in enumerate(causal_factors[:3], 1):
                lines.append(f"{i}. **{factor.factor_name}** (influence: {factor.influence_score:.2f})")
                lines.append(f"   {factor.description}")
                if detail_level in (DetailLevel.DETAILED, DetailLevel.INTERACTIVE) and factor.evidence:
                    lines.append(f"   Evidence: {factor.evidence[:150]}...")
            lines.append("")
        
        # Critic contributions
        if critic_contributions and detail_level != DetailLevel.SUMMARY:
            lines.append("### Critic Contributions")
            for contrib in critic_contributions[:3]:
                decisive_marker = " ⭐ (Decisive)" if contrib.was_decisive else ""
                lines.append(f"- **{contrib.critic_name}** (contribution: {contrib.contribution_score:.2f}){decisive_marker}")
                lines.append(f"  Severity: {contrib.severity_weight:.2f}, Violations: {contrib.violations_count}")
                if detail_level == DetailLevel.INTERACTIVE:
                    lines.append(f"  Rationale: {contrib.rationale[:200]}...")
            lines.append("")
        
        # Counterfactuals (detailed level only)
        if counterfactuals and detail_level in (DetailLevel.DETAILED, DetailLevel.INTERACTIVE):
            lines.append("### Counterfactual Analysis")
            lines.append("What would need to change for a different decision:")
            for i, cf in enumerate(counterfactuals, 1):
                lines.append(f"{i}. **{cf.scenario}**")
                lines.append(f"   Would result in: {cf.hypothetical_decision} (confidence: {cf.confidence:.2f})")
                lines.append(f"   Required changes:")
                for change in cf.required_changes:
                    lines.append(f"     - {change}")
            lines.append("")
        
        # Precedent and uncertainty (detailed level only)
        if detail_level in (DetailLevel.DETAILED, DetailLevel.INTERACTIVE):
            precedent_inf = self._extract_precedent_influence(result)
            if precedent_inf:
                lines.append(f"### Precedent Influence: {precedent_inf}")
            
            uncertainty_inf = self._extract_uncertainty_impact(result)
            if uncertainty_inf:
                lines.append(f"### Uncertainty Impact: {uncertainty_inf}")
        
        return "\n".join(lines)
    
    def _build_interactive_data(
        self,
        decision: str,
        causal_factors: List[CausalFactor],
        counterfactuals: List[Counterfactual],
        critic_contributions: List[CriticContribution],
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build interactive data structure for UI visualization."""
        return {
            "decision": decision,
            "causal_tree": {
                "root": decision,
                "factors": [
                    {
                        "name": f.factor_name,
                        "score": f.influence_score,
                        "source": f.source,
                        "description": f.description
                    }
                    for f in causal_factors
                ]
            },
            "critic_breakdown": [
                {
                    "name": c.critic_name,
                    "contribution": c.contribution_score,
                    "severity": c.severity_weight,
                    "violations": c.violations_count,
                    "decisive": c.was_decisive
                }
                for c in critic_contributions
            ],
            "counterfactuals": [
                {
                    "scenario": cf.scenario,
                    "required_changes": cf.required_changes,
                    "hypothetical_decision": cf.hypothetical_decision,
                    "confidence": cf.confidence
                }
                for cf in counterfactuals
            ],
            "precedent_influence": self._extract_precedent_influence(result),
            "uncertainty_impact": self._extract_uncertainty_impact(result)
        }


__all__ = [
    "ExplainableGovernance",
    "Explanation",
    "CausalFactor",
    "Counterfactual",
    "CriticContribution",
    "DetailLevel",
]
