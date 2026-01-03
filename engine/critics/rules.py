"""
ELEANOR V8.1 — Canonical Constitutional Clauses Registry
---------------------------------------------------------

This module defines the 22 canonical constitutional clauses from the
Constitutional Critics & Escalation Governance Handbook v8.1.

These are NOT pattern-matching rules. They are governance escalation clauses
that define:
- Which constitutional principle is at stake
- What tier of human review is required
- What triggers the clause
- What human action satisfies the governance gate

From Handbook v8.1, Section 8.1:
"Critics SHALL emit clause-aware escalation signals directly (e.g., P4, DP2).
Escalation SHALL NOT be inferred at aggregation time."

Usage:
- Critic implementations: Reference these clauses when emitting escalation signals
- Aggregator: Map escalation signals to human review requirements
- OPA enforcement: Validate execution gates satisfied
- Audit: Trace constitutional reasoning
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum


# Reuse canonical escalation tier from runtime schemas to avoid divergence.
from engine.schemas.escalation import EscalationTier as EscalationTier  # noqa: F401


class HumanAction(Enum):
    """Required human actions per tier (Handbook Appendix A)."""

    ACKNOWLEDGMENT = "acknowledgment"  # Tier 2
    DETERMINATION = "determination"  # Tier 3


class CriticDomain(Enum):
    """Canonical critic domains."""

    AUTONOMY = "autonomy"
    DIGNITY = "dignity"
    PRIVACY = "privacy"
    FAIRNESS = "fairness"
    DUE_PROCESS = "due_process"
    PRECEDENT = "precedent"
    UNCERTAINTY = "uncertainty"


@dataclass
class ConstitutionalClause:
    """
    Canonical constitutional clause definition.

    From Handbook v8.1: These clauses define when escalation is triggered
    and what form of human review is required.

    Fields:
    - clause_id: Immutable canonical identifier (A1, D2, P4, etc.)
    - critic: Which critic owns this clause
    - tier: Escalation tier (tier_2 or tier_3)
    - trigger: What constitutional condition triggers this clause
    - human_action: What human action satisfies the governance gate
    - description: Full explanation of the constitutional principle
    - handbook_section: Where this is defined in Handbook v8.1
    """

    clause_id: str
    critic: CriticDomain
    tier: EscalationTier
    trigger: str
    human_action: HumanAction
    description: str
    handbook_section: str
    rationale: str = ""
    examples: List[str] = field(default_factory=list)


# ============================================================
# CANONICAL CONSTITUTIONAL CLAUSES (22 Total)
# ============================================================

CLAUSES: Dict[str, ConstitutionalClause] = {
    # -----------------------------------------------------------
    # AUTONOMY CRITIC (A1, A2, A3)
    # -----------------------------------------------------------
    "A1": ConstitutionalClause(
        clause_id="A1",
        critic=CriticDomain.AUTONOMY,
        tier=EscalationTier.TIER_2,
        trigger="Meaningful consent failure",
        human_action=HumanAction.ACKNOWLEDGMENT,
        description="Action materially affects an individual without informed, voluntary consent/refusal",
        handbook_section="Section 7.1",
        rationale="Material effects without informed voluntary consent/refusal undermine self-determination",
        examples=[
            "User data used without explicit consent",
            "Decision made affecting user without opportunity for refusal",
            "Consent obtained through deception or dark patterns",
        ],
    ),
    "A2": ConstitutionalClause(
        clause_id="A2",
        critic=CriticDomain.AUTONOMY,
        tier=EscalationTier.TIER_3,
        trigger="Coercive or manipulative influence",
        human_action=HumanAction.DETERMINATION,
        description="Coercion, manipulation, deception, or asymmetric pressure meaningfully constrains choice",
        handbook_section="Section 7.1",
        rationale="Legitimacy breach - automation cannot decide acceptability of coercion",
        examples=[
            "Manipulative framing that exploits cognitive biases",
            "Economic coercion through essential service gatekeeping",
            "Deceptive presentation of choices",
        ],
    ),
    "A3": ConstitutionalClause(
        clause_id="A3",
        critic=CriticDomain.AUTONOMY,
        tier=EscalationTier.TIER_3,
        trigger="Irreversible autonomy impact",
        human_action=HumanAction.DETERMINATION,
        description="Non-reversible limitation on future choice or self-determination",
        handbook_section="Section 7.1",
        rationale="Irreversibility demands explicit human decision",
        examples=[
            "Permanent account termination",
            "Irreversible identity linkage",
            "Permanent loss of future choices",
        ],
    ),
    # -----------------------------------------------------------
    # DIGNITY CRITIC (D1, D2, D3)
    # -----------------------------------------------------------
    "D1": ConstitutionalClause(
        clause_id="D1",
        critic=CriticDomain.DIGNITY,
        tier=EscalationTier.TIER_2,
        trigger="Instrumentalization of persons",
        human_action=HumanAction.ACKNOWLEDGMENT,
        description="Persons treated primarily as means rather than ends",
        handbook_section="Section 7.2",
        rationale="When instrumentalization becomes ordinary it becomes dangerous",
        examples=[
            "User value measured purely by profit extraction",
            "Persons reduced to data points for optimization",
            "Human welfare subordinated to system efficiency",
        ],
    ),
    "D2": ConstitutionalClause(
        clause_id="D2",
        critic=CriticDomain.DIGNITY,
        tier=EscalationTier.TIER_3,
        trigger="Degrading / dehumanizing outcome",
        human_action=HumanAction.DETERMINATION,
        description="Humiliation, stigmatization, demeaning or dehumanizing treatment risk",
        handbook_section="Section 7.2",
        rationale="Dignity failures become dangerous when normalized - escalation prevents quiet acceptance",
        examples=[
            "Public shaming or humiliation",
            "Dehumanizing language or categorization",
            "Stigmatizing labels applied to individuals",
        ],
    ),
    "D3": ConstitutionalClause(
        clause_id="D3",
        critic=CriticDomain.DIGNITY,
        tier=EscalationTier.TIER_2,
        trigger="Asymmetric harm without voice",
        human_action=HumanAction.ACKNOWLEDGMENT,
        description="Harm imposed without meaningful participation, representation, or moral presence",
        handbook_section="Section 7.2",
        rationale="Those affected must have standing in deliberation",
        examples=[
            "Decision affecting vulnerable group without their input",
            "Harm to excluded community without representation",
            "Burden imposed on those without voice in process",
        ],
    ),
    # -----------------------------------------------------------
    # PRIVACY & IDENTITY CRITIC (P1, P2, P3, P4)
    # -----------------------------------------------------------
    "P1": ConstitutionalClause(
        clause_id="P1",
        critic=CriticDomain.PRIVACY,
        tier=EscalationTier.TIER_3,
        trigger="Non-consensual identity inference",
        human_action=HumanAction.DETERMINATION,
        description="Identity inferred without explicit consent",
        handbook_section="Section 7.3",
        rationale="Capability does not equal permission",
        examples=[
            "Inferring sensitive attributes from behavior",
            "Demographic profiling without consent",
            "Identity categories assigned without user knowledge",
        ],
    ),
    "P2": ConstitutionalClause(
        clause_id="P2",
        critic=CriticDomain.PRIVACY,
        tier=EscalationTier.TIER_3,
        trigger="Identity persistence or linkage across contexts",
        human_action=HumanAction.DETERMINATION,
        description="Persistent identity linkage across sessions/domains",
        handbook_section="Section 7.3",
        rationale="Cross-context tracking threatens contextual integrity",
        examples=[
            "Linking activity across unrelated services",
            "Persistent tracking across sessions",
            "Identity correlation across contexts",
        ],
    ),
    "P3": ConstitutionalClause(
        clause_id="P3",
        critic=CriticDomain.PRIVACY,
        tier=EscalationTier.TIER_2,
        trigger="Context collapse",
        human_action=HumanAction.ACKNOWLEDGMENT,
        description="Cross-domain use violates contextual integrity",
        handbook_section="Section 7.3",
        rationale="Information norms are context-specific",
        examples=[
            "Data shared for one purpose used in different context",
            "Information appropriate in one domain exposed in another",
            "Context-specific expectations violated",
        ],
    ),
    "P4": ConstitutionalClause(
        clause_id="P4",
        critic=CriticDomain.PRIVACY,
        tier=EscalationTier.TIER_2,
        trigger="Secondary use expansion without renewed consent",
        human_action=HumanAction.ACKNOWLEDGMENT,
        description="Secondary use beyond original authorization without renewed consent",
        handbook_section="Section 7.3, Amendment v8.1",
        rationale="P4 exists to stop scope creep from becoming normal",
        examples=[
            "Data collected for service A used for product B",
            "Original consent scope exceeded",
            "New uses added without re-authorization",
        ],
    ),
    # -----------------------------------------------------------
    # FAIRNESS & NON-DISCRIMINATION CRITIC (F1, F2, F3)
    # -----------------------------------------------------------
    "F1": ConstitutionalClause(
        clause_id="F1",
        critic=CriticDomain.FAIRNESS,
        tier=EscalationTier.TIER_3,
        trigger="Protected class impact",
        human_action=HumanAction.DETERMINATION,
        description="Differential impact affecting protected classes",
        handbook_section="Section 7.4",
        rationale="Math laundering cannot absolve moral responsibility",
        examples=[
            "Disparate impact on protected demographic groups",
            "Outcome differences correlated with protected attributes",
            "Algorithm disadvantages specific protected classes",
        ],
    ),
    "F2": ConstitutionalClause(
        clause_id="F2",
        critic=CriticDomain.FAIRNESS,
        tier=EscalationTier.TIER_2,
        trigger="Structural bias amplification",
        human_action=HumanAction.ACKNOWLEDGMENT,
        description="Feedback loops or design amplify structural inequity",
        handbook_section="Section 7.4",
        rationale="Systems must not compound existing structural disadvantage",
        examples=[
            "Feedback loop entrenches historical bias",
            "Design choice amplifies existing inequities",
            "System perpetuates structural disadvantage",
        ],
    ),
    "F3": ConstitutionalClause(
        clause_id="F3",
        critic=CriticDomain.FAIRNESS,
        tier=EscalationTier.TIER_2,
        trigger="Opaque differential treatment",
        human_action=HumanAction.ACKNOWLEDGMENT,
        description="Materially different outcomes cannot be meaningfully explained",
        handbook_section="Section 7.4",
        rationale="Unexplainable disparity blocks legitimate fairness accountability",
        examples=[
            "Different outcomes for similar cases without explanation",
            "Treatment varies but rationale is opaque",
            "Disparity exists but cannot be justified",
        ],
    ),
    # -----------------------------------------------------------
    # DUE PROCESS & ACCOUNTABILITY CRITIC (DP1, DP2, DP3)
    # -----------------------------------------------------------
    "DP1": ConstitutionalClause(
        clause_id="DP1",
        critic=CriticDomain.DUE_PROCESS,
        tier=EscalationTier.TIER_3,
        trigger="Lack of contestability",
        human_action=HumanAction.DETERMINATION,
        description="No meaningful appeal, challenge, or review path",
        handbook_section="Section 7.5",
        rationale="If no one can challenge it, the system must not act alone",
        examples=[
            "Decision with no appeal mechanism",
            "No path to contest outcome",
            "Challenge process is inaccessible or ineffective",
        ],
    ),
    "DP2": ConstitutionalClause(
        clause_id="DP2",
        critic=CriticDomain.DUE_PROCESS,
        tier=EscalationTier.TIER_3,
        trigger="Decision without attribution",
        human_action=HumanAction.DETERMINATION,
        description="Responsibility cannot be assigned to an accountable authority",
        handbook_section="Section 7.5",
        rationale="If no one can own it, the system must not act alone",
        examples=[
            "No accountable party for decision",
            "Responsibility diffused across system",
            "Cannot identify who authorized action",
        ],
    ),
    "DP3": ConstitutionalClause(
        clause_id="DP3",
        critic=CriticDomain.DUE_PROCESS,
        tier=EscalationTier.TIER_2,
        trigger="Unreviewable automation",
        human_action=HumanAction.ACKNOWLEDGMENT,
        description="Decision cannot be reconstructed, audited, or explained post-hoc",
        handbook_section="Section 7.5",
        rationale="Auditability is minimum threshold for accountability",
        examples=[
            "Decision process not logged",
            "Cannot explain how outcome was reached",
            "Reasoning cannot be reconstructed for review",
        ],
    ),
    # -----------------------------------------------------------
    # PRECEDENT & LEGITIMACY CRITIC (PR1, PR2, PR3)
    # -----------------------------------------------------------
    "PR1": ConstitutionalClause(
        clause_id="PR1",
        critic=CriticDomain.PRECEDENT,
        tier=EscalationTier.TIER_3,
        trigger="Precedent void in high-impact domain",
        human_action=HumanAction.DETERMINATION,
        description="No relevant precedent in high-impact domain",
        handbook_section="Section 7.6",
        rationale="Precedent failures compound - this critic watches tomorrow not just today",
        examples=[
            "Novel high-stakes decision without precedent",
            "First instance in critical domain",
            "No established norms for guidance",
        ],
    ),
    "PR2": ConstitutionalClause(
        clause_id="PR2",
        critic=CriticDomain.PRECEDENT,
        tier=EscalationTier.TIER_2,
        trigger="Precedent conflict",
        human_action=HumanAction.ACKNOWLEDGMENT,
        description="Conflicting precedents require interpretation and explicit acknowledgment",
        handbook_section="Section 7.6",
        rationale="Conflicts must be acknowledged not hidden",
        examples=[
            "Prior decisions point in opposite directions",
            "Precedents contradict each other",
            "No clear guidance from historical cases",
        ],
    ),
    "PR3": ConstitutionalClause(
        clause_id="PR3",
        critic=CriticDomain.PRECEDENT,
        tier=EscalationTier.TIER_3,
        trigger="Uncontrolled precedent creation",
        human_action=HumanAction.DETERMINATION,
        description="Decision creates an operational norm without explicit authorization",
        handbook_section="Section 7.6",
        rationale="Norm creation must be intentional not accidental",
        examples=[
            "Decision establishes new standard unintentionally",
            "Creates precedent without deliberate authorization",
            "Becomes norm through repetition not explicit choice",
        ],
    ),
    # -----------------------------------------------------------
    # UNCERTAINTY CRITIC (U1, U2, U3)
    # -----------------------------------------------------------
    "U1": ConstitutionalClause(
        clause_id="U1",
        critic=CriticDomain.UNCERTAINTY,
        tier=EscalationTier.TIER_2,
        trigger="Epistemic insufficiency",
        human_action=HumanAction.ACKNOWLEDGMENT,
        description="Critical context missing for responsible judgment",
        handbook_section="Section 7.7",
        rationale="Uncertainty is not a flaw - unacknowledged uncertainty is",
        examples=[
            "Insufficient information for confident decision",
            "Key context unavailable",
            "Gaps in understanding material to outcome",
        ],
    ),
    "U2": ConstitutionalClause(
        clause_id="U2",
        critic=CriticDomain.UNCERTAINTY,
        tier=EscalationTier.TIER_3,
        trigger="High impact × high uncertainty",
        human_action=HumanAction.DETERMINATION,
        description="High impact + high uncertainty exceeds safety bounds",
        handbook_section="Section 7.7",
        rationale="Risk magnitude with epistemic uncertainty requires human determination",
        examples=[
            "High-stakes decision with substantial unknowns",
            "Significant impact coupled with low confidence",
            "Critical outcome with epistemic gaps",
        ],
    ),
    "U3": ConstitutionalClause(
        clause_id="U3",
        critic=CriticDomain.UNCERTAINTY,
        tier=EscalationTier.TIER_3,
        trigger="Competence boundary exceeded",
        human_action=HumanAction.DETERMINATION,
        description="Scenario outside validated competence envelope / assumptions",
        handbook_section="Section 7.7, U3 Competence Envelope",
        rationale="System must know when it doesn't know",
        examples=[
            "Domain assumptions do not hold",
            "Required evidence sources unavailable",
            "Outside validated scope",
            "Charter-defined exclusions apply",
        ],
    ),
}


# ============================================================
# CANONICAL HUMAN REVIEW STATEMENTS (Binding)
# ============================================================

HUMAN_REVIEW_STATEMENTS = {
    HumanAction.ACKNOWLEDGMENT: "I acknowledge the identified constitutional risks and accept responsibility for proceeding.",
    HumanAction.DETERMINATION: "I affirmatively determine the appropriate course of action in light of the identified constitutional risks.",
}


# ============================================================
# REGISTRY UTILITIES
# ============================================================


def get_clause(clause_id: str) -> Optional[ConstitutionalClause]:
    """Retrieve a clause by its canonical ID."""
    return CLAUSES.get(clause_id)


def get_clauses_by_critic(critic: CriticDomain) -> List[ConstitutionalClause]:
    """Get all clauses owned by a specific critic."""
    return [clause for clause in CLAUSES.values() if clause.critic == critic]


def get_clauses_by_tier(tier: EscalationTier) -> List[ConstitutionalClause]:
    """Get all clauses at a specific escalation tier."""
    return [clause for clause in CLAUSES.values() if clause.tier == tier]


def get_required_human_action(clause_id: str) -> Optional[str]:
    """Get the canonical human action statement required for a clause."""
    clause = CLAUSES.get(clause_id)
    if not clause:
        return None
    return HUMAN_REVIEW_STATEMENTS.get(clause.human_action)


def validate_clause_id(clause_id: str, critic: CriticDomain) -> Dict[str, Any]:
    """
    Validate that a clause_id is valid for a given critic.

    Used by ConsistencyEngine to validate charter compliance.
    """
    clause = CLAUSES.get(clause_id)

    if not clause:
        return {
            "valid": False,
            "error": f"Unknown clause ID: {clause_id}",
            "suggestion": "Use canonical clause IDs (A1-A3, D1-D3, P1-P4, F1-F3, DP1-DP3, PR1-PR3, U1-U3)",
        }

    if clause.critic != critic:
        return {
            "valid": False,
            "error": f"Clause {clause_id} belongs to {clause.critic.value}, not {critic.value}",
            "suggestion": f"Valid clauses for {critic.value}: {[c.clause_id for c in get_clauses_by_critic(critic)]}",
        }

    return {
        "valid": True,
        "clause": clause,
        "tier": clause.tier.value,
        "human_action_required": clause.human_action.value,
    }


def get_clause_statistics() -> Dict[str, Any]:
    """Get statistics about the clause registry."""
    by_critic = {}
    for critic in CriticDomain:
        by_critic[critic.value] = len(get_clauses_by_critic(critic))

    by_tier = {}
    for tier in EscalationTier:
        by_tier[tier.value] = len(get_clauses_by_tier(tier))

    return {
        "total_clauses": len(CLAUSES),
        "clauses_by_critic": by_critic,
        "clauses_by_tier": by_tier,
        "tier_2_count": len(get_clauses_by_tier(EscalationTier.TIER_2)),
        "tier_3_count": len(get_clauses_by_tier(EscalationTier.TIER_3)),
        "handbook_version": "v8.1",
        "canonical_status": "BINDING",
    }


# Backward compatibility - keep RULES name but point to CLAUSES
RULES: Dict[str, Any] = CLAUSES
