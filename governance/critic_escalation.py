"""
Charter-aligned escalation clauses per critic.

These clauses define when a critic can unilaterally trigger human review.
They DO NOT modify aggregator reasoning; they only gate automatic execution.
"""

from enum import Enum
from typing import List, Dict, Any


class EscalationTier(str, Enum):
    ACKNOWLEDGMENT = "human_acknowledgment"  # Tier 2
    DETERMINATION = "human_determination"  # Tier 3


# Canonical clauses (aligned to escalation doctrine; critics may be added as they exist)
CRITIC_CLAUSES: Dict[str, List[Dict[str, Any]]] = {
    "autonomy": [
        {
            "id": "A1",
            "name": "Meaningful Consent Failure",
            "definition": "Material effect without reasonable opportunity for informed, voluntary consent or refusal.",
            "tier": EscalationTier.ACKNOWLEDGMENT,
            "udhr": ["Art 1", "Art 3", "Art 12"],
        },
        {
            "id": "A2",
            "name": "Coercive or Manipulative Influence",
            "definition": "Pressure, deception, or asymmetric information that meaningfully constrains free choice.",
            "tier": EscalationTier.DETERMINATION,
            "udhr": ["Art 1", "Art 3", "Art 12"],
        },
        {
            "id": "A3",
            "name": "Irreversible Autonomy Impact",
            "definition": "Non-reversible limitation on future choice, agency, or self-determination.",
            "tier": EscalationTier.DETERMINATION,
            "udhr": ["Art 1", "Art 3", "Art 12"],
        },
    ],
    "dignity": [
        {
            "id": "D1",
            "name": "Instrumentalization of Persons",
            "definition": "Individuals treated primarily as means to an end.",
            "tier": EscalationTier.ACKNOWLEDGMENT,
            "udhr": ["Art 1", "Art 22"],
        },
        {
            "id": "D2",
            "name": "Degrading or Dehumanizing Outcome",
            "definition": "Humiliation, stigmatization, or erosion of intrinsic human worth.",
            "tier": EscalationTier.DETERMINATION,
            "udhr": ["Art 1", "Art 22"],
        },
        {
            "id": "D3",
            "name": "Harm Without Voice",
            "definition": "Imposed harm without meaningful participation, representation, or recourse.",
            "tier": EscalationTier.ACKNOWLEDGMENT,
            "udhr": ["Art 1", "Art 2"],
        },
    ],
    "privacy": [
        {
            "id": "P1",
            "name": "Non-Consensual Identity Inference",
            "definition": "Infers/acts on sensitive attributes without explicit consent.",
            "tier": EscalationTier.DETERMINATION,
            "udhr": ["Art 12", "Art 6"],
        },
        {
            "id": "P2",
            "name": "Persistent or Linked Identity",
            "definition": "Enables durable identification, tracking, or cross-context linkage.",
            "tier": EscalationTier.DETERMINATION,
            "udhr": ["Art 12"],
        },
        {
            "id": "P3",
            "name": "Context Collapse",
            "definition": "Information repurposed across contexts with materially different expectations.",
            "tier": EscalationTier.ACKNOWLEDGMENT,
            "udhr": ["Art 12"],
        },
    ],
    "fairness": [
        {
            "id": "F1",
            "name": "Protected Class Impact",
            "definition": "Differential treatment/outcomes affecting protected classes.",
            "tier": EscalationTier.DETERMINATION,
            "udhr": ["Art 2", "Art 7"],
        },
        {
            "id": "F2",
            "name": "Structural Bias Amplification",
            "definition": "Existing inequities reinforced or magnified through automation.",
            "tier": EscalationTier.ACKNOWLEDGMENT,
            "udhr": ["Art 2", "Art 7"],
        },
        {
            "id": "F3",
            "name": "Opaque Differential Treatment",
            "definition": "Material differences cannot be meaningfully explained to affected parties.",
            "tier": EscalationTier.ACKNOWLEDGMENT,
            "udhr": ["Art 2", "Art 7"],
        },
    ],
    "due_process": [
        {
            "id": "DP1",
            "name": "No Contestability",
            "definition": "No meaningful mechanism to challenge, appeal, or seek explanation.",
            "tier": EscalationTier.DETERMINATION,
            "udhr": ["Art 8", "Art 10"],
        },
        {
            "id": "DP2",
            "name": "No Attribution",
            "definition": "Responsibility cannot be clearly assigned to a human or accountable authority.",
            "tier": EscalationTier.DETERMINATION,
            "udhr": ["Art 8", "Art 10"],
        },
        {
            "id": "DP3",
            "name": "Unreviewable Automation",
            "definition": "Decision cannot be reconstructed, audited, or explained post-hoc.",
            "tier": EscalationTier.ACKNOWLEDGMENT,
            "udhr": ["Art 8", "Art 10"],
        },
    ],
    "precedent": [
        {
            "id": "PR1",
            "name": "No Precedent + High Impact",
            "definition": "High-impact decision with no relevant precedent.",
            "tier": EscalationTier.DETERMINATION,
            "udhr": [],
        },
        {
            "id": "PR2",
            "name": "Conflicting Precedent",
            "definition": "Existing precedents point to materially different conclusions.",
            "tier": EscalationTier.ACKNOWLEDGMENT,
            "udhr": [],
        },
        {
            "id": "PR3",
            "name": "Uncontrolled Norm Creation",
            "definition": "Action would establish a new norm/standard without explicit human authorization.",
            "tier": EscalationTier.DETERMINATION,
            "udhr": [],
        },
    ],
    "uncertainty": [
        {
            "id": "U1",
            "name": "Epistemic Insufficiency",
            "definition": "Critical contextual information is missing for responsible judgment.",
            "tier": EscalationTier.ACKNOWLEDGMENT,
            "udhr": [],
        },
        {
            "id": "U2",
            "name": "High Impact × High Uncertainty",
            "definition": "Combination of uncertainty and potential harm exceeds defined bounds.",
            "tier": EscalationTier.DETERMINATION,
            "udhr": [],
        },
        {
            "id": "U3",
            "name": "Competence Boundary Exceeded",
            "definition": "Scenario falls outside validated competence envelope.",
            "tier": EscalationTier.DETERMINATION,
            "udhr": [],
        },
    ],
}


def get_clauses(critic: str) -> List[Dict[str, Any]]:
    """Return escalation clauses for a critic (empty list if none registered)."""
    return CRITIC_CLAUSES.get(critic.lower(), [])


def get_clause(critic: str, clause_id: str) -> Dict[str, Any]:
    """Return a specific clause dict, or {} if not found."""
    for clause in get_clauses(critic):
        if clause.get("id") == clause_id:
            return clause
    return {}
