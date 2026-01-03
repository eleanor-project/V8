"""
Immutable contract between Eleanor runtime and human reviewers.
This packet captures reasoning context — never outcomes.
"""

from pydantic import BaseModel
from typing import Dict, List, Optional


class ReviewPacket(BaseModel):
    case_id: str
    domain: str

    severity: float
    uncertainty_flags: List[str]

    critic_outputs: Dict[str, dict]
    aggregator_summary: str
    dissent: Optional[str]

    citations: Dict[str, List[str]]
    triggers: List[str]


def build_review_packet(case, review_decision: dict) -> ReviewPacket:
    return ReviewPacket(
        case_id=case.id,
        domain=getattr(case, "domain", "unspecified"),
        severity=case.severity,
        uncertainty_flags=getattr(case.uncertainty, "flags", []),
        critic_outputs=case.critic_outputs,
        aggregator_summary=case.aggregator_summary,
        dissent=getattr(case, "dissent", None),
        citations=case.citations,
        triggers=review_decision["triggers"],
    )
