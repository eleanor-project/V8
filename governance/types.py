from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal


Route = Literal["green", "amber", "red"]
Outcome = Literal["permit", "modify", "refuse", "route_to_human"]
BindingLevel = Literal["hard", "soft", "advisory"]
RiskTier = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class RequestContext:
    request_id: str
    text: str
    jurisdiction: str
    product: str
    workflow: str
    user_role: str
    domains: List[str]
    risk_tier: RiskTier
    core_rights_tags: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class RouterSignals:
    # Precedent coverage score in [0,1]. 0 means no match.
    coverage_score: float = 0.0
    # Uncertainty proxy score in [0,1], where larger = more divergent/uncertain.
    divergence_score: float = 0.0
    # Set true if a policy-violation trigger already fired (from upstream classifier/rules).
    policy_violation: bool = False
    # Optional telemetry summaries (e.g., logprob aggregates) if available.
    telemetry: Optional[Dict[str, Any]] = None
    # Optional reason labels from upstream detectors.
    flags: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class PrecedentRef:
    precedent_id: str
    version: int
    binding_level: BindingLevel
    match_score: float
    summary: str = ""


@dataclass(frozen=True)
class RouterDecision:
    route: Route
    reason: str
    risk_tier: RiskTier
    coverage_score: float
    divergence_score: float


@dataclass(frozen=True)
class ConstraintsBundle:
    route: Route
    outcome: Outcome
    applied_precedents: List[Dict[str, Any]]
    constraints: Dict[str, Any]
    audit_labels: Dict[str, Any]
    human_review: Optional[Dict[str, Any]] = None
