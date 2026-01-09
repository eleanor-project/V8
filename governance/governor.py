from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Dict, Any

from .types import RequestContext, RouterSignals, ConstraintsBundle, RouterDecision
from .router import load_router_config, decide_route, build_constraints_bundle
from .precedent_engine import PrecedentCandidate, match_precedents


CandidateProvider = Callable[[RequestContext], List[PrecedentCandidate]]
DivergenceProvider = Callable[[RequestContext], float]


@dataclass
class GovernorResult:
    constraints: ConstraintsBundle
    router_decision: RouterDecision
    best_precedent: Optional[PrecedentCandidate]
    coverage_score: float
    ranked_precedents: List[PrecedentCandidate]
    case_packet: Optional[Dict[str, Any]] = None


def default_divergence_provider(_: RequestContext) -> float:
    """Fallback uncertainty proxy.

    Replace with:
      - multi-sample divergence (2–3 candidate answers)
      - cross-model agreement score
      - a small dedicated risk/uncertainty model
    """
    return 0.0


def build_case_packet(
    ctx: RequestContext,
    route: str,
    risk_domain: str,
    coverage_score: float,
    uncertainty_telemetry: Optional[Dict[str, Any]],
    divergence_score: float,
    ranked: List[PrecedentCandidate],
    top_k: int = 5
) -> Dict[str, Any]:
    """Create a reviewer-safe Case Packet.

    Sanctity rule:
      - No critic internals
      - No runtime 'verdict override' controls
      - Reviewers author *future precedent*, not change outcomes
    """
    return {
        "case_id": f"CASE-{ctx.request_id}",
        "request_summary": ctx.text[:1200],
        "context": {
            "jurisdiction": ctx.jurisdiction,
            "product": ctx.product,
            "workflow": ctx.workflow,
            "user_role": ctx.user_role
        },
        "signals": {
            "route": route,
            "risk_domain": risk_domain,
            "coverage_score": float(coverage_score),
            "uncertainty_telemetry": uncertainty_telemetry,
            "uncertainty_proxy": {"divergence_score": float(divergence_score)}
        },
        "candidate_precedents": [
            {
                "precedent_id": c.precedent_id,
                "version": c.version,
                "match_score": float(c.match_score),
                "binding_level": c.binding_level,
                "summary": (c.rationale or {}).get("summary", "")
            }
            for c in ranked[:top_k]
        ],
        "explicitly_excluded": [
            "critic_ensemble_outputs",
            "critic_internal_prompts",
            "aggregator_state",
            "model_hidden_chain_of_thought"
        ]
    }


def evaluate(
    ctx: RequestContext,
    router_config_path: str,
    candidate_provider: CandidateProvider,
    divergence_provider: DivergenceProvider = default_divergence_provider,
    signals: Optional[RouterSignals] = None,
    risk_domain: str = "unknown"
) -> GovernorResult:
    """One-call governance entry point.

    Pipeline:
      1) Fetch precedent candidates (ledger/search layer)
      2) Match & resolve precedents (B)
      3) Compute router signals (coverage + uncertainty proxy + flags)
      4) Decide route (C router)
      5) Build Constraints Bundle
      6) If route requires human review, emit Case Packet (reviewer-safe)

    This preserves the sanctity rule: review influences *future precedent*, not runtime decisions.
    """
    cfg = load_router_config(router_config_path)

    # Get candidates and compute coverage
    candidates = candidate_provider(ctx)
    best, coverage, ranked = match_precedents(ctx, candidates)

    # Signals: allow upstream injection, but always set coverage/divergence
    if signals is None:
        signals = RouterSignals()

    divergence = float(divergence_provider(ctx))
    merged_signals = RouterSignals(
        coverage_score=float(coverage),
        divergence_score=float(divergence),
        policy_violation=bool(getattr(signals, "policy_violation", False)),
        telemetry=getattr(signals, "telemetry", None),
        flags=list(getattr(signals, "flags", []))
    )

    router_decision = decide_route(ctx, merged_signals, cfg)

    best_dict = best.to_best_match_dict() if best else None
    constraints = build_constraints_bundle(ctx, router_decision, best_dict)

    case_packet = None
    if constraints.human_review and constraints.human_review.get("required"):
        case_packet = build_case_packet(
            ctx=ctx,
            route=constraints.route,
            risk_domain=risk_domain,
            coverage_score=coverage,
            uncertainty_telemetry=merged_signals.telemetry,
            divergence_score=divergence,
            ranked=ranked
        )

    return GovernorResult(
        constraints=constraints,
        router_decision=router_decision,
        best_precedent=best,
        coverage_score=float(coverage),
        ranked_precedents=ranked,
        case_packet=case_packet
    )
