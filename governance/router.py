from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .types import RequestContext, RouterSignals, RouterDecision, ConstraintsBundle, Route, Outcome


@dataclass(frozen=True)
class RouterConfig:
    # Coverage thresholds
    high_risk_green_min: float
    medium_risk_green_min: float
    low_risk_green_min: float

    # Divergence thresholds
    high_uncertainty_min: float
    medium_uncertainty_min: float

    # Routing behaviors
    hard_red_on_policy_violation: bool = True
    amber_on_no_precedent_high_risk: bool = True
    conservative_high_risk_bias: bool = True

    # Output knobs
    include_precedent_ids: bool = True
    include_overlay_refs: bool = True


def load_router_config(path: str | Path) -> RouterConfig:
    """Load router config from YAML-like file.

    We intentionally keep this loader minimal and dependency-free.
    It supports a small subset of YAML used in router_config.yaml.

    If you prefer, replace this with a real YAML loader in your stack.
    """
    text = Path(path).read_text(encoding="utf-8")
    # Tiny parser: read 'key: value' lines under known sections.
    kv: Dict[str, Any] = {}
    section_stack: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith(":") and ":" not in line[:-1]:
            # section header
            section_stack.append(line[:-1])
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            key = ".".join(section_stack + [k.strip()])
            val = v.strip()
            # parse booleans / numbers / strings
            if val.lower() in ("true", "false"):
                kv[key] = val.lower() == "true"
            else:
                try:
                    kv[key] = float(val) if "." in val else int(val)
                except ValueError:
                    kv[key] = val.strip('"').strip("'")
        # pop sections when encountering top-level keys (very small heuristic)
        if raw and not raw.startswith(" ") and section_stack:
            # if we hit a non-indented line that isn't a section continuation,
            # we likely started a new top-level key; reset stack conservatively.
            if not raw.strip().endswith(":") and raw.split(":", 1)[0].strip() not in ("version",):
                # don't clear for normal kv pairs; only clear if next line is top-level.
                pass

    return RouterConfig(
        high_risk_green_min=float(kv.get("thresholds.coverage.high_risk_green_min", 0.75)),
        medium_risk_green_min=float(kv.get("thresholds.coverage.medium_risk_green_min", 0.65)),
        low_risk_green_min=float(kv.get("thresholds.coverage.low_risk_green_min", 0.55)),
        high_uncertainty_min=float(kv.get("thresholds.divergence.high_uncertainty_min", 0.35)),
        medium_uncertainty_min=float(kv.get("thresholds.divergence.medium_uncertainty_min", 0.20)),
        hard_red_on_policy_violation=bool(kv.get("routing.hard_red_on_policy_violation", True)),
        amber_on_no_precedent_high_risk=bool(kv.get("routing.amber_on_no_precedent_high_risk", True)),
        conservative_high_risk_bias=bool(kv.get("routing.conservative_high_risk_bias", True)),
        include_precedent_ids=bool(kv.get("outputs.include_precedent_ids", True)),
        include_overlay_refs=bool(kv.get("outputs.include_overlay_refs", True)),
    )


def decide_route(ctx: RequestContext, sig: RouterSignals, cfg: RouterConfig) -> RouterDecision:
    """Return the Traffic Light decision.

    Sanctity rule reminder:
    - This router *routes*; it does not change critic ensemble outcomes.
    - Human reviewers do not interact with this function and cannot override runtime decisions.
    """
    coverage = max(0.0, min(1.0, float(sig.coverage_score)))
    div = max(0.0, min(1.0, float(sig.divergence_score)))

    if cfg.hard_red_on_policy_violation and sig.policy_violation:
        return RouterDecision(route="red", reason="policy_violation", risk_tier=ctx.risk_tier,
                             coverage_score=coverage, divergence_score=div)

    # Determine uncertainty band
    uncertainty = "low"
    if div >= cfg.high_uncertainty_min:
        uncertainty = "high"
    elif div >= cfg.medium_uncertainty_min:
        uncertainty = "medium"

    # Coverage thresholds by risk tier
    if ctx.risk_tier == "high":
        green_min = cfg.high_risk_green_min
    elif ctx.risk_tier == "medium":
        green_min = cfg.medium_risk_green_min
    else:
        green_min = cfg.low_risk_green_min

    # No precedent case
    if ctx.risk_tier == "high" and coverage <= 0.0001 and cfg.amber_on_no_precedent_high_risk:
        return RouterDecision(route="amber", reason="no_precedent_high_risk", risk_tier=ctx.risk_tier,
                             coverage_score=coverage, divergence_score=div)

    # AMBER conditions
    if ctx.risk_tier == "high" and (coverage < green_min or uncertainty in ("medium", "high")):
        reason = "high_risk_low_coverage" if coverage < green_min else "high_risk_uncertainty"
        return RouterDecision(route="amber", reason=reason, risk_tier=ctx.risk_tier,
                             coverage_score=coverage, divergence_score=div)

    if ctx.risk_tier == "medium" and (coverage < green_min and uncertainty in ("medium", "high")):
        return RouterDecision(route="amber", reason="medium_risk_low_coverage_and_uncertainty", risk_tier=ctx.risk_tier,
                             coverage_score=coverage, divergence_score=div)

    # GREEN
    return RouterDecision(route="green", reason="sufficient_coverage_or_low_risk", risk_tier=ctx.risk_tier,
                         coverage_score=coverage, divergence_score=div)


def build_constraints_bundle(
    ctx: RequestContext,
    decision: RouterDecision,
    best_precedent: Optional[Dict[str, Any]] = None
) -> ConstraintsBundle:
    """Build a constraints bundle in the shape of governance/schemas/constraints.bundle.schema.json."""
    route: Route = decision.route

    # Default outcome by route
    if route == "red":
        outcome: Outcome = "refuse"
    elif route == "amber":
        outcome = "route_to_human" if not best_precedent else str(best_precedent.get("decision", {}).get("outcome", "route_to_human"))  # type: ignore
        if outcome not in ("permit", "modify", "refuse", "route_to_human"):
            outcome = "route_to_human"
    else:
        outcome = "permit" if not best_precedent else str(best_precedent.get("decision", {}).get("outcome", "permit"))  # type: ignore
        if outcome not in ("permit", "modify", "refuse", "route_to_human"):
            outcome = "permit"

    applied = []
    if best_precedent:
        applied.append({
            "precedent_id": best_precedent.get("precedent_id", ""),
            "version": int(best_precedent.get("version", 1)),
            "binding_level": best_precedent.get("binding_level", "advisory"),
            "match_score": float(best_precedent.get("match_score", 0.0)),
        })

    # Minimal constraints defaults; precedent engine can override/merge these.
    constraints = {
        "must_include": [],
        "must_not": [],
        "required_checks": [],
        "style": {
            "tone": "cautious" if route != "green" else "neutral",
            "structure": ["answer", "limitations", "rationale", "next_steps"] if route != "green" else ["answer", "next_steps"]
        },
        "citations": {
            "include_precedent_ids": True,
            "include_overlay_refs": True
        }
    }

    # If precedent contains decision constraints, merge them in.
    if best_precedent:
        dec = best_precedent.get("decision", {}) or {}
        for k in ("must_include", "must_not", "required_checks"):
            if isinstance(dec.get(k), list):
                constraints[k].extend([str(x) for x in dec.get(k)])
        if isinstance(dec.get("conditions"), list):
            # treat conditions as required checks unless you split them later
            constraints["required_checks"].extend([str(x) for x in dec.get("conditions")])

    audit_labels = {
        "jurisdiction": ctx.jurisdiction,
        "domains": ctx.domains,
        "core_rights_tags": ctx.core_rights_tags,
        "risk_tier": ctx.risk_tier
    }

    human_review = None
    if outcome == "route_to_human":
        human_review = {
            "required": True,
            "queue": "human-review-general",
            "reason": decision.reason
        }

    return ConstraintsBundle(
        route=route,
        outcome=outcome,
        applied_precedents=applied,
        constraints=constraints,
        audit_labels=audit_labels,
        human_review=human_review
    )
