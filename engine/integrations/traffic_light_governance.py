from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Traffic Light governance routing package (Patch Packs 1-3)
from governance.types import RequestContext, RouterSignals
from governance.governor import evaluate
from governance.audit import make_governance_event
from governance.audit_sink import append_jsonl_safely
from governance.precedent_engine import PrecedentCandidate

logger = logging.getLogger(__name__)


def _truthy(val: Optional[str]) -> bool:
    if val is None:
        return False
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_list(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x) for x in val if x is not None]
    if isinstance(val, str):
        return [val]
    return []


def _candidate_from_case(case: Dict[str, Any], *, fallback: Dict[str, Any]) -> PrecedentCandidate:
    """Coerce a V8 precedent case intoளம் a PrecedentCandidate.

    We fill missing fields with conservative defaults so the router can still compute coverage.
    """
    meta = (case or {}).get("metadata", {}) or {}
    pid = str(meta.get("precedent_id") or case.get("precedent_id") or case.get("id") or meta.get("id") or "UNKNOWN")

    # Versioning is optional in current V8 examples; default to 1.
    try:
        version = int(meta.get("version", 1))
    except Exception:
        version = 1

    binding = str(meta.get("binding_level") or meta.get("binding") or "advisory").lower()
    if binding not in {"hard", "soft", "advisory"}:
        binding = "advisory"

    # Outcome mapping (optional)
    outcome_raw = (meta.get("decision") or {}).get("outcome") if isinstance(meta.get("decision"), dict) else meta.get("outcome")
    outcome = str(outcome_raw or "permit")
    if outcome in {"allow", "green"}:
        outcome = "permit"
    if outcome in {"deny", "block", "red"}:
        outcome = "refuse"
    if outcome in {"escalate", "amber", "route"}:
        outcome = "route_to_human"

    # Retrieval score if present
    try:
        retrieval_score = float(meta.get("retrieval_score", meta.get("score", 0.0)) or 0.0)
    except Exception:
        retrieval_score = 0.0

    rationale_summary = str(meta.get("summary") or (meta.get("rationale") or {}).get("summary") or "")
    if not rationale_summary:
        text = str(case.get("text") or case.get("content") or "")
        rationale_summary = text[:400]

    domains = _safe_list(meta.get("domains")) or _safe_list(fallback.get("domains"))
    jurisdiction = str(meta.get("jurisdiction") or fallback.get("jurisdiction") or "global")
    risk_tier = str(meta.get("risk_tier") or fallback.get("risk_tier") or "medium")

    scope = meta.get("scope")
    if not isinstance(scope, dict):
        scope = {
            "applies_to": {
                "products": _safe_list(fallback.get("product")),
                "workflows": _safe_list(fallback.get("workflow")),
                "user_roles": _safe_list(fallback.get("user_role")),
            }
        }

    triggers = meta.get("triggers")
    if not isinstance(triggers, dict):
        triggers = {"trigger_patterns": []}

    decision = meta.get("decision")
    if not isinstance(decision, dict):
        decision = {"outcome": outcome}
    else:
        decision = {**decision, "outcome": decision.get("outcome") or outcome}

    return PrecedentCandidate(
        precedent_id=pid,
        version=version,
        binding_level=binding,
        jurisdiction=jurisdiction,
        domains=domains or ["general"],
        scope=scope,
        risk_tier=risk_tier,
        core_rights_tags=_safe_list(meta.get("core_rights_tags")) or [],
        triggers=triggers,
        decision=decision,
        rationale={"summary": rationale_summary},
        examples=meta.get("examples") if isinstance(meta.get("examples"), dict) else {},
        retrieval_score=retrieval_score,
        final_score=0.0,
        match_score=0.0,
    )


def candidates_from_precedent_bundle(precedent_data: Optional[Dict[str, Any]], *, fallback: Dict[str, Any]) -> List[PrecedentCandidate]:
    """Extract candidate precedents from V8 precedent alignment output.

    This function is intentionally permissive — it works with multiple shapes:
      - {"cases": [...]}
      - {"retrieval": {"cases": [...]}}
      - {"retrieval": {"precedent_cases": [...]}}
      - {"top_case": {...}}
    """
    if not precedent_data or not isinstance(precedent_data, dict):
        return []

    candidates: List[Dict[str, Any]] = []

    # Common shapes
    if isinstance(precedent_data.get("cases"), list):
        candidates.extend(precedent_data["cases"])

    retrieval = precedent_data.get("retrieval")
    if isinstance(retrieval, dict):
        for k in ("cases", "precedent_cases", "candidates"):
            if isinstance(retrieval.get(k), list):
                candidates.extend(retrieval[k])
        top = retrieval.get("top_case")
        if isinstance(top, dict):
            candidates.append(top)

    top_case = precedent_data.get("top_case")
    if isinstance(top_case, dict):
        candidates.append(top_case)

    # De-dupe by a best-effort id
    seen = set()
    out: List[PrecedentCandidate] = []
    for c in candidates:
        if not isinstance(c, dict):
            continue
        meta = c.get("metadata") if isinstance(c.get("metadata"), dict) else {}
        pid = str((meta or {}).get("precedent_id") or c.get("precedent_id") or c.get("id") or (meta or {}).get("id") or "")
        if pid and pid in seen:
            continue
        if pid:
            seen.add(pid)
        out.append(_candidate_from_case(c, fallback=fallback))

    return out


def divergence_from_uncertainty(uncertainty: Optional[Dict[str, Any]]) -> float:
    if not uncertainty or not isinstance(uncertainty, dict):
        return 0.0
    for key in ("critic_divergence", "overall_uncertainty", "epistemic_uncertainty"):
        if key in uncertainty:
            try:
                return float(uncertainty.get(key) or 0.0)
            except Exception as e:
                logger.debug(f"Failed to extract uncertainty value for key '{key}': {e}")
    return 0.0


@dataclass
class TrafficLightGovernanceHook:
    """Runtime hook that:

    - Computes a Traffic Light route using precedent coverage + uncertainty
    - Emits a governance event (append-only)
    - Returns reviewer-safe metadata for attachment to the response

    Sanctity rule preserved:
      - Reviewers never see critic internals
      - Reviewers cannot override the critic ensemble's runtime output
      - Reviewers author *future precedent* only
    """

    enabled: bool = True
    router_config_path: str = "governance/router_config.yaml"
    events_jsonl_path: Optional[str] = "governance_events.jsonl"
    mode: str = "observe"  # observe | enforce (enforce is intentionally not used by this hook)

    @classmethod
    def from_env(cls, **kwargs) -> "TrafficLightGovernanceHook":
        enabled = kwargs.pop("enabled", True)
        env = os.getenv("ELEANOR_TRAFFIC_LIGHT_ENABLED")
        if env is not None:
            enabled = _truthy(env)
        return cls(enabled=enabled, **kwargs)

    async def apply(
        self,
        *,
        trace_id: str,
        text: str,
        context: Optional[Dict[str, Any]],
        aggregated: Optional[Dict[str, Any]],
        precedent_data: Optional[Dict[str, Any]],
        uncertainty_data: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        if not os.path.exists(self.router_config_path):
            # Router config not present; treat hook as disabled.
            return None

        ctx_dict = context or {}
        domains = _safe_list(ctx_dict.get("domains")) or ["general"]
        core_rights = _safe_list(ctx_dict.get("core_rights_tags"))
        risk_tier = str(ctx_dict.get("risk_tier") or "medium")
        if risk_tier not in {"low", "medium", "high"}:
            risk_tier = "medium"

        req = RequestContext(
            request_id=trace_id,
            text=text,
            jurisdiction=str(ctx_dict.get("jurisdiction") or "global"),
            product=str(ctx_dict.get("product") or "unknown"),
            workflow=str(ctx_dict.get("workflow") or "default"),
            user_role=str(ctx_dict.get("user_role") or "user"),
            risk_tier=risk_tier,
            domains=domains,
            core_rights_tags=core_rights,
        )

        fallback = {
            "jurisdiction": req.jurisdiction,
            "product": req.product,
            "workflow": req.workflow,
            "user_role": req.user_role,
            "risk_tier": req.risk_tier,
            "domains": req.domains,
        }

        candidates = candidates_from_precedent_bundle(precedent_data, fallback=fallback)

        # Policy violation proxy (observer-safe): treat DENY as policy violation.
        policy_violation = False
        if aggregated and isinstance(aggregated, dict):
            decision = str(aggregated.get("decision") or "").lower()
            policy_violation = decision in {"deny", "block"}

        divergence = divergence_from_uncertainty(uncertainty_data)

        sig = RouterSignals(
            coverage_score=0.0,
            divergence_score=float(divergence),
            policy_violation=bool(policy_violation),
            telemetry={
                "overall_uncertainty": (uncertainty_data or {}).get("overall_uncertainty") if isinstance(uncertainty_data, dict) else None,
                "needs_escalation": (uncertainty_data or {}).get("needs_escalation") if isinstance(uncertainty_data, dict) else None,
            },
            flags=[]
        )

        def candidate_provider(_: RequestContext) -> List[PrecedentCandidate]:
            return candidates

        def divergence_provider(_: RequestContext) -> float:
            return float(divergence)

        result = evaluate(
            ctx=req,
            router_config_path=self.router_config_path,
            candidate_provider=candidate_provider,
            divergence_provider=divergence_provider,
            signals=sig,
            risk_domain=domains[0] if domains else "unknown",
        )

        # Rehydrate signals with computed coverage for event creation
        event_sig = RouterSignals(
            coverage_score=float(result.coverage_score),
            divergence_score=float(divergence),
            policy_violation=bool(policy_violation),
            telemetry=sig.telemetry,
            flags=sig.flags,
        )

        event = make_governance_event(
            ctx=req,
            decision=result.router_decision,
            bundle=result.constraints,
            signals=event_sig,
            applied_precedents=None,
            created_by="eleanor-traffic-light-hook",
        )

        err = None
        if self.events_jsonl_path:
            err = append_jsonl_safely(self.events_jsonl_path, event, swallow_errors=True)

        applied = list(event.get("applied_precedents") or [])

        # Metadata safe for user-facing attachment
        return {
            "route": result.constraints.route,
            "outcome": result.constraints.outcome,
            "reason": result.router_decision.reason,
            "applied_precedent_ids": applied,
            "event_id": event.get("event_id"),
            "event_sink_error": err,
        }
