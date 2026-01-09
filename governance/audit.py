from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

from .types import RequestContext, RouterSignals, ConstraintsBundle, RouterDecision


def make_governance_event(
    ctx: RequestContext,
    decision: RouterDecision,
    bundle: ConstraintsBundle,
    signals: RouterSignals,
    applied_precedents: Optional[List[Dict[str, Any]]] = None,
    created_by: str = "eleanor-governor"
) -> Dict[str, Any]:
    """Create an append-only governance event record.

    This is designed to map cleanly onto the governance_events table described in docs.
    Persist this event as JSONB or as a row with evidence_json JSONB.
    """
    now = datetime.now(timezone.utc)
    event_id = str(uuid.uuid4())

    evidence = {
        "router": {
            "reason": decision.reason,
            "coverage_score": float(signals.coverage_score),
            "divergence_score": float(signals.divergence_score),
            "policy_violation": bool(signals.policy_violation),
            "flags": list(signals.flags or []),
            "telemetry": signals.telemetry
        },
        "constraints_bundle": {
            "route": bundle.route,
            "outcome": bundle.outcome,
            "constraints": bundle.constraints,
            "audit_labels": bundle.audit_labels,
            "human_review": bundle.human_review
        }
    }

    applied = applied_precedents if applied_precedents is not None else list(bundle.applied_precedents or [])
    applied_ids = []
    for p in applied:
        pid = p.get("precedent_id")
        ver = p.get("version")
        if pid and ver is not None:
            applied_ids.append(f"{pid}v{ver}")

    return {
        "event_id": event_id,
        "occurred_at": now.isoformat(),
        "request_id": ctx.request_id,
        "user_role": ctx.user_role,
        "product": ctx.product,
        "workflow": ctx.workflow,
        "route_decision": bundle.route,
        "outcome": bundle.outcome,
        "applied_precedents": applied_ids,
        "notes": None,
        "evidence_json": evidence,
        "created_by": created_by
    }
