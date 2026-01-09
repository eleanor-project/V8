from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from .types import RequestContext, RouterSignals
from .precedent_engine import PrecedentCandidate
from .governor import evaluate
from .audit import make_governance_event


def load_candidates(path: str) -> List[PrecedentCandidate]:
    """Load precedent candidates from a JSON file.

    Expected format: a JSON array of objects compatible with PrecedentCandidate fields
    (see governance/precedent_engine.py). This is a dry-run convenience; production uses
    the Precedent Ledger + search index.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Candidates file must be a JSON array.")
    out: List[PrecedentCandidate] = []
    for obj in raw:
        out.append(PrecedentCandidate(**obj))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="ELEANOR governance dry-run CLI (router + precedent matching).")
    p.add_argument("--text", required=True, help="User prompt / request text")
    p.add_argument("--request-id", default="dryrun-0001")
    p.add_argument("--jurisdiction", default="Org-Policy-Set-A")
    p.add_argument("--product", default="unknown-product")
    p.add_argument("--workflow", default="unknown-workflow")
    p.add_argument("--user-role", default="end_user")
    p.add_argument("--domains", default="general", help="Comma-separated domains, e.g. health,privacy")
    p.add_argument("--risk-tier", default="low", choices=["low","medium","high"])
    p.add_argument("--router-config", default="governance/router_config.yaml")
    p.add_argument("--candidates", help="Path to JSON array of PrecedentCandidate objects")
    p.add_argument("--policy-violation", action="store_true", help="Force a policy violation trigger (RED)")
    p.add_argument("--divergence", type=float, default=0.0, help="Set uncertainty proxy divergence score [0..1]")
    p.add_argument("--print-event", action="store_true", help="Also print a governance event record")

    args = p.parse_args()

    ctx = RequestContext(
        request_id=args.request_id,
        text=args.text,
        jurisdiction=args.jurisdiction,
        product=args.product,
        workflow=args.workflow,
        user_role=args.user_role,
        domains=[d.strip() for d in args.domains.split(",") if d.strip()],
        risk_tier=args.risk_tier
    )

    sig = RouterSignals(
        coverage_score=0.0,
        divergence_score=float(args.divergence),
        policy_violation=bool(args.policy_violation),
        telemetry=None,
        flags=[]
    )

    candidates = load_candidates(args.candidates) if args.candidates else []

    def provider(_: RequestContext):
        return candidates

    def divergence_provider(_: RequestContext):
        return float(args.divergence)

    result = evaluate(
        ctx=ctx,
        router_config_path=args.router_config,
        candidate_provider=provider,
        divergence_provider=divergence_provider,
        signals=sig,
        risk_domain=args.domains.split(",")[0].strip() if args.domains else "general"
    )

    print(json.dumps({
        "constraints_bundle": result.constraints.__dict__,
        "router_decision": result.router_decision.__dict__,
        "coverage_score": result.coverage_score,
        "best_precedent": (result.best_precedent.to_best_match_dict() if result.best_precedent else None),
        "case_packet": result.case_packet
    }, indent=2, default=str))

    if args.print_event:
        event = make_governance_event(
            ctx=ctx,
            decision=result.router_decision,
            bundle=result.constraints,
            signals=sig
        )
        print("\n--- governance_event ---")
        print(json.dumps(event, indent=2, default=str))


if __name__ == "__main__":
    main()
