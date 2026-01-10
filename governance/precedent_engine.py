from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .types import RequestContext, PrecedentRef


@dataclass(frozen=True)
class PrecedentCandidate:
    precedent_id: str
    version: int
    binding_level: str
    jurisdiction: str
    domains: List[str]
    scope: Dict[str, Any]
    risk_tier: str
    core_rights_tags: List[str]
    triggers: Dict[str, Any]
    decision: Dict[str, Any]
    rationale: Dict[str, Any]
    examples: Dict[str, Any]

    # Scores
    retrieval_score: float = 0.0
    final_score: float = 0.0
    match_score: float = 0.0  # normalized 0..1

    def to_best_match_dict(self) -> Dict[str, Any]:
        return {
            "precedent_id": self.precedent_id,
            "version": self.version,
            "binding_level": self.binding_level,
            "match_score": self.match_score,
            "decision": self.decision,
            "summary": (self.rationale or {}).get("summary", "")
        }


# ---- Retrieval primitives (stubs) ----
def vector_similarity(query_text: str, candidate_text: str) -> float:
    """Stub: replace with real embedding similarity."""
    return 0.0


def bm25_normalized(query_text: str, candidate_text: str) -> float:
    """Stub: replace with real keyword retrieval."""
    return 0.0


def trigger_hits(query_text: str, patterns: List[str]) -> float:
    """Stub: basic string containment scoring; replace with regex / pattern engine."""
    hits = 0
    for p in patterns or []:
        if p and p.lower() in query_text.lower():
            hits += 1
    if not patterns:
        return 0.0
    return min(1.0, hits / max(1, len(patterns)))


# ---- Candidate filtering ----
def filter_candidates(all_candidates: List[PrecedentCandidate], ctx: RequestContext) -> List[PrecedentCandidate]:
    out: List[PrecedentCandidate] = []
    for c in all_candidates:
        if c.jurisdiction != ctx.jurisdiction:
            continue
        if not set(c.domains).intersection(set(ctx.domains)):
            continue
        out.append(c)
    return out


# ---- Hybrid retrieval ----
def hybrid_rank(candidates: List[PrecedentCandidate], query_text: str) -> List[PrecedentCandidate]:
    ranked: List[PrecedentCandidate] = []
    for c in candidates:
        # candidate text fields for retrieval
        candidate_text = " ".join([
            c.precedent_id,
            (c.rationale or {}).get("summary", ""),
            " ".join((c.triggers or {}).get("trigger_patterns", []) or []),
            " ".join((c.examples or {}).get("green", []) or []),
            " ".join((c.examples or {}).get("amber", []) or []),
            " ".join((c.examples or {}).get("red", []) or []),
        ])

        v = vector_similarity(query_text, candidate_text)
        k = bm25_normalized(query_text, candidate_text)
        t = trigger_hits(query_text, (c.triggers or {}).get("trigger_patterns", []) or [])

        retrieval = 0.55 * v + 0.30 * k + 0.15 * t
        ranked.append(PrecedentCandidate(**{**c.__dict__, "retrieval_score": retrieval}))

    ranked.sort(key=lambda x: x.retrieval_score, reverse=True)
    return ranked


def binding_rank(level: str) -> int:
    return {"hard": 3, "soft": 2, "advisory": 1}.get(level, 1)


def scope_specificity(scope: Dict[str, Any], ctx: RequestContext) -> int:
    """Rough specificity score: workflow-specific > product-specific > global."""
    applies = (scope or {}).get("applies_to", {}) or {}
    workflows = applies.get("workflows", []) or []
    products = applies.get("products", []) or []
    roles = applies.get("user_roles", []) or []

    score = 0
    if ctx.workflow in workflows:
        score += 3
    if ctx.product in products:
        score += 2
    if ctx.user_role in roles:
        score += 1
    return score


def conservatism_rank(outcome: str, risk_tier: str) -> int:
    """Higher rank = more conservative."""
    if risk_tier != "high":
        return 0
    return {
        "route_to_human": 4,
        "refuse": 3,
        "modify": 2,
        "permit": 1
    }.get(outcome, 0)


def apply_precedence_resolver(candidates: List[PrecedentCandidate], ctx: RequestContext) -> List[PrecedentCandidate]:
    """Lexicographic ordering implementing: hard>soft>advisory, specificity, recency, conservatism, retrieval."""
    def key(c: PrecedentCandidate):
        b = binding_rank(c.binding_level)
        s = scope_specificity(c.scope, ctx)
        r = c.version
        cons = conservatism_rank((c.decision or {}).get("outcome", "permit"), ctx.risk_tier)
        return (b, s, r, cons, c.retrieval_score)

    sorted_list = sorted(candidates, key=key, reverse=True)

    if not sorted_list:
        return sorted_list

    max_binding = max(binding_rank(c.binding_level) for c in sorted_list) or 1
    max_specificity = max(scope_specificity(c.scope, ctx) for c in sorted_list) or 1
    max_version = max(c.version for c in sorted_list) or 1
    max_conservatism = max(conservatism_rank((c.decision or {}).get("outcome", "permit"), ctx.risk_tier) for c in sorted_list) or 1
    max_retrieval = max(c.retrieval_score for c in sorted_list) or 1

    def _norm(value: float, max_value: float) -> float:
        return value / max_value if max_value else 0.0

    weighted: List[PrecedentCandidate] = []
    for c in sorted_list:
        b = binding_rank(c.binding_level)
        s = scope_specificity(c.scope, ctx)
        r = c.version
        cons = conservatism_rank((c.decision or {}).get("outcome", "permit"), ctx.risk_tier)
        retrieval = c.retrieval_score

        composite = (
            0.35 * _norm(b, max_binding)
            + 0.25 * _norm(s, max_specificity)
            + 0.15 * _norm(r, max_version)
            + 0.10 * _norm(cons, max_conservatism)
            + 0.15 * _norm(retrieval, max_retrieval)
        )
        composite = max(0.0, min(1.0, composite))

        weighted.append(
            PrecedentCandidate(
                **{
                    **c.__dict__,
                    "final_score": composite,
                    "match_score": composite,
                }
            )
        )
    return weighted


def match_precedents(
    ctx: RequestContext,
    all_candidates: List[PrecedentCandidate]
) -> Tuple[Optional[PrecedentCandidate], float, List[PrecedentCandidate]]:
    """Main entry point.

    In production, all_candidates comes from the Precedent Ledger store (Postgres) + search index.
    """
    filtered = filter_candidates(all_candidates, ctx)
    if not filtered:
        return None, 0.0, []

    ranked = hybrid_rank(filtered, ctx.text)
    resolved = apply_precedence_resolver(ranked, ctx)
    best = resolved[0] if resolved else None
    best_score = float(best.match_score) if best else 0.0
    return best, best_score, resolved
