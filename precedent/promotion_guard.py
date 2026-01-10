"""
Hard governance guardrails for precedent promotion.
This is the constitutional backstop.
"""

from engine.replay_store import load_human_reviews


class PromotionBlocked(Exception):
    """Raised when a case cannot be promoted due to unresolved review."""


def assert_promotion_allowed(case_id: str):
    """
    Raises if a case has an unresolved human review requirement.
    """
    reviews = load_human_reviews(case_id)

    if not reviews:
        raise PromotionBlocked(f"Case {case_id} has no completed human review")

    latest = sorted(
        reviews,
        key=lambda r: r.get("timestamp") or "",
        reverse=True,
    )[0]

    if latest.get("outcome") in ("contested", "rejected"):
        raise PromotionBlocked(f"Case {case_id} outcome blocks promotion: {latest.get('outcome')}")

    return True
