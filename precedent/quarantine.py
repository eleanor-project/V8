"""
Quarantined cases:
- rejected
- contested
- explicitly excluded from automation
"""

from replay_store import load_human_reviews


def is_quarantined(case_id: str) -> bool:
    reviews = load_human_reviews(case_id)
    if not reviews:
        return False

    latest = sorted(
        reviews,
        key=lambda r: r.get("timestamp") or "",
        reverse=True,
    )[0]

    return latest.get("outcome") in ("contested", "rejected")
