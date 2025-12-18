"""
Governance metrics export.
Safe for dashboards and audits.
"""

from replay_store import load_human_reviews


def review_metrics(case_id: str):
    reviews = load_human_reviews(case_id)
    if not reviews:
        return {}

    outcomes: dict[str, int] = {}
    for r in reviews:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1

    dissent_preserved = sum(
        1
        for r in reviews
        if r.get("dissent_evaluation", {}).get("present")
        and r.get("dissent_evaluation", {}).get("preserved")
    )

    return {
        "total_reviews": len(reviews),
        "outcomes": outcomes,
        "dissent_preservation_rate": dissent_preserved / len(reviews),
    }
