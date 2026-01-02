"""
End-to-end demo:
- run a synthetic case
- trigger human review
- submit a mock review
- attempt promotion
"""

from uuid import uuid4
from datetime import datetime, timezone

from governance.review_triggers import ReviewTriggerEvaluator
from governance.review_packets import build_review_packet
from replay_store import store_review_packet, store_human_review
from precedent.promotion_guard import assert_promotion_allowed
from governance.human_review.schemas import (
    HumanReviewRecord,
    SeverityAssessment,
    DissentEvaluation,
    ReviewOutcome,
)


# --- Fake Case Object (minimal) ---
class DemoCase:
    def __init__(self):
        self.id = str(uuid4())
        self.domain = "employment"
        self.severity = 1.7
        self.critic_disagreement = 0.7
        self.novel_precedent = True
        self.rights_impacted = ["dignity"]
        self.uncertainty = type("U", (), {"flags": ["context_missing"]})
        self.critic_outputs = {"dignity": {"analysis": "Possible exclusion"}}
        self.aggregator_summary = "Risk to dignity noted"
        self.dissent = "Autonomy critic raised concern"
        self.citations = {"UDHR": ["Art 1", "Art 23"]}


case = DemoCase()

# --- Trigger Review ---
trigger = ReviewTriggerEvaluator()
decision = trigger.evaluate(case)

print("Review required:", decision)

packet = build_review_packet(case, decision)
store_review_packet(packet)

# --- Submit Human Review ---
review = HumanReviewRecord(
    review_id=str(uuid4()),
    case_id=case.id,
    reviewer_role="constitutional_reviewer",
    timestamp=datetime.now(timezone.utc).isoformat(),
    coverage_issues=[],
    severity_assessment=SeverityAssessment(
        original=1.7,
        adjusted=1.4,
        justification="Impact narrower than initial assessment",
    ),
    dissent_evaluation=DissentEvaluation(
        present=True,
        preserved=True,
        notes="Minority concern remains valid",
    ),
    uncertainty_adequate=True,
    outcome=ReviewOutcome.ADJUSTED,
    reviewer_justification="Reasoning sound; severity calibrated",
)

store_human_review(review)

# --- Attempt Promotion ---
try:
    assert_promotion_allowed(case.id)
    print("✅ Promotion allowed")
except Exception as e:
    print("❌ Promotion blocked:", e)

# 🎯 Result: perfect demo story in under 60 seconds.
