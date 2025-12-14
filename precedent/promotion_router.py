"""
Promotion router for Eleanor precedent system.

Routes cases based on human review outcomes.
Promotion is NOT automatic - this assigns the lane.

Separation of powers: Jurisprudence decisions stay OUT of governance and engine logic.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json
import uuid

from governance.human_review.schemas import ReviewOutcome, HumanReviewRecord
from precedent.promotion_guard import assert_promotion_allowed


class PromotionRouter:
    """
    Routes reviewed cases to appropriate promotion lanes.

    Lanes:
    - precedent_candidate: Affirmed reasoning, candidate for precedent promotion
    - training_calibration: Adjusted reasoning, use for severity calibration
    - policy_insight: Contested reasoning, route to policy/governance insights
    - quarantine: Rejected reasoning, do not promote or learn from
    """

    def __init__(self, sunset_months: int = 12):
        """
        Initialize promotion router.

        Args:
            sunset_months: How many months before precedent sunset review (default 12)
        """
        self.sunset_months = sunset_months

        # Storage paths for different lanes
        self.lanes = {
            "precedent_candidate": Path("precedent/candidates"),
            "training_calibration": Path("precedent/calibration"),
            "policy_insight": Path("precedent/policy_insights"),
            "quarantine": Path("precedent/quarantine"),
        }

        # Ensure all lanes exist
        for lane_path in self.lanes.values():
            lane_path.mkdir(parents=True, exist_ok=True)

    def route(self, review_record: HumanReviewRecord) -> str:
        """
        Route a reviewed case to appropriate lane based on outcome.

        Args:
            review_record: Human review record

        Returns:
            str indicating which lane the case was routed to
        """
        if review_record.outcome == ReviewOutcome.AFFIRMED:
            lane = "precedent_candidate"

        elif review_record.outcome == ReviewOutcome.CLARIFIED:
            # Clarified cases can also be precedent candidates
            lane = "precedent_candidate"

        elif review_record.outcome == ReviewOutcome.ADJUSTED:
            lane = "training_calibration"

        elif review_record.outcome == ReviewOutcome.CONTESTED:
            lane = "policy_insight"

        elif review_record.outcome == ReviewOutcome.REJECTED:
            lane = "quarantine"

        else:
            # Unknown outcome, quarantine by default
            lane = "quarantine"

        # Store in appropriate lane
        self._store_in_lane(review_record, lane)

        return lane

    def _store_in_lane(self, review_record: HumanReviewRecord, lane: str) -> None:
        """
        Store review record in appropriate lane.

        Args:
            review_record: Human review record
            lane: Which lane to store in
        """
        lane_path = self.lanes[lane]
        record_file = lane_path / f"{review_record.case_id}.json"

        with open(record_file, "w") as f:
            json.dump(review_record.dict(), f, indent=2, default=str)

    def create_precedent_entry(
        self,
        case_id: str,
        reasoning_pattern: str,
        citations: Dict[str, Any],
        dissent_retained: bool,
        review_record: HumanReviewRecord,
    ) -> Dict[str, Any]:
        """
        Create a precedent entry with safeguards.

        Sunsets are MANDATORY. If someone argues otherwise, they don't understand jurisprudence.

        Args:
            case_id: Original case ID
            reasoning_pattern: The reasoning signature/pattern
            citations: Precedent citations used
            dissent_retained: Whether dissent was preserved
            review_record: Human review that affirmed this case

        Returns:
            dict representing precedent entry
        """
        precedent_id = str(uuid.uuid4())
        sunset_date = datetime.utcnow() + timedelta(days=self.sunset_months * 30)

        return {
            "precedent_id": precedent_id,
            "origin_case": case_id,
            "reasoning_pattern": reasoning_pattern,
            "citations": citations,
            "dissent_retained": dissent_retained,
            "confidence": "provisional",  # All precedents start as provisional
            "sunset_review": sunset_date.isoformat(),
            "created_at": datetime.utcnow().isoformat(),
            "reviewed_by": review_record.reviewer_role,
            "review_id": review_record.review_id,
        }

    def get_lane_contents(self, lane: str) -> list[Dict[str, Any]]:
        """
        Get all cases in a specific lane.

        Args:
            lane: Lane name

        Returns:
            List of case dicts in that lane
        """
        if lane not in self.lanes:
            return []

        lane_path = self.lanes[lane]
        cases = []

        for case_file in lane_path.glob("*.json"):
            with open(case_file, "r") as f:
                data = json.load(f)
            cases.append(data)

        return cases

    def promote_to_precedent(
        self,
        case_id: str,
        reasoning_pattern: str,
        citations: Dict[str, Any],
        dissent_retained: bool,
    ) -> Optional[Dict[str, Any]]:
        """
        Promote a case from candidate lane to actual precedent.

        This is the ONLY way to create new precedent.

        Args:
            case_id: Case to promote
            reasoning_pattern: Reasoning signature
            citations: Citations used
            dissent_retained: Whether dissent was preserved

        Returns:
            Precedent entry if successful, None if case not found in candidates
        """
        # Governance guardrail: ensure human review is completed and allows promotion
        assert_promotion_allowed(case_id)

        # Check if case is in precedent_candidate lane
        candidate_file = self.lanes["precedent_candidate"] / f"{case_id}.json"

        if not candidate_file.exists():
            return None

        # Load review record
        with open(candidate_file, "r") as f:
            review_data = json.load(f)

        review_record = HumanReviewRecord(**review_data)

        # Create precedent entry
        precedent = self.create_precedent_entry(
            case_id=case_id,
            reasoning_pattern=reasoning_pattern,
            citations=citations,
            dissent_retained=dissent_retained,
            review_record=review_record,
        )

        # Store in precedent store
        precedent_store_path = Path("precedent/store")
        precedent_store_path.mkdir(parents=True, exist_ok=True)

        precedent_file = precedent_store_path / f"{precedent['precedent_id']}.json"
        with open(precedent_file, "w") as f:
            json.dump(precedent, f, indent=2, default=str)

        return precedent
