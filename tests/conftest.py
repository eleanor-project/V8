import pytest

from engine.engine import EleanorEngineV8, EngineConfig
from engine.aggregator.aggregator import AggregatorV8
from engine.detectors.engine import DetectorEngineV8
from engine.aggregator.escalation import resolve_escalation
from engine.execution.human_review import enforce_human_review
from engine.schemas.escalation import (
    Concern,
    CriticEvaluation,
    EscalationSignal,
    EscalationTier,
    HumanAction,
    HumanActionType,
)


@pytest.fixture
def engine():
    """
    Lightweight engine instance for async tests.
    Uses empty detector set and default aggregator to avoid network or model calls.
    """

    class _FakeRouter:
        def route(self, text: str, context: dict):
            return {
                "model_name": "fake-model",
                "model_version": "v0",
                "reason": "fixture",
                "health_score": 1.0,
                "cost": {"usd": 0.0},
                "response_text": "mock response",
                "usage": {},
            }

    return EleanorEngineV8(
        config=EngineConfig(),
        detector_engine=DetectorEngineV8(detectors={}),
        aggregator=AggregatorV8(),
        router_backend=_FakeRouter,
    )


@pytest.fixture
def base_concern():
    return Concern(
        summary="Test concern",
        rationale="Test rationale",
        impacted_rights=["UDHR Article 1"],
        confidence=0.9,
    )


@pytest.fixture
def tier2_escalation():
    return EscalationSignal.for_tier(
        tier=EscalationTier.TIER_2,
        critic_id="autonomy",
        clause_id="A1",
        clause_description="Meaningful consent failure",
        doctrine_ref="Doctrine §4",
        rationale="Consent not clearly established",
    )


@pytest.fixture
def tier3_escalation():
    return EscalationSignal.for_tier(
        tier=EscalationTier.TIER_3,
        critic_id="privacy",
        clause_id="P1",
        clause_description="Non-consensual identity inference",
        doctrine_ref="Doctrine §4",
        rationale="Sensitive inference without consent",
    )


@pytest.fixture
def critic_eval_no_escalation(base_concern):
    return CriticEvaluation(
        critic_id="fairness",
        charter_version="v1",
        concerns=[base_concern],
        escalation=None,
        severity_score=0.3,
        citations=["UDHR Article 2"],
        uncertainty=None,
    )


@pytest.fixture
def critic_eval_tier2(base_concern, tier2_escalation):
    return CriticEvaluation(
        critic_id="autonomy",
        charter_version="v1",
        concerns=[base_concern],
        escalation=tier2_escalation,
        severity_score=0.6,
        citations=["UDHR Article 1"],
        uncertainty=None,
    )


@pytest.fixture
def critic_eval_tier3(base_concern, tier3_escalation):
    return CriticEvaluation(
        critic_id="privacy",
        charter_version="v1",
        concerns=[base_concern],
        escalation=tier3_escalation,
        severity_score=0.9,
        citations=["UDHR Article 12"],
        uncertainty=None,
    )


@pytest.fixture
def valid_tier2_human_action(tier2_escalation):
    return HumanAction(
        action_type=HumanActionType.HUMAN_ACK,
        actor_id="human-123",
        statement=(
            "I acknowledge the identified constitutional risks and accept responsibility for proceeding."
        ),
        linked_escalations=[tier2_escalation],
    )


@pytest.fixture
def valid_tier3_human_action(tier3_escalation):
    return HumanAction(
        action_type=HumanActionType.HUMAN_DETERMINATION,
        actor_id="human-456",
        statement=(
            "I affirmatively determine the appropriate course of action in light of the identified constitutional risks."
        ),
        linked_escalations=[tier3_escalation],
    )


@pytest.fixture
def aggregation_result_fixture(critic_eval_no_escalation):
    return resolve_escalation(
        critic_evaluations=[critic_eval_no_escalation],
        synthesis="noop",
    )


@pytest.fixture
def blocked_decision_fixture(critic_eval_tier2):
    aggregation = resolve_escalation(
        critic_evaluations=[critic_eval_tier2],
        synthesis="Consent issue",
    )
    return enforce_human_review(
        aggregation_result=aggregation,
        human_action=None,
    )
