"""Pytest configuration and shared fixtures."""

import pytest
import asyncio
from unittest.mock import AsyncMock

from engine.engine import EleanorEngineV8, EngineConfig
from engine.factory import EngineDependencies
from engine.mocks import (
    MockRouter,
    MockCritic,
    MockDetectorEngine,
    MockEvidenceRecorder,
    MockAggregator,
    MockReviewTriggerEvaluator,
)
from engine.aggregator.escalation import resolve_escalation
from engine.execution.human_review import (
    enforce_human_review,
    TIER_2_ACK_STATEMENT,
    TIER_3_DETERMINATION_STATEMENT,
)
from engine.schemas.escalation import (
    CriticEvaluation,
    EscalationSignal,
    EscalationTier,
    HumanAction,
    HumanActionType,
)


@pytest.fixture
def mock_llm_adapter():
    """Mock LLM adapter for testing."""
    adapter = AsyncMock()
    adapter.generate = AsyncMock(return_value="Mock LLM response")
    return adapter


@pytest.fixture
def sample_context():
    """Sample context for testing."""
    return {
        "domain": "test",
        "user_id": "test_user",
        "priority": "medium",
    }


@pytest.fixture
def sample_critic_result():
    """Sample critic evaluation result."""
    return {
        "critic": "rights",
        "score": 0.3,
        "severity": "LOW",
        "violations": [
            {
                "rule_id": "privacy_001",
                "description": "Potential privacy concern",
                "confidence": 0.7,
            }
        ],
        "justification": "Test justification",
        "evaluated_rules": ["privacy_001", "privacy_002"],
    }


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def engine():
    """Engine configured with lightweight mocks for fast tests."""
    deps = EngineDependencies(
        router=MockRouter(model_name="fake-model", response_text="mock response"),
        detector_engine=MockDetectorEngine(),
        evidence_recorder=MockEvidenceRecorder(),
        critics={"mock": MockCritic},
        precedent_engine=None,
        precedent_retriever=None,
        uncertainty_engine=None,
        aggregator=MockAggregator(),
        review_trigger_evaluator=MockReviewTriggerEvaluator(),
        critic_models=None,
    )

    return EleanorEngineV8(
        config=EngineConfig(enable_precedent_analysis=False, enable_reflection=False),
        dependencies=deps,
    )


@pytest.fixture
def critic_eval_no_escalation():
    return CriticEvaluation(
        critic_id="rights",
        charter_version="v1",
        severity_score=0.1,
    )


@pytest.fixture
def critic_eval_tier2():
    signal = EscalationSignal.for_tier(
        tier=EscalationTier.TIER_2,
        critic_id="rights",
        clause_id="R1",
        clause_description="Consent required",
        doctrine_ref="UDHR",
        rationale="Consent missing",
    )
    return CriticEvaluation(
        critic_id="rights",
        charter_version="v1",
        severity_score=0.7,
        escalation=signal,
    )


@pytest.fixture
def critic_eval_tier3():
    signal = EscalationSignal.for_tier(
        tier=EscalationTier.TIER_3,
        critic_id="privacy",
        clause_id="P9",
        clause_description="Material privacy violation",
        doctrine_ref="UDHR",
        rationale="Sensitive data exposure",
    )
    return CriticEvaluation(
        critic_id="privacy",
        charter_version="v1",
        severity_score=0.9,
        escalation=signal,
    )


@pytest.fixture
def valid_tier2_human_action(critic_eval_tier2):
    return HumanAction(
        action_type=HumanActionType.HUMAN_ACK,
        actor_id="reviewer-1",
        statement=TIER_2_ACK_STATEMENT,
        linked_escalations=[critic_eval_tier2.escalation],
    )


@pytest.fixture
def valid_tier3_human_action(critic_eval_tier3):
    return HumanAction(
        action_type=HumanActionType.HUMAN_DETERMINATION,
        actor_id="reviewer-2",
        statement=TIER_3_DETERMINATION_STATEMENT,
        linked_escalations=[critic_eval_tier3.escalation],
    )


@pytest.fixture
def aggregation_result_fixture(critic_eval_no_escalation):
    return resolve_escalation(
        critic_evaluations=[critic_eval_no_escalation],
        synthesis="All good",
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


try:
    import pytest_benchmark  # noqa: F401
    _HAS_PYTEST_BENCHMARK = True
except Exception:
    _HAS_PYTEST_BENCHMARK = False


# Provide a lightweight benchmark fixture when pytest-benchmark is unavailable
if not _HAS_PYTEST_BENCHMARK:

    @pytest.fixture
    def benchmark():
        def _run(func, *args, **kwargs):
            return func(*args, **kwargs)

        return _run
