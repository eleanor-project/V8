import importlib
import os
import sys
import types

from engine.execution.human_review import TIER_2_ACK_STATEMENT
from engine.schemas.escalation import (
    AggregationResult,
    CriticEvaluation,
    EscalationSignal,
    EscalationSummary,
    ExecutionGate,
    EscalationTier,
    HumanAction,
    HumanActionType,
)


def _load_main():
    os.environ.setdefault("AUTH_ENABLED", "false")
    weaviate_stub = types.ModuleType("weaviate")
    weaviate_stub.Client = None
    sys.modules.setdefault("weaviate", weaviate_stub)

    fake_sql = types.ModuleType("psycopg2.sql")
    sys.modules.setdefault("psycopg2.sql", fake_sql)

    psycopg2_stub = types.ModuleType("psycopg2")
    psycopg2_stub.connect = None
    psycopg2_stub.Error = Exception
    psycopg2_stub.sql = fake_sql
    sys.modules.setdefault("psycopg2", psycopg2_stub)
    main = importlib.import_module("api.rest.main")
    return main


def _aggregation_payload():
    signal = EscalationSignal.for_tier(
        tier=EscalationTier.TIER_2,
        critic_id="rights",
        clause_id="R1",
        clause_description="Rights threshold exceeded",
        doctrine_ref="doc-1",
        rationale="Test escalation",
    )
    summary = EscalationSummary(
        highest_tier=EscalationTier.TIER_2,
        triggering_signals=[signal],
        critics_triggered=["rights"],
        explanation="Triggered for test",
    )
    gate = ExecutionGate(
        gated=True,
        required_action=HumanActionType.HUMAN_ACK,
        reason="test",
        escalation_tier=EscalationTier.TIER_2,
    )
    critic_eval = CriticEvaluation(
        critic_id="rights",
        charter_version="8.0",
        concerns=[],
        escalation=signal,
        severity_score=0.5,
        citations=[],
        uncertainty=None,
    )
    agg = AggregationResult(
        synthesis="test",
        critic_evaluations=[critic_eval],
        escalation_summary=summary,
        execution_gate=gate,
        dissent_present=False,
        audit_hash="hash",
    )
    return {"aggregation_result": agg.model_dump(mode="json")}, signal


def test_execution_gate_requires_human_action():
    main = _load_main()
    aggregated, _ = _aggregation_payload()

    execution = main.resolve_execution_decision(aggregated, None)
    assert execution is not None
    assert execution.executable is False
    assert main.apply_execution_gate("allow", execution) == "escalate"


def test_execution_gate_accepts_human_action():
    main = _load_main()
    aggregated, signal = _aggregation_payload()

    human_action = HumanAction(
        action_type=HumanActionType.HUMAN_ACK,
        actor_id="reviewer-1",
        statement=TIER_2_ACK_STATEMENT,
        linked_escalations=[signal],
    )
    execution = main.resolve_execution_decision(aggregated, human_action)
    assert execution is not None
    assert execution.executable is True
    assert main.apply_execution_gate("allow", execution) == "allow"
