import pytest

from engine.aggregator.aggregator import AggregatorV8
from engine.aggregator import escalation as escalation_module


def test_normalize_and_lexicographic():
    agg = AggregatorV8()
    critics = {"rights": {"severity": "2.2", "violations": ["v"], "justification": "x"}}
    normalized = agg._normalize_critics(critics)
    assert normalized["rights"]["severity"] == pytest.approx(2.2)

    lex = agg._compute_lexicographic(normalized)
    assert lex["highest_priority_violation"]["critic"] == "rights"


def test_precedent_and_uncertainty_adjustments():
    agg = AggregatorV8()
    critics = {"rights": {"severity": 1.0, "violations": [], "justification": ""}}
    adjusted = agg._apply_precedent(critics, {"alignment_score": -1.0})
    assert adjusted["rights"]["precedent_adjusted_severity"] == pytest.approx(2.0)

    adjusted = agg._apply_uncertainty(adjusted, {"overall_uncertainty": 0.6})
    assert adjusted["rights"]["final_severity"] > 2.0


def test_decision_logic_variants():
    agg = AggregatorV8()
    assert (
        agg._decision_logic(
            {"average_severity": 0.2},
            {"highest_priority_violation": None},
            {"overall_uncertainty": 0.9},
        )
        == "allow"
    )
    assert (
        agg._decision_logic(
            {"average_severity": 1.2},
            {"highest_priority_violation": None},
            {"overall_uncertainty": 0.7},
        )
        == "escalate"
    )
    assert (
        agg._decision_logic(
            {"average_severity": 1.2},
            {"highest_priority_violation": None},
            {"overall_uncertainty": 0.1},
        )
        == "constrained_allow"
    )
    assert (
        agg._decision_logic(
            {"average_severity": 1.2},
            {"highest_priority_violation": {"severity": 2.6}},
            {"overall_uncertainty": 0.1},
        )
        == "deny"
    )


def test_aggregate_handles_bad_scores_and_empty():
    agg = AggregatorV8()
    critics = {
        "rights": {"severity": "bad", "violations": [], "justification": "x"},
        "risk": {"severity": None, "score": "bad", "violations": [], "justification": "y"},
    }
    result = agg.aggregate(
        critics=critics,
        precedent={"alignment_score": 0.0},
        uncertainty={"overall_uncertainty": 0.0},
        model_output="out",
    )
    assert result["decision"] in {"allow", "constrained_allow"}

    score = agg._final_score({}, {})
    assert score["average_severity"] == 0.0


def test_aggregate_handles_invalid_severity_score(monkeypatch):
    agg = AggregatorV8()

    def _fake_uncertainty(_critics, _uncertainty):
        return {
            "crit": {
                "severity": "bad",
                "score": "bad",
                "violations": [],
                "justification": "x",
                "final_severity": 0.0,
            }
        }

    monkeypatch.setattr(agg, "_apply_uncertainty", _fake_uncertainty)
    result = agg.aggregate(
        critics={"crit": {"severity": 0.1, "violations": [], "justification": "ok"}},
        precedent={},
        uncertainty={},
        model_output="out",
    )
    severity_score = result["aggregation_result"]["critic_evaluations"][0]["severity_score"]
    assert severity_score == 0.0


def test_aggregate_handles_missing_severity_bad_score(monkeypatch):
    agg = AggregatorV8()

    def _fake_uncertainty(_critics, _uncertainty):
        return {
            "crit": {
                "severity": None,
                "score": "bad",
                "violations": [],
                "justification": "x",
                "final_severity": 0.0,
            }
        }

    monkeypatch.setattr(agg, "_apply_uncertainty", _fake_uncertainty)
    result = agg.aggregate(
        critics={"crit": {"severity": 0.1, "violations": [], "justification": "ok"}},
        precedent={},
        uncertainty={},
        model_output="out",
    )
    severity_score = result["aggregation_result"]["critic_evaluations"][0]["severity_score"]
    assert severity_score == 0.0


def test_escalation_unhandled_tier():
    with pytest.raises(RuntimeError):
        escalation_module._build_execution_gate(highest_tier="bad", escalation_signals=[])
