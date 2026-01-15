import asyncio
import os

import pytest

from engine.integrations import traffic_light_governance as tl


class DummyConstraints:
    def __init__(self):
        self.route = "traffic-light-route"
        self.outcome = "traffic-light-outcome"
        self.human_review = {"required": False}


class DummyRouterDecision:
    def __init__(self):
        self.reason = "test reason"


class DummyGovernorResult:
    def __init__(self):
        self.constraints = DummyConstraints()
        self.router_decision = DummyRouterDecision()
        self.coverage_score = 0.75


@pytest.mark.asyncio
async def test_traffic_light_hook_apply(monkeypatch, tmp_path):
    router_config = tmp_path / "router_config.yaml"
    router_config.write_text("dummy config")

    called = {"evaluate": False, "append": False}

    def fake_evaluate(*args, **kwargs):
        called["evaluate"] = True
        return DummyGovernorResult()

    def fake_make_governance_event(*args, **kwargs):
        return {"event_id": "evt-123", "applied_precedents": []}

    def fake_append_jsonl(path, event, swallow_errors=True):
        called["append"] = True
        return None

    monkeypatch.setattr(tl, "evaluate", fake_evaluate)
    monkeypatch.setattr(tl, "make_governance_event", fake_make_governance_event)
    monkeypatch.setattr(tl, "append_jsonl_safely", fake_append_jsonl)

    hook = tl.TrafficLightGovernanceHook(
        enabled=True,
        router_config_path=str(router_config),
        events_jsonl_path=str(tmp_path / "events.jsonl"),
        mode="observe",
    )

    result = await hook.apply(
        trace_id="trace-test",
        text="test input",
        context={"domains": ["general"], "jurisdiction": "global"},
        aggregated={"decision": "deny"},
        precedent_data={"cases": []},
        uncertainty_data={"overall_uncertainty": 0.2},
    )

    assert called["evaluate"] is True
    assert called["append"] is True
    assert result["route"] == "traffic-light-route"
    assert result["outcome"] == "traffic-light-outcome"
    assert result["reason"] == "test reason"
    assert result["event_id"] == "evt-123"


@pytest.mark.asyncio
async def test_hook_disabled_without_router(monkeypatch, tmp_path):
    router_config = tmp_path / "missing.yaml"
    hook = tl.TrafficLightGovernanceHook(
        enabled=True,
        router_config_path=str(router_config),
        events_jsonl_path=None,
        mode="observe",
    )

    res = await hook.apply(
        trace_id="trace-no-config",
        text="input",
        context={},
        aggregated={},
        precedent_data=None,
        uncertainty_data=None,
    )

    assert res is None
