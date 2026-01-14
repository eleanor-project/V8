import asyncio
import os

import pytest

from api.bootstrap import evaluate_opa


@pytest.mark.asyncio
async def test_evaluate_opa_calls_callback(monkeypatch):
    async def fake_callback(payload):
        assert payload["trace_id"] == "test-trace"
        return {"allow": True, "escalate": False}

    result = await evaluate_opa(fake_callback, {"trace_id": "test-trace"})
    assert result["allow"] is True
    assert result["escalate"] is False


@pytest.mark.asyncio
async def test_evaluate_opa_fallback_escalate(monkeypatch):
    monkeypatch.setenv("OPA_FAIL_STRATEGY", "escalate")
    result = await evaluate_opa(None, {"trace_id": "test-fallback"})
    assert result["allow"] is False
    assert result["escalate"] is True


@pytest.mark.asyncio
async def test_evaluate_opa_fallback_deny(monkeypatch):
    monkeypatch.setenv("OPA_FAIL_STRATEGY", "deny")
    result = await evaluate_opa(None, {"trace_id": "test-fallback"})
    assert result["allow"] is False
    assert result["escalate"] is False


@pytest.mark.asyncio
async def test_evaluate_opa_fallback_allow(monkeypatch):
    monkeypatch.setenv("OPA_FAIL_STRATEGY", "allow")
    result = await evaluate_opa(None, {"trace_id": "test-fallback"})
    assert result["allow"] is True
    assert result["escalate"] is False
