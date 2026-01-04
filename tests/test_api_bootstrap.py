import json
from types import SimpleNamespace

import pytest

from api import bootstrap


class DummyEngine:
    def __init__(self):
        self.critics = {"rights": None, "risk": None}
        self.critic_models = {}
        self.router = SimpleNamespace(adapters={"ollama": object(), "other": object()})


def test_load_constitutional_config(tmp_path):
    config = {"a": 1}
    path = tmp_path / "config.yaml"
    path.write_text("a: 1\n", encoding="utf-8")
    assert bootstrap.load_constitutional_config(str(path)) == config

    with pytest.raises(FileNotFoundError):
        bootstrap.load_constitutional_config(str(tmp_path / "missing.yaml"))

    with pytest.raises(ValueError):
        bootstrap.load_constitutional_config(str(tmp_path))


def test_parse_critic_bindings_json_and_fallback(caplog):
    data = bootstrap._parse_critic_bindings(json.dumps({"rights": "adapter"}))
    assert data == {"rights": "adapter"}

    with caplog.at_level("WARNING"):
        data = bootstrap._parse_critic_bindings("rights=adapter,risk=adapter2")
        assert data["rights"] == "adapter"
        assert data["risk"] == "adapter2"


def test_bind_critic_models_explicit(monkeypatch):
    engine = DummyEngine()
    engine.router.adapters["adapter1"] = object()
    bootstrap.bind_critic_models(engine, {"rights": "adapter1", "risk": "missing"})

    assert engine.critic_models["rights"] is engine.router.adapters["adapter1"]
    assert "risk" not in engine.critic_models


def test_bind_critic_models_default_adapter(monkeypatch):
    engine = DummyEngine()
    engine.router.adapters["adapter1"] = object()
    monkeypatch.setenv("CRITIC_DEFAULT_ADAPTER", "adapter1")
    bootstrap.bind_critic_models(engine)
    assert engine.critic_models["rights"] is engine.router.adapters["adapter1"]


def test_build_engine_uses_builder(monkeypatch):
    dummy_engine = DummyEngine()
    monkeypatch.setattr(bootstrap, "build_eleanor_engine_v8", lambda **_kwargs: dummy_engine)
    monkeypatch.setattr(bootstrap, "load_constitutional_config", lambda *_args: {"ok": True})
    monkeypatch.setenv("ELEANOR_SEED", "not-an-int")

    engine = bootstrap.build_engine()
    assert engine is dummy_engine


@pytest.mark.asyncio
async def test_evaluate_opa_fallbacks(monkeypatch):
    monkeypatch.setenv("OPA_FAIL_STRATEGY", "allow")
    result = await bootstrap.evaluate_opa(None, {"input": 1})
    assert result["allow"] is True

    monkeypatch.setenv("OPA_FAIL_STRATEGY", "deny")
    result = await bootstrap.evaluate_opa(None, {"input": 1})
    assert result["allow"] is False
    assert result["escalate"] is False

    async def async_callback(_payload):
        return {"allow": True}

    result = await bootstrap.evaluate_opa(async_callback, {"input": 1})
    assert result["allow"] is True

    def sync_callback(_payload):
        return None

    result = await bootstrap.evaluate_opa(sync_callback, {"input": 1})
    assert result["allow"] is False

    def error_callback(_payload):
        raise RuntimeError("boom")

    monkeypatch.setenv("OPA_FAIL_STRATEGY", "escalate")
    result = await bootstrap.evaluate_opa(error_callback, {"input": 1})
    assert result["escalate"] is True
