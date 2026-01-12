import asyncio
import types

import pytest
import json
import yaml

from api.rest.services import config_proposals as proposals
from engine.config import settings as settings_module
from engine.config.manager import ConfigManager
from engine.security import ledger as ledger_module


def _reset_state():
    settings_module._settings = None
    ConfigManager._instance = None
    ledger_module._writer = None
    ledger_module._reader = None


def _setup_env(monkeypatch, tmp_path, *, config_path=None, overlay_path=None):
    _reset_state()
    monkeypatch.setenv("ELEANOR_LEDGER_BACKEND", "stone_tablet_ledger")
    monkeypatch.setenv("ELEANOR_LEDGER_PATH", str(tmp_path / "ledger.jsonl"))
    monkeypatch.delenv("ELEANOR_OPA_POLICY_BUNDLE_HASH", raising=False)
    monkeypatch.delenv("ELEANOR_PRECEDENT_INDEX_HASH", raising=False)
    monkeypatch.delenv("ELEANOR_ENABLE_REFLECTION", raising=False)

    if config_path is not None:
        monkeypatch.setenv("ELEANOR_CONFIG_PATH", str(config_path))
    else:
        monkeypatch.delenv("ELEANOR_CONFIG_PATH", raising=False)

    if overlay_path is not None:
        monkeypatch.setenv("ELEANOR_CONFIG_OVERLAY_PATH", str(overlay_path))
    else:
        monkeypatch.delenv("ELEANOR_CONFIG_OVERLAY_PATH", raising=False)


def _make_engine():
    router = types.SimpleNamespace(policy={"primary": "default"})
    critic_model = types.SimpleNamespace(model="model-a")
    return types.SimpleNamespace(router=router, critic_models={"critic": critic_model})


def _submit_proposal(changes):
    return proposals.submit_proposal(
        proposal={"proposal_type": "config_change", "title": "t", "changes": changes},
        actor="user",
    )


def test_preview_stores_ledger_event(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}", encoding="utf-8")
    _setup_env(monkeypatch, tmp_path, config_path=config_path, overlay_path=tmp_path / "overlay.yaml")

    submission = _submit_proposal({"enable_reflection": False})
    engine = _make_engine()
    artifact = proposals.build_preview_artifact(
        proposal_id=submission["proposal_id"],
        proposal={"changes": {"enable_reflection": False}},
        mode="policy_only",
        window={"type": "time", "duration": "4h"},
        limits={},
        engine=engine,
    )
    proposals.store_preview_artifact(
        proposal_id=submission["proposal_id"], artifact=artifact, actor="user"
    )

    reader = ledger_module.get_ledger_reader(path=str(tmp_path / "ledger.jsonl"))
    preview_records = [r for r in reader.read_all() if r.event == "config_proposal_previewed"]
    assert preview_records
    payload = preview_records[-1].payload
    assert payload["artifact_hash"] == artifact["artifact_hash"]
    assert proposals._artifact_hash(artifact) == artifact["artifact_hash"]


def test_apply_requires_preview(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}", encoding="utf-8")
    _setup_env(monkeypatch, tmp_path, config_path=config_path, overlay_path=tmp_path / "overlay.yaml")

    submission = _submit_proposal({"enable_reflection": False})
    engine = _make_engine()

    with pytest.raises(proposals.PreviewValidationError) as excinfo:
        proposals.apply_proposal(
            proposal_id=submission["proposal_id"],
            artifact_hash="sha256:deadbeef",
            engine=engine,
            actor="admin",
        )
    assert excinfo.value.code == "PREVIEW_REQUIRED"
    assert excinfo.value.status_code == 428


def test_apply_rejects_wrong_hash(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}", encoding="utf-8")
    _setup_env(monkeypatch, tmp_path, config_path=config_path, overlay_path=tmp_path / "overlay.yaml")

    submission = _submit_proposal({"enable_reflection": False})
    engine = _make_engine()
    artifact = proposals.build_preview_artifact(
        proposal_id=submission["proposal_id"],
        proposal={"changes": {"enable_reflection": False}},
        mode="policy_only",
        window={"type": "time", "duration": "4h"},
        limits={},
        engine=engine,
    )
    proposals.store_preview_artifact(
        proposal_id=submission["proposal_id"], artifact=artifact, actor="user"
    )

    with pytest.raises(proposals.PreviewValidationError) as excinfo:
        proposals.apply_proposal(
            proposal_id=submission["proposal_id"],
            artifact_hash="sha256:bad",
            engine=engine,
            actor="admin",
        )
    assert excinfo.value.code == "INVALID_PREVIEW_ARTIFACT"
    assert excinfo.value.status_code == 400


def test_apply_detects_baseline_drift(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}", encoding="utf-8")
    _setup_env(monkeypatch, tmp_path, config_path=config_path, overlay_path=tmp_path / "overlay.yaml")

    submission = _submit_proposal({"enable_reflection": False})
    engine = _make_engine()
    artifact = proposals.build_preview_artifact(
        proposal_id=submission["proposal_id"],
        proposal={"changes": {"enable_reflection": False}},
        mode="policy_only",
        window={"type": "time", "duration": "4h"},
        limits={},
        engine=engine,
    )
    proposals.store_preview_artifact(
        proposal_id=submission["proposal_id"], artifact=artifact, actor="user"
    )

    monkeypatch.setenv("ELEANOR_ENABLE_REFLECTION", "false")
    _reset_state()
    ConfigManager()

    with pytest.raises(proposals.PreviewValidationError) as excinfo:
        proposals.apply_proposal(
            proposal_id=submission["proposal_id"],
            artifact_hash=artifact["artifact_hash"],
            engine=engine,
            actor="admin",
        )
    assert excinfo.value.code == "BASELINE_DIVERGED"
    assert excinfo.value.status_code == 412


def test_apply_detects_continuity_drift(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}", encoding="utf-8")
    _setup_env(monkeypatch, tmp_path, config_path=config_path, overlay_path=tmp_path / "overlay.yaml")

    monkeypatch.setenv("ELEANOR_OPA_POLICY_BUNDLE_HASH", "sha256:old")
    submission = _submit_proposal({"enable_reflection": False})
    engine = _make_engine()
    artifact = proposals.build_preview_artifact(
        proposal_id=submission["proposal_id"],
        proposal={"changes": {"enable_reflection": False}},
        mode="policy_only",
        window={"type": "time", "duration": "4h"},
        limits={},
        engine=engine,
    )
    proposals.store_preview_artifact(
        proposal_id=submission["proposal_id"], artifact=artifact, actor="user"
    )

    monkeypatch.setenv("ELEANOR_OPA_POLICY_BUNDLE_HASH", "sha256:new")

    with pytest.raises(proposals.PreviewValidationError) as excinfo:
        proposals.apply_proposal(
            proposal_id=submission["proposal_id"],
            artifact_hash=artifact["artifact_hash"],
            engine=engine,
            actor="admin",
        )
    assert excinfo.value.code == "PREVIEW_REQUIRED"
    assert excinfo.value.status_code == 428


def test_apply_persists_overlay_and_is_idempotent(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}", encoding="utf-8")
    _setup_env(monkeypatch, tmp_path, config_path=config_path, overlay_path=None)

    submission = _submit_proposal({"enable_reflection": False})
    engine = _make_engine()
    artifact = proposals.build_preview_artifact(
        proposal_id=submission["proposal_id"],
        proposal={"changes": {"enable_reflection": False}},
        mode="policy_only",
        window={"type": "time", "duration": "4h"},
        limits={},
        engine=engine,
    )
    proposals.store_preview_artifact(
        proposal_id=submission["proposal_id"], artifact=artifact, actor="user"
    )

    result = proposals.apply_proposal(
        proposal_id=submission["proposal_id"],
        artifact_hash=artifact["artifact_hash"],
        engine=engine,
        actor="admin",
    )
    assert result["status"] == "applied"

    overlay_path = config_path.with_suffix(".overlay.yaml")
    assert overlay_path.exists()
    overlay_payload = yaml.safe_load(overlay_path.read_text(encoding="utf-8"))
    assert overlay_payload == {"enable_reflection": False}

    repeat = proposals.apply_proposal(
        proposal_id=submission["proposal_id"],
        artifact_hash=artifact["artifact_hash"],
        engine=engine,
        actor="admin",
    )
    assert repeat["status"] == "already_applied"

    reader = ledger_module.get_ledger_reader(path=str(tmp_path / "ledger.jsonl"))
    applied_records = [r for r in reader.read_all() if r.event == "config_proposal_applied"]
    assert len(applied_records) == 1


def test_full_replay_preview_counts_changes(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}", encoding="utf-8")
    replay_path = tmp_path / "replay.jsonl"
    monkeypatch.setenv("ELEANOR_REPLAY_LOG_PATH", str(replay_path))
    _setup_env(monkeypatch, tmp_path, config_path=config_path, overlay_path=tmp_path / "overlay.yaml")

    entries = [
        {"trace_id": "t1", "input": "good", "context": {}, "timestamp": 1},
        {"trace_id": "t2", "input": "bad", "context": {}, "timestamp": 2},
    ]
    with replay_path.open("w", encoding="utf-8") as handle:
        for record in entries:
            handle.write(json.dumps(record) + "\n")

    submission = _submit_proposal({"enable_reflection": False})

    class StubEngine:
        async def run(self, input_text, context=None, trace_id=None, detail_level=3):
            decision = "allow" if "good" in input_text else "deny"
            return {"trace_id": trace_id, "aggregator_output": {"decision": decision}}

    class StubCandidateEngine(StubEngine):
        async def run(self, input_text, context=None, trace_id=None, detail_level=3):
            decision = "deny" if "good" in input_text else "deny"
            return {"trace_id": trace_id, "aggregator_output": {"decision": decision}}

    monkeypatch.setattr(proposals, "build_preview_engine", lambda **_: StubCandidateEngine())

    artifact = asyncio.run(
        proposals.run_full_replay_preview(
            proposal_id=submission["proposal_id"],
            proposal={"changes": {"enable_reflection": False}},
            window={"type": "count", "limit": 2},
            limits={"max_changed_traces": 5},
            engine=StubEngine(),
        )
    )

    assert artifact["metrics"]["trace_count"] == 2
    assert artifact["metrics"]["changed_trace_count"] == 1
    assert artifact["top_changed_traces"][0]["trace_id"] == "t1"
