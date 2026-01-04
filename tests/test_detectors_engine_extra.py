import asyncio
import importlib

import pytest

from engine.detectors.engine import DetectorEngineV8
from engine.detectors.signals import DetectorSignal, SeverityLevel


class DummyDetector:
    name = "dummy"

    async def detect(self, text, context):
        return DetectorSignal(detector_name="dummy", severity=0.2, violations=["v"])


class ErrorDetector:
    name = "boom"

    async def detect(self, text, context):
        raise RuntimeError("boom")


class SlowDetector:
    name = "slow"

    async def detect(self, text, context):
        await asyncio.sleep(0.05)
        return DetectorSignal(detector_name="slow", severity=0.1)


@pytest.mark.asyncio
async def test_detector_engine_runs_and_aggregates():
    engine = DetectorEngineV8(detectors={"dummy": DummyDetector()}, timeout_seconds=1.0)
    signals = await engine.detect_all("text", {})
    assert signals["dummy"].violation is True

    summary = engine.aggregate_signals(signals)
    assert summary["total_violations"] == 1
    assert "low" in summary["by_severity"]


@pytest.mark.asyncio
async def test_detector_engine_error_and_timeout():
    engine = DetectorEngineV8(detectors={"boom": ErrorDetector()}, timeout_seconds=1.0)
    signals = await engine.detect_all("text", {})
    assert signals["boom"].evidence["error"] == "boom"

    engine = DetectorEngineV8(detectors={"slow": SlowDetector()}, timeout_seconds=0.001)
    signals = await engine.detect_all("text", {})
    assert "TIMEOUT" in signals["slow"].flags


def test_detector_engine_load_failure(monkeypatch):
    original_import = importlib.import_module

    def _fake_import(name):
        raise ImportError("boom")

    monkeypatch.setattr(importlib, "import_module", _fake_import)
    engine = DetectorEngineV8(detectors=None)
    assert engine.detectors == {}

    monkeypatch.setattr(importlib, "import_module", original_import)


def test_detector_signal_properties():
    signal = DetectorSignal(detector_name="d1", severity=0.5, violations=[])
    assert isinstance(signal.severity, SeverityLevel)
    assert signal.severity_label == "S2"
    assert signal.violation is False

    signal = DetectorSignal(
        detector_name="d1",
        severity=0.8,
        violations=[],
        evidence={"violation": True, "mitigation": "mask"},
    )
    assert signal.violation is True
    assert signal.mitigation == "mask"
