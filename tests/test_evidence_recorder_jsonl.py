import json

import pytest

from engine.recorder.evidence_recorder import EvidenceRecorder


class DummyDBSink:
    def __init__(self):
        self.records = []

    async def write(self, record):
        self.records.append(record)


@pytest.mark.asyncio
async def test_evidence_recorder_flushes_jsonl(tmp_path):
    path = tmp_path / "evidence.jsonl"
    recorder = EvidenceRecorder(jsonl_path=str(path), buffer_size=2, flush_interval=0)
    await recorder.initialize()

    await recorder.record(
        critic="test",
        rule_id="rule-1",
        severity="INFO",
        violation_description="desc",
        confidence=0.5,
        trace_id="t1",
    )
    await recorder.record(
        critic="test",
        rule_id="rule-2",
        severity="INFO",
        violation_description="desc2",
        confidence=0.7,
        trace_id="t2",
    )

    await recorder.close()

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    payload = json.loads(lines[0])
    assert payload["trace_id"] == "t1"


@pytest.mark.asyncio
async def test_evidence_recorder_writes_db_sink(tmp_path):
    path = tmp_path / "evidence.jsonl"
    sink = DummyDBSink()
    recorder = EvidenceRecorder(jsonl_path=str(path), db_sink=sink, buffer_size=10, flush_interval=0)
    await recorder.initialize()

    record = await recorder.record(
        critic="db",
        rule_id="rule",
        severity="WARN",
        violation_description="desc",
        confidence=0.2,
    )

    assert sink.records == [record]
    await recorder.close()


@pytest.mark.asyncio
async def test_evidence_recorder_latest_and_flush(tmp_path, monkeypatch):
    monkeypatch.setenv("ELEANOR_EVIDENCE_BUFFER_SIZE", "not-an-int")
    recorder = EvidenceRecorder(jsonl_path=None, buffer_size=3, flush_interval=10)

    await recorder.record(
        critic="test",
        rule_id="rule",
        severity="INFO",
        violation_description="desc",
        confidence=0.1,
        trace_id="t1",
    )
    await recorder.record(
        critic="test",
        rule_id="rule",
        severity="INFO",
        violation_description="desc",
        confidence=0.2,
        trace_id="t2",
    )

    latest = recorder.latest(1)
    assert latest[0].trace_id == "t2"
    await recorder.flush_to_jsonl()
