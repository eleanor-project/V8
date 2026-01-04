import json

import pytest

from engine import evidence_recorder_async as recorder_module


@pytest.mark.asyncio
async def test_async_recorder_sync_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(recorder_module, "AIOFILES_AVAILABLE", False)
    path = tmp_path / "evidence.jsonl"
    recorder = recorder_module.AsyncEvidenceRecorder(str(path), buffer_size=1, flush_interval=0.1)

    await recorder.initialize()
    await recorder.record(event="one")
    await recorder.flush()
    await recorder.close()

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(lines[0])["event"] == "one"


@pytest.mark.asyncio
async def test_async_recorder_record_after_shutdown(tmp_path):
    path = tmp_path / "evidence.jsonl"
    recorder = recorder_module.AsyncEvidenceRecorder(str(path), buffer_size=1, flush_interval=0.1)
    await recorder.initialize()
    await recorder.close()

    await recorder.record(event="ignored")
    assert recorder.buffer == []


@pytest.mark.asyncio
async def test_async_recorder_flush_noop(tmp_path):
    path = tmp_path / "evidence.jsonl"
    recorder = recorder_module.AsyncEvidenceRecorder(str(path), buffer_size=1, flush_interval=0.1)
    await recorder.flush()
