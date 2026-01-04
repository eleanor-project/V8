import asyncio
from types import SimpleNamespace

import pytest

import engine.evidence_recorder_async as async_module


class DummyAsyncFile:
    def __init__(self):
        self.closed = False
        self.writes = []

    async def write(self, data):
        self.writes.append(data)

    async def flush(self):
        return None

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_async_recorder_async_io(tmp_path, monkeypatch):
    async def _open(*_args, **_kwargs):
        return DummyAsyncFile()

    monkeypatch.setattr(async_module, "AIOFILES_AVAILABLE", True)
    monkeypatch.setattr(async_module, "aiofiles", SimpleNamespace(open=_open), raising=False)

    recorder = async_module.AsyncEvidenceRecorder(str(tmp_path / "evidence.jsonl"), buffer_size=1)
    await recorder.initialize()
    await recorder.record(test="value")
    await recorder.flush()
    await recorder.close()


def test_async_recorder_sync_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(async_module, "AIOFILES_AVAILABLE", False)
    recorder = async_module.AsyncEvidenceRecorder(str(tmp_path / "evidence.jsonl"), buffer_size=1)

    async def _run():
        await recorder.initialize()
        await recorder.record(test="value")
        await recorder.close()

    asyncio.run(_run())


def test_periodic_flush_error(tmp_path, monkeypatch):
    monkeypatch.setattr(async_module, "AIOFILES_AVAILABLE", False)
    recorder = async_module.AsyncEvidenceRecorder(
        str(tmp_path / "evidence.jsonl"), buffer_size=1, flush_interval=0.01
    )
    recorder.buffer = [{"k": "v"}]

    async def _flush_fail():
        raise RuntimeError("boom")

    recorder.flush = _flush_fail

    async def _run():
        task = asyncio.create_task(recorder._periodic_flush())
        await asyncio.sleep(0.02)
        recorder._shutdown = True
        await task

    asyncio.run(_run())


def test_close_flush_error(tmp_path, monkeypatch):
    monkeypatch.setattr(async_module, "AIOFILES_AVAILABLE", False)
    recorder = async_module.AsyncEvidenceRecorder(str(tmp_path / "evidence.jsonl"))

    async def _run():
        await recorder.initialize()

        async def _fail_flush():
            raise RuntimeError("boom")

        recorder.flush = _fail_flush

        class DummyFile:
            def __init__(self):
                self.closed = False

            def close(self):
                raise RuntimeError("close boom")

        recorder._file_handle = DummyFile()
        await recorder.close()

    asyncio.run(_run())
