import asyncio
import signal

import pytest

from engine.resource_manager import ResourceManager, ShutdownHandler, TimeoutProtection


@pytest.mark.asyncio
async def test_resource_manager_shutdown_and_cleanup():
    manager = ResourceManager()
    task = asyncio.create_task(asyncio.sleep(10))
    manager.track_task(task)

    waiter = asyncio.create_task(manager.wait_for_shutdown())
    manager.signal_shutdown()
    await waiter
    assert manager.is_shutdown() is True

    await manager.cleanup_tasks(timeout=0.1)
    assert task.done()


@pytest.mark.asyncio
async def test_timeout_protection_returns_none_on_timeout():
    result = await TimeoutProtection.with_timeout(
        asyncio.sleep(0.1), timeout=0.01, operation="slow", raise_on_timeout=False
    )
    assert result is None


@pytest.mark.asyncio
async def test_shutdown_handler_handles_callback(monkeypatch):
    called = []

    async def callback():
        called.append("done")

    handler = ShutdownHandler(callback)
    await handler._handle_shutdown(signal.SIGTERM)
    assert called == ["done"]

    await handler._handle_shutdown(signal.SIGTERM)
    assert called == ["done"]


def test_shutdown_handler_setup_and_restore(monkeypatch):
    handler = ShutdownHandler(lambda: None)

    loop = asyncio.new_event_loop()
    monkeypatch.setattr(asyncio, "get_event_loop", lambda: loop)
    monkeypatch.setattr(loop, "add_signal_handler", lambda *_a, **_k: None)
    handler.setup_handlers()

    handler._original_handlers[signal.SIGTERM] = lambda *_a, **_k: None
    handler._original_handlers[signal.SIGINT] = lambda *_a, **_k: None
    handler.restore_handlers()
    loop.close()
