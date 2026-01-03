import pytest


@pytest.mark.asyncio
async def test_streaming_api(engine):
    events = []
    async for event in engine.run_stream("hello"):
        events.append(event)
    assert any(e["event"] == "router_selected" for e in events)
    assert any(e["event"] == "final_output" for e in events)
