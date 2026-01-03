import pytest


@pytest.mark.asyncio
async def test_critics_execute(engine):
    model_response = "Sample output"
    context = {}
    trace_id = "test-trace"

    results = await engine._run_critics_parallel(model_response, context, trace_id)
    assert isinstance(results, dict)
    assert set(results.keys()) == set(engine.critics.keys())

    for k, v in results.items():
        assert "violations" in v
        assert "duration_ms" in v
