import pytest

@pytest.mark.asyncio
async def test_uncertainty_engine_optional(engine):
    critic_results = await engine._run_critics_parallel("text", {}, "trace")
    aggregated = await engine._aggregate_results(critic_results, "text")
    out = await engine._run_uncertainty_engine(aggregated, critic_results)
    assert out is None or isinstance(out, dict)
