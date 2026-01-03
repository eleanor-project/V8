import pytest


@pytest.mark.asyncio
async def test_aggregation_structure(engine):
    critic_results = await engine._run_critics_parallel("text", {}, "trace")
    aggregated = await engine._aggregate_results(critic_results, "text")
    assert isinstance(aggregated, dict)
    assert "final_output" in aggregated
