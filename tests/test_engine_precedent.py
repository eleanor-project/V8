import pytest


@pytest.mark.asyncio
async def test_precedent_alignment_optional(engine):
    results = await engine._run_critics_parallel("text", {}, "trace")
    out = await engine._run_precedent_alignment(results, "trace")
    assert out is None or isinstance(out, dict)
