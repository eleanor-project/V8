import pytest


@pytest.mark.asyncio
async def test_detail_levels(engine):
    for level in [1, 2, 3]:
        result = await engine.run("hello", detail_level=level)
        assert result.trace_id is not None
        assert result.output_text is not None
        if level == 3:
            assert result.forensic is not None
