import pytest


@pytest.mark.asyncio
async def test_model_routing(engine):
    out = await engine._select_model("hello", {})
    assert "model_info" in out
    assert "response_text" in out
    assert out["model_info"]["model_name"] == "fake-model"
