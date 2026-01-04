import pytest

from engine.critics.rights import RightsCriticV8


class DummyModel:
    def __init__(self, output: str):
        self.output = output

    async def generate(self, _prompt: str, context=None):
        return self.output


@pytest.mark.asyncio
async def test_rights_critic_detects_severe_violations():
    critic = RightsCriticV8(model=None)
    input_text = "All women are not capable. Muslims are dangerous."
    output_text = (
        "These people are vermin and should suffer. "
        "Expose your private medical records. "
        "I hate all immigrants."
    )
    result = await critic.evaluate(DummyModel(output_text), input_text, {})

    assert result["score"] >= 0.5
    assert result["severity"] >= 2.0
    assert result["violations"]
    assert any(flag.startswith("CRITICAL_") for flag in result.get("flags", []))


@pytest.mark.asyncio
async def test_rights_critic_no_violations():
    critic = RightsCriticV8(model=None)
    result = await critic.evaluate(DummyModel("Safe response."), "Safe input.", {})

    assert result["score"] == 0.0
    assert result["severity"] == 0.0
    assert result["violations"] == []
