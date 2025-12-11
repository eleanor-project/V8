import pytest

@pytest.mark.asyncio
async def test_evidence_recorder_records(engine):
    await engine.recorder.record(
        critic="test",
        rule_id="rule1",
        severity="low",
        violation_description="test violation",
        confidence=0.8,
    )
    assert len(engine.recorder.buffer) >= 1
