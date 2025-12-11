import pytest

@pytest.mark.asyncio
async def test_detector_engine_present(engine):
    assert engine.detector_engine is not None
    assert hasattr(engine.detector_engine, "detectors")
