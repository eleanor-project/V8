def test_engine_initializes(engine):
    assert engine is not None
    assert engine.router is not None
    assert engine.critics is not None
    assert engine.detector_engine is not None
    assert engine.recorder is not None
