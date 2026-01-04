from engine.detectors.signals import DetectorSignal, SeverityLevel, _severity_label


def test_severity_label_boundaries():
    assert _severity_label(0.0) == "S0"
    assert _severity_label(0.32) == "S1"
    assert _severity_label(0.33) == "S2"
    assert _severity_label(0.66) == "S3"


def test_severity_level_comparisons_and_strings():
    level = SeverityLevel(0.5)
    assert level > "S1"
    assert level < "S3"
    assert (level > "bogus") is False
    assert SeverityLevel(0.0) == "S0"

    other = SeverityLevel(0.2)
    assert level > other

    class BadFloat:
        def __float__(self):
            raise TypeError("boom")

    assert (level < BadFloat()) is False


def test_detector_signal_coercion_and_label():
    signal = DetectorSignal(detector_name="det", severity="bad", violations=[])
    assert signal.severity_label == "S0"

    signal = DetectorSignal(detector_name="det", severity="0.7", violations=[])
    assert signal.severity_label == "S3"

    signal = DetectorSignal(detector_name="det", severity=0.2, violations=[])
    assert isinstance(signal.severity, SeverityLevel)
