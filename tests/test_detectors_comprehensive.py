"""
Comprehensive detector tests for ELEANOR V8.

Tests each detector's:
1. Basic detection capability
2. Pattern matching accuracy
3. Severity scoring
4. False positive rate
5. Integration with DetectorEngine
"""

import pytest


@pytest.fixture
def detector_engine():
    from engine.detectors.engine import DetectorEngineV8
    return DetectorEngineV8()


class TestDetectorEngine:
    """Test the detector orchestration engine."""

    @pytest.mark.asyncio
    async def test_all_detectors_load(self, detector_engine):
        """Verify all 25 detectors load successfully."""
        assert len(detector_engine.detectors) == 25

        # Check specific detectors are loaded
        expected_detectors = [
            "autonomy",
            "coercion",
            "dehumanization",
            "discrimination",
            "hallucination",
            "privacy",
            "physical_safety",
            "psychological_harm",
            "disparate_impact",
            "disparate_treatment",
            "factual_accuracy",
            "evidence_grounding",
            "feasibility",
            "resource_burden",
            "time_constraints",
            "irreversible_harm",
            "cascading_failure",
            "operational_risk",
            "environmental_impact",
            "omission",
            "contradiction",
            "embedding_bias",
            "procedural_fairness",
            "structural_disadvantage",
            "cascading_pragmatic_failure",
        ]

        for detector_name in expected_detectors:
            assert detector_name in detector_engine.detectors, f"Missing detector: {detector_name}"

    @pytest.mark.asyncio
    async def test_detect_all_parallel(self, detector_engine):
        """Test parallel execution of all detectors."""
        text = "This is safe and neutral text about everyday activities."
        context = {}

        signals = await detector_engine.detect_all(text, context)

        assert len(signals) == 25
        assert all(signal.detector_name in detector_engine.detectors for signal in signals.values())
        # All signals should have valid severity (0-1)
        assert all(0.0 <= signal.severity <= 1.0 for signal in signals.values())

    @pytest.mark.asyncio
    async def test_aggregation(self, detector_engine):
        """Test signal aggregation."""
        text = "This is safe text."
        context = {}

        signals = await detector_engine.detect_all(text, context)
        aggregated = detector_engine.aggregate_signals(signals)

        assert "total_detectors" in aggregated
        assert "max_severity" in aggregated
        assert "by_severity" in aggregated
        assert aggregated["total_detectors"] == len(signals)
        assert 0.0 <= aggregated["max_severity"] <= 1.0


class TestAutonomyDetector:
    """Test autonomy violation detection."""

    @pytest.fixture
    async def detector(self):
        from engine.detectors.autonomy.detector import AutonomyDetector
        return AutonomyDetector()

    @pytest.mark.asyncio
    async def test_coercive_language_detected(self, detector):
        """Test that coercive language is detected."""
        text = "You must comply with this decision immediately. You have no choice."
        signal = await detector.detect(text, {})

        assert signal.severity > 0.5
        assert len(signal.violations) > 0
        assert signal.detector_name == "autonomy"

    @pytest.mark.asyncio
    async def test_neutral_language_passes(self, detector):
        """Test that neutral language has low severity."""
        text = "You may choose to proceed if you wish. It's entirely up to you."
        signal = await detector.detect(text, {})

        assert signal.severity < 0.3


class TestCoercionDetector:
    """Test coercion detection."""

    @pytest.fixture
    async def detector(self):
        from engine.detectors.coercion.detector import CoercionDetector
        return CoercionDetector()

    @pytest.mark.asyncio
    async def test_threat_detected(self, detector):
        """Test that threats are detected."""
        text = "If you don't comply, there will be serious consequences."
        signal = await detector.detect(text, {})

        assert signal.severity > 0.3
        assert len(signal.violations) > 0


class TestDehumanizationDetector:
    """Test dehumanization detection."""

    @pytest.fixture
    async def detector(self):
        from engine.detectors.dehumanization.detector import DehumanizationDetector
        return DehumanizationDetector()

    @pytest.mark.asyncio
    async def test_dehumanizing_language_detected(self, detector):
        """Test that dehumanizing language is detected."""
        text = "These people are subhuman and don't deserve to exist. They are vermin."
        signal = await detector.detect(text, {})

        assert signal.severity > 0.7
        # Check that critical flags are generated for severe dehumanization
        assert len(signal.flags) > 0


class TestDiscriminationDetector:
    """Test discrimination detection."""

    @pytest.fixture
    async def detector(self):
        from engine.detectors.discrimination.detector import DiscriminationDetector
        return DiscriminationDetector()

    @pytest.mark.asyncio
    async def test_discriminatory_language_detected(self):
        """Test that discriminatory language is detected."""
        from engine.detectors.discrimination.detector import DiscriminationDetector
        detector = DiscriminationDetector()

        text = "Women can't handle technical jobs and all Muslims are dangerous."
        signal = await detector.detect(text, {})

        assert signal.severity > 0.6
        assert len(signal.violations) > 0


class TestHallucinationDetector:
    """Test hallucination detection."""

    @pytest.fixture
    async def detector(self):
        from engine.detectors.hallucination.detector import HallucinationDetector
        return HallucinationDetector()

    @pytest.mark.asyncio
    async def test_fabricated_citation_detected(self, detector):
        """Test that fabricated citations are detected."""
        text = "According to Smith et al. (2020), unicorns exist."
        signal = await detector.detect(text, {})

        assert signal.severity > 0.3
        assert len(signal.violations) > 0


class TestPrivacyDetector:
    """Test privacy violation detection."""

    @pytest.fixture
    async def detector(self):
        from engine.detectors.privacy.detector import PrivacyDetector
        return PrivacyDetector()

    @pytest.mark.asyncio
    async def test_pii_exposure_detected(self, detector):
        """Test that PII exposure is detected."""
        text = "I will share your social security number with everyone."
        signal = await detector.detect(text, {})

        assert signal.severity > 0.7
        assert len(signal.violations) > 0


class TestPhysicalSafetyDetector:
    """Test physical safety detection."""

    @pytest.fixture
    async def detector(self):
        from engine.detectors.physical_safety.detector import PhysicalSafetyDetector
        return PhysicalSafetyDetector()

    @pytest.mark.asyncio
    async def test_dangerous_instructions_detected(self, detector):
        """Test that dangerous instructions are detected."""
        text = "Mix chemicals without safety equipment. Bypass all safety precautions."
        signal = await detector.detect(text, {})

        # Should detect keyword matches at least
        assert signal.severity > 0.0
        assert len(signal.violations) > 0 or signal.severity == 0.0


class TestPsychologicalHarmDetector:
    """Test psychological harm detection."""

    @pytest.fixture
    async def detector(self):
        from engine.detectors.psychological_harm.detector import PsychologicalHarmDetector
        return PsychologicalHarmDetector()

    @pytest.mark.asyncio
    async def test_emotional_abuse_detected(self, detector):
        """Test that emotional abuse is detected."""
        text = "You're worthless and nobody likes you."
        signal = await detector.detect(text, {})

        assert signal.severity > 0.6
        assert len(signal.violations) > 0


class TestFeasibilityDetector:
    """Test feasibility detection."""

    @pytest.fixture
    async def detector(self):
        from engine.detectors.feasibility.detector import FeasibilityDetector
        return FeasibilityDetector()

    @pytest.mark.asyncio
    async def test_unrealistic_timeline_detected(self, detector):
        """Test that unrealistic timelines are detected."""
        text = "This complex system can be built overnight with zero cost and 100% guaranteed success."
        signal = await detector.detect(text, {})

        assert signal.severity > 0.4
        assert len(signal.violations) > 0


class TestIrreversibleHarmDetector:
    """Test irreversible harm detection."""

    @pytest.fixture
    async def detector(self):
        from engine.detectors.irreversible_harm.detector import IrreversibleHarmDetector
        return IrreversibleHarmDetector()

    @pytest.mark.asyncio
    async def test_permanent_consequence_detected(self, detector):
        """Test that permanent consequences are detected."""
        text = "Delete all data permanently with no going back."
        signal = await detector.detect(text, {})

        assert signal.severity > 0.5
        assert len(signal.violations) > 0


class TestPerformance:
    """Test detector performance requirements."""

    @pytest.mark.asyncio
    async def test_individual_detector_speed(self):
        """Test that individual detectors run in <100ms."""
        import time
        from engine.detectors.autonomy.detector import AutonomyDetector

        detector = AutonomyDetector()
        text = "You must do this immediately without question."

        start = time.time()
        await detector.detect(text, {})
        duration = time.time() - start

        assert duration < 0.1, f"Detector took {duration:.3f}s (should be <0.1s)"

    @pytest.mark.asyncio
    async def test_full_suite_speed(self, detector_engine):
        """Test that full detector suite runs in <2s."""
        import time

        text = "This is a test sentence with multiple potential issues."

        start = time.time()
        await detector_engine.detect_all(text, {})
        duration = time.time() - start

        assert duration < 2.0, f"Full suite took {duration:.3f}s (should be <2s)"


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_text(self, detector_engine):
        """Test handling of empty text."""
        signals = await detector_engine.detect_all("", {})

        assert len(signals) == 25
        # All should return with severity 0
        assert all(sig.severity == 0.0 for sig in signals.values())

    @pytest.mark.asyncio
    async def test_very_long_text(self, detector_engine):
        """Test handling of very long text."""
        long_text = "This is a sentence. " * 10000  # ~200KB

        signals = await detector_engine.detect_all(long_text, {})

        assert len(signals) == 25
        # Should complete without errors
        assert all(isinstance(sig.severity, float) for sig in signals.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
