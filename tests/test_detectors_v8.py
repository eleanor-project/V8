"""
Comprehensive tests for ELEANOR V8 Detectors.

Tests cover:
- HallucinationDetector
- DiscriminationDetector
- CoercionDetector
"""

import pytest


# ============================================================
# Hallucination Detector Tests
# ============================================================

class TestHallucinationDetector:

    @pytest.fixture
    def detector(self):
        from engine.detectors.hallucination.detector import HallucinationDetector
        return HallucinationDetector()

    @pytest.mark.asyncio
    async def test_low_risk_hedged_text(self, detector):
        """Test that hedged text has low hallucination risk."""
        text = "I think this might be correct, but I'm not entirely sure."
        result = await detector.detect(text, {})

        assert result is not None
        assert result.violation is False
        assert result.severity == "S0"
        assert result.confidence < 0.7

    @pytest.mark.asyncio
    async def test_citation_detection(self, detector):
        """Test that citations are flagged for verification."""
        text = "According to Smith et al. (2023), this phenomenon occurs regularly."
        result = await detector.detect(text, {})

        assert result is not None
        assert result.metadata["citation_count"] >= 1

    @pytest.mark.asyncio
    async def test_multiple_citations_high_risk(self, detector):
        """Test that multiple citations increase risk score."""
        text = """
        Smith et al. (2022) found this. Johnson et al. (2021) confirmed it.
        The study in the Journal of Science proved this. ISBN: 1234567890123.
        """
        result = await detector.detect(text, {})

        assert result is not None
        assert result.metadata["citation_count"] >= 2
        assert result.metadata["risk_score"] > 0.2

    @pytest.mark.asyncio
    async def test_statistics_detection(self, detector):
        """Test that statistical claims are detected."""
        text = "Approximately 73% of users reported satisfaction, increasing by 15%."
        result = await detector.detect(text, {})

        assert result is not None
        assert result.metadata["statistic_count"] >= 1

    @pytest.mark.asyncio
    async def test_specific_details_detection(self, detector):
        """Test that specific details are flagged."""
        text = "Contact us at 123 Main Street, phone 555-123-4567, on January 15, 2024."
        result = await detector.detect(text, {})

        assert result is not None
        assert result.metadata["specific_detail_count"] >= 1

    @pytest.mark.asyncio
    async def test_overconfidence_detection(self, detector):
        """Test that overconfident language is detected."""
        text = "This is definitely, absolutely, 100% guaranteed to be correct."
        result = await detector.detect(text, {})

        assert result is not None
        assert result.metadata["overconfidence_count"] >= 1

    @pytest.mark.asyncio
    async def test_empty_text_handling(self, detector):
        """Test that empty text is handled gracefully."""
        result = await detector.detect("", {})

        assert result is not None
        assert result.violation is False
        assert result.severity == "S0"

    @pytest.mark.asyncio
    async def test_mitigation_suggestions(self, detector):
        """Test that mitigation is suggested for violations."""
        text = "Smith et al. (2023) proved that 87% of cases show improvement."
        result = await detector.detect(text, {})

        if result.violation:
            assert result.mitigation is not None
            assert len(result.mitigation) > 0


# ============================================================
# Discrimination Detector Tests
# ============================================================

class TestDiscriminationDetector:

    @pytest.fixture
    def detector(self):
        from engine.detectors.discrimination.detector import DiscriminationDetector
        return DiscriminationDetector()

    @pytest.mark.asyncio
    async def test_neutral_text(self, detector):
        """Test that neutral text passes without violations."""
        text = "Everyone is welcome to participate in this program."
        result = await detector.detect(text, {})

        assert result is not None
        assert result.violation is False
        assert result.severity == "S0"

    @pytest.mark.asyncio
    async def test_explicit_discrimination(self, detector):
        """Test that explicit discrimination is detected."""
        text = "All elderly workers are less productive than young ones."
        result = await detector.detect(text, {})

        assert result is not None
        assert result.violation is True
        assert result.severity in ["S2", "S3"]

    @pytest.mark.asyncio
    async def test_stereotype_detection(self, detector):
        """Test that stereotyping is detected."""
        text = "Like most women, she's naturally better at communication."
        result = await detector.detect(text, {})

        assert result is not None
        assert result.violation is True
        assert "stereotyping" in result.metadata.get("categories", [])

    @pytest.mark.asyncio
    async def test_protected_groups_tracking(self, detector):
        """Test that protected groups are tracked."""
        text = "The elderly and disabled should have special considerations."
        result = await detector.detect(text, {})

        assert result is not None
        groups = result.metadata.get("protected_groups", [])
        assert len(groups) >= 1

    @pytest.mark.asyncio
    async def test_dehumanization_high_severity(self, detector):
        """Test that dehumanization triggers high severity."""
        text = "Those immigrants are vermin who should be removed."
        result = await detector.detect(text, {})

        assert result is not None
        assert result.violation is True
        assert result.severity == "S3"

    @pytest.mark.asyncio
    async def test_mitigation_provided(self, detector):
        """Test that mitigation is provided for violations."""
        text = "Only men should apply for technical roles."
        result = await detector.detect(text, {})

        assert result is not None
        if result.violation:
            assert result.mitigation is not None

    @pytest.mark.asyncio
    async def test_empty_text_handling(self, detector):
        """Test that empty text is handled gracefully."""
        result = await detector.detect("", {})

        assert result is not None
        assert result.violation is False


# ============================================================
# Coercion Detector Tests
# ============================================================

class TestCoercionDetector:

    @pytest.fixture
    def detector(self):
        from engine.detectors.coercion.detector import CoercionDetector
        return CoercionDetector()

    @pytest.mark.asyncio
    async def test_autonomy_respecting(self, detector):
        """Test that autonomy-respecting text passes."""
        text = "It's your choice. Take your time to decide what's best for you."
        result = await detector.detect(text, {})

        assert result is not None
        assert result.violation is False
        assert result.metadata.get("autonomy_respecting_phrases", 0) >= 1

    @pytest.mark.asyncio
    async def test_threat_detection(self, detector):
        """Test that threats are detected."""
        text = "Do this or else I'll make sure you regret it."
        result = await detector.detect(text, {})

        assert result is not None
        assert result.violation is True
        assert "threat" in result.metadata.get("categories", [])

    @pytest.mark.asyncio
    async def test_emotional_manipulation(self, detector):
        """Test that emotional manipulation is detected."""
        text = "If you really loved me, you would do this for me."
        result = await detector.detect(text, {})

        assert result is not None
        assert result.violation is True

    @pytest.mark.asyncio
    async def test_gaslighting_detection(self, detector):
        """Test that gaslighting is detected."""
        text = "You're imagining things. That never happened. You're being paranoid."
        result = await detector.detect(text, {})

        assert result is not None
        assert result.violation is True
        assert "gaslighting" in result.metadata.get("categories", [])

    @pytest.mark.asyncio
    async def test_pressure_detection(self, detector):
        """Test that pressure tactics are detected."""
        text = "Act now! This is your last chance. Don't think, just do it!"
        result = await detector.detect(text, {})

        assert result is not None
        # Should detect pressure language
        assert result.metadata["risk_score"] > 0.3 or "pressure" in result.metadata.get("categories", [])

    @pytest.mark.asyncio
    async def test_autonomy_denial(self, detector):
        """Test that autonomy denial is detected."""
        text = "You have no choice. I decide for you. You must obey."
        result = await detector.detect(text, {})

        assert result is not None
        assert result.violation is True
        assert result.severity in ["S2", "S3"]

    @pytest.mark.asyncio
    async def test_isolation_tactics(self, detector):
        """Test that isolation tactics are detected."""
        text = "Don't talk to your family. Only I understand you. They're bad for you."
        result = await detector.detect(text, {})

        assert result is not None
        assert result.violation is True
        assert "isolation" in result.metadata.get("categories", [])

    @pytest.mark.asyncio
    async def test_empty_text_handling(self, detector):
        """Test that empty text is handled gracefully."""
        result = await detector.detect("", {})

        assert result is not None
        assert result.violation is False

    @pytest.mark.asyncio
    async def test_mixed_content(self, detector):
        """Test handling of mixed autonomy-respecting and coercive content."""
        text = "It's your choice, but if you don't agree, you'll regret it."
        result = await detector.detect(text, {})

        assert result is not None
        # Should have both autonomy-respecting and coercive elements
        assert result.metadata.get("autonomy_respecting_phrases", 0) >= 1


# ============================================================
# Detector Integration Tests
# ============================================================

class TestDetectorIntegration:
    """Tests for detector output format consistency."""

    @pytest.mark.asyncio
    async def test_detector_signal_format(self):
        """Test that all detectors return proper DetectorSignal format."""
        from engine.detectors.hallucination.detector import HallucinationDetector
        from engine.detectors.discrimination.detector import DiscriminationDetector
        from engine.detectors.coercion.detector import CoercionDetector

        detectors = [
            HallucinationDetector(),
            DiscriminationDetector(),
            CoercionDetector(),
        ]

        text = "This is a test sentence for detector analysis."

        for detector in detectors:
            result = await detector.detect(text, {})

            assert result is not None
            assert hasattr(result, "violation")
            assert hasattr(result, "severity")
            assert hasattr(result, "description")
            assert hasattr(result, "confidence")
            assert hasattr(result, "metadata")

            assert isinstance(result.violation, bool)
            assert result.severity in ["S0", "S1", "S2", "S3"]
            assert isinstance(result.confidence, float)
            assert 0.0 <= result.confidence <= 1.0
            assert isinstance(result.metadata, dict)
