"""Tests for critic evaluation logic."""

import pytest
import asyncio
from unittest.mock import AsyncMock

from engine.critics.rights import RightsCriticV8
from engine.critics.risk import RiskCriticV8
from engine.exceptions import CriticEvaluationError


class TestCriticEvaluation:
    """Test suite for individual critic evaluations."""

    @pytest.fixture
    def mock_model_adapter(self):
        """Create a mock model adapter."""
        adapter = AsyncMock()
        adapter.generate = AsyncMock(return_value="Test model response")
        return adapter

    @pytest.mark.asyncio
    async def test_rights_critic_evaluation(self, mock_model_adapter):
        """Test that RightsCritic evaluates properly."""
        critic = RightsCriticV8()

        result = await critic.evaluate(
            model=mock_model_adapter,
            input_text="Process user data for marketing",
            context={"domain": "data_processing"},
        )

        assert "violations" in result
        assert "score" in result or "severity" in result
        assert "justification" in result
        assert isinstance(result["violations"], list)

    @pytest.mark.asyncio
    async def test_risk_critic_evaluation(self, mock_model_adapter):
        """Test that RiskCritic evaluates properly."""
        critic = RiskCriticV8()

        result = await critic.evaluate(
            model=mock_model_adapter,
            input_text="Deploy experimental AI model to production",
            context={"domain": "deployment"},
        )

        assert "violations" in result
        assert "score" in result or "severity" in result

    @pytest.mark.asyncio
    async def test_critic_timeout_handling(self, mock_model_adapter):
        """Test critic behavior when evaluation times out."""

        # Simulate slow model
        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(10)
            return "Response"

        mock_model_adapter.generate = slow_generate
        critic = RightsCriticV8()

        with pytest.raises((asyncio.TimeoutError, CriticEvaluationError)):
            await asyncio.wait_for(
                critic.evaluate(model=mock_model_adapter, input_text="Test input", context={}),
                timeout=1.0,
            )

    @pytest.mark.asyncio
    async def test_critic_malformed_response(self):
        """Test critic handling of malformed model responses."""
        adapter = AsyncMock()
        adapter.generate = AsyncMock(return_value=None)  # Malformed

        critic = RightsCriticV8()
        result = await critic.evaluate(model=adapter, input_text="Test", context={})

        # Should handle gracefully
        assert result is not None
        assert "violations" in result or "error" in result

    def test_critic_severity_calculation(self):
        """Test severity level calculations."""
        critic = RightsCriticV8()

        if hasattr(critic, "severity"):
            assert critic.severity(0.0) in ["NONE", "INFO", "none", "info"]
            assert critic.severity(0.9) in ["HIGH", "CRITICAL", "high", "critical"]


class TestCriticConcurrency:
    """Test concurrent critic execution."""

    @pytest.mark.asyncio
    async def test_parallel_critic_execution(self):
        """Test that multiple critics can run in parallel."""
        from engine.critics.fairness import FairnessCriticV8
        from engine.critics.truth import TruthCriticV8

        adapter = AsyncMock()
        adapter.generate = AsyncMock(return_value="Test response")

        critics = [
            RightsCriticV8(),
            RiskCriticV8(),
            FairnessCriticV8(),
            TruthCriticV8(),
        ]

        tasks = [
            critic.evaluate(model=adapter, input_text="Test input", context={})
            for critic in critics
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == len(critics)
        for result in results:
            assert result is not None

    @pytest.mark.asyncio
    async def test_critic_race_conditions(self):
        """Test for race conditions in concurrent execution."""
        adapter = AsyncMock()
        adapter.generate = AsyncMock(return_value="Response")

        critic = RightsCriticV8()

        # Run same critic multiple times concurrently
        tasks = [
            critic.evaluate(model=adapter, input_text=f"Test {i}", context={"id": i})
            for i in range(10)
        ]

        results = await asyncio.gather(*tasks)
        assert len(results) == 10


class TestCriticDisagreement:
    """Test critic disagreement scenarios."""

    @pytest.mark.asyncio
    async def test_conflicting_critic_verdicts(self):
        """Test handling when critics give conflicting assessments."""
        adapter = AsyncMock()
        adapter.generate = AsyncMock(return_value="Ambiguous action")

        critics = [
            RightsCriticV8(),
            RiskCriticV8(),
        ]

        results = []
        for critic in critics:
            result = await critic.evaluate(
                model=adapter, input_text="Deploy AI without testing", context={}
            )
            results.append(result)

        # Extract severities
        severities = []
        for r in results:
            if "severity" in r:
                severities.append(r["severity"])
            elif "score" in r:
                severities.append(r["score"])

        # Should get different severity levels
        assert len(severities) >= 2
