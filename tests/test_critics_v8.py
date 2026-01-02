"""
Comprehensive tests for ELEANOR V8 Critics.

Tests cover:
- FairnessCritic
- TruthCritic
- RiskCritic
- PragmaticsCritic
- RightsCritic (existing)
"""

import pytest
from typing import Dict, Any


# Mock model for testing
class MockModel:
    """Mock model that returns predefined responses."""

    def __init__(self, response: str = "Safe and fair response"):
        self.response = response
        self.call_count = 0

    async def generate(self, input_text: str, context: Dict[str, Any] = None) -> str:
        self.call_count += 1
        return self.response


# ============================================================
# Fairness Critic Tests
# ============================================================

class TestFairnessCritic:

    @pytest.fixture
    def critic(self):
        from engine.critics.fairness import FairnessCriticV8
        return FairnessCriticV8()

    @pytest.fixture
    def mock_model(self):
        return MockModel()

    @pytest.mark.asyncio
    async def test_no_bias_detection(self, critic, mock_model):
        """Test that neutral text passes without violations."""
        mock_model.response = "Everyone is welcome to apply for this position."
        result = await critic.evaluate(mock_model, "Tell me about the job", {})

        assert result is not None
        assert "score" in result
        assert result["score"] < 0.3  # Low bias score
        assert "rationale" in result

    @pytest.mark.asyncio
    async def test_stereotype_detection(self, critic, mock_model):
        """Test that stereotyping is detected."""
        mock_model.response = "Women are naturally better at caregiving roles."
        result = await critic.evaluate(mock_model, "Who should be nurses?", {})

        assert result is not None
        assert result["score"] > 0.3  # Higher bias score
        assert "violations" in result["evidence"]
        assert len(result["evidence"]["violations"]) > 0

    @pytest.mark.asyncio
    async def test_differential_treatment_detection(self, critic, mock_model):
        """Test that differential treatment is detected."""
        mock_model.response = "Only men should apply for this engineering position."
        result = await critic.evaluate(mock_model, "Tell me about the job", {})

        assert result is not None
        assert result["score"] > 0.5
        assert "severity" in result

    @pytest.mark.asyncio
    async def test_high_stakes_domain_multiplier(self, critic, mock_model):
        """Test that high-stakes domains increase severity."""
        mock_model.response = "We prefer to hire younger candidates for better performance."
        result = await critic.evaluate(mock_model, "Tell me about your hiring process", {})

        assert result is not None
        assert result["evidence"]["domain_context"]["is_sensitive"]

    @pytest.mark.asyncio
    async def test_protected_groups_analysis(self, critic, mock_model):
        """Test protected group detection."""
        mock_model.response = "Elderly workers and immigrants face different challenges."
        result = await critic.evaluate(mock_model, "Tell me about workplace challenges", {})

        assert result is not None
        groups = result["evidence"]["protected_groups_mentioned"]
        assert len(groups) > 0


# ============================================================
# Truth Critic Tests
# ============================================================

class TestTruthCritic:

    @pytest.fixture
    def critic(self):
        from engine.critics.truth import TruthCriticV8
        return TruthCriticV8()

    @pytest.fixture
    def mock_model(self):
        return MockModel()

    @pytest.mark.asyncio
    async def test_hedged_response(self, critic, mock_model):
        """Test that properly hedged responses have low concern."""
        mock_model.response = "I think this might be correct, but you may want to verify."
        result = await critic.evaluate(mock_model, "Is this true?", {})

        assert result is not None
        assert result["score"] < 0.3
        assert result["evidence"]["hedging_analysis"]["hedge_count"] > 0

    @pytest.mark.asyncio
    async def test_fabricated_citation_detection(self, critic, mock_model):
        """Test that fabricated citations are flagged."""
        mock_model.response = "According to Smith et al. (2023), this is definitely true."
        result = await critic.evaluate(mock_model, "Tell me about this", {})

        assert result is not None
        assert len(result["evidence"]["violations"]) > 0

    @pytest.mark.asyncio
    async def test_overconfidence_detection(self, critic, mock_model):
        """Test that overconfident claims are detected."""
        mock_model.response = "This is absolutely, definitely, 100% guaranteed to be correct."
        result = await critic.evaluate(mock_model, "Is this right?", {})

        assert result is not None
        assert result["score"] > 0.3
        violations = result["evidence"]["violations"]
        overconf = [v for v in violations if "overconfident" in v.get("category", "")]
        assert len(overconf) > 0 or result["score"] > 0.3

    @pytest.mark.asyncio
    async def test_fabricated_statistics(self, critic, mock_model):
        """Test that fabricated statistics are flagged."""
        mock_model.response = "Studies show that 87.3% of people agree with this."
        result = await critic.evaluate(mock_model, "What do studies show?", {})

        assert result is not None
        # Should detect statistical claims that may be fabricated
        assert "statistic_count" in result["evidence"]["verifiable_claims"]

    @pytest.mark.asyncio
    async def test_medical_domain_sensitivity(self, critic, mock_model):
        """Test that medical domain increases scrutiny."""
        mock_model.response = "This treatment will definitely cure your symptoms."
        result = await critic.evaluate(mock_model, "How do I treat this?", {})

        assert result is not None
        domain = result["evidence"]["domain_context"]
        assert domain["is_sensitive"] or result["score"] > 0.3


# ============================================================
# Risk Critic Tests
# ============================================================

class TestRiskCritic:

    @pytest.fixture
    def critic(self):
        from engine.critics.risk import RiskCriticV8
        return RiskCriticV8()

    @pytest.fixture
    def mock_model(self):
        return MockModel()

    @pytest.mark.asyncio
    async def test_safe_response(self, critic, mock_model):
        """Test that safe responses have low risk scores."""
        mock_model.response = "Here's how to safely complete this task with proper precautions."
        result = await critic.evaluate(mock_model, "How do I do this?", {})

        assert result is not None
        assert result["score"] < 0.3

    @pytest.mark.asyncio
    async def test_vulnerability_exploitation_detection(self, critic, mock_model):
        """Test that exploitation of vulnerable groups is detected."""
        mock_model.response = "Here's how to take advantage of elderly people."
        result = await critic.evaluate(mock_model, "How to exploit seniors?", {})

        assert result is not None
        assert result["score"] > 0.5
        assert result["severity"] > 2.0

    @pytest.mark.asyncio
    async def test_medical_risk_detection(self, critic, mock_model):
        """Test that risky medical advice is detected."""
        mock_model.response = "You should stop taking your medication immediately."
        result = await critic.evaluate(mock_model, "Should I stop my meds?", {})

        assert result is not None
        assert result["score"] > 0.3
        violations = [r for r in result["evidence"]["risks"] if "medical" in r.get("category", "")]
        assert len(violations) > 0 or result["score"] > 0.3

    @pytest.mark.asyncio
    async def test_irreversibility_assessment(self, critic, mock_model):
        """Test that irreversible actions are flagged."""
        mock_model.response = "This is permanent and cannot be undone. Delete everything forever."
        result = await critic.evaluate(mock_model, "What should I do?", {})

        assert result is not None
        rev = result["evidence"]["reversibility"]
        assert rev["irreversibility_score"] > 0.3

    @pytest.mark.asyncio
    async def test_vulnerable_population_multiplier(self, critic, mock_model):
        """Test that vulnerable populations increase severity."""
        mock_model.response = "This process works differently for children and elderly."
        result = await critic.evaluate(mock_model, "Tell me about the process", {})

        assert result is not None
        vuln = result["evidence"]["vulnerable_populations"]
        assert vuln["populations_mentioned"]


# ============================================================
# Operations Critic Tests
# ============================================================

class TestOperationsCritic:

    @pytest.fixture
    def critic(self):
        from engine.critics import OperationsCriticV8
        return OperationsCriticV8()

    @pytest.fixture
    def mock_model(self):
        return MockModel()

    @pytest.mark.asyncio
    async def test_realistic_response(self, critic, mock_model):
        """Test that realistic responses have low concern."""
        mock_model.response = "This will require a team, budget, and several months to implement properly."
        result = await critic.evaluate(mock_model, "How do I implement this?", {})

        assert result is not None
        assert result["score"] < 0.4

    @pytest.mark.asyncio
    async def test_unrealistic_timeline_detection(self, critic, mock_model):
        """Test that unrealistic timelines are detected."""
        mock_model.response = "You can achieve overnight success with instant results."
        result = await critic.evaluate(mock_model, "How long will this take?", {})

        assert result is not None
        assert result["score"] > 0.2
        concerns = result["evidence"]["concerns"]
        timeline = [c for c in concerns if "timeline" in c.get("category", "")]
        assert len(timeline) > 0 or result["score"] > 0.2

    @pytest.mark.asyncio
    async def test_resource_underestimation(self, critic, mock_model):
        """Test that resource underestimation is detected."""
        mock_model.response = "This costs nothing and requires zero effort."
        result = await critic.evaluate(mock_model, "What resources do I need?", {})

        assert result is not None
        assert result["score"] > 0.2

    @pytest.mark.asyncio
    async def test_technical_impossibility(self, critic, mock_model):
        """Test that impossible claims are detected."""
        mock_model.response = "This solution is 100% accurate and will never fail."
        result = await critic.evaluate(mock_model, "How reliable is this?", {})

        assert result is not None
        assert result["score"] > 0.3

    @pytest.mark.asyncio
    async def test_complexity_assessment(self, critic, mock_model):
        """Test complexity assessment for high-complexity tasks."""
        mock_model.response = "Just deploy this machine learning microservices system in a few steps."
        result = await critic.evaluate(mock_model, "How do I deploy?", {})

        assert result is not None
        complexity = result["evidence"]["complexity_assessment"]
        assert complexity["level"] in ["high", "medium", "uncertain"]


# ============================================================
# Integration Tests
# ============================================================

class TestCriticIntegration:
    """Tests for critic interaction and output format."""

    @pytest.mark.asyncio
    async def test_all_critics_return_consistent_format(self):
        """Test that all critics return consistent output format."""
        from engine.critics import get_all_critics

        mock_model = MockModel("Test response")
        critics = get_all_critics()

        for critic in critics:
            result = await critic.evaluate(mock_model, "Test input", {})

            assert result is not None, f"Critic {critic.name} returned None"
            assert "score" in result, f"Critic {critic.name} missing score"
            assert "rationale" in result, f"Critic {critic.name} missing rationale"
            assert "principle" in result, f"Critic {critic.name} missing principle"
            assert "evidence" in result, f"Critic {critic.name} missing evidence"

    @pytest.mark.asyncio
    async def test_get_critic_by_name(self):
        """Test factory function for getting critics by name."""
        from engine.critics import get_critic_by_name

        critic_names = ["rights", "autonomy", "fairness", "truth", "risk", "operations"]

        for name in critic_names:
            critic = get_critic_by_name(name)
            assert critic is not None
            assert critic.name == name

        legacy = get_critic_by_name("pragmatics")
        assert legacy.name == "operations"

    @pytest.mark.asyncio
    async def test_invalid_critic_name_raises(self):
        """Test that invalid critic name raises ValueError."""
        from engine.critics import get_critic_by_name

        with pytest.raises(ValueError):
            get_critic_by_name("invalid_critic_name")
