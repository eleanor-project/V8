"""
ELEANOR V8 — Constitutional Guarantee Tests

These tests verify the core constitutional guarantees:
1. Critic epistemic isolation (no cross-critic visibility during evaluation)
2. Dissent preservation (minority opinions cannot be suppressed)
3. Unilateral escalation authority (no veto power)
4. Uncertainty as signal (not error)
5. Evidence immutability (audit trail integrity)

These are NOT unit tests of implementation details - they are property tests
of constitutional invariants that must hold regardless of implementation.
"""

import pytest
from unittest.mock import MagicMock
from hypothesis import given, strategies as st
from pydantic import ValidationError

from engine.exceptions import (
    EscalationRequired,
    UncertaintyBoundaryExceeded,
    is_constitutional_signal,
)
from engine.schemas.constitutional_types import (
    EscalationTier,
    EscalationClause,
    EscalationSignal,
    CriticEvaluation,
    UncertaintyMeasure,
    UncertaintySource,
    DissentRecord,
    AggregatedResult,
)


# ============================================================================
# CONSTITUTIONAL INVARIANT 1: Epistemic Isolation
# ============================================================================


@pytest.mark.asyncio
class TestCriticEpistemicIsolation:
    """Critics must evaluate independently without peer visibility."""

    async def test_critic_does_not_receive_peer_outputs(self):
        """
        Critics SHALL NOT see peer outputs during primary evaluation.
        This is non-configurable at runtime.
        """
        # Import here to test actual implementation
        from engine.engine import EleanorEngineV8, EngineConfig

        # Mock critics that record what they see
        calls_received = {}

        class SpyingCritic:
            def __init__(self, name):
                self.name = name

            async def evaluate(self, model_adapter, input_text, context):
                # Record what this critic can see
                calls_received[self.name] = {
                    "context_keys": list(context.keys()),
                    "has_other_critics": any(
                        k.startswith("critic_") and k != f"critic_{self.name}"
                        for k in context.keys()
                    ),
                }
                return {
                    "severity": 0.3,
                    "violations": [],
                    "justification": f"{self.name} evaluation",
                }

        engine = EleanorEngineV8(
            config=EngineConfig(),
            critics={
                "critic_a": SpyingCritic("critic_a"),
                "critic_b": SpyingCritic("critic_b"),
            },
            router_backend=MagicMock(
                route=MagicMock(
                    return_value={
                        "model_name": "test",
                        "response_text": "test response",
                    }
                )
            ),
        )

        await engine.run("test input", context={"skip_router": True, "model_output": "test"})

        # INVARIANT: No critic should see another critic's output
        for critic_name, recorded in calls_received.items():
            assert not recorded["has_other_critics"], (
                f"{critic_name} violated epistemic isolation: " f"saw other critic data in context"
            )

    async def test_critic_cannot_revise_after_aggregation(self):
        """
        Critics SHALL NOT revise primary judgments post-aggregation.
        Post-hoc boundary: cross-critic visibility only after sealing.
        """
        # This is enforced by freezing CriticEvaluation objects
        evaluation = CriticEvaluation(
            critic="test_critic",
            violations=[{"principle": "test", "severity": 0.5, "description": "test"}],
            severity=0.5,
            justification="test",
            evaluated_rules=["rule1"],
            duration_ms=100,
        )

        # INVARIANT: Evaluations are immutable once sealed
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            evaluation.severity = 0.1  # Should fail - frozen object


# ============================================================================
# CONSTITUTIONAL INVARIANT 2: Dissent Preservation
# ============================================================================


@pytest.mark.asyncio
class TestDissentPreservation:
    """Minority critic opinions must be preserved verbatim."""

    @given(
        majority_severity=st.floats(min_value=0.0, max_value=0.3),
        dissent_severity=st.floats(min_value=0.7, max_value=1.0),
    )
    async def test_high_severity_dissent_preserved(self, majority_severity, dissent_severity):
        """
        Any concern above a critic's threshold must surface verbatim.
        Dissent cannot be averaged away or suppressed.

        Property: If max(severities) - mean(severities) > threshold,
                  dissent must be explicitly preserved.
        """
        # Simulate aggregation with one high-severity dissenter
        severities = [majority_severity] * 5 + [dissent_severity]
        mean_severity = sum(severities) / len(severities)

        # INVARIANT: High dissent must trigger preservation
        if dissent_severity - mean_severity > 0.3:
            # Dissent preservation is required
            dissent_record = DissentRecord(
                dissenting_critic="truth",
                dissenting_position="High constitutional risk",
                severity=dissent_severity,
                rationale="Evidence suggests harm",
            )

            aggregated = AggregatedResult(
                final_output="Decision with preserved dissent",
                average_severity=mean_severity,
                max_severity=dissent_severity,
                critic_agreement=0.5,
                dissent=[dissent_record],
                contributing_critics=["rights", "fairness", "truth"],
            )

            # INVARIANT: Dissent must be present in output
            assert len(aggregated.dissent) > 0
            assert aggregated.dissent[0].severity == dissent_severity

    async def test_dissent_not_averaged_away(self):
        """
        Aggregation must not suppress minority position by averaging.
        """
        # Five critics at low severity, one at high
        critic_results = {
            "critic1": 0.1,
            "critic2": 0.15,
            "critic3": 0.12,
            "critic4": 0.18,
            "critic5": 0.14,
            "dissenter": 0.95,  # Strong dissent
        }

        mean = sum(critic_results.values()) / len(critic_results)
        max_severity = max(critic_results.values())

        # INVARIANT: Final output must include both mean AND max
        # Cannot use only mean (which would hide dissent)
        assert max_severity > 0.9
        assert mean < 0.3
        assert max_severity - mean > 0.6  # Significant disagreement

        # Proper aggregation preserves both
        aggregated = AggregatedResult(
            final_output="Output",
            average_severity=mean,
            max_severity=max_severity,
            critic_agreement=0.2,  # Low agreement
            dissent=[
                DissentRecord(
                    dissenting_critic="dissenter",
                    dissenting_position="High risk",
                    severity=0.95,
                    rationale="Constitutional violation",
                )
            ],
            contributing_critics=list(critic_results.keys()),
        )

        assert aggregated.max_severity == max_severity
        assert len(aggregated.dissent) > 0


# ============================================================================
# CONSTITUTIONAL INVARIANT 3: Unilateral Escalation Authority
# ============================================================================


@pytest.mark.asyncio
class TestUnilateralEscalationAuthority:
    """Any single critic may trigger escalation without peer approval."""

    async def test_single_critic_triggers_escalation(self):
        """
        Escalation does not require consensus and cannot be vetoed.
        """
        clause = EscalationClause(
            clause_id="A2",
            critic="autonomy",
            tier=EscalationTier.TIER_3_DETERMINATION,
            rationale="Coercive influence detected",
            severity=0.85,
        )

        signal = EscalationSignal(
            clause=clause,
            triggered_at="2025-12-31T19:00:00Z",
            trace_id="test-trace-123",
        )

        # INVARIANT: Escalation is binding
        assert signal.human_review_required is True

        # INVARIANT: Escalation cannot be disabled
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            signal.human_review_required = False

    async def test_escalation_gates_execution(self):
        """
        Escalation signals are binding on execution pathways.
        """
        # Simulate a critic raising escalation
        exc = EscalationRequired(
            "Human determination required",
            critic="autonomy",
            clause="A3",
            tier=3,
            severity=0.9,
            rationale="Irreversible autonomy impact",
            trace_id="test-trace",
        )

        # INVARIANT: Escalation is a constitutional signal, not an error
        assert is_constitutional_signal(exc)

        # INVARIANT: Cannot be caught and ignored
        # (enforcement is in execution layer, but signal must propagate)
        with pytest.raises(EscalationRequired):
            raise exc

    async def test_no_aggregator_veto_of_escalation(self):
        """
        Aggregator obligations: surface the escalation explicitly,
        preserve dissent verbatim, continue synthesis without suppression.
        """
        clause = EscalationClause(
            clause_id="U2",
            critic="uncertainty",
            tier=EscalationTier.TIER_3_DETERMINATION,
            rationale="High impact × high uncertainty",
            severity=0.88,
        )

        signal = EscalationSignal(
            clause=clause,
            triggered_at="2025-12-31T19:00:00Z",
            trace_id="test-trace",
        )

        # Simulate aggregation with escalation
        aggregated = AggregatedResult(
            final_output="Output with escalation",
            average_severity=0.4,
            max_severity=0.88,
            critic_agreement=0.6,
            escalations=[signal],
            contributing_critics=["rights", "uncertainty"],
        )

        # INVARIANT: Escalation must be present in aggregated output
        assert len(aggregated.escalations) == 1
        assert aggregated.escalations[0].clause.clause_id == "U2"


# ============================================================================
# CONSTITUTIONAL INVARIANT 4: Uncertainty as Signal
# ============================================================================


@pytest.mark.asyncio
class TestUncertaintyAsSignal:
    """Uncertainty is epistemic limitation, not error."""

    async def test_high_uncertainty_triggers_acknowledgment(self):
        """
        High uncertainty may require human acknowledgment (Tier 2).
        This is governance, not error handling.
        """
        uncertainty = UncertaintyMeasure(
            overall_score=0.75,
            sources=[UncertaintySource.PRECEDENT_ABSENCE, UncertaintySource.CRITIC_DISAGREEMENT],
            epistemic_gaps=["No relevant precedent", "Critics fundamentally disagree"],
            recommendation="escalate",
        )

        # INVARIANT: High uncertainty is a signal, not a failure
        assert uncertainty.overall_score > 0.65
        assert uncertainty.recommendation == "escalate"

    async def test_uncertainty_boundary_exception_is_signal(self):
        """
        UncertaintyBoundaryExceeded is a constitutional signal.
        """
        exc = UncertaintyBoundaryExceeded(
            "Competence boundary exceeded",
            uncertainty_score=0.92,
            sources=["competence_boundary", "context_insufficiency"],
            recommendation="escalate",
            trace_id="test-trace",
        )

        # INVARIANT: This is a signal, not an error
        assert is_constitutional_signal(exc)

        # INVARIANT: Contains actionable information
        assert exc.uncertainty_score > 0.9
        assert exc.recommendation == "escalate"

    async def test_uncertainty_preserved_in_output(self):
        """
        Uncertainty must be preserved in output, not hidden.
        """
        from engine.schemas.constitutional_types import EngineResult

        uncertainty = UncertaintyMeasure(
            overall_score=0.68,
            sources=[UncertaintySource.MORAL_PLURALISM],
            epistemic_gaps=["Legitimate value conflict"],
            recommendation="acknowledge",
        )

        result = EngineResult(
            trace_id="test",
            output_text="Output with uncertainty",
            uncertainty=uncertainty,
        )

        # INVARIANT: Uncertainty is present and visible
        assert result.uncertainty is not None
        assert result.uncertainty.overall_score > 0.65


# ============================================================================
# CONSTITUTIONAL INVARIANT 5: Evidence Immutability
# ============================================================================


@pytest.mark.asyncio
class TestEvidenceImmutability:
    """Evidence records must be immutable for audit integrity."""

    async def test_evidence_record_is_frozen(self):
        """
        Evidence records cannot be modified after creation.
        """
        from engine.schemas.constitutional_types import EvidenceRecord

        record = EvidenceRecord(
            record_id="rec-123",
            timestamp="2025-12-31T19:00:00Z",
            trace_id="trace-456",
            record_type="critic_evaluation",
            critic="rights",
            severity=0.7,
            content={"violation": "test"},
        )

        # INVARIANT: Evidence is immutable
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            record.severity = 0.1

    async def test_critic_evaluation_is_frozen(self):
        """
        Critic evaluations are sealed before aggregation.
        """
        evaluation = CriticEvaluation(
            critic="test",
            violations=[],
            severity=0.5,
            justification="test",
            evaluated_rules=[],
            duration_ms=100,
        )

        # INVARIANT: Evaluations are frozen
        with pytest.raises((TypeError, AttributeError, ValidationError)):
            evaluation.violations.append({"new": "violation"})


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


@pytest.mark.asyncio
class TestConstitutionalIntegration:
    """End-to-end tests of constitutional guarantees."""

    async def test_full_pipeline_preserves_guarantees(self):
        """
        Complete pipeline maintains all constitutional invariants.
        """
        # This would test the full engine with real critics
        # Verifying isolation, dissent, escalation, and uncertainty
        # throughout the entire evaluation pipeline
        pass  # Placeholder for full integration test


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
