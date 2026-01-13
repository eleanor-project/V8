"""Comprehensive tests for engine error handling and degradation paths."""
import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

from engine.engine import EleanorEngineV8
from engine.config import EngineConfig
from engine.exceptions import (
    ValidationError,
    RouterSelectionError,
    CriticEvaluationError,
    CircuitBreakerOpenError
)


class TestEngineErrorPaths:
    """Test engine error handling paths."""

    @pytest.mark.asyncio
    async def test_router_failure_with_degradation_enabled(self):
        """Test graceful degradation when router fails."""
        config = EngineConfig(enable_graceful_degradation=True)
        engine = EleanorEngineV8(config=config)
        
        # Mock router to fail
        with patch.object(
            engine,
            '_select_router',
            side_effect=RouterSelectionError("Router service unavailable")
        ):
            result = await engine.run(
                "test input",
                context={},
                detail_level=2
            )
            
            assert result is not None
            assert result.is_degraded
            assert "router" in result.degraded_components

    @pytest.mark.asyncio
    async def test_router_failure_without_degradation_raises(self):
        """Test that router failure raises when degradation disabled."""
        config = EngineConfig(enable_graceful_degradation=False)
        engine = EleanorEngineV8(config=config)
        
        with patch.object(
            engine,
            '_select_router',
            side_effect=RouterSelectionError("Router failed")
        ):
            with pytest.raises(RouterSelectionError):
                await engine.run("test input", context={}, detail_level=2)

    @pytest.mark.asyncio
    async def test_critic_timeout_handling(self):
        """Test critic execution timeout handling."""
        config = EngineConfig(timeout_seconds=0.1)
        engine = EleanorEngineV8(config=config)
        
        # Mock slow critic
        async def slow_critic_evaluate(*args, **kwargs):
            await asyncio.sleep(5)  # Exceeds timeout
            return {"violations": []}
        
        mock_critic = MagicMock()
        mock_critic.evaluate = slow_critic_evaluate
        
        with patch.object(engine, '_critics', {"slow_critic": mock_critic}):
            with pytest.raises(asyncio.TimeoutError):
                await engine._run_single_critic(
                    name="slow_critic",
                    critic_ref=mock_critic,
                    model_response="test",
                    input_text="test",
                    context={},
                    trace_id="test-123"
                )

    @pytest.mark.asyncio
    async def test_critic_exception_handling(self):
        """Test critic exception handling."""
        engine = EleanorEngineV8()
        
        # Mock critic that raises exception
        async def failing_critic_evaluate(*args, **kwargs):
            raise ValueError("Critic internal error")
        
        mock_critic = MagicMock()
        mock_critic.evaluate = failing_critic_evaluate
        
        with patch.object(engine, '_critics', {"failing_critic": mock_critic}):
            result = await engine._run_single_critic(
                name="failing_critic",
                critic_ref=mock_critic,
                model_response="test",
                input_text="test",
                context={},
                trace_id="test-456"
            )
            
            # Should return error result instead of raising
            assert "error" in result or result.get("status") == "error"

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_state(self):
        """Test behavior when circuit breaker is open."""
        engine = EleanorEngineV8()
        
        # Mock circuit breaker in open state
        with patch.object(
            engine._circuit_breakers.get("router", MagicMock()),
            'is_open',
            return_value=True
        ):
            with pytest.raises(CircuitBreakerOpenError):
                await engine._select_router("test input")

    @pytest.mark.asyncio
    async def test_partial_critic_failures(self):
        """Test engine continues with partial critic failures."""
        engine = EleanorEngineV8()
        
        # Mock some critics to fail
        good_critic = MagicMock()
        good_critic.evaluate = AsyncMock(return_value={"violations": []})
        
        bad_critic = MagicMock()
        bad_critic.evaluate = AsyncMock(side_effect=ValueError("Critic failed"))
        
        with patch.object(engine, '_critics', {
            "good_critic_1": good_critic,
            "bad_critic_1": bad_critic,
            "good_critic_2": good_critic,
        }):
            result = await engine.run("test", context={}, detail_level=2)
            
            # Should complete with available critics
            assert result is not None
            assert result.critic_findings is not None

    @pytest.mark.asyncio
    async def test_precedent_retrieval_failure(self):
        """Test handling of precedent retrieval failure."""
        config = EngineConfig(
            enable_precedent_analysis=True,
            enable_graceful_degradation=True
        )
        engine = EleanorEngineV8(config=config)
        
        # Mock precedent retriever to fail
        with patch.object(
            engine._precedent_retriever,
            'retrieve',
            side_effect=Exception("Precedent DB unavailable")
        ):
            result = await engine.run("test", context={}, detail_level=3)
            
            # Should complete without precedent analysis
            assert result is not None
            assert result.precedent_alignment is None or result.is_degraded

    @pytest.mark.asyncio
    async def test_uncertainty_computation_failure(self):
        """Test handling of uncertainty computation failure."""
        config = EngineConfig(
            enable_uncertainty=True,
            enable_graceful_degradation=True
        )
        engine = EleanorEngineV8(config=config)
        
        # Mock uncertainty engine to fail
        with patch.object(
            engine._uncertainty_engine,
            'compute',
            side_effect=Exception("Uncertainty computation error")
        ):
            result = await engine.run("test", context={}, detail_level=2)
            
            # Should complete without uncertainty
            assert result is not None
            assert result.uncertainty is None or result.is_degraded

    @pytest.mark.asyncio
    async def test_aggregation_fallback(self):
        """Test aggregation fallback on failure."""
        engine = EleanorEngineV8()
        
        # Mock aggregator to fail
        with patch.object(
            engine._aggregator,
            'aggregate',
            side_effect=Exception("Aggregation failed")
        ):
            result = await engine.run("test", context={}, detail_level=2)
            
            # Should use fallback aggregation
            assert result is not None
            assert result.aggregated is not None  # Fallback result

    @pytest.mark.asyncio
    async def test_governance_evaluation_failure(self):
        """Test governance evaluation failure handling."""
        config = EngineConfig(enable_governance=True)
        engine = EleanorEngineV8(config=config)
        
        # Mock governance to fail
        with patch.object(
            engine._governance,
            'should_escalate',
            side_effect=Exception("Governance service error")
        ):
            result = await engine.run("test", context={}, detail_level=3)
            
            # Should complete without governance check
            assert result is not None

    @pytest.mark.asyncio
    async def test_validation_error_emission(self):
        """Test validation error emission."""
        engine = EleanorEngineV8()
        
        # Mock validation to fail
        with patch(
            'engine.validation.validate_input',
            side_effect=ValidationError("Invalid input")
        ):
            with pytest.raises(ValidationError) as exc_info:
                await engine.run("", context={}, detail_level=1)
            
            assert "Invalid input" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_evidence_recording_failure_doesnt_block(self):
        """Test that evidence recording failure doesn't block execution."""
        engine = EleanorEngineV8()
        
        # Mock evidence recorder to fail
        with patch.object(
            engine._evidence_recorder,
            'record',
            side_effect=Exception("Evidence recording failed")
        ):
            # Should still complete successfully
            result = await engine.run("test", context={}, detail_level=1)
            assert result is not None


class TestCircuitBreakerStateMachine:
    """Test circuit breaker state transitions."""

    @pytest.mark.asyncio
    async def test_closed_to_open_transition(self):
        """Test circuit breaker transitions from closed to open."""
        from engine.utils.circuit_breaker import CircuitBreaker
        
        cb = CircuitBreaker(failure_threshold=3, timeout=1.0)
        
        # Trigger failures
        for _ in range(3):
            cb.record_failure()
        
        assert cb.is_open()

    @pytest.mark.asyncio
    async def test_open_to_half_open_transition(self):
        """Test circuit breaker transitions from open to half-open."""
        from engine.utils.circuit_breaker import CircuitBreaker
        
        cb = CircuitBreaker(failure_threshold=2, timeout=0.1)
        
        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open()
        
        # Wait for timeout
        await asyncio.sleep(0.2)
        
        # Should be half-open now
        assert cb.is_half_open()

    @pytest.mark.asyncio
    async def test_half_open_to_closed_on_success(self):
        """Test circuit breaker closes from half-open on success."""
        from engine.utils.circuit_breaker import CircuitBreaker
        
        cb = CircuitBreaker(failure_threshold=2, timeout=0.1, success_threshold=2)
        
        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        
        # Wait for half-open
        await asyncio.sleep(0.2)
        
        # Record successes
        cb.record_success()
        cb.record_success()
        
        assert cb.is_closed()

    @pytest.mark.asyncio
    async def test_half_open_to_open_on_failure(self):
        """Test circuit breaker reopens from half-open on failure."""
        from engine.utils.circuit_breaker import CircuitBreaker
        
        cb = CircuitBreaker(failure_threshold=2, timeout=0.1)
        
        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        
        # Wait for half-open
        await asyncio.sleep(0.2)
        
        # Record failure in half-open state
        cb.record_failure()
        
        assert cb.is_open()
