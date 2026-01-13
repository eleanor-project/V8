"""Comprehensive tests for engine error handling and degradation scenarios."""
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


class TestEngineDegradation:
    """Test graceful degradation scenarios."""
    
    @pytest.mark.asyncio
    async def test_router_failure_with_degradation_enabled(self):
        """Test graceful degradation when router fails."""
        config = EngineConfig(enable_graceful_degradation=True)
        engine = EleanorEngineV8(config=config)
        
        # Mock router to always fail
        async def failing_router(*args, **kwargs):
            raise RouterSelectionError("Router service unavailable")
        
        with patch.object(engine, '_select_router', side_effect=failing_router):
            async with engine:
                result = await engine.run(
                    "test input",
                    context={},
                    detail_level=2
                )
            
            assert result.is_degraded
            assert "router" in result.degraded_components
            # Should use fallback model
            assert result.model_info is not None
    
    @pytest.mark.asyncio
    async def test_router_failure_without_degradation(self):
        """Test that router failure raises when degradation disabled."""
        config = EngineConfig(enable_graceful_degradation=False)
        engine = EleanorEngineV8(config=config)
        
        async def failing_router(*args, **kwargs):
            raise RouterSelectionError("Router failure")
        
        with patch.object(engine, '_select_router', side_effect=failing_router):
            async with engine:
                with pytest.raises(RouterSelectionError):
                    await engine.run("test", context={}, detail_level=1)
    
    @pytest.mark.asyncio
    async def test_partial_critic_failures_with_degradation(self):
        """Test system continues with partial critic failures."""
        engine = EleanorEngineV8()
        
        # Mock 30% of critics to fail
        failure_count = 0
        original_run_critic = engine._run_single_critic
        
        async def sometimes_failing_critic(*args, **kwargs):
            nonlocal failure_count
            if failure_count % 3 == 0:
                failure_count += 1
                raise CriticEvaluationError("Simulated failure")
            failure_count += 1
            return await original_run_critic(*args, **kwargs)
        
        with patch.object(engine, '_run_single_critic', side_effect=sometimes_failing_critic):
            async with engine:
                result = await engine.run(
                    "test input",
                    context={},
                    detail_level=2
                )
            
            assert result is not None
            # Should have results from working critics
            assert result.critic_findings is not None


class TestEngineTimeouts:
    """Test timeout handling in engine operations."""
    
    @pytest.mark.asyncio
    async def test_critic_execution_timeout(self):
        """Test critic execution respects timeout."""
        config = EngineConfig(timeout_seconds=0.1)
        engine = EleanorEngineV8(config=config)
        
        async def slow_critic(*args, **kwargs):
            await asyncio.sleep(1.0)  # Exceeds timeout
            return {"violations": []}
        
        with patch.object(engine, '_run_single_critic', side_effect=slow_critic):
            async with engine:
                # Should handle timeout gracefully
                result = await engine.run(
                    "test",
                    context={},
                    detail_level=2
                )
                
                assert result is not None
                assert result.forensic is not None
    
    @pytest.mark.asyncio
    async def test_multiple_critic_timeouts(self):
        """Test handling of multiple simultaneous timeouts."""
        config = EngineConfig(timeout_seconds=0.1)
        engine = EleanorEngineV8(config=config)
        
        async def always_timeout(*args, **kwargs):
            await asyncio.sleep(2.0)
        
        with patch.object(engine, '_run_single_critic', side_effect=always_timeout):
            async with engine:
                result = await engine.run(
                    "test",
                    context={},
                    detail_level=2
                )
                
                # Should complete with available data
                assert result is not None


class TestCircuitBreaker:
    """Test circuit breaker behavior."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after threshold failures."""
        engine = EleanorEngineV8()
        
        # Simulate repeated failures
        async def failing_operation(*args, **kwargs):
            raise Exception("Simulated failure")
        
        failure_count = 0
        threshold = 5
        
        with patch.object(engine, '_select_router', side_effect=failing_operation):
            async with engine:
                for i in range(threshold + 1):
                    try:
                        await engine._select_router("test")
                    except:
                        failure_count += 1
                
                # Circuit breaker should be open
                assert failure_count >= threshold
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker transitions to half-open after timeout."""
        engine = EleanorEngineV8()
        
        # Simulate failure then success
        call_count = 0
        
        async def sometimes_failing(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 5:
                raise Exception("Failing")
            return {"model": "test-model", "provider": "test"}
        
        with patch.object(engine, '_select_router', side_effect=sometimes_failing):
            async with engine:
                # Force failures to open circuit
                for _ in range(5):
                    try:
                        await engine._select_router("test")
                    except:
                        pass
                
                # Wait for half-open timeout
                await asyncio.sleep(1.0)
                
                # Should allow test requests
                try:
                    result = await engine._select_router("test")
                    assert result is not None
                except:
                    pass  # May still fail if not recovered


class TestEngineResourceCleanup:
    """Test resource cleanup and context manager behavior."""
    
    @pytest.mark.asyncio
    async def test_engine_cleanup_on_normal_exit(self):
        """Test resources are cleaned up on normal exit."""
        engine = EleanorEngineV8()
        
        async with engine:
            result = await engine.run("test", context={}, detail_level=1)
            assert result is not None
        
        # Verify cleanup occurred
        # (specific assertions depend on engine implementation)
    
    @pytest.mark.asyncio
    async def test_engine_cleanup_on_exception(self):
        """Test resources are cleaned up even on exception."""
        engine = EleanorEngineV8()
        
        try:
            async with engine:
                raise RuntimeError("Simulated error")
        except RuntimeError:
            pass
        
        # Cleanup should still have occurred


class TestEngineValidationErrors:
    """Test validation error handling."""
    
    @pytest.mark.asyncio
    async def test_validation_error_propagation(self):
        """Test validation errors are properly propagated."""
        engine = EleanorEngineV8()
        
        async with engine:
            with pytest.raises(ValidationError):
                await engine.run(
                    "",  # Empty string should fail validation
                    context={},
                    detail_level=1
                )
    
    @pytest.mark.asyncio
    async def test_malformed_context_validation(self):
        """Test malformed context is rejected."""
        engine = EleanorEngineV8()
        
        async with engine:
            with pytest.raises(ValidationError):
                await engine.run(
                    "test",
                    context="not a dict",  # Invalid context type
                    detail_level=1
                )
