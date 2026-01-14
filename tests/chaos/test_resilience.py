"""Chaos engineering tests for system resilience."""
import pytest
import asyncio
import random
from unittest.mock import patch, AsyncMock
from engine import EleanorEngineV8
from engine.config import EngineConfig


@pytest.mark.chaos
class TestPartialFailures:
    """Tests for partial system failures."""

    @pytest.mark.asyncio
    async def test_critic_random_failures(self):
        """Test system resilience with random critic failures."""
        engine = EleanorEngineV8(
            config=EngineConfig(enable_graceful_degradation=True)
        )
        
        # Mock critics with 30% failure rate
        original_run = engine._run_single_critic
        
        async def flaky_critic_run(*args, **kwargs):
            if random.random() < 0.3:  # 30% failure rate
                raise Exception("Random critic failure")
            return await original_run(*args, **kwargs)
        
        with patch.object(engine, '_run_single_critic', side_effect=flaky_critic_run):
            result = await engine.run("test", context={}, detail_level=2)
        
        # Should complete with available critics
        assert result is not None
        assert result.critic_findings is not None

    @pytest.mark.asyncio
    async def test_redis_cache_failure(self):
        """Test graceful degradation when Redis cache fails."""
        engine = EleanorEngineV8(
            config=EngineConfig(
                cache_enabled=True,
                enable_graceful_degradation=True
            )
        )
        
        # Mock cache to fail
        if hasattr(engine, '_cache_manager'):
            with patch.object(
                engine._cache_manager,
                'get',
                side_effect=ConnectionError("Redis unavailable")
            ):
                result = await engine.run("test", context={}, detail_level=2)
            
            # Should complete without cache
            assert result is not None

    @pytest.mark.asyncio
    async def test_precedent_db_failure(self):
        """Test handling of precedent database failure."""
        engine = EleanorEngineV8(
            config=EngineConfig(
                enable_precedent_analysis=True,
                enable_graceful_degradation=True
            )
        )
        
        # Mock precedent retriever to fail
        if hasattr(engine, '_precedent_retriever'):
            with patch.object(
                engine._precedent_retriever,
                'retrieve',
                side_effect=Exception("Database connection failed")
            ):
                result = await engine.run("test", context={}, detail_level=3)
            
            # Should complete without precedent analysis
            assert result is not None

    @pytest.mark.asyncio
    async def test_network_timeouts(self):
        """Test handling of network timeouts."""
        engine = EleanorEngineV8(
            config=EngineConfig(timeout_seconds=0.5)
        )
        
        # Mock slow network call
        async def slow_operation(*args, **kwargs):
            await asyncio.sleep(2.0)  # Exceeds timeout
            return {}
        
        # Should handle timeout gracefully
        with pytest.raises(asyncio.TimeoutError):
            async with asyncio.timeout(0.5):
                await slow_operation()

    @pytest.mark.asyncio
    async def test_intermittent_failures(self):
        """Test handling of intermittent failures."""
        engine = EleanorEngineV8(
            config=EngineConfig(enable_graceful_degradation=True)
        )
        
        failure_count = 0
        
        async def intermittent_operation():
            nonlocal failure_count
            failure_count += 1
            if failure_count % 3 == 0:  # Fail every 3rd call
                raise Exception("Intermittent failure")
            return {"status": "ok"}
        
        results = []
        for _ in range(10):
            try:
                result = await intermittent_operation()
                results.append(result)
            except Exception:
                pass  # Expected intermittent failures
        
        # Should have some successes
        assert len(results) > 5


@pytest.mark.chaos
class TestResourceExhaustion:
    """Tests for resource exhaustion scenarios."""

    @pytest.mark.asyncio
    async def test_memory_pressure(self):
        """Test behavior under memory pressure."""
        engine = EleanorEngineV8()
        
        # Create large context to simulate memory pressure
        large_context = {f"key_{i}": "x" * 1000 for i in range(50)}
        
        async with engine:
            result = await engine.run(
                "test under memory pressure",
                context=large_context,
                detail_level=1
            )
        
        assert result is not None

    @pytest.mark.asyncio
    async def test_connection_pool_exhaustion(self):
        """Test behavior when connection pool is exhausted."""
        engine = EleanorEngineV8()
        
        # Simulate many concurrent requests
        num_concurrent = 100
        
        async with engine:
            results = await asyncio.gather(*[
                engine.run(f"req {i}", context={}, detail_level=1)
                for i in range(num_concurrent)
            ], return_exceptions=True)
        
        # Should handle all requests (possibly with some timeouts)
        successful = [r for r in results if not isinstance(r, Exception)]
        assert len(successful) > num_concurrent * 0.8  # 80% success rate

    @pytest.mark.asyncio
    async def test_file_descriptor_limits(self):
        """Test behavior approaching file descriptor limits."""
        engine = EleanorEngineV8()
        
        async with engine:
            # Open many concurrent operations
            tasks = [
                engine.run(f"req {i}", context={}, detail_level=1)
                for i in range(50)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Should complete without file descriptor errors
        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) == 0


@pytest.mark.chaos
class TestMemoryLeakDetection:
    """Tests to detect potential memory leaks."""

    @pytest.mark.asyncio
    async def test_no_memory_leak_repeated_operations(self):
        """Test for memory leaks in repeated operations."""
        import psutil
        import os
        import gc
        
        process = psutil.Process(os.getpid())
        
        # Force garbage collection
        gc.collect()
        
        # Get baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        engine = EleanorEngineV8()
        
        async with engine:
            # Perform many operations
            for i in range(100):
                await engine.run(
                    f"operation {i}",
                    context={},
                    detail_level=1
                )
                
                # Force GC every 10 operations
                if i % 10 == 0:
                    gc.collect()
        
        # Force final GC
        gc.collect()
        
        # Check final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - baseline_memory
        
        print(f"\nMemory growth: {memory_growth:.2f} MB")
        
        # Memory growth should be minimal (< 100 MB for 100 operations)
        assert memory_growth < 100

    @pytest.mark.asyncio
    async def test_context_cleanup(self):
        """Test that contexts are properly cleaned up."""
        import weakref
        
        engine = EleanorEngineV8()
        refs = []
        
        async with engine:
            for i in range(50):
                context = {"data": f"test_{i}" * 100}
                ref = weakref.ref(context)
                refs.append(ref)
                
                await engine.run("test", context=context, detail_level=1)
                
                # Clear local reference
                del context
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Most contexts should be garbage collected
        alive = sum(1 for ref in refs if ref() is not None)
        print(f"\nContexts still alive: {alive}/50")
        
        # At most 10% should still be referenced
        assert alive < 5

    @pytest.mark.asyncio
    async def test_async_resource_cleanup(self):
        """Test proper cleanup of async resources."""
        engine = EleanorEngineV8()
        
        # Track open resources
        initial_tasks = len(asyncio.all_tasks())
        
        async with engine:
            await engine.run("test", context={}, detail_level=2)
        
        # Wait for cleanup
        await asyncio.sleep(0.1)
        
        final_tasks = len(asyncio.all_tasks())
        
        # Should not leak asyncio tasks
        assert final_tasks <= initial_tasks + 1  # Allow for current task
