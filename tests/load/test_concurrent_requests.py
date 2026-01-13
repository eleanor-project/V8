"""Load tests for concurrent request handling."""
import pytest
import asyncio
from engine import EleanorEngineV8
from engine.config import EngineConfig


@pytest.mark.load
class TestConcurrentRequestHandling:
    """Tests for concurrent request processing."""

    @pytest.mark.asyncio
    async def test_concurrent_request_handling_50(self):
        """Test engine handles 50 concurrent requests correctly."""
        engine = EleanorEngineV8()
        num_requests = 50
        
        async def make_request(request_id):
            return await engine.run(
                f"Test request {request_id}",
                context={"request_id": request_id},
                detail_level=1
            )
        
        async with engine:
            results = await asyncio.gather(*[
                make_request(i) for i in range(num_requests)
            ])
        
        assert len(results) == num_requests
        assert all(r is not None for r in results)
        assert all(not r.is_degraded for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_high_detail_requests(self):
        """Test concurrent requests with high detail level."""
        engine = EleanorEngineV8(
            config=EngineConfig(
                enable_precedent_analysis=True,
                enable_uncertainty=True
            )
        )
        num_requests = 20
        
        async def make_request(i):
            return await engine.run(
                f"Complex request {i}",
                context={"complexity": "high"},
                detail_level=3
            )
        
        async with engine:
            results = await asyncio.gather(*[
                make_request(i) for i in range(num_requests)
            ])
        
        assert len(results) == num_requests
        # All should complete successfully
        assert all(r.model_info is not None for r in results)

    @pytest.mark.asyncio
    async def test_request_isolation(self):
        """Test that concurrent requests are properly isolated."""
        engine = EleanorEngineV8()
        
        async def make_request_with_context(request_id, value):
            result = await engine.run(
                "test",
                context={"request_id": request_id, "value": value},
                detail_level=1
            )
            return result
        
        async with engine:
            results = await asyncio.gather(*[
                make_request_with_context(i, i * 100)
                for i in range(30)
            ])
        
        # Verify each result has correct context
        for i, result in enumerate(results):
            assert result.context["request_id"] == i
            assert result.context["value"] == i * 100

    @pytest.mark.asyncio
    async def test_resource_cleanup_under_load(self):
        """Test proper resource cleanup under concurrent load."""
        engine = EleanorEngineV8()
        
        async with engine:
            # Make multiple batches of concurrent requests
            for batch in range(5):
                requests = [
                    engine.run(f"batch {batch} request {i}", context={}, detail_level=1)
                    for i in range(10)
                ]
                await asyncio.gather(*requests)
        
        # Engine should clean up properly
        assert engine._is_closed

    @pytest.mark.asyncio
    async def test_throughput_measurement(self):
        """Measure throughput under concurrent load."""
        engine = EleanorEngineV8()
        num_requests = 100
        
        import time
        start_time = time.time()
        
        async with engine:
            await asyncio.gather(*[
                engine.run(f"req {i}", context={}, detail_level=1)
                for i in range(num_requests)
            ])
        
        elapsed = time.time() - start_time
        throughput = num_requests / elapsed
        
        print(f"\nThroughput: {throughput:.2f} requests/sec")
        # Should handle at least 10 req/sec
        assert throughput > 10.0


@pytest.mark.load
@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    def test_validation_performance(self, benchmark):
        """Benchmark validation performance."""
        from engine.validation import validate_input
        
        text = "Sample input text for validation " * 50
        context = {"domain": "test", "key": "value"}
        
        result = benchmark(validate_input, text, context)
        assert result is not None

    def test_critic_evaluation_performance(self, benchmark):
        """Benchmark single critic evaluation."""
        from engine.critics.truth_critic import TruthCritic
        
        critic = TruthCritic()
        model_response = "Test model output " * 20
        input_text = "Test input"
        context = {}
        
        async def evaluate():
            return await critic.evaluate(model_response, input_text, context)
        
        result = benchmark(asyncio.run, evaluate())
        assert result is not None

    @pytest.mark.asyncio
    async def test_end_to_end_latency(self):
        """Test end-to-end latency for typical request."""
        engine = EleanorEngineV8()
        
        import time
        latencies = []
        
        async with engine:
            for i in range(10):
                start = time.time()
                await engine.run(
                    "Test request for latency measurement",
                    context={},
                    detail_level=2
                )
                latencies.append(time.time() - start)
        
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
        
        print(f"\nAvg latency: {avg_latency*1000:.2f}ms")
        print(f"P95 latency: {p95_latency*1000:.2f}ms")
        
        # P95 should be under 5 seconds
        assert p95_latency < 5.0

    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self):
        """Test memory usage remains stable under load."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        engine = EleanorEngineV8()
        
        async with engine:
            # Process 100 requests
            for batch in range(10):
                await asyncio.gather(*[
                    engine.run(f"req {i}", context={}, detail_level=1)
                    for i in range(10)
                ])
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"\nMemory increase: {memory_increase:.2f} MB")
        
        # Memory increase should be reasonable (< 500 MB)
        assert memory_increase < 500
