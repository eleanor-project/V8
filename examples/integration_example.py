"""
ELEANOR V8 - Integration Example

Demonstrates caching, observability, and resilience features.
"""

import asyncio
from engine.cache import CacheManager, AdaptiveConcurrencyManager, CacheKey
from engine.observability import configure_logging, configure_tracing, get_logger, TraceContext
from engine.resilience import CircuitBreaker, DegradationStrategy, ComponentHealthChecker


# Configure observability
configure_logging(
    log_level="INFO",
    json_logs=False,  # Use colored console for demo
    development_mode=True,
)

configure_tracing(
    service_name="eleanor-demo",
    jaeger_endpoint="localhost:6831",
    enabled=False,  # Set to True if Jaeger is running
)

logger = get_logger(__name__)


# Example: Cached operation
async def expensive_operation(query: str) -> dict:
    """Simulate expensive operation."""
    await asyncio.sleep(0.5)  # Simulate latency
    return {"result": f"processed_{query}", "cached": False}


async def demo_caching():
    """Demonstrate caching."""
    logger.info("=== Caching Demo ===")

    cache = CacheManager()

    # First call (miss)
    key1 = CacheKey.from_data("demo", query="test1")
    result1 = await cache.get_or_compute(key1, expensive_operation, query="test1")
    logger.info("First call", result=result1)

    # Second call (hit)
    result2 = await cache.get_or_compute(key1, expensive_operation, query="test1")
    logger.info("Second call (cached)", result=result2)

    # Stats
    stats = cache.get_stats()
    logger.info("Cache stats", stats=stats)


async def demo_circuit_breaker():
    """Demonstrate circuit breaker."""
    logger.info("=== Circuit Breaker Demo ===")

    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=5)

    # Simulate failures
    for i in range(5):
        try:
            result = await breaker.call(
                lambda: 1 / 0  # Will fail
            )
        except Exception as e:
            logger.warning(f"Attempt {i+1} failed", error=str(e), state=breaker.state.value)

    # Circuit should be open now
    logger.info("Circuit state", state=breaker.get_state())

    # Try with fallback
    try:
        result = await breaker.call(lambda: "success")
    except Exception:
        logger.info("Circuit open, using fallback")
        result = await DegradationStrategy.router_fallback(Exception("Circuit open"))
        logger.info("Fallback result", result=result)


async def demo_adaptive_concurrency():
    """Demonstrate adaptive concurrency."""
    logger.info("=== Adaptive Concurrency Demo ===")

    concurrency = AdaptiveConcurrencyManager(initial_limit=3, target_latency_ms=100)

    # Simulate operations with varying latency
    for i in range(50):
        async with concurrency:
            # Simulate work
            latency = 50 if i < 25 else 200  # Increase latency halfway
            await asyncio.sleep(latency / 1000)

        concurrency.record_latency(latency)

        if i % 10 == 0:
            stats = concurrency.get_stats()
            logger.info(
                f"Iteration {i}",
                current_limit=stats["current_limit"],
                p95_latency=stats.get("p95_latency_ms", 0),
            )


async def demo_health_monitoring():
    """Demonstrate health monitoring."""
    logger.info("=== Health Monitoring Demo ===")

    # Create breakers
    breakers = {
        "router": CircuitBreaker(),
        "precedent": CircuitBreaker(),
        "uncertainty": CircuitBreaker(),
    }

    health_checker = ComponentHealthChecker(breakers)

    # Initial health
    status = health_checker.get_health_status()
    logger.info("Initial health", status=status)

    # Simulate precedent failure
    for _ in range(5):
        try:
            await breakers["precedent"].call(lambda: 1 / 0)
        except Exception:
            pass

    # Check health again
    status = health_checker.get_health_status()
    logger.info("After precedent failures", status=status)

    unhealthy = health_checker.get_unhealthy_components()
    logger.info("Unhealthy components", components=unhealthy)


async def main():
    """Run all demos."""
    TraceContext.set_trace_id("demo-trace-123")

    await demo_caching()
    print()

    await demo_circuit_breaker()
    print()

    await demo_adaptive_concurrency()
    print()

    await demo_health_monitoring()


if __name__ == "__main__":
    asyncio.run(main())
