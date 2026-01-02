import asyncio
import os

import pytest


if not os.getenv("ELEANOR_RUN_BENCHMARKS"):
    pytest.skip("Benchmarks disabled by default", allow_module_level=True)


@pytest.mark.performance
def test_engine_run_benchmark(benchmark, engine):
    """Benchmark a minimal engine run using mock dependencies."""

    async def run_once():
        return await engine.run("hello", detail_level=1)

    def runner():
        return asyncio.run(run_once())

    benchmark(runner)
