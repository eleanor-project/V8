"""
Integration tests for GPU batching and precedent embedding cache wiring.
"""

from __future__ import annotations

import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from engine.engine import EleanorEngineV8, EngineConfig
from engine.factory import EngineDependencies
from engine.mocks import (
    MockAggregator,
    MockCritic,
    MockDetectorEngine,
    MockEvidenceRecorder,
    MockReviewTriggerEvaluator,
    MockRouter,
)
from engine.precedent.retrieval import PrecedentRetrievalV8


def _build_dependencies(*, critics, precedent_retriever=None) -> EngineDependencies:
    return EngineDependencies(
        router=MockRouter(),
        detector_engine=MockDetectorEngine(),
        evidence_recorder=MockEvidenceRecorder(),
        critics=critics,
        review_trigger_evaluator=MockReviewTriggerEvaluator(),
        precedent_engine=None,
        precedent_retriever=precedent_retriever,
        uncertainty_engine=None,
        aggregator=MockAggregator(),
        critic_models={},
    )


@pytest.mark.asyncio
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
async def test_engine_critic_batching_uses_batch_processor():
    from engine.gpu.batch_processor import BatchProcessor

    critics = {"alpha": MockCritic, "beta": MockCritic}
    deps = _build_dependencies(critics=critics)
    config = EngineConfig(enable_precedent_analysis=False, enable_reflection=False)
    engine = EleanorEngineV8(config=config, dependencies=deps)
    engine.critic_batcher = BatchProcessor(
        process_fn=engine._process_critic_batch,
        device=torch.device("cpu"),
        initial_batch_size=2,
        max_batch_size=2,
        min_batch_size=1,
        dynamic_sizing=False,
    )

    calls: dict[str, int] = {}
    original = engine.critic_batcher.process_batch

    async def wrapped(items):
        calls["count"] = len(items)
        return await original(items)

    engine.critic_batcher.process_batch = wrapped

    results = await engine._run_critics_parallel(
        "model response",
        context={},
        trace_id="trace-123",
        input_text="input",
    )

    assert calls["count"] == 2
    assert set(results.keys()) == {"alpha", "beta"}


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_engine_precedent_cache_hits():
    from engine.gpu.embeddings import GPUEmbeddingCache

    class DummyStore:
        def __init__(self):
            self.embeddings = []

        def search(self, query_text, top_k=5, embedding=None):
            self.embeddings.append(embedding)
            return [{"values": [], "aggregate_score": 0.5, "text": "case"}]

    embed_calls = {"count": 0}

    def embed_fn(text: str):
        embed_calls["count"] += 1
        return [0.1, 0.2]

    store = DummyStore()
    retriever = PrecedentRetrievalV8(store_client=store, embedding_fn=embed_fn)
    deps = _build_dependencies(critics={}, precedent_retriever=retriever)
    config = EngineConfig(enable_precedent_analysis=False, enable_reflection=False)

    engine = EleanorEngineV8(config=config, dependencies=deps)
    cache = GPUEmbeddingCache(device=torch.device("cpu"), max_cache_size=8, embedding_dim=2)
    engine.gpu_embedding_cache = cache
    engine.precedent_retriever.embedding_cache = cache

    assert engine.precedent_retriever.embedding_cache is cache

    engine.precedent_retriever.retrieve("query", [], top_k=1)
    engine.precedent_retriever.retrieve("query", [], top_k=1)

    assert embed_calls["count"] == 1
    assert store.embeddings[0] == [0.1, 0.2]
