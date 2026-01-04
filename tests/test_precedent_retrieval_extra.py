import sys
from types import SimpleNamespace

from engine.precedent.retrieval import PrecedentRetrievalV8


def test_retrieval_no_results():
    store = SimpleNamespace(search=lambda *_args, **_kwargs: [])
    retrieval = PrecedentRetrievalV8(store_client=store)
    result = retrieval.retrieve("query", [])
    assert result["precedent_cases"] == []
    assert result["alignment_score"] == 1.0


def test_retrieval_with_results_and_embedding():
    store = SimpleNamespace(
        search=lambda *_args, **_kwargs: [
            {"values": ["v1"], "aggregate_score": 0.4, "embedding": [0.1]}
        ]
    )
    retrieval = PrecedentRetrievalV8(store_client=store, embedding_fn=lambda _t: [0.2])
    result = retrieval.retrieve("query", [{"value": "v1", "score": 0.4}])
    assert result["top_case"]
    assert result["query_embedding"]


def test_search_store_embedding_param_fallback():
    class Store:
        def __init__(self):
            self.calls = 0

        def search(self, query_text, top_k=5):
            self.calls += 1
            return [{"embedding": [0.3], "aggregate_score": 0.1}]

    store = Store()
    retrieval = PrecedentRetrievalV8(store_client=store)
    result = retrieval._search_store("query", 5, [0.1])
    assert result
    assert store.calls == 1


def test_cached_embedding_handling():
    class Cache:
        def __init__(self, value):
            self.value = value

        def get_cached_embedding(self, _text):
            return self.value

    class TensorLike:
        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return [0.1, 0.2]

    retrieval = PrecedentRetrievalV8(store_client=SimpleNamespace(search=lambda *_a, **_k: []))
    retrieval.embedding_cache = Cache(TensorLike())
    assert retrieval._get_cached_embedding("query") == [0.1, 0.2]

    retrieval.embedding_cache = Cache(None)
    assert retrieval._get_cached_embedding("query") == []


def test_cache_embedding_uses_torch(monkeypatch):
    class Cache:
        def __init__(self):
            self.called = False

        def cache_embedding(self, text, tensor):
            self.called = True

    cache = Cache()
    retrieval = PrecedentRetrievalV8(store_client=SimpleNamespace(search=lambda *_a, **_k: []))
    retrieval.embedding_cache = cache

    torch_stub = SimpleNamespace(tensor=lambda embedding, dtype=None: embedding, float32=object())
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    retrieval._cache_embedding("query", [0.1])
    assert cache.called is True
