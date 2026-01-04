import builtins
from types import SimpleNamespace

import pytest

import engine.precedent.retrieval as retrieval_module
from engine.precedent.retrieval import PrecedentRetrievalV8


def test_retrieval_embedding_fn_error():
    def _fail(_text):
        raise RuntimeError("boom")

    store = SimpleNamespace(search=lambda *_args, **_kwargs: [])
    retrieval = PrecedentRetrievalV8(store_client=store, embedding_fn=_fail)
    result = retrieval.retrieve("query", [])
    assert result["query_embedding"] == []


def test_cached_embedding_error_handling():
    class BadCache:
        def get_cached_embedding(self, _text):
            raise RuntimeError("boom")

    retrieval = PrecedentRetrievalV8(store_client=SimpleNamespace(search=lambda *_a, **_k: []))
    retrieval.embedding_cache = BadCache()
    assert retrieval._get_cached_embedding("query") == []


def test_cached_embedding_not_callable():
    retrieval = PrecedentRetrievalV8(store_client=SimpleNamespace(search=lambda *_a, **_k: []))
    retrieval.embedding_cache = SimpleNamespace(get_cached_embedding=object())
    assert retrieval._get_cached_embedding("query") == []


def test_cached_embedding_tolist_error():
    class BadTensor:
        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            raise RuntimeError("boom")

    retrieval = PrecedentRetrievalV8(store_client=SimpleNamespace(search=lambda *_a, **_k: []))
    retrieval.embedding_cache = SimpleNamespace(get_cached_embedding=lambda _t: BadTensor())
    assert retrieval._get_cached_embedding("query") == []


def test_cached_embedding_tolist_non_list():
    class TupleTensor:
        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return (1.0, 2.0)

    retrieval = PrecedentRetrievalV8(store_client=SimpleNamespace(search=lambda *_a, **_k: []))
    retrieval.embedding_cache = SimpleNamespace(get_cached_embedding=lambda _t: TupleTensor())
    assert retrieval._get_cached_embedding("query") == []


def test_cache_embedding_missing_torch(monkeypatch):
    class Cache:
        def __init__(self):
            self.called = False

        def cache_embedding(self, _text, _tensor):
            self.called = True

    cache = Cache()
    retrieval = PrecedentRetrievalV8(store_client=SimpleNamespace(search=lambda *_a, **_k: []))
    retrieval.embedding_cache = cache

    original_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("nope")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    retrieval._cache_embedding("query", [0.1])
    assert cache.called is False


def test_cache_embedding_not_callable():
    retrieval = PrecedentRetrievalV8(store_client=SimpleNamespace(search=lambda *_a, **_k: []))
    retrieval.embedding_cache = SimpleNamespace(cache_embedding=None)
    assert retrieval._cache_embedding("query", [0.1]) is None


def test_cache_embedding_cache_fn_error(monkeypatch):
    class Cache:
        def cache_embedding(self, _text, _tensor):
            raise RuntimeError("boom")

    cache = Cache()
    retrieval = PrecedentRetrievalV8(store_client=SimpleNamespace(search=lambda *_a, **_k: []))
    retrieval.embedding_cache = cache

    torch_stub = SimpleNamespace(tensor=lambda embedding, dtype=None: embedding, float32=object())
    monkeypatch.setitem(__import__("sys").modules, "torch", torch_stub)
    retrieval._cache_embedding("query", [0.1])


def test_search_store_missing_search():
    store = SimpleNamespace()
    retrieval = PrecedentRetrievalV8(store_client=store)
    assert retrieval._search_store("query", 5, [0.1]) == []


def test_search_store_signature_error(monkeypatch):
    store = SimpleNamespace(search=lambda *_a, **_k: [{"embedding": [0.1]}])
    retrieval = PrecedentRetrievalV8(store_client=store)

    def _raise(*_args, **_kwargs):
        raise TypeError("boom")

    monkeypatch.setattr(retrieval_module.inspect, "signature", _raise)
    result = retrieval._search_store("query", 5, [0.1])
    assert result


def test_retrieval_uses_candidate_embedding():
    store = SimpleNamespace(search=lambda *_a, **_k: [{"embedding": [0.2], "aggregate_score": 0.1}])
    retrieval = PrecedentRetrievalV8(store_client=store, embedding_fn=lambda _t: [])
    result = retrieval.retrieve("query", [{"score": 0.1}])
    assert result["query_embedding"] == [0.2]


@pytest.mark.asyncio
async def test_retrieval_close_awaitable():
    class Store:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    store = Store()
    retrieval = PrecedentRetrievalV8(store_client=store)
    await retrieval.close()
    assert store.closed is True
