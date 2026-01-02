from engine.cache.router_cache import RouterSelectionCache


def test_router_cache_exact_hit():
    cache = RouterSelectionCache(maxsize=10, ttl=60, similarity_threshold=0.9)
    selection = {"model_info": {"model_name": "test"}, "response_text": "ok"}

    cache.set("Hello world", {"user": "a"}, selection)
    assert cache.get("Hello world", {"user": "a"}) == selection
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 0


def test_router_cache_similarity_hit():
    cache = RouterSelectionCache(maxsize=10, ttl=60, similarity_threshold=0.8)
    selection = {"model_info": {"model_name": "test"}, "response_text": "ok"}

    cache.set("policy update", {"tier": "a"}, selection)
    assert cache.get_similar("policy updates", {"tier": "a"}) == selection
    stats = cache.stats()
    assert stats["similarity_hits"] == 1
