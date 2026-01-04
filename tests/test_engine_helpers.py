import builtins
import sys
import types

import pytest

from engine import engine as engine_module


def test_resolve_router_backend_variants(monkeypatch):
    sentinel = object()

    def fake_create_router(**kwargs):
        return ("router", kwargs)

    monkeypatch.setattr(engine_module.DependencyFactory, "create_router", fake_create_router)

    assert engine_module._resolve_router_backend(None)[0] == "router"
    assert engine_module._resolve_router_backend("local")[1]["backend"] == "local"

    class DummyRouter:
        pass

    resolved = engine_module._resolve_router_backend(DummyRouter)
    assert isinstance(resolved, DummyRouter)

    resolved = engine_module._resolve_router_backend(lambda: sentinel)
    assert resolved is sentinel

    obj = object()
    assert engine_module._resolve_router_backend(obj) is obj


def test_init_cache_redis_import_error(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "redis":
            raise ImportError("no redis")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert engine_module._init_cache_redis("redis://localhost") is None


def test_init_cache_redis_success(monkeypatch):
    stub = types.SimpleNamespace(Redis=types.SimpleNamespace(from_url=lambda *_a, **_k: "client"))
    monkeypatch.setitem(sys.modules, "redis", stub)
    assert engine_module._init_cache_redis("redis://localhost") == "client"


def test_estimate_embedding_cache_entries_and_parse_memory():
    assert engine_module._estimate_embedding_cache_entries(0, 10) == 0
    assert engine_module._estimate_embedding_cache_entries(1, 2, bytes_per_value=4) > 0

    assert engine_module._parse_memory_gb(None) is None
    assert engine_module._parse_memory_gb("  ") is None
    assert engine_module._parse_memory_gb("24GB") == 24.0
    assert engine_module._parse_memory_gb("512mb") == pytest.approx(0.5, rel=1e-3)
    assert engine_module._parse_memory_gb("bad") is None
    assert engine_module._parse_memory_gb("badmb") is None
    assert engine_module._parse_memory_gb(8) == 8.0
