import sys
from types import SimpleNamespace

import pytest

from engine.precedent import stores as stores_module


def test_embedder():
    embedder = stores_module.Embedder()
    assert embedder.embed("text") == []

    embedder = stores_module.Embedder(lambda text: [float(len(text))])
    assert embedder.embed("hi") == [2.0]


def test_weaviate_store_missing_embedding(monkeypatch):
    monkeypatch.setattr(stores_module, "weaviate", None)
    store = stores_module.WeaviatePrecedentStore(client=SimpleNamespace(query=None))
    assert store.search("query") == []


def test_weaviate_store_search_success(monkeypatch):
    class DummyQuery:
        def __init__(self):
            self._result = {
                "data": {"Get": {"Precedent": [{"text": "doc", "metadata": {"k": 1}}]}}
            }

        def get(self, *_args, **_kwargs):
            return self

        def with_near_vector(self, *_args, **_kwargs):
            return self

        def with_limit(self, *_args, **_kwargs):
            return self

        def do(self):
            return self._result

    client = SimpleNamespace(query=DummyQuery())
    store = stores_module.WeaviatePrecedentStore(client=client, class_name="Precedent")
    results = store.search("query", embedding=[0.1])
    assert results[0]["text"] == "doc"


def test_weaviate_store_empty_result():
    class DummyQuery:
        def __init__(self):
            self._result = {"data": {"Get": {"Precedent": []}}}

        def get(self, *_args, **_kwargs):
            return self

        def with_near_vector(self, *_args, **_kwargs):
            return self

        def with_limit(self, *_args, **_kwargs):
            return self

        def do(self):
            return self._result

    client = SimpleNamespace(query=DummyQuery())
    store = stores_module.WeaviatePrecedentStore(client=client, class_name="Precedent")
    assert store.search("query", embedding=[0.1]) == []


def test_pgvector_store_validation_and_search(monkeypatch):
    with pytest.raises(ValueError):
        stores_module.PGVectorPrecedentStore(conn_string="pg", table_name="bad-name")

    class DummyCursor:
        def __init__(self, rows=None):
            self._rows = rows or [("text", "{}")]

        def execute(self, *_args, **_kwargs):
            return None

        def fetchall(self):
            return self._rows

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyConn:
        def __init__(self):
            self.closed = False

        def cursor(self):
            return DummyCursor()

        def close(self):
            self.closed = True

    psycopg2_stub = SimpleNamespace(connect=lambda _conn: DummyConn())
    monkeypatch.setattr(stores_module, "psycopg2", psycopg2_stub)
    monkeypatch.setattr(stores_module, "ThreadedConnectionPool", None)
    monkeypatch.setattr(stores_module, "sql", None)

    store = stores_module.PGVectorPrecedentStore(conn_string="pg", table_name="precedent")
    assert store.search("query", embedding=[0.1])

    store.pool = None
    store.conn = DummyConn()
    results = store.search("query", embedding=[0.1])
    assert results[0]["text"] == "text"

    store.close()


def test_pgvector_store_pool(monkeypatch):
    class DummyCursor:
        def __init__(self):
            self._rows = [("text", "{}")]

        def execute(self, *_args, **_kwargs):
            return None

        def fetchall(self):
            return self._rows

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyConn:
        def cursor(self):
            return DummyCursor()

    class DummyPool:
        def __init__(self, *_args, **_kwargs):
            self.closed = False

        def getconn(self):
            return DummyConn()

        def putconn(self, _conn):
            return None

        def closeall(self):
            self.closed = True

    psycopg2_stub = SimpleNamespace(connect=lambda _conn: DummyConn())
    monkeypatch.setattr(stores_module, "psycopg2", psycopg2_stub)
    monkeypatch.setattr(stores_module, "ThreadedConnectionPool", DummyPool)

    store = stores_module.PGVectorPrecedentStore(conn_string="pg", table_name="precedent")
    results = store.search("query", embedding=[0.1])
    assert results
    store.close()


def test_env_int(monkeypatch):
    monkeypatch.delenv("PG_POOL_MIN", raising=False)
    assert stores_module._env_int("PG_POOL_MIN", 2) == 2
    monkeypatch.setenv("PG_POOL_MIN", "0")
    assert stores_module._env_int("PG_POOL_MIN", 2) == 1
    monkeypatch.setenv("PG_POOL_MIN", "bad")
    assert stores_module._env_int("PG_POOL_MIN", 2) == 2
