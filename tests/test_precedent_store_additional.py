import builtins
import json
import sys

import pytest

from engine.precedent.store import (
    BasePrecedentStore,
    ChromaStore,
    InMemoryStore,
    JSONFileStore,
    PgVectorStore,
    PrecedentCase,
)


class _BaseStore(BasePrecedentStore):
    def add(self, case, embedding):
        return BasePrecedentStore.add(self, case, embedding)

    def search(self, query, top_k=5, embedding=None):
        return BasePrecedentStore.search(self, query, top_k=top_k, embedding=embedding)

    def get(self, case_id):
        return BasePrecedentStore.get(self, case_id)

    def delete(self, case_id):
        return BasePrecedentStore.delete(self, case_id)

    def count(self):
        return BasePrecedentStore.count(self)


def test_inmemory_store_similarity_and_recent():
    store = InMemoryStore()
    case1 = PrecedentCase(
        case_id=None,
        query_text="q1",
        decision="allow",
        timestamp=1.0,
    )
    case2 = PrecedentCase(
        case_id="c2",
        query_text="q2",
        decision="deny",
        timestamp=2.0,
    )

    case1_id = store.add(case1, [1.0, 0.0])
    store.add(case2, [0.0, 1.0])

    recent = store.search("q", top_k=1)
    assert recent[0]["case_id"] == "c2"

    ranked = store.search("q", top_k=2, embedding=[1.0, 0.0])
    assert ranked[0]["case_id"] == case1_id
    assert ranked[0]["similarity_score"] == pytest.approx(1.0)

    assert InMemoryStore._cosine_similarity([1.0], [0.0, 1.0]) == 0.0
    assert InMemoryStore._cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0


def test_json_store_load_and_delete(tmp_path):
    path = tmp_path / "precedents.json"
    payload = {
        "cases": {
            "c1": {"case_id": "c1", "query_text": "q", "decision": "allow", "timestamp": 1.0}
        },
        "embeddings": {"c1": [1.0, 0.0]},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    store = JSONFileStore(file_path=str(path))
    assert store.get("c1")["decision"] == "allow"

    results = store.search("q", embedding=[1.0, 0.0], top_k=1)
    assert results[0]["case_id"] == "c1"

    assert store.delete("c1") is True
    assert store.delete("missing") is False


def test_json_store_empty_search_and_generate_id(tmp_path):
    path = tmp_path / "empty.json"
    store = JSONFileStore(file_path=str(path))
    assert store.search("q") == []

    case = PrecedentCase(case_id=None, query_text="q", decision="allow")
    case_id = store.add(case, [0.1])
    assert case_id


def test_json_store_load_bad_file(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{bad", encoding="utf-8")
    store = JSONFileStore(file_path=str(path))
    assert store.count() == 0


def test_pgvector_get_connection_import_error(monkeypatch):
    store = PgVectorStore(connection_string="postgresql://", table_name="cases")
    original_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name.startswith("psycopg2"):
            raise ImportError("nope")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(ImportError):
        store._get_connection()


def test_pgvector_get_connection_success(monkeypatch):
    original_import = builtins.__import__
    class DummyCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyConn:
        def cursor(self):
            return DummyCursor()

    psycopg2_module = type(
        "Psycopg2",
        (),
        {
            "connect": lambda *_a, **_k: DummyConn(),
            "extras": type("Extras", (), {"RealDictCursor": object}),
        },
    )

    def _fake_import(name, *args, **kwargs):
        if name == "psycopg2":
            return psycopg2_module
        if name == "psycopg2.extras":
            return psycopg2_module.extras
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    store = PgVectorStore(connection_string="postgresql://", table_name="cases")
    assert store._get_connection()


def test_pgvector_generate_id():
    generated = PgVectorStore._generate_id("query")
    assert isinstance(generated, str)
    assert len(generated) == 16


def test_pgvector_get_returns_none(monkeypatch):
    class DummyCursor:
        rowcount = 0

        def execute(self, *_args, **_kwargs):
            return None

        def fetchone(self):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyConn:
        def cursor(self):
            return DummyCursor()

        def close(self):
            return None

    store = PgVectorStore.__new__(PgVectorStore)
    store.table_name = "cases"
    store.embedding_dim = 2
    store._initialized = True
    store._get_connection = lambda: DummyConn()
    assert store.get("missing") is None


def test_chroma_store_missing_dependency(monkeypatch):
    original_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "chromadb":
            raise ImportError("nope")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(ImportError):
        ChromaStore()


def test_chroma_store_branches(monkeypatch, tmp_path):
    class DummyCollection:
        def __init__(self):
            self.add_calls = []
            self.query_calls = []
            self.get_calls = []

        def add(self, **kwargs):
            self.add_calls.append(kwargs)

        def query(self, **kwargs):
            self.query_calls.append(kwargs)
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        def get(self, **kwargs):
            self.get_calls.append(kwargs)
            return {"ids": []}

        def delete(self, **_kwargs):
            raise RuntimeError("boom")

        def count(self):
            return 0

    class DummyClient:
        def __init__(self):
            self.collection = DummyCollection()

        def get_or_create_collection(self, **_kwargs):
            return self.collection

    chroma_module = type(
        "Chroma",
        (),
        {
            "Client": lambda *_a, **_k: DummyClient(),
            "PersistentClient": lambda *_a, **_k: DummyClient(),
        },
    )

    monkeypatch.setitem(sys.modules, "chromadb", chroma_module)
    store = ChromaStore(persist_directory=str(tmp_path))
    case = PrecedentCase(case_id=None, query_text="q", decision="allow")
    store.add(case, [0.1])
    assert store.search("q") == []
    assert store.get("missing") is None
    assert store.delete("missing") is False


def test_base_precedent_store_passes():
    store = _BaseStore()
    assert store.add(None, []) is None
    assert store.search("q") is None
    assert store.get("x") is None
    assert store.delete("x") is None
    assert store.count() is None
