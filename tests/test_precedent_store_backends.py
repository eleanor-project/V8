import sys
from types import SimpleNamespace

import pytest

from engine.precedent.store import (
    ChromaStore,
    PgVectorStore,
    PrecedentCase,
    create_store,
)


class DummyCursor:
    def __init__(self):
        self.rowcount = 1
        self._fetchone = None
        self._fetchall = []

    def execute(self, *_args, **_kwargs):
        return None

    def fetchall(self):
        return self._fetchall

    def fetchone(self):
        return self._fetchone

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class DummyConn:
    def __init__(self, cursor):
        self._cursor = cursor
        self.commits = 0
        self.closed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.commits += 1

    def close(self):
        self.closed = True


def test_pgvector_store_crud(monkeypatch):
    cursor = DummyCursor()
    store = PgVectorStore(connection_string="postgres://test", table_name="cases")
    monkeypatch.setattr(store, "_get_connection", lambda: DummyConn(cursor))

    case = PrecedentCase(case_id="c1", query_text="q", decision="allow")
    case_id = store.add(case, embedding=[0.1, 0.2])
    assert case_id == "c1"

    cursor._fetchall = [
        {
            "case_id": "c1",
            "query_text": "q",
            "decision": "allow",
            "values": [],
            "aggregate_score": 0.5,
            "critic_outputs": "{}",
            "rationale": "",
            "timestamp": 0,
            "metadata": "{}",
            "similarity_score": 0.9,
        }
    ]
    results = store.search("q", top_k=5, embedding=[0.1, 0.2])
    assert results[0]["similarity_score"] == 0.9

    cursor._fetchall = [
        {
            "case_id": "c1",
            "query_text": "q",
            "decision": "allow",
            "values": [],
            "aggregate_score": 0.5,
            "critic_outputs": "{}",
            "rationale": "",
            "timestamp": 0,
            "metadata": "{}",
            "similarity_score": 1.0,
        }
    ]
    assert store.search("q", top_k=1, embedding=None)

    cursor._fetchone = {
        "case_id": "c1",
        "query_text": "q",
        "decision": "allow",
        "values": [],
        "aggregate_score": 0.5,
        "critic_outputs": "{}",
        "rationale": "",
        "timestamp": 0,
        "metadata": "{}",
    }
    assert store.get("c1") is not None

    cursor.rowcount = 1
    assert store.delete("c1") is True

    cursor._fetchone = {"cnt": 3}
    assert store.count() == 3


def test_chroma_store_stub(monkeypatch):
    class DummyCollection:
        def __init__(self):
            self._ids = ["id1"]

        def add(self, **_kwargs):
            return None

        def query(self, **_kwargs):
            return {
                "ids": [["id1"]],
                "documents": [["doc"]],
                "metadatas": [
                    [
                        {
                            "decision": "allow",
                            "values": "[]",
                            "aggregate_score": 0.1,
                            "critic_outputs": "{}",
                            "rationale": "",
                            "timestamp": 0,
                            "metadata": "{}",
                        }
                    ]
                ],
                "distances": [[0.1]],
            }

        def get(self, **_kwargs):
            return {
                "ids": ["id1"],
                "documents": ["doc"],
                "metadatas": [
                    {
                        "decision": "allow",
                        "values": "[]",
                        "aggregate_score": 0.1,
                        "critic_outputs": "{}",
                        "rationale": "",
                        "timestamp": 0,
                        "metadata": "{}",
                    }
                ],
            }

        def delete(self, **_kwargs):
            return None

        def count(self):
            return 1

    class DummyClient:
        def get_or_create_collection(self, **_kwargs):
            return DummyCollection()

    chroma_stub = SimpleNamespace(
        Client=lambda *_a, **_k: DummyClient(),
        PersistentClient=lambda *_a, **_k: DummyClient(),
    )
    monkeypatch.setitem(sys.modules, "chromadb", chroma_stub)

    store = ChromaStore(collection_name="cases")
    case = PrecedentCase(case_id="c1", query_text="q", decision="allow")
    store.add(case, embedding=[0.1, 0.2])
    assert store.search("q", top_k=1, embedding=[0.1, 0.2])
    assert store.get("id1") is not None
    assert store.delete("id1") is True
    assert store.count() == 1


def test_create_store_backends(monkeypatch):
    chroma_stub = SimpleNamespace(
        Client=lambda *_a, **_k: SimpleNamespace(
            get_or_create_collection=lambda **_kw: SimpleNamespace(
                add=lambda **_kwargs: None,
                query=lambda **_kwargs: {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]},
                get=lambda **_kwargs: {"ids": [], "documents": [], "metadatas": []},
                delete=lambda **_kwargs: None,
                count=lambda: 0,
            )
        ),
        PersistentClient=lambda *_a, **_k: SimpleNamespace(
            get_or_create_collection=lambda **_kw: SimpleNamespace(
                add=lambda **_kwargs: None,
                query=lambda **_kwargs: {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]},
                get=lambda **_kwargs: {"ids": [], "documents": [], "metadatas": []},
                delete=lambda **_kwargs: None,
                count=lambda: 0,
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "chromadb", chroma_stub)

    assert isinstance(create_store("pgvector"), PgVectorStore)
    assert isinstance(create_store("chroma"), ChromaStore)
