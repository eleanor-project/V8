import json
import os
from pathlib import Path

import pytest

from engine.precedent.store import InMemoryStore, JSONFileStore, PrecedentCase, create_store


def test_in_memory_store_search_and_delete():
    store = InMemoryStore()
    case = PrecedentCase(case_id="c1", query_text="q", decision="allow")
    store.add(case, embedding=[0.0, 1.0])
    assert store.count() == 1

    results = store.search("q", top_k=1, embedding=None)
    assert results[0]["case_id"] == "c1"

    results = store.search("q", top_k=1, embedding=[0.0, 1.0])
    assert results[0]["similarity_score"] == 1.0

    assert store.get("c1")["case_id"] == "c1"
    assert store.delete("c1") is True
    assert store.delete("missing") is False


def test_in_memory_store_cosine_similarity_mismatch():
    store = InMemoryStore()
    case = PrecedentCase(case_id="c1", query_text="q", decision="allow")
    store.add(case, embedding=[0.0, 1.0])
    results = store.search("q", top_k=1, embedding=[0.1])
    assert results[0]["similarity_score"] == 0.0


def test_json_file_store_load_and_save(tmp_path, monkeypatch):
    file_path = tmp_path / "precedents.json"
    file_path.write_text("{bad json", encoding="utf-8")

    store = JSONFileStore(file_path=str(file_path))
    case = PrecedentCase(case_id="c1", query_text="q", decision="allow")
    store.add(case, embedding=[0.1, 0.2])

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "cases" in data

    def _raise(*_args, **_kwargs):
        raise OSError("boom")

    monkeypatch.setattr("builtins.open", _raise)
    store._save()


def test_create_store_invalid_backend():
    with pytest.raises(ValueError):
        create_store("unknown")

    assert isinstance(create_store("memory"), InMemoryStore)
    assert isinstance(create_store("json"), JSONFileStore)
