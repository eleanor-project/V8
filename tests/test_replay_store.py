import json
import os
from pathlib import Path

import pytest

from api import replay_store as rs


def _configure_dirs(tmp_path, monkeypatch):
    packet_dir = tmp_path / "packets"
    review_dir = tmp_path / "reviews"
    monkeypatch.setattr(rs, "REVIEW_PACKET_DIR", str(packet_dir))
    monkeypatch.setattr(rs, "REVIEW_RECORD_DIR", str(review_dir))
    os.makedirs(packet_dir, exist_ok=True)
    os.makedirs(review_dir, exist_ok=True)
    return packet_dir, review_dir


def test_replay_store_save_get_and_cache(tmp_path):
    path = tmp_path / "replay.jsonl"
    store = rs.ReplayStore(path=str(path), max_cache=2)

    record1 = {"trace_id": "t1", "payload": {"step": 1}}
    record2 = {"trace_id": "t2", "payload": {"step": 2}}
    store.save(record1)
    store.save(record2)

    assert store.get("t1") == record1
    assert store.get("missing") is None

    store._cache.clear()
    result = store.get("t1")
    assert result is not None
    assert result["trace_id"] == "t1"


def test_replay_store_load_existing_skips_bad_lines(tmp_path):
    path = tmp_path / "replay.jsonl"
    path.write_text(
        json.dumps({"trace_id": "good", "payload": 1})
        + "\n"
        + "{bad json}\n"
        + json.dumps({"trace_id": "good2", "payload": 2})
        + "\n",
        encoding="utf-8",
    )

    store = rs.ReplayStore(path=str(path), max_cache=10)
    assert store.get("good") is not None
    assert store.get("good2") is not None


def test_replay_store_trims_log(tmp_path):
    path = tmp_path / "replay.jsonl"
    store = rs.ReplayStore(path=str(path), max_cache=3, max_log_bytes=180)

    for i in range(5):
        store.save({"trace_id": f"t{i}", "payload": "x" * 50})

    assert store._file_size <= store.max_log_bytes
    assert len(store._cache) <= store.max_cache


@pytest.mark.asyncio
async def test_replay_store_async_save_get(tmp_path):
    path = tmp_path / "replay.jsonl"
    store = rs.ReplayStore(path=str(path), max_cache=2)
    await store.save_async({"trace_id": "async", "payload": {"step": "a"}})
    result = await store.get_async("async")
    assert result is not None
    assert result["trace_id"] == "async"


def test_review_packet_and_listing(tmp_path, monkeypatch):
    packet_dir, _ = _configure_dirs(tmp_path, monkeypatch)

    record1 = {"case_id": "case-1", "packet_id": "p1", "stored_at": "2022-01-01T00:00:00+00:00"}
    record2 = {"case_id": "case-1", "packet_id": "p2", "stored_at": "2023-01-01T00:00:00+00:00"}
    (packet_dir / "case-1_p1.json").write_text(json.dumps(record1), encoding="utf-8")
    (packet_dir / "case-1_p2.json").write_text(json.dumps(record2), encoding="utf-8")
    (packet_dir / "case-1_bad.json").write_text("{bad json", encoding="utf-8")

    latest = rs.load_review_packet("case-1")
    assert latest is not None
    assert latest["packet_id"] == "p2"

    packets = rs.list_review_packets("case-1")
    assert len(packets) == 2
    assert packets[0]["packet_id"] == "p2"


def test_store_review_packet_and_human_review(tmp_path, monkeypatch):
    _configure_dirs(tmp_path, monkeypatch)

    assert rs.store_review_packet({"packet_id": "p0"}) is None
    assert rs.store_human_review({"case_id": "case"}) is None

    packet_path = rs.store_review_packet({"case_id": "case", "data": 1})
    assert packet_path is not None
    assert Path(packet_path).exists()

    review_path = rs.store_human_review({"review_id": "r1", "case_id": "case"})
    assert review_path is not None
    assert Path(review_path).exists()

    reviews = rs.load_human_reviews("case")
    assert len(reviews) == 1
    assert reviews[0]["review_id"] == "r1"
