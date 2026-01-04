import json
import os

import pytest

import api.replay_store as replay_store


class _ModelDump:
    def model_dump(self):
        return {"a": 1}


class _DictLike:
    def dict(self):
        return {"b": 2}


def test_as_dict_variants():
    assert replay_store._as_dict(_ModelDump()) == {"a": 1}
    assert replay_store._as_dict(_DictLike()) == {"b": 2}
    assert replay_store._as_dict({"c": 3}) == {"c": 3}


def test_env_int_parsing(monkeypatch):
    monkeypatch.delenv("REPLAY_LOG_MAX_BYTES", raising=False)
    assert replay_store._env_int("REPLAY_LOG_MAX_BYTES") is None

    monkeypatch.setenv("REPLAY_LOG_MAX_BYTES", "0")
    assert replay_store._env_int("REPLAY_LOG_MAX_BYTES") is None

    monkeypatch.setenv("REPLAY_LOG_MAX_BYTES", "bad")
    assert replay_store._env_int("REPLAY_LOG_MAX_BYTES") is None

    monkeypatch.setenv("REPLAY_LOG_MAX_BYTES", "5")
    assert replay_store._env_int("REPLAY_LOG_MAX_BYTES") == 5


def test_replay_store_load_existing_directory(tmp_path):
    path = tmp_path / "replay_dir"
    path.mkdir()
    store = replay_store.ReplayStore(path=str(path))
    assert store._file_size == 0


def test_replay_store_index_entries_adjustment(tmp_path):
    store = replay_store.ReplayStore(
        path=str(tmp_path / "replay.jsonl"),
        max_cache=5,
        max_index_entries=1,
    )
    assert store.max_index_entries == 5


def test_replay_store_load_existing_stat_error(tmp_path, monkeypatch):
    path = tmp_path / "replay.jsonl"
    path.write_text(json.dumps({"trace_id": "t1"}) + "\n", encoding="utf-8")

    def _raise(*_args, **_kwargs):
        raise OSError("boom")

    monkeypatch.setattr(replay_store.Path, "stat", _raise)
    store = replay_store.ReplayStore(path=str(path))
    assert store._file_size == 0


def test_replay_store_read_at_offset_and_scan(tmp_path):
    path = tmp_path / "replay.jsonl"
    lines = [
        json.dumps({"trace_id": "t1", "value": 1}),
        "bad json",
        json.dumps({"trace_id": "t2", "value": 2}),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    store = replay_store.ReplayStore(path=str(path), max_cache=1, max_index_entries=1)
    assert store._read_at_offset_locked("t2", 0) is None
    assert store.get("t1")["value"] == 1

    missing_path = tmp_path / "missing.jsonl"
    store_missing = replay_store.ReplayStore(path=str(missing_path))
    assert store_missing._read_at_offset_locked("t1", 0) is None


def test_replay_store_save_missing_trace_id(tmp_path):
    path = tmp_path / "empty.jsonl"
    store = replay_store.ReplayStore(path=str(path))
    store.save({"value": 1})
    assert path.exists() is False


def test_rewrite_log_empty_records_handles_errors(tmp_path, monkeypatch):
    path = tmp_path / "empty.jsonl"
    store = replay_store.ReplayStore(path=str(path))
    store._cache.clear()

    def _raise(*_args, **_kwargs):
        raise OSError("boom")

    monkeypatch.setattr(replay_store.Path, "write_text", _raise)
    store._rewrite_log_locked()


def test_rewrite_log_trims_records_and_stat_error(tmp_path, monkeypatch):
    path = tmp_path / "replay.jsonl"
    store = replay_store.ReplayStore(path=str(path), max_log_bytes=80)
    store._cache.clear()
    store._cache["big"] = {"trace_id": "big", "payload": "x" * 200}
    store._cache["a"] = {"trace_id": "a", "payload": "x" * 20}
    store._cache["b"] = {"trace_id": "b", "payload": "x" * 20}

    def _raise(*_args, **_kwargs):
        raise OSError("boom")

    monkeypatch.setattr(replay_store.Path, "stat", _raise)
    store._rewrite_log_locked()


def test_get_missing_path_and_bad_json(tmp_path, monkeypatch):
    missing_store = replay_store.ReplayStore(path=str(tmp_path / "missing.jsonl"))
    assert missing_store.get("t1") is None

    path = tmp_path / "replay.jsonl"
    path.write_text("bad json\n", encoding="utf-8")
    store = replay_store.ReplayStore(path=str(path))
    store._offsets.clear()
    assert store.get("nope") is None

    def _raise(*_args, **_kwargs):
        raise FileNotFoundError("gone")

    monkeypatch.setattr(replay_store.Path, "open", _raise)
    assert store.get("nope") is None


def test_review_packet_storage_and_loading(tmp_path, monkeypatch):
    packet_dir = tmp_path / "packets"
    review_dir = tmp_path / "reviews"
    packet_dir.mkdir()
    review_dir.mkdir()
    monkeypatch.setattr(replay_store, "REVIEW_PACKET_DIR", str(packet_dir))
    monkeypatch.setattr(replay_store, "REVIEW_RECORD_DIR", str(review_dir))

    assert replay_store.store_review_packet({"foo": "bar"}) is None
    assert replay_store.store_human_review({"case_id": "c1"}) is None

    packet_path = replay_store.store_review_packet({"case_id": "case-1"})
    assert packet_path and os.path.exists(packet_path)

    (packet_dir / "case-1.txt").write_text("skip", encoding="utf-8")
    (packet_dir / "other.json").write_text("{}", encoding="utf-8")

    bad_ts = {"case_id": "case-1", "stored_at": "not-a-date"}
    (packet_dir / "case-1_bad_ts.json").write_text(json.dumps(bad_ts), encoding="utf-8")
    (packet_dir / "case-1_bad.json").write_text("{bad", encoding="utf-8")
    latest = replay_store.load_review_packet("case-1")
    assert latest["case_id"] == "case-1"

    packets = replay_store.list_review_packets("case-1")
    assert packets

    review_path = replay_store.store_human_review({"review_id": "r1", "case_id": "case-1"})
    assert review_path and os.path.exists(review_path)

    reviews = replay_store.load_human_reviews("case-1")
    assert reviews and reviews[0]["review_id"] == "r1"


def test_review_packet_missing_dirs(tmp_path, monkeypatch):
    missing_packets = tmp_path / "missing_packets"
    missing_reviews = tmp_path / "missing_reviews"
    monkeypatch.setattr(replay_store, "REVIEW_PACKET_DIR", str(missing_packets))
    monkeypatch.setattr(replay_store, "REVIEW_RECORD_DIR", str(missing_reviews))

    assert replay_store.load_review_packet("case-1") is None
    assert replay_store.list_review_packets("case-1") == []
    assert replay_store.load_human_reviews("case-1") == []


def test_rewrite_log_skips_oversized_record(tmp_path):
    path = tmp_path / "replay.jsonl"
    store = replay_store.ReplayStore(path=str(path), max_log_bytes=10)
    store._cache.clear()
    store._cache["big"] = {"trace_id": "big", "payload": "x" * 50}
    store._rewrite_log_locked()
    assert not store._cache


def test_get_cache_filled_inside_lock(tmp_path):
    path = tmp_path / "replay.jsonl"
    path.write_text("", encoding="utf-8")
    store = replay_store.ReplayStore(path=str(path))
    record = {"trace_id": "t1", "payload": "x"}

    class FlakyCache:
        def __init__(self, item):
            self.item = item
            self.calls = 0

        def get(self, key):
            self.calls += 1
            if self.calls > 1 and key == self.item["trace_id"]:
                return self.item
            return None

    store._cache = FlakyCache(record)
    store._offsets.clear()
    assert store.get("t1") == record
