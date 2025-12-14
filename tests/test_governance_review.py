import os
from types import SimpleNamespace

from api import replay_store as rs
from governance.review_packets import ReviewPacket
from governance.review_triggers import ReviewTriggerEvaluator, Case


def _configure_dirs(tmp_path, monkeypatch):
    packet_dir = tmp_path / "packets"
    review_dir = tmp_path / "reviews"
    monkeypatch.setattr(rs, "REVIEW_PACKET_DIR", str(packet_dir))
    monkeypatch.setattr(rs, "REVIEW_RECORD_DIR", str(review_dir))
    os.makedirs(packet_dir, exist_ok=True)
    os.makedirs(review_dir, exist_ok=True)


def test_review_triggers_accept_uncertainty_flags(monkeypatch):
    evaluator = ReviewTriggerEvaluator()

    case_with_flags = Case(severity=0.0, uncertainty_flags=["u1"], rights_impacted=[])
    decision = evaluator.evaluate(case_with_flags)
    assert decision["review_required"]
    assert "uncertainty_present" in decision["triggers"]

    case_with_nested = Case(severity=0.0, rights_impacted=[])
    case_with_nested.uncertainty = SimpleNamespace(flags=["nested"])
    decision_nested = evaluator.evaluate(case_with_nested)
    assert decision_nested["review_required"]
    assert "uncertainty_present" in decision_nested["triggers"]


def test_store_review_packet_append_only(tmp_path, monkeypatch):
    _configure_dirs(tmp_path, monkeypatch)

    packet1 = ReviewPacket(
        case_id="case123",
        domain="test",
        severity=1.0,
        uncertainty_flags=[],
        critic_outputs={"c1": {"severity": 1.0}},
        aggregator_summary="first",
        dissent=None,
        citations={},
        triggers=["t1"],
    )
    packet2 = ReviewPacket(
        case_id="case123",
        domain="test",
        severity=2.0,
        uncertainty_flags=["u1"],
        critic_outputs={"c1": {"severity": 2.0}},
        aggregator_summary="second",
        dissent=None,
        citations={},
        triggers=["t2"],
    )

    path1 = rs.store_review_packet(packet1)
    path2 = rs.store_review_packet(packet2)

    assert os.path.exists(path1)
    assert os.path.exists(path2)
    files = list((tmp_path / "packets").glob("case123_*.json"))
    assert len(files) == 2

    latest = rs.load_review_packet("case123")
    assert latest["triggers"] == ["t2"]

    packets = rs.list_review_packets("case123")
    assert len(packets) == 2
    assert packets[0]["triggers"] == ["t2"]
