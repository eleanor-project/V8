import types

import pytest
from fastapi import HTTPException

from api.rest import governance as governance_module
from api.rest import review as review_module


def _call_endpoint(func, *args, **kwargs):
    target = getattr(func, "__wrapped__", func)
    return target(*args, **kwargs)


def test_governance_review_metrics_not_found(monkeypatch):
    monkeypatch.setattr(governance_module, "review_metrics", lambda _cid: {})
    monkeypatch.setattr(governance_module, "severity_drift", lambda _cid: None)
    monkeypatch.setattr(governance_module, "dissent_suppression_rate", lambda _cid: None)

    with pytest.raises(HTTPException):
        _call_endpoint(governance_module.get_review_metrics, "case-1")


def test_governance_review_metrics_success(monkeypatch):
    monkeypatch.setattr(governance_module, "review_metrics", lambda _cid: {"count": 2})
    monkeypatch.setattr(governance_module, "severity_drift", lambda _cid: 0.2)
    monkeypatch.setattr(governance_module, "dissent_suppression_rate", lambda _cid: 0.1)

    result = _call_endpoint(governance_module.get_review_metrics, "case-1")
    assert result["metrics"]["count"] == 2
    assert result["severity_drift"] == 0.2


def test_governance_quarantine_list(monkeypatch):
    monkeypatch.setattr(governance_module, "list_quarantined_cases", lambda: ["c1"])
    result = _call_endpoint(governance_module.get_quarantined_cases)
    assert result == ["c1"]


def test_submit_review_endpoint_success(monkeypatch):
    review = types.SimpleNamespace(case_id="case-1")
    monkeypatch.setattr(review_module, "submit_review", lambda _review: {"status": "ok"})
    monkeypatch.setattr(review_module.promotion_router, "route", lambda _review: "lane-a")
    calls = []
    monkeypatch.setattr(review_module, "resolve_review", lambda cid: calls.append(cid))

    result = _call_endpoint(review_module.submit_review_endpoint, review)
    assert result["status"] == "ok"
    assert result["promotion_lane"] == "lane-a"
    assert calls == ["case-1"]


def test_submit_review_endpoint_error(monkeypatch):
    review = types.SimpleNamespace(case_id="case-1")

    def _raise(_review):
        raise ValueError("bad")

    monkeypatch.setattr(review_module, "submit_review", _raise)
    with pytest.raises(HTTPException):
        _call_endpoint(review_module.submit_review_endpoint, review)


def test_get_review_endpoint(monkeypatch):
    monkeypatch.setattr(review_module, "get_review", lambda _rid: None)
    with pytest.raises(HTTPException):
        _call_endpoint(review_module.get_review_endpoint, "r1")

    monkeypatch.setattr(review_module, "get_review", lambda _rid: {"review_id": "r1"})
    result = _call_endpoint(review_module.get_review_endpoint, "r1")
    assert result["review_id"] == "r1"


def test_review_case_endpoints(monkeypatch):
    monkeypatch.setattr(review_module, "get_reviews_for_case", lambda _cid: [{"id": 1}])
    result = _call_endpoint(review_module.get_case_reviews_endpoint, "case-1")
    assert result["count"] == 1

    monkeypatch.setattr(review_module, "get_pending_reviews", lambda: [{"id": 2}])
    pending = _call_endpoint(review_module.get_pending_reviews_endpoint)
    assert pending["count"] == 1

    monkeypatch.setattr(review_module, "get_review_stats", lambda _cid: {"total": 3})
    stats = _call_endpoint(review_module.get_stats_endpoint, "case-1")
    assert stats["total"] == 3

    monkeypatch.setattr(review_module, "validate_review_chain", lambda _cid: {"valid": True})
    validation = _call_endpoint(review_module.validate_case_endpoint, "case-1")
    assert validation["valid"] is True


def test_lane_contents(monkeypatch):
    with pytest.raises(HTTPException):
        _call_endpoint(review_module.get_lane_contents_endpoint, "bad")

    monkeypatch.setattr(review_module.promotion_router, "get_lane_contents", lambda _lane: ["c1"])
    result = _call_endpoint(review_module.get_lane_contents_endpoint, "quarantine")
    assert result["count"] == 1
