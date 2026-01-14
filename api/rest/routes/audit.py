from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from engine.logging_config import get_logger

from api.middleware.auth import require_authenticated_user
from api.middleware.rate_limit import check_rate_limit
from api.rest.deps import get_engine, get_replay_store
from api.rest.services.audit_utils import fetch_trace, search_traces, replay_trace
from engine.security.ledger import get_ledger_reader

logger = get_logger(__name__)
router = APIRouter(tags=["Audit"])


class AuditSearchRequest(BaseModel):
    query: Optional[str] = Field(default=None, description="Search text (trace_id or input substring)")
    user_id: Optional[str] = Field(default=None, description="Filter by user_id if stored")
    decision: Optional[str] = Field(default=None, description="Filter by final_decision label")
    limit: int = Field(default=50, ge=1, le=500)


class AuditSearchResponse(BaseModel):
    results: List[Dict[str, Any]]


class AuditQueryResponse(BaseModel):
    """Response model for ledger-backed audit queries."""

    results: List[Dict[str, Any]]
    count: int


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        # fromisoformat handles both with and without tz offsets
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _ledger_records_as_dicts(records: List[Any]) -> List[Dict[str, Any]]:
    """Convert LedgerRecord objects to plain dicts."""
    output: List[Dict[str, Any]] = []
    for rec in records:
        try:
            row = rec.to_dict() if hasattr(rec, "to_dict") else dict(rec)
            output.append(row)
        except Exception:
            continue
    return output


def _filter_ledger_records(
    *,
    records: List[Any],
    start_time: Optional[datetime],
    end_time: Optional[datetime],
    event: Optional[str],
    actor_id: Optional[str],
    trace_id: Optional[str],
    query: Optional[str],
) -> List[Any]:
    filtered: List[Any] = []
    for rec in records:
        ts_raw = getattr(rec, "timestamp", None)
        try:
            ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00")) if ts_raw else None
        except Exception:
            ts = None

        if start_time and ts and ts < start_time:
            continue
        if end_time and ts and ts > end_time:
            continue

        if event:
            rec_event = getattr(rec, "event", None) or getattr(rec, "event_id", None)
            if str(rec_event) != str(event):
                continue

        if actor_id:
            rec_actor = getattr(rec, "actor_id", None) or getattr(rec, "payload", {}).get("actor_id")
            if rec_actor is None or str(rec_actor) != str(actor_id):
                continue

        if trace_id:
            rec_trace = getattr(rec, "trace_id", None) or getattr(rec, "payload", {}).get("trace_id")
            if rec_trace is None or str(rec_trace) != str(trace_id):
                continue

        if query:
            payload_blob = json.dumps(getattr(rec, "payload", {}) or {}, default=str)
            combined = f"{getattr(rec, 'event', '')} {payload_blob}"
            if query.lower() not in combined.lower():
                continue

        filtered.append(rec)
    return filtered


@router.get("/trace/{trace_id}")
async def get_trace(
    trace_id: str,
    user: str = Depends(require_authenticated_user),
    _rate: None = Depends(check_rate_limit),
    replay_store=Depends(get_replay_store),
):
    stored = await fetch_trace(replay_store, trace_id)
    if not stored:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trace not found")

    owner = stored.get("user_id")
    if owner and str(owner) != str(user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized for this trace")

    return stored


@router.post("/audit/search", response_model=AuditSearchResponse)
async def audit_search(
    payload: AuditSearchRequest,
    user: str = Depends(require_authenticated_user),
    _rate: None = Depends(check_rate_limit),
    replay_store=Depends(get_replay_store),
):
    user_id = payload.user_id or user
    results = await search_traces(
        replay_store,
        query=payload.query,
        user_id=user_id,
        decision=payload.decision,
        limit=payload.limit,
    )
    return AuditSearchResponse(results=results)


@router.get("/audit/query", response_model=AuditQueryResponse)
async def audit_query(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    event: Optional[str] = None,
    actor_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    query: Optional[str] = None,
    limit: int = 200,
    user: str = Depends(require_authenticated_user),
    _rate: None = Depends(check_rate_limit),
):
    """
    Query immutable audit ledger records with optional filters.
    """
    start_dt = _parse_dt(start_time)
    end_dt = _parse_dt(end_time)

    # Pull from ledger; handle missing ledger gracefully
    try:
        reader = get_ledger_reader()
        records = reader.read_all()
    except Exception as exc:
        logger.warning("audit_query_ledger_unavailable", extra={"error": str(exc)})
        records = []

    filtered = _filter_ledger_records(
        records=records,
        start_time=start_dt,
        end_time=end_dt,
        event=event,
        actor_id=actor_id,
        trace_id=trace_id,
        query=query,
    )

    # Enforce limit after filtering to keep responses predictable
    limited = filtered[: max(1, min(limit, 1000))]
    results = _ledger_records_as_dicts(limited)

    return AuditQueryResponse(results=results, count=len(results))


@router.get("/audit/export")
async def audit_export(
    format: str = "jsonl",
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    event: Optional[str] = None,
    actor_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    query: Optional[str] = None,
    limit: int = 1000,
    user: str = Depends(require_authenticated_user),
    _rate: None = Depends(check_rate_limit),
):
    """
    Export audit ledger records in jsonl or csv format.
    """
    start_dt = _parse_dt(start_time)
    end_dt = _parse_dt(end_time)

    try:
        reader = get_ledger_reader()
        records = reader.read_all()
    except Exception as exc:
        logger.warning("audit_export_ledger_unavailable", extra={"error": str(exc)})
        records = []

    filtered = _filter_ledger_records(
        records=records,
        start_time=start_dt,
        end_time=end_dt,
        event=event,
        actor_id=actor_id,
        trace_id=trace_id,
        query=query,
    )
    limited = filtered[: max(1, min(limit, 5000))]
    rows = _ledger_records_as_dicts(limited)

    if format.lower() == "csv":
        if not rows:
            content = ""
        else:
            output = io.StringIO()
            fieldnames = sorted(rows[0].keys())
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            content = output.getvalue()
        return StreamingResponse(
            iter([content]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=audit_export.csv"},
        )

    # default jsonl
    content = "\n".join(json.dumps(r, default=str) for r in rows)
    return StreamingResponse(
        iter([content]),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": "attachment; filename=audit_export.jsonl"},
    )


@router.post("/replay/{trace_id}")
async def replay(
    trace_id: str,
    user: str = Depends(require_authenticated_user),
    _rate: None = Depends(check_rate_limit),
    engine=Depends(get_engine),
    replay_store=Depends(get_replay_store),
):
    stored = await fetch_trace(replay_store, trace_id)
    if not stored:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Trace not found")

    owner = stored.get("user_id")
    if owner and str(owner) != str(user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized for this trace")

    try:
        rerun = await replay_trace(engine, stored)
        return {
            "trace_id": trace_id,
            "replayed": True,
            "result": rerun,
        }
    except Exception as exc:
        logger.error("replay_failed", extra={"trace_id": trace_id, "error": str(exc)})
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Replay failed")
