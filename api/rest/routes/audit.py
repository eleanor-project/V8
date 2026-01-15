"""
Audit query API endpoints.

Provides REST API for querying and exporting audit logs, plus trace fetch/replay utilities.
"""

import csv
import io
import json
import logging
import inspect
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from engine.security.ledger_reader import (
    get_ledger_reader,
    AuditEventType,
    AuditSeverity,
)

from api.middleware.auth import require_authenticated_user
from api.middleware.rate_limit import check_rate_limit
from api.rest.deps import get_engine, get_replay_store
from api.rest.services.audit_utils import fetch_trace, search_traces, replay_trace

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/audit", tags=["audit"])


class AuditQueryRequest(BaseModel):
    """Audit query request parameters."""
    start_time: Optional[datetime] = Field(None, description="Filter events after this time")
    end_time: Optional[datetime] = Field(None, description="Filter events before this time")
    event_types: Optional[List[str]] = Field(None, description="Filter by event types")
    severity: Optional[str] = Field(None, description="Filter by severity level")
    user: Optional[str] = Field(None, description="Filter by user/accessor")
    request_id: Optional[str] = Field(None, description="Filter by request ID")
    resource: Optional[str] = Field(None, description="Filter by resource name")
    search_text: Optional[str] = Field(None, description="Full-text search")
    limit: int = Field(100, ge=1, le=10000, description="Maximum results")
    offset: int = Field(0, ge=0, description="Results offset")
    sort_desc: bool = Field(True, description="Sort descending (newest first)")


class AuditQueryResponse(BaseModel):
    """Audit query response."""
    events: List[dict]
    total: int
    limit: int
    offset: int
    has_more: bool


class AuditStatsResponse(BaseModel):
    """Audit statistics response."""
    period_hours: int
    total_events: int
    by_type: dict
    by_severity: dict
    start_time: str
    end_time: str


# -----------------------------
# Compatibility helpers
# -----------------------------

def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
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


def _payload_value(payload: Any, key: str) -> Any:
    if isinstance(payload, dict):
        return payload.get(key)
    return getattr(payload, key, None)


def _filter_ledger_records(
    *,
    records: List[Any],
    start_time: Optional[datetime],
    end_time: Optional[datetime],
    event_types: Optional[List[str]],
    severity: Optional[str],
    user: Optional[str],
    request_id: Optional[str],
    resource: Optional[str],
    search_text: Optional[str],
) -> List[Any]:
    """
    Best-effort filter for readers that only expose read_all().
    Works even if record schema varies.
    """
    filtered: List[Any] = []
    for rec in records:
        payload = getattr(rec, "payload", None) or {}
        try:
            ts_raw = getattr(rec, "timestamp", None) or payload.get("timestamp")
            ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00")) if ts_raw else None
        except Exception:
            ts = None

        if start_time and ts and ts < start_time:
            continue
        if end_time and ts and ts > end_time:
            continue

        # event type
        if event_types:
            rec_event = (
                getattr(rec, "event", None)
                or getattr(rec, "event_type", None)
                or getattr(rec, "event_id", None)
                or payload.get("event")
                or payload.get("event_type")
            )
            if rec_event is None or str(rec_event) not in {str(x) for x in event_types}:
                continue

        # severity
        if severity:
            rec_sev = getattr(rec, "severity", None) or payload.get("severity")
            if rec_sev is None or str(rec_sev).lower() != str(severity).lower():
                continue

        # user/accessor
        if user:
            rec_user = (
                getattr(rec, "actor_id", None)
                or getattr(rec, "user", None)
                or payload.get("actor_id")
                or payload.get("user")
            )
            if rec_user is None or str(rec_user) != str(user):
                continue

        # request_id / trace_id-ish
        if request_id:
            rec_req = (
                getattr(rec, "request_id", None)
                or getattr(rec, "trace_id", None)
                or payload.get("request_id")
                or payload.get("trace_id")
            )
            if rec_req is None or str(rec_req) != str(request_id):
                continue

        # resource
        if resource:
            rec_res = getattr(rec, "resource", None) or payload.get("resource")
            if rec_res is None or str(rec_res) != str(resource):
                continue

        # full-text search
        if search_text:
            blob = json.dumps(payload or {}, default=str)
            combined = f"{getattr(rec, 'event', '')} {getattr(rec, 'event_type', '')} {blob}"
            if str(search_text).lower() not in combined.lower():
                continue

        filtered.append(rec)

    return filtered


def _safe_query(reader: Any, request: AuditQueryRequest) -> Dict[str, Any]:
    """
    Use reader.query() if available, else fallback to read_all()+filtering.
    Returns a dict matching AuditQueryResponse fields.
    """
    if hasattr(reader, "query") and callable(getattr(reader, "query")):
        return reader.query(
            start_time=request.start_time,
            end_time=request.end_time,
            event_types=request.event_types,
            severity=request.severity,
            user=request.user,
            request_id=request.request_id,
            resource=request.resource,
            search_text=request.search_text,
            limit=request.limit,
            offset=request.offset,
            sort_desc=request.sort_desc,
        )

    # Fallback path
    records = reader.read_all() if hasattr(reader, "read_all") else []
    filtered = _filter_ledger_records(
        records=records,
        start_time=request.start_time,
        end_time=request.end_time,
        event_types=request.event_types,
        severity=request.severity,
        user=request.user,
        request_id=request.request_id,
        resource=request.resource,
        search_text=request.search_text,
    )

    # sort (best-effort)
    def _ts_key(rec: Any) -> str:
        payload = getattr(rec, "payload", None) or {}
        ts = getattr(rec, "timestamp", None) or payload.get("timestamp") or ""
        return str(ts)

    filtered.sort(key=_ts_key, reverse=bool(request.sort_desc))

    total = len(filtered)
    start = int(request.offset)
    end = start + int(request.limit)
    page = filtered[start:end]
    events = _ledger_records_as_dicts(page)

    return {
        "events": events,
        "total": total,
        "limit": int(request.limit),
        "offset": int(request.offset),
        "has_more": end < total,
    }


def _safe_export(reader: Any, request: AuditQueryRequest, format: str) -> str:
    """
    Use reader.export() if available, else implement export over _safe_query fallback.
    Returns serialized content.
    """
    fmt = (format or "json").lower()
    if hasattr(reader, "export") and callable(getattr(reader, "export")):
        export_params = request.model_dump()
        export_params["limit"] = min(int(export_params.get("limit", 10000)), 10000)
        return reader.export(format=fmt, **export_params)

    # Fallback: get up to 10k records via safe query
    req2 = request.model_copy()
    req2.limit = min(int(req2.limit), 10000)
    req2.offset = 0
    result = _safe_query(reader, req2)
    events = result.get("events", [])

    if fmt == "csv":
        if not events:
            return ""
        output = io.StringIO()
        fieldnames = sorted(events[0].keys())
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(events)
        return output.getvalue()

    # default json
    return json.dumps(events, default=str)


# -----------------------------
# Trace endpoints (from HEAD side)
# -----------------------------

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


@router.post("/search")
async def audit_search(
    payload,
    user: str = Depends(require_authenticated_user),
    _rate: None = Depends(check_rate_limit),
    replay_store=Depends(get_replay_store),
):
    """
    Trace search endpoint. Kept generic because AuditSearchRequest/Response may live elsewhere.
    If you have typed models, change `payload` to AuditSearchRequest and add response_model.
    """
    user_id = _payload_value(payload, "user_id") or user
    results = await search_traces(
        replay_store,
        query=_payload_value(payload, "query"),
        user_id=user_id,
        decision=_payload_value(payload, "decision"),
        limit=_payload_value(payload, "limit") or 50,
    )
    return {"results": results}


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

    # Call replay_trace in a signature-flexible way (repo may have changed)
    try:
        sig = inspect.signature(replay_trace)
        kwargs = {}
        params = sig.parameters

        if "engine" in params:
            kwargs["engine"] = engine
        if "replay_store" in params:
            kwargs["replay_store"] = replay_store
        if "trace_id" in params:
            kwargs["trace_id"] = trace_id
        if "trace" in params:
            kwargs["trace"] = stored
        if "stored" in params:
            kwargs["stored"] = stored
        if "user" in params:
            kwargs["user"] = user
        if "user_id" in params:
            kwargs["user_id"] = user

        result = replay_trace(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result
    except Exception as e:
        logger.error("replay_failed", extra={"error": str(e), "trace_id": trace_id})
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Ledger query/export API (from other side, secured)
# -----------------------------

@router.post("/query", response_model=AuditQueryResponse)
async def query_audit_logs(
    request: AuditQueryRequest,
    user: str = Depends(require_authenticated_user),
    _rate: None = Depends(check_rate_limit),
) -> AuditQueryResponse:
    """
    Query audit logs with filters. Returns paginated results with total count.
    """
    try:
        reader = get_ledger_reader()
        result = _safe_query(reader, request)

        if isinstance(result, dict) and "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return AuditQueryResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("audit_query_failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/event-types", response_model=List[str])
async def get_event_types(
    user: str = Depends(require_authenticated_user),
    _rate: None = Depends(check_rate_limit),
) -> List[str]:
    """Get list of all unique event types in the audit ledger."""
    try:
        reader = get_ledger_reader()
        if hasattr(reader, "get_event_types") and callable(getattr(reader, "get_event_types")):
            return reader.get_event_types()

        # fallback: derive from read_all
        records = reader.read_all() if hasattr(reader, "read_all") else []
        rows = _ledger_records_as_dicts(records)
        found = set()
        for r in rows:
            ev = r.get("event_type") or r.get("event") or r.get("event_id")
            if ev:
                found.add(str(ev))
        return sorted(found)
    except Exception as e:
        logger.error("get_event_types_failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=AuditStatsResponse)
async def get_audit_stats(
    hours: int = Query(24, ge=1, le=168, description="Number of hours to analyze"),
    user: str = Depends(require_authenticated_user),
    _rate: None = Depends(check_rate_limit),
) -> AuditStatsResponse:
    """Get audit statistics for recent period."""
    try:
        reader = get_ledger_reader()
        if hasattr(reader, "get_stats") and callable(getattr(reader, "get_stats")):
            stats = reader.get_stats(hours=hours)
            if isinstance(stats, dict) and "error" in stats:
                raise HTTPException(status_code=500, detail=stats["error"])
            return AuditStatsResponse(**stats)

        # fallback: minimal stats from read_all()
        now = datetime.utcnow()
        start = now.timestamp() - (hours * 3600)

        records = reader.read_all() if hasattr(reader, "read_all") else []
        rows = _ledger_records_as_dicts(records)

        by_type: Dict[str, int] = {}
        by_sev: Dict[str, int] = {}
        total = 0

        for r in rows:
            ts_raw = r.get("timestamp") or r.get("time") or r.get("ts")
            try:
                ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00")).timestamp() if ts_raw else None
            except Exception:
                ts = None
            if ts is not None and ts < start:
                continue

            total += 1
            et = str(r.get("event_type") or r.get("event") or "unknown")
            sv = str(r.get("severity") or "unknown").lower()
            by_type[et] = by_type.get(et, 0) + 1
            by_sev[sv] = by_sev.get(sv, 0) + 1

        return AuditStatsResponse(
            period_hours=hours,
            total_events=total,
            by_type=by_type,
            by_severity=by_sev,
            start_time=datetime.utcfromtimestamp(start).isoformat() + "Z",
            end_time=now.isoformat() + "Z",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_stats_failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export")
async def export_audit_logs(
    request: AuditQueryRequest,
    format: str = Query("json", description="Export format: json or csv"),
    user: str = Depends(require_authenticated_user),
    _rate: None = Depends(check_rate_limit),
) -> Response:
    """
    Export audit logs to JSON or CSV format.
    Applies the same filters as query endpoint but returns up to 10,000 events.
    """
    fmt = (format or "json").lower()
    if fmt not in {"json", "csv"}:
        raise HTTPException(status_code=400, detail="format must be 'json' or 'csv'")

    try:
        reader = get_ledger_reader()

        # Override limit for export (cap at 10k)
        export_req = request.model_copy()
        export_req.limit = min(int(export_req.limit), 10000)
        export_req.offset = 0

        content = _safe_export(reader, export_req, fmt)

        media_type = "application/json" if fmt == "json" else "text/csv"
        filename = f"audit_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.{fmt}"

        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except Exception as e:
        logger.error("export_failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def audit_health() -> dict:
    """Health check endpoint for audit query service."""
    try:
        reader = get_ledger_reader()
        event_types = []
        if hasattr(reader, "get_event_types") and callable(getattr(reader, "get_event_types")):
            event_types = reader.get_event_types()
        return {
            "status": "healthy",
            "ledger": reader.__class__.__name__,
            "event_types_count": len(event_types),
            "service": "audit-query",
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "service": "audit-query"}
