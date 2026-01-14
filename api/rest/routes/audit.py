"""
Audit query API endpoints.

Provides REST API for querying and exporting audit logs.
"""

import logging
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Query, HTTPException, Response
from pydantic import BaseModel, Field

from engine.security.ledger_reader import (
    get_ledger_reader,
    AuditEventType,
    AuditSeverity,
)

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
    limit: int = Field(100, ge=1, le=1000, description="Maximum results")
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


@router.post("/query", response_model=AuditQueryResponse)
async def query_audit_logs(request: AuditQueryRequest) -> AuditQueryResponse:
    """
    Query audit logs with filters.
    
    Supports filtering by:
    - Time range (start_time, end_time)
    - Event types (access_log, secret_access_log, etc.)
    - Severity level (info, warning, error, critical)
    - User/accessor
    - Request ID
    - Resource name
    - Full-text search
    
    Returns paginated results with total count.
    """
    try:
        reader = get_ledger_reader()
        result = reader.query(
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
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return AuditQueryResponse(**result)
    except Exception as e:
        logger.error(f"Audit query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/event-types", response_model=List[str])
async def get_event_types() -> List[str]:
    """
    Get list of all unique event types in the audit ledger.
    
    Useful for populating filter dropdowns in UI.
    """
    try:
        reader = get_ledger_reader()
        return reader.get_event_types()
    except Exception as e:
        logger.error(f"Failed to get event types: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=AuditStatsResponse)
async def get_audit_stats(
    hours: int = Query(24, ge=1, le=168, description="Number of hours to analyze")
) -> AuditStatsResponse:
    """
    Get audit statistics for recent period.
    
    Returns counts by event type and severity for the specified time window.
    Default is last 24 hours, max is 7 days (168 hours).
    """
    try:
        reader = get_ledger_reader()
        stats = reader.get_stats(hours=hours)
        
        if "error" in stats:
            raise HTTPException(status_code=500, detail=stats["error"])
        
        return AuditStatsResponse(**stats)
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export")
async def export_audit_logs(
    request: AuditQueryRequest,
    format: str = Query("json", regex="^(json|csv)$", description="Export format")
) -> Response:
    """
    Export audit logs to JSON or CSV format.
    
    Applies the same filters as query endpoint but returns all matching
    events (up to 10,000) in the specified format.
    """
    try:
        reader = get_ledger_reader()
        
        # Override limit for export
        export_params = request.dict()
        export_params["limit"] = 10000
        
        content = reader.export(format=format, **export_params)
        
        # Set content type and filename
        media_type = "application/json" if format == "json" else "text/csv"
        filename = f"audit_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.{format}"
        
        return Response(
            content=content,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def audit_health() -> dict:
    """
    Health check endpoint for audit query service.
    
    Returns status and basic ledger information.
    """
    try:
        reader = get_ledger_reader()
        event_types = reader.get_event_types()
        
        return {
            "status": "healthy",
            "ledger_path": str(reader.ledger_path),
            "event_types_count": len(event_types),
            "service": "audit-query",
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "service": "audit-query",
        }
