from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from engine.logging_config import get_logger

from api.middleware.auth import require_authenticated_user
from api.middleware.rate_limit import check_rate_limit
from api.rest.deps import get_engine, get_replay_store
from api.rest.services.audit_utils import fetch_trace, search_traces, replay_trace

logger = get_logger(__name__)
router = APIRouter(tags=["Audit"])


class AuditSearchRequest(BaseModel):
    query: Optional[str] = Field(default=None, description="Search text (trace_id or input substring)")
    user_id: Optional[str] = Field(default=None, description="Filter by user_id if stored")
    decision: Optional[str] = Field(default=None, description="Filter by final_decision label")
    limit: int = Field(default=50, ge=1, le=500)


class AuditSearchResponse(BaseModel):
    results: List[Dict[str, Any]]


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
