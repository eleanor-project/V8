from __future__ import annotations

import os
import time
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status

from api.middleware.auth import require_authenticated_user, require_role
from api.rest.admin_write import require_admin_write_enabled
from api.rest.deps import get_engine
from api.schemas import (
    ConfigProposalApplyRequest,
    ConfigProposalApplyResponse,
    ConfigProposalListResponse,
    ConfigProposalPreviewRequest,
)
from api.rest.services.config_proposals import (
    PreviewValidationError,
    apply_proposal,
    build_preview_artifact,
    get_proposal,
    list_proposals,
    run_full_replay_preview,
    store_preview_artifact,
)
from engine.config import ConfigManager

router = APIRouter(tags=["Admin"])
ADMIN_ROLE = os.getenv("ADMIN_ROLE", "admin")


@router.get("/config/proposals", response_model=ConfigProposalListResponse)
@require_role(ADMIN_ROLE)
async def list_config_proposals():
    items = list_proposals()
    return ConfigProposalListResponse(
        schema_version=1,
        environment=ConfigManager().settings.environment,
        items=items,
    )


@router.post("/config/proposals/{proposal_id}/preview")
@require_role(ADMIN_ROLE)
async def preview_config_proposal(
    proposal_id: str,
    payload: ConfigProposalPreviewRequest,
    user: str = Depends(require_authenticated_user),
    engine=Depends(get_engine),
):
    start = time.monotonic()
    window = payload.window.model_dump(mode="json") if payload.window else {"type": "time", "duration": "4h"}
    limits = payload.limits.model_dump(mode="json") if payload.limits else {}
    proposal = get_proposal(proposal_id)
    if proposal is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Proposal not found",
        )
    if payload.mode == "full_replay":
        artifact = await run_full_replay_preview(
            proposal_id=proposal_id,
            proposal=proposal,
            window=window,
            limits=limits,
            engine=engine,
        )
    else:
        artifact = build_preview_artifact(
            proposal_id=proposal_id,
            proposal=proposal,
            mode=payload.mode,
            window=window,
            limits=limits,
            engine=engine,
        )
    artifact["metrics"]["preview_duration_ms"] = int((time.monotonic() - start) * 1000)

    store_preview_artifact(proposal_id=proposal_id, artifact=artifact, actor=user)
    return artifact


@router.post("/config/proposals/{proposal_id}/apply", response_model=ConfigProposalApplyResponse)
@require_role(ADMIN_ROLE)
async def apply_config_proposal(
    proposal_id: str,
    payload: ConfigProposalApplyRequest,
    user: str = Depends(require_authenticated_user),
    engine=Depends(get_engine),
    _write_enabled: None = Depends(require_admin_write_enabled),
    preview_hash: Optional[str] = Header(default=None, alias="X-Preview-Artifact-Hash"),
):
    if preview_hash and payload.artifact_hash and preview_hash != payload.artifact_hash:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "INVALID_PREVIEW_ARTIFACT",
                "message": "Preview artifact hash mismatch between header and body.",
            },
        )
    artifact_hash = preview_hash or payload.artifact_hash
    if not artifact_hash:
        raise HTTPException(
            status_code=status.HTTP_428_PRECONDITION_REQUIRED,
            detail={"error": "PREVIEW_REQUIRED", "message": "Missing preview artifact hash"},
        )

    try:
        result = apply_proposal(
            proposal_id=proposal_id,
            artifact_hash=artifact_hash,
            engine=engine,
            actor=user,
        )
    except PreviewValidationError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={
                "error": exc.code,
                "message": exc.message,
                "details": exc.details,
            },
        )
    return ConfigProposalApplyResponse(
        schema_version=1,
        proposal_id=result["proposal_id"],
        status=result["status"],
        applied_at=result["applied_at"],
        fingerprints=result["fingerprints"],
        ledger=result["ledger"],
    )
