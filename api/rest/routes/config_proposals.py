from __future__ import annotations

from fastapi import APIRouter, Depends, status

from api.middleware.auth import require_authenticated_user
from api.middleware.rate_limit import check_rate_limit
from api.schemas import ConfigProposalRequest, ConfigProposalResponse
from api.rest.services.config_proposals import submit_proposal

router = APIRouter(tags=["Config"])


@router.post(
    "/config/proposals",
    response_model=ConfigProposalResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_config_proposal(
    payload: ConfigProposalRequest,
    user: str = Depends(require_authenticated_user),
    _rate: None = Depends(check_rate_limit),
):
    record = submit_proposal(proposal=payload.model_dump(mode="json"), actor=user)
    return ConfigProposalResponse(
        proposal_id=record["proposal_id"],
        status="submitted",
        submitted_at=record["submitted_at"],
    )
