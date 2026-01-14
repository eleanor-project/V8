from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from api.middleware.auth import require_authenticated_user
from api.rest.deps import get_engine, get_replay_store
from engine.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Governance"])


@router.get("/explanation/{trace_id}", tags=["Governance"])
async def get_explanation(
    trace_id: str,
    detail_level: str = "summary",
    user: str = Depends(require_authenticated_user),
    engine=Depends(get_engine),
    replay_store=Depends(get_replay_store),
):
    """
    Get explainable governance explanation for a trace.
    detail_level: summary | detailed | interactive
    """
    if not hasattr(engine, "explainable_governance"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Explainable Governance is not enabled. Enable via feature flags.",
        )

    try:
        trace_data = await replay_store.get_async(trace_id)
        if not trace_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Trace {trace_id} not found",
            )

        response_data = trace_data.get("response") or trace_data

        from engine.core.feature_integration import get_explanation_for_result

        explanation = get_explanation_for_result(engine, response_data, detail_level=detail_level)
        if not explanation:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate explanation",
            )

        return explanation

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "explanation_retrieval_failed",
            extra={"trace_id": trace_id, "error": str(exc)},
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve explanation: {str(exc)}",
        )
