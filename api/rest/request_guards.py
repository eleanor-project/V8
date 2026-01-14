import os
from typing import Optional

from fastapi import HTTPException, Request, status
from engine.logging_config import get_logger

logger = get_logger(__name__)


def _resolve_environment() -> str:
    return os.getenv("ELEANOR_ENVIRONMENT") or os.getenv("ELEANOR_ENV") or "development"


def check_content_length(request: Request, max_bytes: Optional[int] = None):
    """
    Guardrail: reject overly large requests early.

    In production, enforces stricter limits (512KB default).
    In development, allows larger requests (1MB default).
    """
    env = _resolve_environment()
    is_production = env == "production"

    default_max = 524288 if is_production else 1048576  # 512KB vs 1MB
    max_allowed = max_bytes or int(os.getenv("MAX_REQUEST_BYTES", str(default_max)))

    if is_production and max_allowed > 1048576:
        logger.warning(
            "MAX_REQUEST_BYTES exceeds 1MB in production, using 1MB limit",
            extra={"configured": max_allowed},
        )
        max_allowed = 1048576

    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            length = int(content_length)
            if length > max_allowed:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Request too large ({length} bytes, max: {max_allowed} bytes)",
                )
        except ValueError:
            logger.warning("Invalid content-length header; continuing without size enforcement")
