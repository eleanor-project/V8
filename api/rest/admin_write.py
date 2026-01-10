import os
from fastapi import HTTPException, status


def _truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def admin_write_enabled() -> bool:
    """
    Kill-switch for admin write operations (POST/PUT/PATCH/DELETE).
    Default: disabled. Enable explicitly with ELEANOR_ENABLE_ADMIN_WRITE=true.
    """
    return _truthy(os.getenv("ELEANOR_ENABLE_ADMIN_WRITE", "false"))


def require_admin_write_enabled():
    if not admin_write_enabled():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin write operations are disabled (ELEANOR_ENABLE_ADMIN_WRITE=false).",
        )
