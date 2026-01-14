from __future__ import annotations

import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from api.rest.admin_write import admin_write_enabled
from engine.security.ledger import ledger_backend_id
from engine.version import ELEANOR_VERSION


class AdminWriteHeaderMiddleware(BaseHTTPMiddleware):
    """
    Adds operational headers to every /admin/* response.
    """

    @staticmethod
    def _environment() -> str:
        return os.getenv("ELEANOR_ENVIRONMENT") or os.getenv("ELEANOR_ENV") or "development"

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)

        path = request.url.path or ""
        if path.startswith("/admin"):
            response.headers["X-Admin-Write-Enabled"] = (
                "true" if admin_write_enabled() else "false"
            )
            response.headers["X-Eleanor-Environment"] = self._environment()
            response.headers["X-Eleanor-Ledger"] = ledger_backend_id()
            response.headers["X-Eleanor-Version"] = ELEANOR_VERSION

        return response
