"""
ELEANOR V8 — Security Headers Middleware
----------------------------------------

Adds security headers to all API responses:
- HSTS (HTTP Strict Transport Security)
- CSP (Content Security Policy)
- X-Content-Type-Options
- X-Frame-Options
- X-XSS-Protection
- Referrer-Policy
"""

import os
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.environment = (
            os.getenv("ELEANOR_ENVIRONMENT") or os.getenv("ELEANOR_ENV") or "development"
        ).lower()
        self.is_production = self.environment == "production"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response."""
        response = await call_next(request)

        # HSTS - Only in production with HTTPS
        if self.is_production and request.url.scheme == "https":
            max_age = int(os.getenv("HSTS_MAX_AGE", "31536000"))  # 1 year default
            include_subdomains = os.getenv("HSTS_INCLUDE_SUBDOMAINS", "true").lower() == "true"
            hsts_value = f"max-age={max_age}"
            if include_subdomains:
                hsts_value += "; includeSubDomains"
            if os.getenv("HSTS_PRELOAD", "false").lower() == "true":
                hsts_value += "; preload"
            response.headers["Strict-Transport-Security"] = hsts_value

        # Content Security Policy
        csp_policy = os.getenv(
            "CSP_POLICY",
            "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self'",
        )
        response.headers["Content-Security-Policy"] = csp_policy

        # X-Content-Type-Options
        response.headers["X-Content-Type-Options"] = "nosniff"

        # X-Frame-Options
        frame_options = os.getenv("X_FRAME_OPTIONS", "DENY")
        response.headers["X-Frame-Options"] = frame_options

        # X-XSS-Protection (legacy, but still useful)
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Referrer-Policy
        referrer_policy = os.getenv("REFERRER_POLICY", "strict-origin-when-cross-origin")
        response.headers["Referrer-Policy"] = referrer_policy

        # Permissions-Policy (formerly Feature-Policy)
        permissions_policy = os.getenv(
            "PERMISSIONS_POLICY",
            "geolocation=(), microphone=(), camera=(), payment=(), usb=(), magnetometer=(), gyroscope=(), accelerometer=()",
        )
        response.headers["Permissions-Policy"] = permissions_policy

        # Remove server header (security through obscurity)
        if "server" in response.headers:
            # MutableHeaders does not implement pop()
            del response.headers["server"]

        return response
