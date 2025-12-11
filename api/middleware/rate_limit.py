"""
ELEANOR V8 — Rate Limiting Middleware
-------------------------------------

Simple in-memory rate limiting for the ELEANOR V8 API.

For production deployments, consider using Redis-backed rate limiting.

Configuration via environment variables:
- RATE_LIMIT_ENABLED: Enable/disable rate limiting (default: true)
- RATE_LIMIT_REQUESTS: Max requests per window (default: 100)
- RATE_LIMIT_WINDOW: Window size in seconds (default: 60)
"""

import os
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
from typing import Dict, Tuple, Optional, Callable
import threading

from fastapi import HTTPException, Request, status


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    enabled: bool = True
    requests_per_window: int = 100
    window_seconds: int = 60

    @classmethod
    def from_env(cls) -> "RateLimitConfig":
        """Load configuration from environment variables."""
        return cls(
            enabled=os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true",
            requests_per_window=int(os.getenv("RATE_LIMIT_REQUESTS", "100")),
            window_seconds=int(os.getenv("RATE_LIMIT_WINDOW", "60"))
        )


class RateLimiter:
    """
    In-memory sliding window rate limiter.

    Thread-safe implementation using a lock.
    For distributed deployments, use Redis-backed rate limiting.
    """

    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig.from_env()
        self._requests: Dict[str, list] = defaultdict(list)
        self._lock = threading.Lock()

    def _get_client_id(self, request: Request) -> str:
        """
        Get a unique identifier for the client.

        Uses X-Forwarded-For if behind a proxy, otherwise uses client host.
        """
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Take the first IP in the chain (original client)
            return forwarded.split(",")[0].strip()

        return request.client.host if request.client else "unknown"

    def _cleanup_old_requests(self, client_id: str, now: float) -> None:
        """Remove requests outside the current window."""
        cutoff = now - self.config.window_seconds
        self._requests[client_id] = [
            ts for ts in self._requests[client_id] if ts > cutoff
        ]

    def check(self, request: Request) -> Tuple[bool, Dict[str, str]]:
        """
        Check if a request is allowed under rate limiting.

        Returns:
            Tuple of (allowed: bool, headers: dict)
        """
        if not self.config.enabled:
            return True, {}

        client_id = self._get_client_id(request)
        now = time.time()

        with self._lock:
            self._cleanup_old_requests(client_id, now)

            current_count = len(self._requests[client_id])
            remaining = max(0, self.config.requests_per_window - current_count)
            reset_time = int(now + self.config.window_seconds)

            headers = {
                "X-RateLimit-Limit": str(self.config.requests_per_window),
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Reset": str(reset_time)
            }

            if current_count >= self.config.requests_per_window:
                headers["Retry-After"] = str(self.config.window_seconds)
                return False, headers

            # Record this request
            self._requests[client_id].append(now)
            headers["X-RateLimit-Remaining"] = str(remaining - 1)

            return True, headers

    def reset(self, client_id: str = None) -> None:
        """Reset rate limit counters. Useful for testing."""
        with self._lock:
            if client_id:
                self._requests.pop(client_id, None)
            else:
                self._requests.clear()


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create the global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


async def check_rate_limit(request: Request) -> None:
    """
    FastAPI dependency for rate limiting.

    Usage:
        @app.post("/deliberate")
        async def deliberate(request: Request, _: None = Depends(check_rate_limit)):
            ...
    """
    limiter = get_rate_limiter()
    allowed, headers = limiter.check(request)

    # Always add rate limit headers to response
    # Note: This requires middleware to actually add headers to response

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
            headers=headers
        )


def rate_limit(
    requests_per_window: int = None,
    window_seconds: int = None
):
    """
    Decorator for rate limiting specific endpoints.

    Usage:
        @app.post("/deliberate")
        @rate_limit(requests_per_window=10, window_seconds=60)
        async def deliberate(request: Request):
            ...
    """
    def decorator(func: Callable):
        # Create endpoint-specific limiter if custom config provided
        if requests_per_window or window_seconds:
            config = RateLimitConfig(
                enabled=True,
                requests_per_window=requests_per_window or 100,
                window_seconds=window_seconds or 60
            )
            limiter = RateLimiter(config)
        else:
            limiter = get_rate_limiter()

        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            allowed, headers = limiter.check(request)

            if not allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded. Please try again later.",
                    headers=headers
                )

            return await func(request, *args, **kwargs)

        return wrapper
    return decorator


class RateLimitMiddleware:
    """
    ASGI middleware for global rate limiting.

    Adds rate limit headers to all responses.
    """

    def __init__(self, app, limiter: RateLimiter = None):
        self.app = app
        self.limiter = limiter or get_rate_limiter()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Create a mock request to check rate limit
        from starlette.requests import Request
        request = Request(scope)

        allowed, headers = self.limiter.check(request)

        if not allowed:
            # Send 429 response
            response_headers = [
                (k.encode(), v.encode()) for k, v in headers.items()
            ]
            response_headers.append((b"content-type", b"application/json"))

            await send({
                "type": "http.response.start",
                "status": 429,
                "headers": response_headers
            })
            await send({
                "type": "http.response.body",
                "body": b'{"error": "Rate limit exceeded", "detail": "Please try again later."}'
            })
            return

        # Add rate limit headers to response
        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                existing_headers = list(message.get("headers", []))
                for key, value in headers.items():
                    existing_headers.append((key.encode(), value.encode()))
                message["headers"] = existing_headers
            await send(message)

        await self.app(scope, receive, send_with_headers)
