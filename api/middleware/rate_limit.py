"""
ELEANOR V8 — Rate Limiting Middleware
-------------------------------------

Token-bucket rate limiting for the ELEANOR V8 API with an optional Redis backend.

Configuration via environment variables:
- RATE_LIMIT_ENABLED: Enable/disable rate limiting (default: true)
- RATE_LIMIT_REQUESTS: Token bucket capacity (default: 100)
- RATE_LIMIT_WINDOW: Seconds to refill to full capacity (default: 60)
- RATE_LIMIT_MAX_CLIENTS: Optional cap on tracked clients (default: unset)
- RATE_LIMIT_CLIENT_TTL: Optional idle eviction in seconds (default: window*10)
- RATE_LIMIT_REDIS_URL: Optional Redis URL for distributed limiting
- RATE_LIMIT_KEY_PREFIX: Optional Redis key prefix
"""

import asyncio
import logging
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
from typing import Dict, Tuple, Optional, Callable, Any
import threading

from fastapi import HTTPException, Request, status

logger = logging.getLogger(__name__)


def _get_client_id(request: Request) -> str:
    """
    Get a unique identifier for the client.

    Uses X-Forwarded-For if behind a proxy, otherwise uses client host.
    """
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _build_headers(
    tokens: float, capacity: float, refill_rate: float, now: float
) -> Dict[str, str]:
    remaining = max(0, int(math.floor(tokens)))
    if refill_rate > 0:
        reset_in = max(0.0, (capacity - tokens) / refill_rate)
    else:
        reset_in = 0.0
    headers = {
        "X-RateLimit-Limit": str(int(capacity)),
        "X-RateLimit-Remaining": str(remaining),
        "X-RateLimit-Reset": str(int(now + reset_in)),
    }
    if tokens < 1.0 and refill_rate > 0:
        retry_after = max(1, int(math.ceil((1.0 - tokens) / refill_rate)))
        headers["Retry-After"] = str(retry_after)
    return headers


def _env_positive_int(name: str) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    enabled: bool = True
    requests_per_window: int = 100
    window_seconds: int = 60
    max_clients: Optional[int] = None
    client_ttl_seconds: Optional[int] = None
    redis_url: Optional[str] = None
    key_prefix: str = "eleanor:rate_limit:"

    @classmethod
    def from_env(cls) -> "RateLimitConfig":
        """Load configuration from environment variables."""
        env = os.getenv("ELEANOR_ENVIRONMENT") or os.getenv("ELEANOR_ENV") or "development"
        enabled = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
        if env != "development" and not enabled:
            raise ValueError("RATE_LIMIT_ENABLED cannot be false in production environments")
        window_seconds = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
        ttl_raw = os.getenv("RATE_LIMIT_CLIENT_TTL")
        if ttl_raw is None:
            ttl_seconds = window_seconds * 10
        else:
            try:
                ttl_seconds = int(ttl_raw)
            except ValueError:
                ttl_seconds = window_seconds * 10
            if ttl_seconds <= 0:
                ttl_seconds = None
        return cls(
            enabled=enabled,
            requests_per_window=int(os.getenv("RATE_LIMIT_REQUESTS", "100")),
            window_seconds=window_seconds,
            max_clients=_env_positive_int("RATE_LIMIT_MAX_CLIENTS"),
            client_ttl_seconds=ttl_seconds,
            redis_url=os.getenv("RATE_LIMIT_REDIS_URL") or None,
            key_prefix=os.getenv("RATE_LIMIT_KEY_PREFIX", "eleanor:rate_limit:"),
        )


class TokenBucketRateLimiter:
    """
    In-memory token bucket rate limiter.

    Thread-safe implementation using a lock.
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.capacity = float(config.requests_per_window)
        self.refill_rate = self.capacity / float(config.window_seconds)
        self._tokens: Dict[str, float] = defaultdict(lambda: self.capacity)
        self._timestamps: Dict[str, float] = {}
        self._last_cleanup = time.time()
        self._cleanup_interval = 60.0
        self.max_clients = config.max_clients
        self.client_ttl_seconds = config.client_ttl_seconds
        if self.client_ttl_seconds:
            self._cleanup_interval = max(1.0, min(60.0, self.client_ttl_seconds / 2))
        self._lock = threading.Lock()

    async def check(self, request: Request) -> Tuple[bool, Dict[str, str]]:
        """
        Check if a request is allowed under rate limiting.

        Returns:
            Tuple of (allowed: bool, headers: dict)
        """
        if not self.config.enabled:
            return True, {}

        client_id = _get_client_id(request)
        now = time.time()

        with self._lock:
            last_ts = self._timestamps.get(client_id, now)
            tokens = self._tokens.get(client_id, self.capacity)
            elapsed = max(0.0, now - last_ts)
            tokens = min(self.capacity, tokens + elapsed * self.refill_rate)
            allowed = tokens >= 1.0
            if allowed:
                tokens -= 1.0
            self._tokens[client_id] = tokens
            self._timestamps[client_id] = now
            if self._should_cleanup(now):
                self._cleanup_locked(now)

        headers = _build_headers(tokens, self.capacity, self.refill_rate, now)
        return allowed, headers

    def reset(self, client_id: Optional[str] = None) -> None:
        """Reset rate limit counters. Useful for testing."""
        with self._lock:
            if client_id:
                self._tokens.pop(client_id, None)
                self._timestamps.pop(client_id, None)
            else:
                self._tokens.clear()
                self._timestamps.clear()
            self._last_cleanup = time.time()

    def _should_cleanup(self, now: float) -> bool:
        if self.max_clients and len(self._timestamps) > self.max_clients:
            return True
        if self.client_ttl_seconds:
            return now - self._last_cleanup >= self._cleanup_interval
        return False

    def _cleanup_locked(self, now: float) -> None:
        if self.client_ttl_seconds:
            cutoff = now - self.client_ttl_seconds
            stale = [client for client, ts in self._timestamps.items() if ts < cutoff]
            for client in stale:
                self._timestamps.pop(client, None)
                self._tokens.pop(client, None)

        if self.max_clients and len(self._timestamps) > self.max_clients:
            excess = len(self._timestamps) - self.max_clients
            oldest = sorted(self._timestamps.items(), key=lambda item: item[1])[:excess]
            for client, _ in oldest:
                self._timestamps.pop(client, None)
                self._tokens.pop(client, None)

        self._last_cleanup = now


_REDIS_SCRIPT = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])

local data = redis.call("HMGET", key, "tokens", "ts")
local tokens = tonumber(data[1])
local ts = tonumber(data[2])

if tokens == nil then
  tokens = capacity
end
if ts == nil then
  ts = now
end
if now < ts then
  ts = now
end

local delta = now - ts
tokens = math.min(capacity, tokens + (delta * refill_rate))
local allowed = tokens >= 1
if allowed then
  tokens = tokens - 1
end

redis.call("HMSET", key, "tokens", tokens, "ts", now)
redis.call("EXPIRE", key, ttl)

return {allowed and 1 or 0, tokens}
"""


class RedisTokenBucketRateLimiter:
    """Redis-backed token bucket rate limiter for distributed deployments."""

    def __init__(self, config: RateLimitConfig):
        if not config.redis_url:
            raise ValueError("redis_url required for RedisTokenBucketRateLimiter")
        self.config = config
        self.capacity = float(config.requests_per_window)
        self.refill_rate = self.capacity / float(config.window_seconds)
        self._redis = self._init_client(config.redis_url)

    @staticmethod
    def _init_client(redis_url: str):
        try:
            import redis  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("redis package is required for Redis rate limiting") from exc
        return redis.Redis.from_url(redis_url, decode_responses=True)

    async def check(self, request: Request) -> Tuple[bool, Dict[str, str]]:
        if not self.config.enabled:
            return True, {}

        client_id = _get_client_id(request)
        now = time.time()
        key = f"{self.config.key_prefix}{client_id}"
        ttl = int(max(self.config.window_seconds * 2, 1))

        try:
            result = await asyncio.to_thread(
                self._redis.eval,
                _REDIS_SCRIPT,
                1,
                key,
                self.capacity,
                self.refill_rate,
                now,
                ttl,
            )
        except Exception as exc:
            logger.warning("rate_limit_redis_error", extra={"error": str(exc)})
            return True, {}

        allowed = bool(result[0])
        tokens = float(result[1])
        headers = _build_headers(tokens, self.capacity, self.refill_rate, now)
        return allowed, headers

    async def reset(self, client_id: Optional[str] = None) -> None:
        if client_id:
            await asyncio.to_thread(self._redis.delete, f"{self.config.key_prefix}{client_id}")


# Global rate limiter instance
_rate_limiter: Optional[Any] = None


def get_rate_limiter() -> Any:
    """Get or create the global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        config = RateLimitConfig.from_env()
        if config.redis_url:
            _rate_limiter = RedisTokenBucketRateLimiter(config)
        else:
            _rate_limiter = TokenBucketRateLimiter(config)
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
    allowed, headers = await limiter.check(request)

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
            headers=headers,
        )


def rate_limit(
    requests_per_window: Optional[int] = None,
    window_seconds: Optional[int] = None,
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
        config = RateLimitConfig(
            enabled=True,
            requests_per_window=requests_per_window or 100,
            window_seconds=window_seconds or 60,
            redis_url=os.getenv("RATE_LIMIT_REDIS_URL") or None,
            key_prefix=os.getenv("RATE_LIMIT_KEY_PREFIX", "eleanor:rate_limit:"),
        )
        if config.redis_url:
            limiter = RedisTokenBucketRateLimiter(config)
        else:
            limiter = TokenBucketRateLimiter(config)

        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            allowed, headers = await limiter.check(request)

            if not allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded. Please try again later.",
                    headers=headers,
                )

            return await func(request, *args, **kwargs)

        return wrapper

    return decorator


class RateLimitMiddleware:
    """
    ASGI middleware for global rate limiting.

    Adds rate limit headers to all responses.
    """

    def __init__(self, app, limiter: Optional[Any] = None):
        self.app = app
        self.limiter = limiter or get_rate_limiter()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        from starlette.requests import Request

        request = Request(scope)

        allowed, headers = await self.limiter.check(request)

        if not allowed:
            response_headers = [(k.encode(), v.encode()) for k, v in headers.items()]
            response_headers.append((b"content-type", b"application/json"))

            await send(
                {
                    "type": "http.response.start",
                    "status": 429,
                    "headers": response_headers,
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b'{"error": "Rate limit exceeded", "detail": "Please try again later."}',
                }
            )
            return

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                existing_headers = list(message.get("headers", []))
                for key, value in headers.items():
                    existing_headers.append((key.encode(), value.encode()))
                message["headers"] = existing_headers
            await send(message)

        await self.app(scope, receive, send_with_headers)


# Backwards-compatible alias for older imports.
RateLimiter = TokenBucketRateLimiter
