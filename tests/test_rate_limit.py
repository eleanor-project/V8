"""Tests for API rate limiting."""

import pytest
from starlette.requests import Request

import importlib
from api.middleware.rate_limit import RateLimitConfig, TokenBucketRateLimiter

rate_limit_module = importlib.import_module("api.middleware.rate_limit")


def _make_request(ip: str = "127.0.0.1", headers=None) -> Request:
    headers = headers or []
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": headers,
        "client": (ip, 12345),
    }
    return Request(scope)


@pytest.mark.asyncio
async def test_token_bucket_blocks_after_capacity():
    config = RateLimitConfig(enabled=True, requests_per_window=2, window_seconds=60)
    limiter = TokenBucketRateLimiter(config)
    request = _make_request()

    allowed_first, _ = await limiter.check(request)
    allowed_second, _ = await limiter.check(request)
    allowed_third, headers = await limiter.check(request)

    assert allowed_first is True
    assert allowed_second is True
    assert allowed_third is False
    assert "Retry-After" in headers


def test_get_client_id_forwarded_header():
    request = _make_request(headers=[(b"x-forwarded-for", b"10.0.0.1, 127.0.0.1")])
    assert rate_limit_module._get_client_id(request) == "10.0.0.1"


def test_build_headers_includes_retry_after():
    headers = rate_limit_module._build_headers(tokens=0.2, capacity=10, refill_rate=1.0, now=100.0)
    assert headers["X-RateLimit-Limit"] == "10"
    assert "Retry-After" in headers


def test_rate_limit_config_from_env(monkeypatch):
    monkeypatch.setenv("ELEANOR_ENVIRONMENT", "production")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    with pytest.raises(ValueError):
        RateLimitConfig.from_env()

    monkeypatch.setenv("ELEANOR_ENVIRONMENT", "development")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "5")
    monkeypatch.setenv("RATE_LIMIT_WINDOW", "10")
    cfg = RateLimitConfig.from_env()
    assert cfg.requests_per_window == 5
    assert cfg.window_seconds == 10


@pytest.mark.asyncio
async def test_redis_rate_limiter_success(monkeypatch):
    class DummyRedis:
        def eval(self, *_args, **_kwargs):
            return [1, 5.0]

    monkeypatch.setattr(rate_limit_module.RedisTokenBucketRateLimiter, "_init_client", lambda *_a, **_k: DummyRedis())
    cfg = RateLimitConfig(enabled=True, requests_per_window=10, window_seconds=60, redis_url="redis://")
    limiter = rate_limit_module.RedisTokenBucketRateLimiter(cfg)
    allowed, headers = await limiter.check(_make_request())
    assert allowed is True
    assert "X-RateLimit-Remaining" in headers


@pytest.mark.asyncio
async def test_redis_rate_limiter_error(monkeypatch):
    class DummyRedis:
        def eval(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(rate_limit_module.RedisTokenBucketRateLimiter, "_init_client", lambda *_a, **_k: DummyRedis())
    cfg = RateLimitConfig(enabled=True, requests_per_window=10, window_seconds=60, redis_url="redis://")
    limiter = rate_limit_module.RedisTokenBucketRateLimiter(cfg)
    allowed, _ = await limiter.check(_make_request())
    assert allowed is True


@pytest.mark.asyncio
async def test_rate_limit_decorator_blocks():
    from fastapi import HTTPException
    limiter = rate_limit_module.rate_limit(requests_per_window=1, window_seconds=60)

    @limiter
    async def handler(request):
        return "ok"

    request = _make_request()
    assert await handler(request) == "ok"
    with pytest.raises(HTTPException):
        await handler(request)


@pytest.mark.asyncio
async def test_rate_limit_middleware_blocks(monkeypatch):
    messages = []

    class DummyLimiter:
        async def check(self, _request):
            return False, {"Retry-After": "1"}

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = rate_limit_module.RateLimitMiddleware(app, limiter=DummyLimiter())

    async def receive():
        return {"type": "http.request"}

    async def send(message):
        messages.append(message)

    await middleware({"type": "http", "method": "GET", "path": "/", "headers": []}, receive, send)
    assert messages[0]["status"] == 429
