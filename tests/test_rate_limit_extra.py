import asyncio
import importlib
import sys

import pytest
from fastapi import HTTPException
from starlette.requests import Request

rate_limit_module = importlib.import_module("api.middleware.rate_limit")


def _make_request(client_host="1.2.3.4", headers=None):
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [],
        "client": (client_host, 1234),
    }
    if headers:
        scope["headers"] = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    return Request(scope)


def test_build_headers_retry_after():
    headers = rate_limit_module._build_headers(tokens=0.2, capacity=10, refill_rate=1.0, now=100)
    assert "Retry-After" in headers
    assert headers["X-RateLimit-Limit"] == "10"


def test_env_positive_int(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_MAX_CLIENTS", "-1")
    assert rate_limit_module._env_positive_int("RATE_LIMIT_MAX_CLIENTS") is None
    monkeypatch.setenv("RATE_LIMIT_MAX_CLIENTS", "abc")
    assert rate_limit_module._env_positive_int("RATE_LIMIT_MAX_CLIENTS") is None
    monkeypatch.setenv("RATE_LIMIT_MAX_CLIENTS", "5")
    assert rate_limit_module._env_positive_int("RATE_LIMIT_MAX_CLIENTS") == 5


def test_rate_limit_config_env(monkeypatch):
    monkeypatch.setenv("ELEANOR_ENVIRONMENT", "production")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    with pytest.raises(ValueError):
        rate_limit_module.RateLimitConfig.from_env()

    monkeypatch.setenv("ELEANOR_ENVIRONMENT", "development")
    monkeypatch.setenv("RATE_LIMIT_CLIENT_TTL", "0")
    config = rate_limit_module.RateLimitConfig.from_env()
    assert config.client_ttl_seconds is None


def test_token_bucket_limiter_check_and_reset(monkeypatch):
    config = rate_limit_module.RateLimitConfig(enabled=True, requests_per_window=2, window_seconds=1)
    limiter = rate_limit_module.TokenBucketRateLimiter(config)
    request = _make_request()
    allowed, _ = asyncio.run(limiter.check(request))
    assert allowed is True
    limiter.reset(client_id="1.2.3.4")
    limiter.reset()

    config_disabled = rate_limit_module.RateLimitConfig(enabled=False)
    limiter_disabled = rate_limit_module.TokenBucketRateLimiter(config_disabled)
    allowed, headers = asyncio.run(limiter_disabled.check(request))
    assert allowed is True
    assert headers == {}


def test_token_bucket_cleanup(monkeypatch):
    config = rate_limit_module.RateLimitConfig(
        enabled=True,
        requests_per_window=1,
        window_seconds=1,
        max_clients=1,
        client_ttl_seconds=1,
    )
    limiter = rate_limit_module.TokenBucketRateLimiter(config)
    req1 = _make_request(client_host="1.1.1.1")
    req2 = _make_request(client_host="2.2.2.2")
    asyncio.run(limiter.check(req1))
    asyncio.run(limiter.check(req2))
    assert len(limiter._timestamps) <= 1


def test_redis_rate_limiter_init_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "redis", None)
    config = rate_limit_module.RateLimitConfig(redis_url="redis://localhost")
    with pytest.raises(RuntimeError):
        rate_limit_module.RedisTokenBucketRateLimiter(config)


def test_redis_rate_limiter_check(monkeypatch):
    class DummyRedis:
        def eval(self, *_args, **_kwargs):
            return [1, 0.5]

    config = rate_limit_module.RateLimitConfig(redis_url="redis://localhost")
    limiter = rate_limit_module.RedisTokenBucketRateLimiter.__new__(
        rate_limit_module.RedisTokenBucketRateLimiter
    )
    limiter.config = config
    limiter.capacity = 10.0
    limiter.refill_rate = 1.0
    limiter._redis = DummyRedis()

    request = _make_request()
    allowed, headers = asyncio.run(limiter.check(request))
    assert allowed is True
    assert headers["X-RateLimit-Limit"] == "10"

    def _raise(*_args, **_kwargs):
        raise RuntimeError("boom")

    limiter._redis.eval = _raise
    allowed, headers = asyncio.run(limiter.check(request))
    assert allowed is True
    assert headers == {}


def test_get_rate_limiter(monkeypatch):
    rate_limit_module._rate_limiter = None
    monkeypatch.setenv("RATE_LIMIT_REDIS_URL", "")
    limiter = rate_limit_module.get_rate_limiter()
    assert limiter is not None

    class DummyRedisLimiter:
        pass

    rate_limit_module._rate_limiter = None
    monkeypatch.setenv("RATE_LIMIT_REDIS_URL", "redis://local")
    monkeypatch.setattr(rate_limit_module, "RedisTokenBucketRateLimiter", lambda _cfg: DummyRedisLimiter())
    limiter = rate_limit_module.get_rate_limiter()
    assert isinstance(limiter, DummyRedisLimiter)


def test_check_rate_limit_dependency(monkeypatch):
    class DummyLimiter:
        async def check(self, _request):
            return False, {"Retry-After": "1"}

    monkeypatch.setattr(rate_limit_module, "get_rate_limiter", lambda: DummyLimiter())
    with pytest.raises(HTTPException):
        asyncio.run(rate_limit_module.check_rate_limit(_make_request()))

    class AllowedLimiter:
        async def check(self, _request):
            return True, {}

    monkeypatch.setattr(rate_limit_module, "get_rate_limiter", lambda: AllowedLimiter())
    asyncio.run(rate_limit_module.check_rate_limit(_make_request()))


def test_rate_limit_decorator(monkeypatch):
    class DummyLimiter:
        async def check(self, _request):
            return True, {}

    monkeypatch.setattr(rate_limit_module, "TokenBucketRateLimiter", lambda _cfg: DummyLimiter())

    @rate_limit_module.rate_limit(requests_per_window=1, window_seconds=1)
    async def handler(request):
        return "ok"

    result = asyncio.run(handler(_make_request()))
    assert result == "ok"

    class BlockedLimiter:
        async def check(self, _request):
            return False, {"Retry-After": "1"}

    monkeypatch.setattr(rate_limit_module, "TokenBucketRateLimiter", lambda _cfg: BlockedLimiter())

    @rate_limit_module.rate_limit(requests_per_window=1, window_seconds=1)
    async def handler_blocked(request):
        return "ok"

    with pytest.raises(HTTPException):
        asyncio.run(handler_blocked(_make_request()))


def test_rate_limit_middleware():
    events = []

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    class BlockedLimiter:
        async def check(self, _request):
            return False, {"Retry-After": "1"}

    middleware = rate_limit_module.RateLimitMiddleware(app, limiter=BlockedLimiter())

    async def send(message):
        events.append(message)

    asyncio.run(
        middleware({"type": "http", "headers": [], "client": ("1.1.1.1", 0)}, None, send)
    )
    assert events[0]["status"] == 429

    events.clear()

    class AllowedLimiter:
        async def check(self, _request):
            return True, {"X-RateLimit-Limit": "1"}

    middleware = rate_limit_module.RateLimitMiddleware(app, limiter=AllowedLimiter())
    asyncio.run(
        middleware({"type": "http", "headers": [], "client": ("1.1.1.1", 0)}, None, send)
    )
    headers = dict((k.decode(), v.decode()) for k, v in events[0]["headers"])
    assert headers["X-RateLimit-Limit"] == "1"
