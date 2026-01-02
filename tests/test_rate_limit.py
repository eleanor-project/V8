"""Tests for API rate limiting."""

import pytest
from starlette.requests import Request

from api.middleware.rate_limit import RateLimitConfig, TokenBucketRateLimiter


def _make_request(ip: str = "127.0.0.1") -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [],
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
