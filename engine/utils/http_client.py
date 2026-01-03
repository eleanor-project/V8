"""
Shared HTTP client pool utilities.

Provides a single AsyncClient with keep-alive pooling for outbound calls.
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

import httpx

_async_client: Optional[httpx.AsyncClient] = None
_client_lock = asyncio.Lock()


def _limits() -> httpx.Limits:
    keepalive = int(os.getenv("ELEANOR_HTTP_KEEPALIVE", "20"))
    max_connections = int(os.getenv("ELEANOR_HTTP_MAX_CONNECTIONS", "100"))
    return httpx.Limits(
        max_keepalive_connections=max(1, keepalive),
        max_connections=max(1, max_connections),
    )


async def get_async_client() -> httpx.AsyncClient:
    """Get or create a shared AsyncClient with pooled connections."""
    global _async_client
    if _async_client is not None and not _async_client.is_closed:
        return _async_client
    async with _client_lock:
        if _async_client is None or _async_client.is_closed:
            _async_client = httpx.AsyncClient(limits=_limits(), timeout=None)
    return _async_client


async def aclose_async_client() -> None:
    """Close the shared AsyncClient pool."""
    global _async_client
    if _async_client is None:
        return
    client = _async_client
    _async_client = None
    if not client.is_closed:
        await client.aclose()


__all__ = ["get_async_client", "aclose_async_client"]
