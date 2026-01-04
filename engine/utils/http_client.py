"""
Shared HTTP client pool utilities.

Provides pooled AsyncClient instances keyed by base URL to accelerate
outbound calls and keep connection reuse consistent across modules.
"""

from __future__ import annotations

import asyncio
import os
from typing import Dict, Optional
from urllib.parse import urlparse

import httpx

_clients: Dict[str, httpx.AsyncClient] = {}
_client_lock = asyncio.Lock()
_DEFAULT_CLIENT_KEY = "__default__"


def _limits() -> httpx.Limits:
    keepalive = int(os.getenv("ELEANOR_HTTP_KEEPALIVE", "20"))
    max_connections = int(os.getenv("ELEANOR_HTTP_MAX_CONNECTIONS", "100"))
    return httpx.Limits(
        max_keepalive_connections=max(1, keepalive),
        max_connections=max(1, max_connections),
    )


def _normalize_base_url(base_url: Optional[str]) -> Optional[str]:
    if not base_url:
        return None
    parsed = urlparse(base_url)
    netloc = parsed.netloc or parsed.path
    scheme = parsed.scheme or "http"
    return f"{scheme}://{netloc}"


def _client_key(base_url: Optional[str]) -> str:
    normalized = _normalize_base_url(base_url)
    return normalized or _DEFAULT_CLIENT_KEY


async def get_async_client(base_url: Optional[str] = None) -> httpx.AsyncClient:
    """Get or create a shared AsyncClient for the provided base_url."""
    return await get_async_client_for(base_url)


async def get_async_client_for(base_url: Optional[str]) -> httpx.AsyncClient:
    """Get or create an AsyncClient instance scoped to a base_url."""
    key = _client_key(base_url)
    client = _clients.get(key)
    if client is not None and not client.is_closed:
        return client
    async with _client_lock:
        client = _clients.get(key)
        if client is None or client.is_closed:
            client_args = {"limits": _limits(), "timeout": None}
            normalized = _normalize_base_url(base_url)
            if normalized:
                client_args["base_url"] = normalized
            client = httpx.AsyncClient(**client_args)
            _clients[key] = client
    return client


async def aclose_async_client(base_url: Optional[str] = None) -> None:
    """Close a specific AsyncClient or all instances if no base_url is provided."""
    if base_url is not None:
        await aclose_async_client_for(base_url)
    else:
        await aclose_all_async_clients()


async def aclose_async_client_for(base_url: Optional[str]) -> None:
    """Close the AsyncClient matching base_url."""
    key = _client_key(base_url)
    client = _clients.pop(key, None)
    if client and not client.is_closed:
        await client.aclose()


async def aclose_all_async_clients() -> None:
    """Close every pooled AsyncClient."""
    async with _client_lock:
        clients = list(_clients.values())
        _clients.clear()
    for client in clients:
        if not client.is_closed:
            await client.aclose()


__all__ = [
    "get_async_client",
    "get_async_client_for",
    "aclose_async_client",
    "aclose_async_client_for",
    "aclose_all_async_clients",
]
