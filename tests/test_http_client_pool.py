import asyncio

import engine.utils.http_client as http_client


def test_http_client_pool_reuse(monkeypatch):
    async def _run():
        client1 = await http_client.get_async_client()
        client2 = await http_client.get_async_client()
        assert client1 is client2
        await http_client.aclose_async_client()
        assert client1.is_closed

    asyncio.run(_run())
