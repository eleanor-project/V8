import asyncio

import engine.utils.http_client as http_client


def test_http_client_pool_reuse():
    async def _run():
        client_default = await http_client.get_async_client()
        client_default_repeat = await http_client.get_async_client()
        assert client_default is client_default_repeat

        client_a = await http_client.get_async_client_for("https://example.com/api")
        client_b = await http_client.get_async_client_for("https://example.com/api")
        assert client_a is client_b

        client_other = await http_client.get_async_client_for("https://other.example.com/auth")
        assert client_other is not client_a

        await http_client.aclose_async_client_for("https://example.com/api")
        assert client_a.is_closed

        await http_client.aclose_async_client()
        assert client_default.is_closed
        assert client_other.is_closed

    asyncio.run(_run())
