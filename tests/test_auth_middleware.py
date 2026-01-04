import time
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from api.middleware import auth as auth_module


class DummyJWT:
    class ExpiredSignatureError(Exception):
        pass

    class InvalidTokenError(Exception):
        pass

    def __init__(self):
        self.decode = lambda *_args, **_kwargs: {}
        self.encode = lambda *_args, **_kwargs: "token"


def _reset_auth_config():
    auth_module._auth_config = None


def test_auth_config_from_env(monkeypatch):
    monkeypatch.setenv("ELEANOR_ENVIRONMENT", "production")
    monkeypatch.setenv("AUTH_ENABLED", "false")
    _reset_auth_config()
    with pytest.raises(ValueError):
        auth_module.AuthConfig.from_env()

    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.delenv("JWT_SECRET", raising=False)
    with pytest.raises(ValueError):
        auth_module.AuthConfig.from_env()

    monkeypatch.setenv("JWT_SECRET", "secret")
    cfg = auth_module.AuthConfig.from_env()
    assert cfg.secret == "secret"


def test_decode_token_no_jwt_library(monkeypatch):
    monkeypatch.setattr(auth_module, "JWT_AVAILABLE", False)
    with pytest.raises(HTTPException):
        auth_module.decode_token("token", auth_module.AuthConfig(secret="dev"))


def test_decode_token_success(monkeypatch):
    jwt_stub = DummyJWT()
    jwt_stub.decode = lambda *_args, **_kwargs: {"sub": "user", "exp": time.time() + 100}
    monkeypatch.setattr(auth_module, "JWT_AVAILABLE", True)
    monkeypatch.setattr(auth_module, "jwt", jwt_stub)

    payload = auth_module.decode_token("token", auth_module.AuthConfig(secret="dev"))
    assert payload.user_id == "user"


def test_decode_token_expired(monkeypatch):
    jwt_stub = DummyJWT()
    jwt_stub.decode = lambda *_args, **_kwargs: {"sub": "user", "exp": time.time() - 10}
    monkeypatch.setattr(auth_module, "JWT_AVAILABLE", True)
    monkeypatch.setattr(auth_module, "jwt", jwt_stub)

    with pytest.raises(HTTPException):
        auth_module.decode_token("token", auth_module.AuthConfig(secret="dev"))


def test_decode_token_invalid(monkeypatch):
    jwt_stub = DummyJWT()

    def _raise(*_args, **_kwargs):
        raise jwt_stub.InvalidTokenError("bad")

    jwt_stub.decode = _raise
    monkeypatch.setattr(auth_module, "JWT_AVAILABLE", True)
    monkeypatch.setattr(auth_module, "jwt", jwt_stub)

    with pytest.raises(HTTPException):
        auth_module.decode_token("token", auth_module.AuthConfig(secret="dev"))


@pytest.mark.asyncio
async def test_verify_token_disabled(monkeypatch):
    cfg = auth_module.AuthConfig(secret="dev-secret", enabled=False)
    monkeypatch.setattr(auth_module, "get_auth_config", lambda: cfg)
    assert await auth_module.verify_token(credentials=None) is None


@pytest.mark.asyncio
async def test_require_role_and_scope(monkeypatch):
    token = auth_module.TokenPayload({"sub": "user", "roles": ["admin"], "scopes": ["read"]})
    cfg = auth_module.AuthConfig(secret="dev-secret", enabled=True)
    monkeypatch.setattr(auth_module, "get_auth_config", lambda: cfg)

    @auth_module.require_role("admin")
    async def handler_role():
        return "ok"

    @auth_module.require_scope("read")
    async def handler_scope():
        return "ok"

    assert await handler_role(token=token) == "ok"
    assert await handler_scope(token=token) == "ok"

    with pytest.raises(HTTPException):
        await handler_role(token=auth_module.TokenPayload({"roles": []}))


def test_create_token(monkeypatch):
    jwt_stub = DummyJWT()
    monkeypatch.setattr(auth_module, "JWT_AVAILABLE", True)
    monkeypatch.setattr(auth_module, "jwt", jwt_stub)
    cfg = auth_module.AuthConfig(secret="dev-secret", token_expiry_seconds=10)

    token = auth_module.create_token("user", roles=["admin"], scopes=["read"], config=cfg)
    assert token == "token"
