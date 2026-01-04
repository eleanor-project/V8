import pytest
from fastapi import HTTPException

from api.middleware import auth as auth_module


def _reset_auth():
    auth_module._auth_config = None


def test_auth_config_dev_disabled(monkeypatch):
    monkeypatch.setenv("ELEANOR_ENVIRONMENT", "development")
    monkeypatch.setenv("AUTH_ENABLED", "false")
    _reset_auth()
    cfg = auth_module.AuthConfig.from_env()
    assert cfg.enabled is False
    assert cfg.secret == "dev-secret"


def test_auth_config_prod_default_secret(monkeypatch):
    monkeypatch.setenv("ELEANOR_ENVIRONMENT", "production")
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("JWT_SECRET", "dev-secret")
    _reset_auth()
    with pytest.raises(ValueError):
        auth_module.AuthConfig.from_env()


@pytest.mark.asyncio
async def test_verify_token_requires_credentials(monkeypatch):
    cfg = auth_module.AuthConfig(secret="dev-secret", enabled=True)
    monkeypatch.setattr(auth_module, "get_auth_config", lambda: cfg)
    with pytest.raises(HTTPException):
        await auth_module.verify_token(credentials=None)


@pytest.mark.asyncio
async def test_require_scope_missing(monkeypatch):
    cfg = auth_module.AuthConfig(secret="dev-secret", enabled=True)
    monkeypatch.setattr(auth_module, "get_auth_config", lambda: cfg)

    @auth_module.require_scope("write")
    async def handler():
        return "ok"

    with pytest.raises(HTTPException):
        await handler(token=auth_module.TokenPayload({"scopes": ["read"]}))


@pytest.mark.asyncio
async def test_get_current_user(monkeypatch):
    token = auth_module.TokenPayload({"sub": "user"})
    assert await auth_module.get_current_user(token) == "user"
    assert await auth_module.get_current_user(None) is None


def test_create_token_no_jwt(monkeypatch):
    monkeypatch.setattr(auth_module, "JWT_AVAILABLE", False)
    with pytest.raises(RuntimeError):
        auth_module.create_token("user")
