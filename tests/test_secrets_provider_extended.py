import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from engine.security import secrets as secrets_module


class DummyClientError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class DummyPaginator:
    def __init__(self, pages):
        self.pages = pages

    def paginate(self):
        return self.pages


class DummyAWSClient:
    def __init__(self, secret_value="value", error_code=None, pages=None):
        self.secret_value = secret_value
        self.error_code = error_code
        self.pages = pages or []
        self.get_secret_value = MagicMock(side_effect=self._get_secret_value)

    def _get_secret_value(self, **_kwargs):
        if self.error_code:
            raise DummyClientError(self.error_code)
        if self.secret_value is None:
            return {"SecretString": None}
        return {"SecretString": self.secret_value}

    def get_paginator(self, _name):
        return DummyPaginator(self.pages)


class DummyVaultKV:
    def __init__(self, value="vault-value", keys=None, raise_exc=False):
        self.value = value
        self.keys = keys or []
        self.raise_exc = raise_exc

    def read_secret_version(self, **_kwargs):
        if self.raise_exc:
            raise RuntimeError("boom")
        return {"data": {"data": {"value": self.value}}}

    def list_secrets(self, **_kwargs):
        if self.raise_exc:
            raise RuntimeError("boom")
        return {"data": {"keys": self.keys}}


class DummyVaultClient:
    def __init__(self, kv):
        self.secrets = SimpleNamespace(kv=SimpleNamespace(v2=kv))
        self._authenticated = True

    def is_authenticated(self):
        return self._authenticated


def test_get_llm_api_key_sync_env_fallback(monkeypatch):
    provider = secrets_module.EnvironmentSecretsProvider(prefix="TEST_")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    assert secrets_module.get_llm_api_key_sync("openai", provider) == "env-key"


@pytest.mark.asyncio
async def test_get_llm_api_key_async(monkeypatch):
    provider = secrets_module.EnvironmentSecretsProvider(prefix="TEST_")
    monkeypatch.setenv("XAI_API_KEY", "xai-key")
    value = await secrets_module.get_llm_api_key("xai", provider)
    assert value == "xai-key"


def test_build_secret_provider_from_settings_env(monkeypatch):
    settings = SimpleNamespace(
        security=SimpleNamespace(
            secret_provider="env",
            secrets_cache_ttl=123,
            aws=SimpleNamespace(region="us-east-1", secret_prefix="eleanor"),
            vault=SimpleNamespace(address=None, token=None, mount_path="secret/eleanor"),
        )
    )
    provider = secrets_module.build_secret_provider_from_settings(settings)
    assert isinstance(provider, secrets_module.EnvironmentSecretProvider)
    assert provider.cache_ttl == 123


def test_build_secret_provider_from_settings_invalid():
    settings = SimpleNamespace(
        security=SimpleNamespace(
            secret_provider="invalid",
            secrets_cache_ttl=10,
            aws=SimpleNamespace(region="us-east-1", secret_prefix="eleanor"),
            vault=SimpleNamespace(address=None, token=None, mount_path="secret/eleanor"),
        )
    )
    with pytest.raises(ValueError, match="Unknown secret_provider"):
        secrets_module.build_secret_provider_from_settings(settings)


def test_aws_secrets_provider_cache_and_list(monkeypatch):
    aws_client = DummyAWSClient(secret_value="secret-value", pages=[{"SecretList": []}])
    boto3_stub = SimpleNamespace(client=lambda *_args, **_kwargs: aws_client)
    monkeypatch.setattr(secrets_module, "boto3", boto3_stub)
    monkeypatch.setattr(secrets_module, "ClientError", DummyClientError)

    provider = secrets_module.AWSSecretsProvider(region_name="us-east-1", cache_ttl=300, prefix="p/")
    value1 = provider.get_secret("key")
    value2 = provider.get_secret("key")
    assert value1 == "secret-value"
    assert value2 == "secret-value"
    assert aws_client.get_secret_value.call_count == 1

    aws_client.pages = [
        {"SecretList": [{"Name": "p/one"}, {"Name": "other"}]},
        {"SecretList": [{"Name": "p/two"}]},
    ]
    assert provider.list_secrets() == ["one", "two"]


def test_aws_secrets_provider_errors(monkeypatch):
    aws_client = DummyAWSClient(secret_value=None, error_code="ResourceNotFoundException")
    boto3_stub = SimpleNamespace(client=lambda *_args, **_kwargs: aws_client)
    monkeypatch.setattr(secrets_module, "boto3", boto3_stub)
    monkeypatch.setattr(secrets_module, "ClientError", DummyClientError)

    provider = secrets_module.AWSSecretsProvider(region_name="us-east-1", cache_ttl=1, prefix="p/")
    assert provider.get_secret("missing") is None

    aws_client.error_code = "AccessDeniedException"
    assert provider.get_secret("denied") is None


def test_vault_secrets_provider(monkeypatch):
    kv = DummyVaultKV(value="vault-secret", keys=["a", "b"])
    vault_client = DummyVaultClient(kv)
    hvac_stub = SimpleNamespace(Client=lambda *_args, **_kwargs: vault_client)
    monkeypatch.setattr(secrets_module, "hvac", hvac_stub)

    provider = secrets_module.VaultSecretsProvider(
        vault_addr="http://vault",
        vault_token="token",
        mount_point="mount",
    )
    assert provider.get_secret("key") == "vault-secret"
    assert provider.list_secrets() == ["a", "b"]


def test_vault_secrets_provider_error_paths(monkeypatch):
    kv = DummyVaultKV(value=None, keys=None, raise_exc=True)
    vault_client = DummyVaultClient(kv)
    hvac_stub = SimpleNamespace(Client=lambda *_args, **_kwargs: vault_client)
    monkeypatch.setattr(secrets_module, "hvac", hvac_stub)

    provider = secrets_module.VaultSecretsProvider(
        vault_addr="http://vault",
        vault_token="token",
        mount_point="mount",
    )
    assert provider.get_secret("key") is None
    assert provider.list_secrets() == []


def test_auto_detect_secrets_provider_fallback(monkeypatch):
    monkeypatch.setenv("AWS_SECRETS_MANAGER", "true")
    monkeypatch.setattr(secrets_module, "boto3", None)
    provider = secrets_module.auto_detect_secrets_provider(cache_ttl=10)
    assert isinstance(provider, secrets_module.EnvironmentSecretProvider)
