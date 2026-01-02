# ELEANOR V8 — Secrets Management Guide

## Overview

ELEANOR supports pluggable secret providers to avoid storing API keys in
environment variables for production workloads. Secrets are cached with a TTL
and refreshed automatically.

Supported providers:
- Environment variables (`env`) — development only
- AWS Secrets Manager (`aws`)
- HashiCorp Vault (`vault`)

Provider dependencies:
- AWS: `boto3`
- Vault: `hvac`

## Secret Naming

Default secret names:
- `openai-api-key`
- `anthropic-api-key`
- `xai-api-key`
- `cohere-api-key`
- `gemini-api-key`

AWS secrets can be prefixed with `ELEANOR_SECURITY__AWS__SECRET_PREFIX`
(`eleanor` by default). Vault secrets are stored under
`ELEANOR_SECURITY__VAULT__MOUNT_PATH` (`secret/eleanor` by default).

## Configuration

### Environment (Development Only)

```bash
ELEANOR_SECURITY__SECRET_PROVIDER=env
```

Environment provider looks up `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`,
etc. It is blocked in production.

### AWS Secrets Manager

```bash
ELEANOR_SECURITY__SECRET_PROVIDER=aws
ELEANOR_SECURITY__AWS__REGION=us-west-2
ELEANOR_SECURITY__AWS__SECRET_PREFIX=eleanor
```

Store secrets as `eleanor/openai-api-key`, `eleanor/anthropic-api-key`, etc.

### HashiCorp Vault

```bash
ELEANOR_SECURITY__SECRET_PROVIDER=vault
ELEANOR_SECURITY__VAULT__ADDRESS=https://vault.example.com
ELEANOR_SECURITY__VAULT__TOKEN=...
ELEANOR_SECURITY__VAULT__MOUNT_PATH=secret/eleanor
```

Store secrets under `secret/eleanor/openai-api-key`, etc.

## Rotation and Refresh

Secrets are cached for `ELEANOR_SECURITY__SECRETS_CACHE_TTL` seconds (default 300).
The API refreshes the cache on a schedule and on demand when a secret expires.
Rotation failures are logged as errors.

## Sanitization

Evidence records and logs are sanitized to prevent credential leakage. Strings
matching common API key and token patterns are redacted, and any dictionary
keys containing `key`, `token`, `password`, or `secret` are replaced with
`[REDACTED]`.

## Validation

Use the config validator to confirm production settings:

```bash
python scripts/validate_config.py --env production
```
