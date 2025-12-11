# Governance Layer

The governance layer enforces policy decisions over engine outputs using OPA (or another PDP).

- **OPA Client**: `engine/governance/opa_client.py` posts the deliberation bundle (`critics`, `aggregator`, `precedent`, `uncertainty`, `model_used`, `user_input`) to `OPA_URL`/`OPA_POLICY_PATH`. Defaults: `http://localhost:8181` and `v1/data/eleanor/decision`.
- **Engine hook**: `engine/core/__init__.py` wires `opa_callback` into the async engine; you can override with a custom callback or use the default OPA client.
- **Error handling**: Non-200 or JSON errors return `allow=False` and `escalate=True` with failure metadata to avoid silent passes.
- **Policy expectations**: OPA should return `{ "result": { "allow": bool, "escalate": bool, "failures": [...] } }`. Deny or escalate overrides aggregator decisions per `_resolve_output` in `engine/core/engine.py`.
- **Configuration**: Environment variables `OPA_URL`, `OPA_POLICY_PATH` control the endpoint. Provide authentication via standard OPA gateway/front-door if needed.

Future hardening: add PDP health checks on startup, JWT-signed evidence payloads, and Merkle-stamped evidence artifacts for audit immutability.
