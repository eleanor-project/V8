# ELEANOR V8 Appliance — Docker Deployment

This directory contains everything needed to run the entire
ELEANOR V8 governance engine as a fully integrated local appliance.

## Components

### 1. ELEANOR API
- FastAPI (REST)
- WebSocket streaming
- CLI integration
- Unified Engine Runtime
- Audit logging

### 2. OPA (Open Policy Agent)
Houses the V8 constitutional policies and enforces allow/deny rules.

### 3. Vector DBs for Precedent Retrieval
Two options (controlled via PRECEDENT_BACKEND):

#### Weaviate (default)
- Flexible
- Fast for prototyping
- No schema migrations

#### PGVector
- Stable for enterprise deployments
- Uses PostgreSQL with pgvector extension
- Great for large precedent stores

### 4. Networking
All services run inside a shared docker network.

### 5. Observability
- Prometheus scrapes:
  - `eleanor:8000/metrics` (API + middleware)
  - `opa:8181/metrics`
- Grafana pre-provisioned with Prometheus datasource
- Optional OpenTelemetry export to an OTLP collector

---

## Running the Appliance

```bash
cd docker
./start.sh
```

or directly:

```bash
docker compose up --build
```

## Environment

Copy the sample env and adjust secrets/keys:

```bash
cp ../.env.sample ../.env
```

Key env vars:
- Model adapters: `OPENAI_KEY`, `ANTHROPIC_KEY`, `XAI_KEY`
- Router policy: `ROUTER_ADAPTER_COSTS`, `ROUTER_MAX_COST`, `ROUTER_ADAPTER_LATENCIES`, `ROUTER_LATENCY_BUDGET_MS`
- Precedent: `PRECEDENT_BACKEND` (`weaviate|pgvector|memory`), `WEAVIATE_URL`, `PG_CONN_STRING`
- Governance: `OPA_URL`, `OPA_POLICY_PATH`, `OPA_FAIL_STRATEGY`
- Metrics/Tracing: `ENABLE_PROMETHEUS_MIDDLEWARE`, `ENABLE_OTEL`, `OTEL_EXPORTER_OTLP_ENDPOINT`
- Evidence/Replays: `EVIDENCE_PATH`, `REPLAY_LOG_PATH`

Prometheus and Grafana are included in `docker-compose.yaml`:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (default admin/admin, override via `GRAFANA_ADMIN_USER/PASSWORD`)
