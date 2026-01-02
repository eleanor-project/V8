# ELEANOR V8 — Production Deployment Checklist

**Version:** 8.0.0
**Last Updated:** 2025-12-19
**Purpose:** Verify all configuration, secrets, and infrastructure before deploying ELEANOR V8 to production

---

## 📋 Pre-Deployment Checklist

Use this checklist to ensure your ELEANOR V8 deployment is production-ready. Check each item before going live.

---

## 🔐 1. Security Configuration

### 1.1 Environment Settings

- [ ] **`ELEANOR_ENVIRONMENT=production`** (CRITICAL: NOT `development`)
  - **Location:** `.env` file
  - **Why:** Ensures production security defaults are active
  - **Verify:** `grep -E "^ELEANOR_ENVIRONMENT=|^ELEANOR_ENV=" .env`

- [ ] **`ELEANOR_SECURITY__SECRET_PROVIDER` is `aws` or `vault`** (NOT `env`)
  - **Location:** `.env` file
  - **Why:** Prevents production secrets from relying on environment variables
  - **Verify:** `grep "^ELEANOR_SECURITY__SECRET_PROVIDER=" .env`

### 1.2 JWT Secret

- [ ] **`JWT_SECRET` is set to a strong random value** (CRITICAL)
  - **Location:** `.env` file
  - **Requirements:**
    - Minimum 32 characters
    - Cryptographically random
    - NOT the default or empty string
  - **Generate:** `openssl rand -base64 32`
  - **Verify:** `[ ! -z "$JWT_SECRET" ] && [ ${#JWT_SECRET} -ge 32 ]`

### 1.3 Grafana Admin Credentials

- [ ] **`GRAFANA_ADMIN_PASSWORD` is changed from default**
  - **Location:** `.env` file or docker-compose environment
  - **Default:** `admin` (INSECURE)
  - **Requirements:** Strong password (12+ chars, mixed case, numbers, symbols)
  - **Verify:** `grep "^GRAFANA_ADMIN_PASSWORD=" .env`

- [ ] **`GRAFANA_ADMIN_USER` is optionally changed from default**
  - **Default:** `admin`
  - **Recommended:** Change to unique username in production

### 1.4 Database Credentials

- [ ] **PostgreSQL password is changed from default**
  - **Location:** `docker-compose.yaml` (line 77)
  - **Current:** `POSTGRES_PASSWORD: postgres` (INSECURE)
  - **Action:** Set `POSTGRES_PASSWORD` environment variable in `.env`
  - **Update:** `PG_CONN_STRING` in `.env` to match new password

- [ ] **Weaviate authentication is enabled for production**
  - **Location:** `docker-compose.yaml` (line 62)
  - **Current:** `AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: "true"` (INSECURE for production)
  - **Action:** Configure Weaviate API key authentication for production
  - **Documentation:** https://weaviate.io/developers/weaviate/configuration/authentication

### 1.5 CORS Configuration

- [ ] **`CORS_ORIGINS` is restricted to production domains only**
  - **Location:** `.env` file
  - **Current:** `http://localhost:3000`
  - **Production Example:** `https://eleanor.yourdomain.com,https://app.yourdomain.com`
  - **Verify:** No wildcards (`*`) allowed in production

### 1.6 Request Size Limits

- [ ] **`MAX_REQUEST_BYTES` is appropriate for your use case**
  - **Current:** `1048576` (1MB)
  - **Verify:** Sufficient but not excessive for deliberation payloads

---

## 🔑 2. API Keys & External Services

### 2.1 LLM Provider Keys

- [ ] **At least one LLM provider API key is configured**
  - **Options:**
    - [ ] `OPENAI_KEY` (OpenAI/GPT models)
    - [ ] `ANTHROPIC_KEY` (Claude models)
    - [ ] `XAI_KEY` (xAI/Grok models)
    - [ ] `GEMINI_KEY` (Google Gemini models)
    - [ ] Ollama (local models - no key required)
  - **Verify:** Keys are valid and not expired
  - **Test:** Make a simple API call to verify connectivity

### 2.2 API Key Validation Script

Run this script to validate all configured API keys:

```bash
#!/bin/bash
# validate_api_keys.sh

source .env

echo "🔍 Validating API Keys..."

# OpenAI
if [ ! -z "$OPENAI_KEY" ]; then
  echo -n "OpenAI: "
  curl -s -H "Authorization: Bearer $OPENAI_KEY" \
    https://api.openai.com/v1/models >/dev/null 2>&1 \
    && echo "✅ Valid" || echo "❌ Invalid or network error"
fi

# Anthropic
if [ ! -z "$ANTHROPIC_KEY" ]; then
  echo -n "Anthropic: "
  curl -s -H "x-api-key: $ANTHROPIC_KEY" \
    https://api.anthropic.com/v1/messages \
    -X POST -H "Content-Type: application/json" \
    -H "anthropic-version: 2023-06-01" \
    -d '{"model":"claude-3-haiku-20240307","max_tokens":1,"messages":[{"role":"user","content":"test"}]}' \
    >/dev/null 2>&1 \
    && echo "✅ Valid" || echo "❌ Invalid or network error"
fi

echo "✅ API key validation complete"
```

---

## 🗄️ 3. Database & Storage Configuration

### 3.1 Precedent Backend

- [ ] **`PRECEDENT_BACKEND` is set to production storage**
  - **Options:** `weaviate`, `pgvector`, `memory`
  - **Production Recommendation:** `weaviate` or `pgvector` (NOT `memory`)
  - **Why:** In-memory storage is lost on container restart

### 3.2 Weaviate Configuration (if using)

- [ ] **`WEAVIATE_URL` is correct**
  - **Docker Compose:** `http://weaviate:8080` (internal network)
  - **External Weaviate:** Full URL with authentication
  - **Verify:** `curl $WEAVIATE_URL/v1/.well-known/ready`

- [ ] **Weaviate data persistence is configured**
  - **Location:** `docker-compose.yaml` line 67
  - **Verify:** Volume mount exists: `./weaviate_data:/data`
  - **Check:** Directory exists and has proper permissions

### 3.3 PgVector Configuration (if using)

- [ ] **`PG_CONN_STRING` is correct**
  - **Format:** `postgresql://user:password@host:port/database`
  - **Docker Compose:** `postgresql://postgres:postgres@pgvector:5432/eleanor`
  - **Production:** Update with secure credentials
  - **Verify:** `psql "$PG_CONN_STRING" -c "SELECT 1;"`

- [ ] **`PG_TABLE` is set**
  - **Default:** `precedent`
  - **Verify:** Table will be auto-created on first run

- [ ] **PostgreSQL data persistence is configured**
  - **Location:** `docker-compose.yaml` line 82
  - **Verify:** Volume mount exists: `./pg_data:/var/lib/postgresql/data`
  - **Check:** Directory exists and has proper permissions

### 3.4 Embedding Backend

- [ ] **`EMBEDDING_BACKEND` is configured**
  - **Current:** `gpt` (uses OpenAI embeddings)
  - **Requires:** Valid `OPENAI_KEY`
  - **Alternatives:** Check if other embedding backends are supported

---

## 🏛️ 4. Governance & OPA Configuration

### 4.1 OPA Service

- [ ] **OPA service is enabled** (unless explicitly disabled for testing)
  - **Verify:** `ELEANOR_DISABLE_OPA` is NOT set to `1` or `true`
  - **Check:** OPA container is running: `docker ps | grep opa`

- [ ] **`OPA_URL` is correct**
  - **Docker Compose:** `http://opa:8181`
  - **External OPA:** Full URL
  - **Verify:** `curl $OPA_URL/health`

- [ ] **`OPA_POLICY_PATH` is correct**
  - **Default:** `v1/data/eleanor/decision`
  - **Verify:** Policy bundle is loaded at this path

- [ ] **`OPA_FAIL_STRATEGY` is set appropriately**
  - **Options:** `allow`, `deny`, `escalate`
  - **Production Recommendation:** `escalate` (enforces human review on OPA failures)
  - **Security:** NEVER use `allow` in production

### 4.2 OPA Policies

- [ ] **All required Rego policies are present**
  - **Location:** `/governance/policies/` or `docker/governance/policies/`
  - **Required Policies:**
    - [ ] `autonomy_check.rego`
    - [ ] `dignity_check.rego`
    - [ ] `fairness_check.rego`
    - [ ] `safety_gate.rego`
    - [ ] `truth_check.rego`
    - [ ] `constitutional_gate.rego`
    - [ ] `escalation.rego`
    - [ ] `reversibility.rego`
    - [ ] `pragmatics.rego`

- [ ] **OPA policies are loaded correctly**
  - **Verify:** `curl $OPA_URL/v1/policies | jq '.result'`
  - **Expected:** JSON listing all policy files

### 4.3 Constitutional Configuration

- [ ] **Constitutional YAML is present and valid**
  - **Location:** `/governance/constitutional.yaml`
  - **Verify:** File exists and parses correctly
  - **Test:** `python3 -c "import yaml; yaml.safe_load(open('governance/constitutional.yaml'))"`

---

## 📊 5. Monitoring & Observability

### 5.1 Prometheus

- [ ] **Prometheus middleware is enabled**
  - **Setting:** `ENABLE_PROMETHEUS_MIDDLEWARE=1`
  - **Verify:** Prometheus container is running: `docker ps | grep prometheus`
  - **Check Metrics:** `curl http://localhost:9090/api/v1/targets`

- [ ] **Prometheus configuration is present**
  - **Location:** `docker/prometheus.yml`
  - **Verify:** File exists and scrape configs are correct

### 5.2 Grafana

- [ ] **Grafana is accessible**
  - **URL:** `http://localhost:3000` (development) or production URL
  - **Login:** Use configured admin credentials
  - **Verify:** Prometheus data source is connected

- [ ] **Grafana dashboards are provisioned**
  - **Location:** `docker/grafana/provisioning/`
  - **Verify:** Dashboards load on Grafana startup

### 5.3 OpenTelemetry (Optional)

- [ ] **`ENABLE_OTEL` is configured if using distributed tracing**
  - **Setting:** `ENABLE_OTEL=1` (if needed)
  - **Requires:** `OTEL_EXPORTER_OTLP_ENDPOINT` and `OTEL_SERVICE_NAME`
  - **Production:** Only enable if you have an OTEL collector running

### 5.4 Audit Logging

- [ ] **Evidence path is configured and writable**
  - **Setting:** `EVIDENCE_PATH=/app/audit/evidence.jsonl`
  - **Docker Mount:** `./audit:/app/audit` (line 31 in docker-compose.yaml)
  - **Verify:** Directory exists: `mkdir -p docker/audit`
  - **Check:** Proper write permissions

- [ ] **Replay log path is configured**
  - **Setting:** `REPLAY_LOG_PATH=/app/audit/replay_log.jsonl`
  - **Verify:** Same directory as evidence path

- [ ] **Audit log rotation is configured** (for production)
  - **Recommendation:** Use logrotate or similar to prevent unbounded growth
  - **Example:** Rotate daily, keep 30 days

---

## 🚀 6. Application Configuration

### 6.1 Critic Model Bindings

- [ ] **Critic models are configured**
  - **Options:**
    - [ ] `OLLAMA_CRITIC_MODEL` (for local Ollama models)
    - [ ] `CRITIC_DEFAULT_ADAPTER` (force all critics to one adapter)
    - [ ] `CRITIC_MODEL_BINDINGS` (per-critic JSON mapping)
  - **Verify:** At least one option is set
  - **Test:** Run a deliberation and check critic outputs

### 6.2 Routing Policy (Optional)

- [ ] **Router cost limits are set (if using cost-based routing)**
  - **Settings:** `ROUTER_ADAPTER_COSTS`, `ROUTER_MAX_COST`
  - **Purpose:** Prevent expensive model calls

- [ ] **Router latency budgets are set (if using latency-based routing)**
  - **Settings:** `ROUTER_ADAPTER_LATENCIES`, `ROUTER_LATENCY_BUDGET_MS`
  - **Purpose:** Ensure response time SLAs

### 6.3 Rate Limiting

- [ ] **Rate limiting is configured appropriately**
  - **Settings:**
    - `RATE_LIMIT_REQUESTS_PER_WINDOW` (default: 30)
    - `RATE_LIMIT_WINDOW_SECONDS` (default: 60)
  - **Production:** Adjust based on expected traffic and capacity

---

## 🏗️ 7. Infrastructure & Docker

### 7.1 Docker Compose

- [ ] **`.env` file exists in project root**
  - **Location:** `/path/to/eleanor-v8/.env`
  - **Verify:** `test -f .env && echo "✅ Exists" || echo "❌ Missing"`

- [ ] **All Docker services start successfully**
  - **Command:** `cd docker && docker-compose up -d`
  - **Verify:** `docker-compose ps`
  - **Expected:** All services show "Up" status

- [ ] **Service dependencies are healthy**
  - **OPA:** `docker exec opa opa version`
  - **Weaviate:** `curl http://localhost:8080/v1/.well-known/ready`
  - **PgVector:** `docker exec pgvector psql -U postgres -d eleanor -c "SELECT version();"`
  - **Prometheus:** `curl http://localhost:9090/-/healthy`
  - **Grafana:** `curl http://localhost:3000/api/health`

### 7.2 Network Configuration

- [ ] **Docker network is created**
  - **Default:** Bridge network (auto-created)
  - **Verify:** `docker network ls | grep bridge`

- [ ] **Port mappings are correct**
  - [ ] ELEANOR API: `8000:8000`
  - [ ] OPA: `8181:8181`
  - [ ] Weaviate: `8080:8080`
  - [ ] PgVector: `5432:5432`
  - [ ] Prometheus: `9090:9090`
  - [ ] Grafana: `3000:3000`

### 7.3 Volume Persistence

- [ ] **Persistent volumes exist and have correct permissions**
  - [ ] `./audit` (evidence & replay logs)
  - [ ] `./weaviate_data` (Weaviate persistence)
  - [ ] `./pg_data` (PostgreSQL persistence)
  - **Verify:** `ls -ld docker/{audit,weaviate_data,pg_data}`

---

## 🧪 8. Pre-Production Testing

### 8.1 Health Checks

- [ ] **API health endpoint responds**
  - **Endpoint:** `GET /health`
  - **Test:** `curl http://localhost:8000/health`
  - **Expected:** `{"status": "healthy", "version": "8.0.0"}`

### 8.2 Functional Tests

- [ ] **Run full test suite**
  - **Command:** `pytest`
  - **Expected:** 313 passed, 5 skipped
  - **Coverage:** 73%+ (target: 80%+)

- [ ] **Test deliberation endpoint**
  - **Endpoint:** `POST /deliberate`
  - **Test:**
    ```bash
    curl -X POST http://localhost:8000/deliberate \
      -H "Content-Type: application/json" \
      -d '{"input": "Should we approve this loan application for a first-time buyer?"}'
    ```
  - **Expected:** Valid deliberation response with critics, aggregation, governance

- [ ] **Test WebSocket streaming**
  - **Endpoint:** `ws://localhost:8000/ws/deliberate`
  - **Test:** Use CLI: `eleanor stream "Test question"`
  - **Expected:** Progressive streaming of critic outputs

### 8.3 Security Tests

- [ ] **Run Bandit security scan**
  - **Command:** `bandit -r engine/ api/ critics/ governance/ -f json -o bandit_report.json`
  - **Expected:** No HIGH severity issues
  - **Address:** All MEDIUM severity issues (HTTP timeouts already fixed ✅)

- [ ] **Run Safety dependency check**
  - **Command:** `safety check --json`
  - **Expected:** No known vulnerabilities
  - **Action:** Update any vulnerable dependencies

### 8.4 Performance Tests

- [ ] **Detector engine performance**
  - **Target:** <2 seconds for all 25 detectors
  - **Test:**
    ```python
    import asyncio
    import time
    from engine.detectors.engine import DetectorEngineV8

    async def benchmark():
        engine = DetectorEngineV8()
        text = "Test deliberation text with ethical considerations"

        start = time.time()
        signals = await engine.detect_all(text, {})
        elapsed = time.time() - start

        print(f"⏱️  Detector suite: {elapsed:.2f}s")
        assert elapsed < 2.0, "Performance regression!"

    asyncio.run(benchmark())
    ```
  - **Expected:** <2s consistently

- [ ] **Load testing (optional but recommended)**
  - **Tool:** Use `locust`, `k6`, or similar
  - **Target:** Handle expected production RPS with <100ms p95 latency
  - **Verify:** No memory leaks or resource exhaustion

---

## 📝 9. Documentation & Compliance

### 9.1 Documentation Review

- [ ] **README.md is up to date**
  - **Check:** Version number, installation steps, deployment guide

- [ ] **INSTALLATION.md reflects production setup**
  - **Verify:** All prerequisites and steps are accurate

- [ ] **API documentation is current**
  - **Check:** Endpoint documentation matches implementation

### 9.2 Audit Trail Compliance

- [ ] **Constitutional governance documentation is accessible**
  - **Files:**
    - [ ] `POLICY_CONSTITUTIONAL_GOVERNANCE.md`
    - [ ] `governance/constitutional.yaml`
    - [ ] OPA policy files

- [ ] **Human review procedures are documented**
  - **Check:** Escalation tiers, reviewer onboarding, authority matrix

### 9.3 Incident Response

- [ ] **Production incident runbook exists**
  - **Contents:**
    - Service restart procedures
    - Database backup/restore
    - Log aggregation access
    - Escalation contacts

---

## ✅ 10. Final Pre-Launch Verification

### 10.1 Security Checklist Summary

Run this automated security check:

```bash
#!/bin/bash
# security_preflight.sh

echo "🔒 ELEANOR V8 Security Preflight Check"
echo "======================================"

ERRORS=0

# 1. Check ELEANOR_ENVIRONMENT
if grep -Eq "^ELEANOR_ENVIRONMENT=production|^ELEANOR_ENV=production" .env; then
  echo "✅ ELEANOR_ENVIRONMENT is set to production"
else
  echo "❌ ELEANOR_ENVIRONMENT must be 'production'"
  ERRORS=$((ERRORS + 1))
fi

# 2. Check JWT_SECRET
JWT_SECRET=$(grep "^JWT_SECRET=" .env | cut -d'=' -f2)
if [ ! -z "$JWT_SECRET" ] && [ ${#JWT_SECRET} -ge 32 ]; then
  echo "✅ JWT_SECRET is set and sufficiently long"
else
  echo "❌ JWT_SECRET must be at least 32 characters"
  ERRORS=$((ERRORS + 1))
fi

# 3. Check Grafana password
if grep -q "^GRAFANA_ADMIN_PASSWORD=admin" .env || \
   grep -q "GRAFANA_ADMIN_PASSWORD:-admin" docker/docker-compose.yaml; then
  echo "❌ GRAFANA_ADMIN_PASSWORD is using default 'admin'"
  ERRORS=$((ERRORS + 1))
else
  echo "✅ GRAFANA_ADMIN_PASSWORD is changed from default"
fi

# 4. Check at least one LLM key is set
if grep -q "^OPENAI_KEY=.\+" .env || \
   grep -q "^ANTHROPIC_KEY=.\+" .env || \
   grep -q "^XAI_KEY=.\+" .env || \
   grep -q "^GEMINI_KEY=.\+" .env; then
  echo "✅ At least one LLM API key is configured"
else
  echo "❌ No LLM API keys configured"
  ERRORS=$((ERRORS + 1))
fi

# 5. Check CORS is not wildcard
if grep -q "CORS_ORIGINS=.*\*" .env; then
  echo "❌ CORS_ORIGINS contains wildcard (*) - insecure for production"
  ERRORS=$((ERRORS + 1))
else
  echo "✅ CORS_ORIGINS does not contain wildcards"
fi

# 6. Check precedent backend is not 'memory'
if grep -q "^PRECEDENT_BACKEND=memory" .env; then
  echo "⚠️  WARNING: PRECEDENT_BACKEND is 'memory' - data will be lost on restart"
else
  echo "✅ PRECEDENT_BACKEND is using persistent storage"
fi

echo ""
echo "======================================"
if [ $ERRORS -eq 0 ]; then
  echo "✅ Security preflight check PASSED"
  exit 0
else
  echo "❌ Security preflight check FAILED with $ERRORS error(s)"
  echo "Fix the issues above before deploying to production"
  exit 1
fi
```

### 10.2 Service Health Check

Run this to verify all services are operational:

```bash
#!/bin/bash
# health_check.sh

echo "🏥 ELEANOR V8 Health Check"
echo "=========================="

# ELEANOR API
curl -sf http://localhost:8000/health >/dev/null && \
  echo "✅ ELEANOR API: Healthy" || \
  echo "❌ ELEANOR API: Unhealthy"

# OPA
curl -sf http://localhost:8181/health >/dev/null && \
  echo "✅ OPA: Healthy" || \
  echo "❌ OPA: Unhealthy"

# Weaviate
curl -sf http://localhost:8080/v1/.well-known/ready >/dev/null && \
  echo "✅ Weaviate: Ready" || \
  echo "❌ Weaviate: Not Ready"

# PgVector
docker exec pgvector psql -U postgres -d eleanor -c "SELECT 1;" >/dev/null 2>&1 && \
  echo "✅ PgVector: Healthy" || \
  echo "❌ PgVector: Unhealthy"

# Prometheus
curl -sf http://localhost:9090/-/healthy >/dev/null && \
  echo "✅ Prometheus: Healthy" || \
  echo "❌ Prometheus: Unhealthy"

# Grafana
curl -sf http://localhost:3000/api/health >/dev/null && \
  echo "✅ Grafana: Healthy" || \
  echo "❌ Grafana: Unhealthy"

echo "=========================="
echo "✅ Health check complete"
```

### 10.3 End-to-End Integration Test

Run a full deliberation to verify the entire pipeline:

```bash
#!/bin/bash
# e2e_test.sh

echo "🧪 Running End-to-End Integration Test"

RESPONSE=$(curl -sf -X POST http://localhost:8000/deliberate \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Should we approve a loan for a first-time homebuyer with good credit but limited savings?"
  }')

if [ $? -eq 0 ]; then
  echo "✅ Deliberation endpoint responded"

  # Check for required fields in response
  echo "$RESPONSE" | jq -e '.model_output' >/dev/null && echo "✅ model_output present" || echo "❌ model_output missing"
  echo "$RESPONSE" | jq -e '.critic_outputs' >/dev/null && echo "✅ critic_outputs present" || echo "❌ critic_outputs missing"
  echo "$RESPONSE" | jq -e '.aggregation_result' >/dev/null && echo "✅ aggregation_result present" || echo "❌ aggregation_result missing"
  echo "$RESPONSE" | jq -e '.governance' >/dev/null && echo "✅ governance present" || echo "❌ governance missing"
  echo "$RESPONSE" | jq -e '.evidence_trace_id' >/dev/null && echo "✅ evidence_trace_id present" || echo "❌ evidence_trace_id missing"

  echo ""
  echo "Trace ID: $(echo "$RESPONSE" | jq -r '.evidence_trace_id')"
else
  echo "❌ Deliberation endpoint failed"
  exit 1
fi
```

---

## 🚦 Launch Decision

### All Critical Items Must Pass

**CRITICAL** items marked with ⚠️ in sections above:
- [ ] `ELEANOR_ENVIRONMENT=production`
- [ ] `JWT_SECRET` is strong and unique
- [ ] Grafana admin password changed
- [ ] At least one LLM API key configured
- [ ] Database passwords changed from defaults
- [ ] OPA fail strategy is NOT `allow` in production
- [ ] All required OPA policies are present
- [ ] Audit logging is configured and writable
- [ ] Security preflight check passes
- [ ] Service health check passes
- [ ] End-to-end integration test passes

### Sign-Off

- [ ] **Security Review Completed** by: ______________ Date: __________
- [ ] **Infrastructure Review Completed** by: ______________ Date: __________
- [ ] **Application Review Completed** by: ______________ Date: __________

---

## 📞 Production Support

### Contacts

- **On-Call Engineer:** [Your contact info]
- **Security Team:** [Security contact]
- **Infrastructure Team:** [Infra contact]

### Monitoring URLs

- **API Health:** http://localhost:8000/health
- **Prometheus:** http://localhost:9090
- **Grafana:** http://localhost:3000
- **OPA:** http://localhost:8181/health

### Emergency Procedures

1. **API Down:** Check Docker logs: `docker logs eleanor_v8`
2. **OPA Failure:** Check policy loading: `curl http://localhost:8181/v1/policies`
3. **Database Connection:** Verify credentials and service status
4. **High Latency:** Check Grafana dashboards for detector performance

---

**Estimated Time to Complete Checklist:** 2-3 hours

**Next Steps After Completion:**
1. Deploy to staging environment
2. Run production load tests
3. Verify monitoring and alerting
4. Execute failover/disaster recovery test
5. Deploy to production with gradual rollout
