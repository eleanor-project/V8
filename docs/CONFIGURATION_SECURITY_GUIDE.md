# ELEANOR V8 — Configuration Security Guide

**Version:** 8.0.0
**Last Updated:** 2025-12-19
**Audience:** DevOps Engineers, Security Teams, System Administrators

---

## 🎯 Quick Start (5 Minutes to Secure Configuration)

```bash
# 1. Generate secure secrets
./scripts/generate_secrets.sh

# 2. Add your LLM API keys to .env
nano .env  # Add OPENAI_KEY, ANTHROPIC_KEY, etc.

# 3. Validate configuration
./scripts/validate_env.sh

# 4. Update docker-compose PG_CONN_STRING with new password
# Edit .env and replace REPLACE_WITH_POSTGRES_PASSWORD

# 5. Deploy
cd docker && docker-compose -f docker-compose.production.yaml up -d
```

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Critical Security Requirements](#critical-security-requirements)
3. [Step-by-Step Configuration](#step-by-step-configuration)
4. [Production Deployment](#production-deployment)
5. [Security Best Practices](#security-best-practices)
6. [Troubleshooting](#troubleshooting)
7. [Secrets Management](#secrets-management)
8. [Compliance & Audit](#compliance--audit)

---

## Overview

### What This Guide Covers

This guide provides comprehensive instructions for securely configuring ELEANOR V8 for production deployment. It covers:

- **Secrets Generation**: Cryptographically secure secret keys and passwords
- **Environment Configuration**: Production-ready .env setup
- **Container Security**: Docker Compose security hardening
- **Validation**: Automated configuration verification
- **Compliance**: Audit logging and constitutional governance requirements

### Security Architecture

ELEANOR V8 requires secure configuration across multiple layers:

```
┌─────────────────────────────────────────────────────┐
│              Secrets & Credentials                  │
├─────────────────────────────────────────────────────┤
│  • JWT_SECRET (256-bit)                             │
│  • LLM API Keys (OpenAI, Anthropic, XAI, Gemini)    │
│  • Database Passwords (PostgreSQL)                  │
│  • Service Credentials (Grafana, Weaviate)          │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│         Environment Configuration (.env)            │
├─────────────────────────────────────────────────────┤
│  • ELEANOR_ENVIRONMENT=production                   │
│  • OPA governance settings                          │
│  • CORS policy                                      │
│  • Rate limiting                                    │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│      Container Orchestration (Docker Compose)       │
├─────────────────────────────────────────────────────┤
│  • Health checks                                    │
│  • Resource limits                                  │
│  • Network isolation                                │
│  • Volume persistence                               │
└─────────────────────────────────────────────────────┘
```

---

## Critical Security Requirements

### ⚠️ MUST-HAVE Before Production

These items are **NON-NEGOTIABLE** for production deployment:

- [ ] **`ELEANOR_ENVIRONMENT=production`** (NOT development)
- [ ] **`JWT_SECRET`** is 32+ characters, cryptographically random
- [ ] **At least ONE LLM API key** configured (OpenAI, Anthropic, XAI, or Gemini)
- [ ] **`POSTGRES_PASSWORD`** changed from default "postgres"
- [ ] **`GRAFANA_ADMIN_PASSWORD`** changed from default "admin"
- [ ] **`OPA_FAIL_STRATEGY`** is "escalate" or "deny" (NOT "allow")
- [ ] **`ELEANOR_DISABLE_OPA`** is 0 or false (OPA governance enabled)
- [ ] **`PRECEDENT_BACKEND`** is "weaviate" or "pgvector" (NOT "memory")
- [ ] **`CORS_ORIGINS`** uses specific domains (NO wildcards or localhost)
- [ ] **All secrets** stored securely (NOT in version control)

### 🔐 Security Posture Matrix

| Component | Development | Staging | Production |
|-----------|-------------|---------|------------|
| `ELEANOR_ENVIRONMENT` | development | staging | **production** |
| `JWT_SECRET` | Can use weak | Strong (32+ chars) | **Cryptographic (32+ chars)** |
| LLM API Keys | Test keys OK | Isolated keys | **Production keys** |
| Database Password | Simple OK | Strong | **Complex (20+ chars)** |
| OPA Governance | Can disable | Enabled | **REQUIRED** |
| CORS Policy | `*` OK | Restricted | **Strict (no wildcards)** |
| Audit Logging | Optional | Enabled | **REQUIRED** |

---

## Step-by-Step Configuration

### Step 1: Generate Secure Secrets

Run the automated secret generation tool:

```bash
cd /path/to/eleanor-v8
./scripts/generate_secrets.sh
```

**What this does:**
- Generates 256-bit JWT secret using OpenSSL
- Creates strong passwords for Grafana and PostgreSQL
- Optionally creates a Weaviate API key
- Offers to write secrets directly to `.env` file

**Example Output:**
```
🔐 ELEANOR V8 - Secret Generation Tool
=======================================

Generating secure secrets for ELEANOR V8...

✅ JWT_SECRET (256-bit):
   xK7mP9vN2jR5wQ8tY3bH6nL0gF4dS1aZ9cE7vM2xK5pR

✅ GRAFANA_ADMIN_PASSWORD:
   Gh7Kp9Lm2Nq5Rt8Wz3

✅ POSTGRES_PASSWORD:
   Pq8Ty3Ru6Vm9Xz2Cb5

⚠️  IMPORTANT SECURITY NOTES:

1. These secrets are displayed ONCE - save them securely
2. Never commit these to version control
3. Store in a secrets manager (AWS Secrets Manager, HashiCorp Vault, etc.)
```

### Step 2: Create .env File

**Option A: Automated (Recommended)**

The `generate_secrets.sh` script will offer to create `.env` automatically.

**Option B: Manual**

```bash
# Copy production template
cp .env.production.template .env

# Edit with your secrets
nano .env
```

### Step 3: Add LLM API Keys

Edit `.env` and add at least ONE provider API key:

```bash
# OpenAI (get from: https://platform.openai.com/api-keys)
OPENAI_KEY=sk-...your-key-here

# OR Anthropic (get from: https://console.anthropic.com/settings/keys)
ANTHROPIC_KEY=sk-ant-...your-key-here

# OR xAI (get from: https://console.x.ai/api-keys)
XAI_KEY=xai-...your-key-here

# OR Google Gemini (get from: https://ai.google.dev)
GEMINI_KEY=...your-key-here
```

**Multi-Provider Configuration:**

For production resilience, configure multiple providers:

```bash
OPENAI_KEY=sk-...
ANTHROPIC_KEY=sk-ant-...
XAI_KEY=xai-...
```

This enables:
- Failover if one provider has an outage
- Cost optimization via routing policies
- Model diversity for critic deliberations

### Step 4: Update Database Connection String

Replace the placeholder password in `PG_CONN_STRING`:

```bash
# BEFORE (from template):
PG_CONN_STRING=postgresql://postgres:REPLACE_WITH_POSTGRES_PASSWORD@pgvector:5432/eleanor

# AFTER (with your POSTGRES_PASSWORD):
PG_CONN_STRING=postgresql://postgres:Pq8Ty3Ru6Vm9Xz2Cb5@pgvector:5432/eleanor
```

### Step 5: Configure CORS for Production

Update `CORS_ORIGINS` with your production domain(s):

```bash
# BEFORE (insecure):
CORS_ORIGINS=http://localhost:3000

# AFTER (production):
CORS_ORIGINS=https://app.yourdomain.com,https://admin.yourdomain.com
```

**Security Rules:**
- ❌ NEVER use wildcards (`*`)
- ❌ NEVER include `localhost` in production
- ✅ Use HTTPS URLs only
- ✅ Specify exact domains (including subdomain)

### Step 6: Validate Configuration

Run the validation script to verify all security requirements:

```bash
./scripts/validate_env.sh
```

**Expected Output (All Pass):**
```
🔐 ELEANOR V8 - Environment Configuration Validator
====================================================

1. Checking .env file existence...
✅ .env file exists

2. Checking ELEANOR_ENVIRONMENT setting...
✅ ELEANOR_ENVIRONMENT=production (correct for production)

3. Checking JWT_SECRET...
✅ JWT_SECRET is set and sufficiently long (44 chars)

4. Checking LLM API Keys...
✅ OPENAI_KEY is configured
✅ ANTHROPIC_KEY is configured

5. Checking Grafana credentials...
✅ GRAFANA_ADMIN_PASSWORD is set to custom value

6. Checking database credentials...
✅ POSTGRES_PASSWORD is set to custom value

7. Checking CORS configuration...
✅ CORS_ORIGINS configured without wildcards or localhost

8. Checking OPA governance configuration...
✅ OPA is enabled
✅ OPA_FAIL_STRATEGY=escalate (recommended)

9. Checking precedent storage configuration...
✅ PRECEDENT_BACKEND=weaviate (persistent storage)

10. Checking monitoring configuration...
✅ Prometheus middleware enabled

11. Checking audit logging configuration...
✅ EVIDENCE_PATH configured: /app/audit/evidence.jsonl
✅ REPLAY_LOG_PATH configured: /app/audit/replay_log.jsonl

12. Checking for security misconfigurations...
✅ .env file has secure permissions (600)

========================================
VALIDATION SUMMARY
========================================
Passed:  15
Warnings: 0
Errors:   0

✅ Configuration is production-ready!
```

**If Errors Occur:**

The script will clearly identify issues:
```
❌ ERROR: JWT_SECRET is too short (24 chars, minimum 32)
   Generate a stronger secret: openssl rand -base64 32
```

Fix all errors before proceeding.

---

## Production Deployment

### Using Production Docker Compose

ELEANOR V8 includes a production-hardened Docker Compose configuration:

```bash
cd docker

# Start all services
docker-compose -f docker-compose.production.yaml up -d

# Check service health
docker-compose -f docker-compose.production.yaml ps

# View logs
docker-compose -f docker-compose.production.yaml logs -f eleanor
```

### Production vs. Development Differences

| Feature | Development | Production |
|---------|-------------|------------|
| Weaviate Auth | Anonymous | API Key Required |
| Health Checks | None | All services monitored |
| Resource Limits | None | Memory/CPU limits set |
| Restart Policy | No | `unless-stopped` |
| Grafana Security | Permissive | HTTPS, secure cookies |
| Data Persistence | Optional | Named volumes |

### Service Health Verification

After deployment, verify all services are healthy:

```bash
# Check service status
docker-compose -f docker-compose.production.yaml ps

# Should show all services "Up (healthy)"
NAME            STATE
eleanor_v8      Up (healthy)
opa             Up (healthy)
weaviate        Up (healthy)
pgvector        Up (healthy)
prometheus      Up (healthy)
grafana         Up (healthy)
```

### Health Check Endpoints

```bash
# ELEANOR API
curl http://localhost:8000/health
# Expected: {"status": "healthy", "version": "8.0.0"}

# OPA
curl http://localhost:8181/health
# Expected: {}

# Weaviate
curl http://localhost:8080/v1/.well-known/ready
# Expected: {"status": "healthy"}

# Prometheus
curl http://localhost:9090/-/healthy
# Expected: Prometheus is Healthy.

# Grafana
curl http://localhost:3000/api/health
# Expected: {"database": "ok", ...}
```

---

## Security Best Practices

### 1. Secrets Rotation

Rotate secrets regularly to minimize breach impact:

**Rotation Schedule:**
- **JWT_SECRET**: Every 90 days
- **Database Passwords**: Every 90 days
- **LLM API Keys**: Every 180 days or on suspected compromise
- **Grafana Password**: Every 90 days

**Rotation Process:**

```bash
# 1. Generate new secrets
./scripts/generate_secrets.sh

# 2. Update .env with NEW secrets (keep old ones temporarily)
JWT_SECRET_NEW=<new-secret>

# 3. Deploy with dual-secret support (if implementing)
# 4. Switch traffic to new secret
# 5. Remove old secret after grace period
```

### 2. File Permissions

Protect sensitive configuration files:

```bash
# .env file (secrets)
chmod 600 .env
chown root:root .env  # or deployment user

# Scripts
chmod 700 scripts/generate_secrets.sh
chmod 700 scripts/validate_env.sh

# Docker compose
chmod 644 docker/docker-compose.production.yaml
```

### 3. Version Control Protection

Ensure secrets never reach version control:

```bash
# Verify .gitignore includes .env
cat .gitignore | grep ".env"

# Check for accidentally committed secrets
git log --all --full-history -- .env

# If found, purge from history (DANGEROUS):
# git filter-branch --force --index-filter \
#   "git rm --cached --ignore-unmatch .env" \
#   --prune-empty --tag-name-filter cat -- --all
```

### 4. Environment Isolation

Use different configurations per environment:

```
development/
  .env.development

staging/
  .env.staging

production/
  .env.production  (NEVER share between environments)
```

### 5. Principle of Least Privilege

- **Database Users**: Create application-specific users with limited permissions
- **API Keys**: Use separate keys per environment
- **Docker**: Run containers as non-root users (where possible)

### 6. Network Security

**Firewall Rules (iptables/AWS Security Groups):**

```bash
# Allow only necessary ports
ALLOW 8000/tcp   # ELEANOR API (from load balancer only)
ALLOW 3000/tcp   # Grafana (from VPN only)
ALLOW 9090/tcp   # Prometheus (from VPN only)

DENY  8181/tcp   # OPA (internal only)
DENY  8080/tcp   # Weaviate (internal only)
DENY  5432/tcp   # PostgreSQL (internal only)
```

**Docker Network Isolation:**

```yaml
# In docker-compose.production.yaml
networks:
  internal:
    internal: true  # No external access
  external:
    # Public-facing services only
```

### 7. Input Validation & Context Hygiene

- Enforce text length limits (≤ 100KB) and context size limits (≤ 1MB).
- Keep context JSON-serializable and shallow (depth ≤ 5) to prevent payload abuse.
- Use safe override keys only (`skip_router` requires `model_output`).
- Normalize Unicode and strip control characters before processing.

Example safe context:

```json
{
  "domain": "finance",
  "priority": "high",
  "constraints": {"max_risk_score": 0.4}
}
```

---

## Troubleshooting

### Common Configuration Issues

#### Issue: "JWT_SECRET too short"

**Symptom:**
```
❌ ERROR: JWT_SECRET is too short (24 chars, minimum 32)
```

**Solution:**
```bash
# Generate new secret
openssl rand -base64 32

# Update .env
JWT_SECRET=<new-secret-here>
```

#### Issue: "No LLM API keys configured"

**Symptom:**
```
⚠️  WARNING: No LLM API keys configured
```

**Solution:**
Add at least one API key to `.env`:
```bash
OPENAI_KEY=sk-...
# OR
ANTHROPIC_KEY=sk-ant-...
```

#### Issue: "CORS contains wildcard"

**Symptom:**
```
❌ ERROR: CORS_ORIGINS contains wildcard (*) - security risk
```

**Solution:**
```bash
# BEFORE:
CORS_ORIGINS=*

# AFTER:
CORS_ORIGINS=https://app.yourdomain.com
```

#### Issue: "Database connection failed"

**Symptom:**
```
psql: FATAL: password authentication failed
```

**Solution:**
Ensure `PG_CONN_STRING` matches `POSTGRES_PASSWORD`:
```bash
# Check .env file
grep "POSTGRES_PASSWORD" .env
grep "PG_CONN_STRING" .env

# Password must match in both places
```

#### Issue: "OPA policies not loaded"

**Symptom:**
```
curl http://localhost:8181/v1/policies
# Returns: {"result": []}
```

**Solution:**
```bash
# Check policy files exist
ls docker/governance/policies/*.rego

# Restart OPA
docker-compose -f docker-compose.production.yaml restart opa

# Verify policies loaded
curl http://localhost:8181/v1/policies | jq
```

---

## Secrets Management

### Production Secrets Managers

Never store production secrets in `.env` files on servers. Use a secrets manager:

#### AWS Secrets Manager

```bash
# Store secrets
aws secretsmanager create-secret \
  --name eleanor/production/jwt \
  --secret-string file://jwt-secret.txt

aws secretsmanager create-secret \
  --name eleanor/production/db-password \
  --secret-string file://db-password.txt

# Retrieve in deployment script
JWT_SECRET=$(aws secretsmanager get-secret-value \
  --secret-id eleanor/production/jwt \
  --query SecretString --output text)
```

#### HashiCorp Vault

```bash
# Store secrets
vault kv put secret/eleanor/production \
  jwt_secret=$JWT_SECRET \
  postgres_password=$POSTGRES_PASSWORD

# Retrieve in deployment
vault kv get -field=jwt_secret secret/eleanor/production
```

#### Kubernetes Secrets

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: eleanor-secrets
type: Opaque
stringData:
  jwt-secret: "..."
  postgres-password: "..."
```

### Environment Variable Injection

**Docker Compose with External Secrets:**

```yaml
# docker-compose.production.yaml
services:
  eleanor:
    environment:
      JWT_SECRET_FILE: /run/secrets/jwt_secret
    secrets:
      - jwt_secret

secrets:
  jwt_secret:
    external: true
```

**Read from file in application:**

```python
# In application startup
import os

def load_secret(secret_name):
    secret_path = os.getenv(f"{secret_name.upper()}_FILE")
    if secret_path and os.path.exists(secret_path):
        with open(secret_path) as f:
            return f.read().strip()
    return os.getenv(secret_name)

JWT_SECRET = load_secret("jwt_secret")
```

---

## Compliance & Audit

### Audit Logging Requirements

ELEANOR V8's constitutional governance requires immutable audit trails:

**Configuration:**
```bash
# .env
EVIDENCE_PATH=/app/audit/evidence.jsonl
REPLAY_LOG_PATH=/app/audit/replay_log.jsonl
```

**Log Rotation:**

```bash
# /etc/logrotate.d/eleanor
/path/to/eleanor-v8/docker/audit/*.jsonl {
    daily
    rotate 90
    compress
    delaycompress
    notifempty
    create 640 root adm
    postrotate
        # Optional: Upload to S3 for long-term retention
        aws s3 cp /path/to/eleanor-v8/docker/audit/*.jsonl.1.gz \
          s3://eleanor-audit-logs/$(date +%Y-%m-%d)/
    endscript
}
```

### Compliance Checklist

- [ ] All deliberation decisions are logged to `EVIDENCE_PATH`
- [ ] Audit logs are backed up off-site (S3, GCS, Azure Blob)
- [ ] Audit log retention meets regulatory requirements (e.g., 7 years for GDPR)
- [ ] Audit logs are immutable (write-once, read-many)
- [ ] Cryptographic signatures for audit log integrity (optional, recommended)

### Regulatory Considerations

| Regulation | Requirement | ELEANOR V8 Configuration |
|------------|-------------|--------------------------|
| **GDPR** | Right to explanation | ✅ Evidence trails with trace IDs |
| **HIPAA** | Audit trails | ✅ EVIDENCE_PATH logging |
| **SOC 2** | Change management | ✅ Replay logs for decision review |
| **ISO 27001** | Access control | ✅ JWT authentication, CORS policy |

---

## Summary Checklist

### Before Production Deployment

```bash
# 1. Generate secrets
./scripts/generate_secrets.sh

# 2. Configure LLM API keys
nano .env  # Add OPENAI_KEY, ANTHROPIC_KEY, etc.

# 3. Update database connection string
# Edit .env: PG_CONN_STRING with POSTGRES_PASSWORD

# 4. Set production CORS origins
# Edit .env: CORS_ORIGINS=https://yourdomain.com

# 5. Validate configuration
./scripts/validate_env.sh

# 6. Review production checklist
cat PRODUCTION_CHECKLIST.md

# 7. Deploy
cd docker && docker-compose -f docker-compose.production.yaml up -d

# 8. Verify health
./scripts/health_check.sh  # From PRODUCTION_CHECKLIST.md
```

---

## Additional Resources

- **Production Checklist:** `PRODUCTION_CHECKLIST.md`
- **Security Scan Report:** `SECURITY_SCAN_REPORT.md`
- **Installation Guide:** `INSTALL.md`
- **API Documentation:** `docs/API.md`
- **Governance Policies:** `governance/`

---

**Document Version:** 1.0
**Last Reviewed:** 2025-12-19
**Next Review:** 2025-03-19 (90 days)
