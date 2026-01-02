# HashiCorp Vault Deployment Guide

## Overview

This guide covers deploying ELEANOR V8 with HashiCorp Vault for production credential management.

---

## Prerequisites

- Vault server deployed and unsealed
- Vault CLI installed
- Admin access to Vault
- ELEANOR V8 with security module installed

---

## Step 1: Install Dependencies

```bash
# Install hvac (Vault Python client)
pip install hvac

# Or install all security dependencies
pip install -r requirements-security.txt
```

---

## Step 2: Deploy Vault Server

### Option A: Docker (Development)

```bash
# Run Vault in dev mode
docker run -d \
    --name vault-dev \
    -p 8200:8200 \
    -e VAULT_DEV_ROOT_TOKEN_ID=root \
    vault:latest

# Set environment
export VAULT_ADDR="http://localhost:8200"
export VAULT_TOKEN="root"

# Verify
vault status
```

### Option B: Production Deployment

**File**: `vault-config.hcl`

```hcl
storage "raft" {
  path = "/vault/data"
  node_id = "vault-1"
}

listener "tcp" {
  address = "0.0.0.0:8200"
  tls_cert_file = "/vault/tls/cert.pem"
  tls_key_file = "/vault/tls/key.pem"
}

api_addr = "https://vault.example.com:8200"
cluster_addr = "https://vault-1.example.com:8201"
ui = true
```

```bash
# Start Vault
vault server -config=vault-config.hcl

# Initialize (first time only)
vault operator init

# Unseal (requires 3 of 5 keys by default)
vault operator unseal <key-1>
vault operator unseal <key-2>
vault operator unseal <key-3>

# Login with root token
vault login <root-token>
```

---

## Step 3: Configure Secrets Engine

### Enable KV v2 Engine

```bash
# Enable secrets engine
vault secrets enable -path=eleanor kv-v2

# Verify
vault secrets list
```

### Store Secrets

```bash
# OpenAI API Key
vault kv put eleanor/prod/openai_key \
    value="sk-proj-..." \
    description="OpenAI API key for production"

# Anthropic API Key
vault kv put eleanor/prod/anthropic_key \
    value="sk-ant-..." \
    description="Anthropic API key for production"

# Optional: Other providers
vault kv put eleanor/prod/google_ai_key \
    value="AIza..."
```

### Verify Secrets

```bash
# List secrets
vault kv list eleanor/prod

# Get secret metadata (not value)
vault kv metadata get eleanor/prod/openai_key

# Get secret value
vault kv get eleanor/prod/openai_key
```

---

## Step 4: Configure Access Control

### Create Policy

**File**: `eleanor-engine-policy.hcl`

```hcl
# Allow reading production secrets
path "eleanor/data/prod/*" {
  capabilities = ["read", "list"]
}

# Allow reading metadata
path "eleanor/metadata/prod/*" {
  capabilities = ["read", "list"]
}

# Deny all other paths
path "eleanor/data/*" {
  capabilities = ["deny"]
}
```

```bash
# Create policy
vault policy write eleanor-engine eleanor-engine-policy.hcl

# Verify
vault policy read eleanor-engine
```

### Create Application Token

```bash
# Create token with policy
vault token create \
    -policy=eleanor-engine \
    -ttl=720h \
    -renewable \
    -display-name="eleanor-engine-prod"

# Save token securely
export VAULT_TOKEN="hvs.CAESIJ..."
```

---

## Step 5: Configure Application

### Update config/security.yaml

```yaml
security:
  secrets:
    provider: "vault"
    vault:
      addr: "https://vault.example.com:8200"
      mount_point: "eleanor"
      # Token from VAULT_TOKEN environment variable
```

### Set Environment Variables

```bash
export VAULT_ADDR="https://vault.example.com:8200"
export VAULT_TOKEN="hvs.CAESIJ..."
```

---

## Step 6: Enable Dynamic Secrets (Advanced)

### Database Credentials

```bash
# Enable database secrets engine
vault secrets enable database

# Configure PostgreSQL connection
vault write database/config/eleanor-db \
    plugin_name=postgresql-database-plugin \
    allowed_roles="eleanor-app" \
    connection_url="postgresql://{{username}}:{{password}}@postgres:5432/eleanor" \
    username="vault-admin" \
    password="vault-password"

# Create role
vault write database/roles/eleanor-app \
    db_name=eleanor-db \
    creation_statements="CREATE ROLE \"{{name}}\" WITH LOGIN PASSWORD '{{password}}' VALID UNTIL '{{expiration}}'; GRANT SELECT ON ALL TABLES IN SCHEMA public TO \"{{name}}\";" \
    default_ttl="1h" \
    max_ttl="24h"
```

### Generate Dynamic Credentials

```bash
# Generate credentials (automatically expire)
vault read database/creds/eleanor-app
```

---

## Step 7: Enable Auto-Rotation

### Configure Secret Rotation

```bash
# Enable secret rotation for long-lived secrets
vault write eleanor/config \
    max_versions=5 \
    cas_required=false

# Manually rotate
vault kv put eleanor/prod/openai_key \
    value="new-sk-proj-..."

# View version history
vault kv metadata get eleanor/prod/openai_key
```

### Automated Rotation Script

**File**: `rotate_secrets.sh`

```bash
#!/bin/bash

# Rotate OpenAI key
NEW_KEY=$(generate_new_openai_key)  # Your key generation logic

vault kv put eleanor/prod/openai_key value="$NEW_KEY"

echo "Secret rotated successfully"
```

```bash
# Schedule with cron (every 30 days)
0 0 1 * * /opt/eleanor/rotate_secrets.sh
```

---

## Step 8: Test Deployment

### Test Secret Retrieval

```python
from engine.security import VaultSecretsProvider

# Initialize provider
provider = VaultSecretsProvider(
    vault_addr="https://vault.example.com:8200",
    mount_point="eleanor",
)

# Test retrieval
api_key = provider.get_secret("prod/openai_key")
assert api_key is not None
assert api_key.startswith("sk-")

print("✅ Secret retrieval successful")
```

### Test Engine Integration

```python
from engine.engine import EleanorEngineV8
from engine.security import VaultSecretsProvider

# Initialize with Vault
secrets = VaultSecretsProvider(
    vault_addr="https://vault.example.com:8200",
    mount_point="eleanor",
)
engine = EleanorEngineV8(secrets_provider=secrets)

# Test query
response = await engine.query("Test query")
assert response is not None

print("✅ Engine integration successful")
```

---

## Step 9: Monitoring and Auditing

### Enable Audit Logging

```bash
# Enable file audit backend
vault audit enable file file_path=/vault/audit/audit.log

# Or syslog
vault audit enable syslog

# Verify
vault audit list
```

### Query Audit Logs

```bash
# View recent access
tail -f /vault/audit/audit.log | jq '.'

# Filter for specific operations
cat /vault/audit/audit.log | jq 'select(.request.path == "eleanor/data/prod/openai_key")'

# Count accesses by user
cat /vault/audit/audit.log | jq -r '.auth.display_name' | sort | uniq -c
```

### Prometheus Metrics

```bash
# Enable telemetry
vault write sys/config/auditing \
    hmac_accessor=false

# Scrape metrics
curl https://vault.example.com:8200/v1/sys/metrics?format=prometheus
```

---

## Step 10: High Availability Setup

### Raft Storage (3-node cluster)

**Node 1**: `vault-1-config.hcl`
```hcl
storage "raft" {
  path = "/vault/data"
  node_id = "vault-1"
  
  retry_join {
    leader_api_addr = "https://vault-2.example.com:8200"
  }
  
  retry_join {
    leader_api_addr = "https://vault-3.example.com:8200"
  }
}
```

```bash
# Initialize cluster on vault-1
vault operator init

# Join vault-2 and vault-3
vault operator raft join https://vault-1.example.com:8200

# Verify cluster
vault operator raft list-peers
```

---

## Step 11: Backup and Disaster Recovery

### Automated Snapshots

```bash
# Create snapshot
vault operator raft snapshot save backup-$(date +%Y%m%d).snap

# Restore snapshot
vault operator raft snapshot restore backup-20260101.snap
```

**Automated backup script**:

```bash
#!/bin/bash
# backup_vault.sh

BACKUP_DIR="/backups/vault"
DATE=$(date +%Y%m%d-%H%M%S)

vault operator raft snapshot save "$BACKUP_DIR/vault-$DATE.snap"

# Encrypt backup
gpg --encrypt --recipient admin@example.com "$BACKUP_DIR/vault-$DATE.snap"

# Upload to S3
aws s3 cp "$BACKUP_DIR/vault-$DATE.snap.gpg" s3://eleanor-vault-backups/

# Clean old backups (keep 30 days)
find "$BACKUP_DIR" -name "vault-*.snap*" -mtime +30 -delete
```

```bash
# Schedule daily backups
0 2 * * * /opt/vault/backup_vault.sh
```

---

## Cost Considerations

### Self-Hosted Vault

- **Compute**: 3 × t3.small instances = ~$50/month
- **Storage**: 100GB EBS = ~$10/month
- **Total**: ~$60/month (vs AWS Secrets Manager ~$7/month for 5 secrets)

### Vault Enterprise Cloud (HCP Vault)

- **Starter**: $0.50/hour = ~$360/month
- **Standard**: $1.00/hour = ~$720/month
- **Plus**: $1.50/hour = ~$1,080/month

**Trade-off**: Higher cost but more features (dynamic secrets, advanced auth, etc.)

---

## Troubleshooting

### "Permission denied"

```bash
# Check current token capabilities
vault token capabilities eleanor/data/prod/openai_key

# Should show: ["read"]

# If not, recreate token with correct policy
vault token create -policy=eleanor-engine
```

### "Vault is sealed"

```bash
# Check status
vault status

# Unseal (requires threshold of unseal keys)
vault operator unseal <key-1>
vault operator unseal <key-2>
vault operator unseal <key-3>
```

### "Connection refused"

```bash
# Check Vault is running
vault status

# Check VAULT_ADDR
echo $VAULT_ADDR

# Test connection
curl $VAULT_ADDR/v1/sys/health
```

---

## Security Best Practices

### 1. Use TLS Always

```hcl
listener "tcp" {
  address = "0.0.0.0:8200"
  tls_cert_file = "/vault/tls/cert.pem"
  tls_key_file = "/vault/tls/key.pem"
  tls_min_version = "tls12"
}
```

### 2. Enable Audit Logging

```bash
vault audit enable file file_path=/var/log/vault/audit.log
```

### 3. Rotate Tokens Regularly

```bash
# Create short-lived tokens
vault token create -policy=eleanor-engine -ttl=24h

# Enable periodic renewal
vault token create -policy=eleanor-engine -period=24h -renewable
```

### 4. Use AppRole Auth

```bash
# Enable AppRole
vault auth enable approle

# Create role
vault write auth/approle/role/eleanor-engine \
    secret_id_ttl=10m \
    token_num_uses=10 \
    token_ttl=20m \
    token_max_ttl=30m \
    secret_id_num_uses=40 \
    policies=eleanor-engine

# Get role ID and secret ID
vault read auth/approle/role/eleanor-engine/role-id
vault write -f auth/approle/role/eleanor-engine/secret-id
```

---

## Next Steps

1. ✅ Vault deployed and configured
2. ✅ Secrets stored securely
3. ✅ Access policies configured
4. ✅ Application integrated
5. ⏭️ Enable audit logging
6. ⏭️ Set up HA cluster
7. ⏭️ Configure automated backups
8. ⏭️ Implement secret rotation

---

**HashiCorp Vault provides enterprise-grade secret management with advanced features for dynamic secrets and fine-grained access control.** 🔐
