# ELEANOR V8 - Security Architecture

## Overview

ELEANOR V8 implements comprehensive security measures to protect:
- **Credentials**: API keys, tokens, passwords
- **Audit Trails**: Evidence recordings, logs, traces
- **User Data**: PII, sensitive context information
- **System Configuration**: Security settings, access controls

---

## Architecture

### Components

```
engine/security/
├── secrets.py        # Secrets provider framework
├── sanitizer.py      # Credential redaction
├── audit.py          # Secure audit logging
└── __init__.py       # Module exports
```

### Data Flow

```
┌─────────────────┐
│  Application    │
│  Requests       │
│  Secrets        │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ SecretsProvider │  (Environment, AWS, Vault)
│  - Get secret   │
│  - Cache TTL    │
│  - Validation   │
└────────┬────────┘
         │
         v
┌─────────────────┐
│   Application   │
│   Uses Secret   │
└────────┬────────┘
         │
         v
┌─────────────────┐
│SecretsSanitizer │  (Before logging)
│  - Pattern match│
│  - Key redaction│
│  - Recursive    │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ SecureAuditLog  │
│  - Sanitized    │
│  - Audit trail  │
└─────────────────┘
```

---

## Secrets Management

### Supported Providers

#### 1. Environment Variables (Development)

**Usage:**
```python
from engine.security import EnvironmentSecretsProvider

provider = EnvironmentSecretsProvider(prefix="ELEANOR_")
api_key = provider.get_secret("OPENAI_API_KEY")
```

**Configuration:**
```bash
export ELEANOR_OPENAI_API_KEY="sk-..."
export ELEANOR_ANTHROPIC_API_KEY="sk-ant-..."
```

**Pros:**
- Simple setup
- No external dependencies
- Good for local development

**Cons:**
- Not secure for production
- No rotation support
- Visible in process list

---

#### 2. AWS Secrets Manager (Production)

**Usage:**
```python
from engine.security import AWSSecretsProvider

provider = AWSSecretsProvider(
    region_name="us-west-2",
    prefix="eleanor/",
)
api_key = provider.get_secret("openai_key")
```

**Setup:**
```bash
# Create secret
aws secretsmanager create-secret \
    --name eleanor/openai_key \
    --secret-string "sk-..."

# Grant IAM access
aws iam attach-role-policy \
    --role-name eleanor-engine-role \
    --policy-arn arn:aws:iam::aws:policy/SecretsManagerReadWrite
```

**Features:**
- ✅ Automatic rotation
- ✅ Encryption at rest (KMS)
- ✅ Fine-grained IAM access control
- ✅ Audit logging (CloudTrail)
- ✅ Version history

**IAM Policy:**
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue",
        "secretsmanager:DescribeSecret"
      ],
      "Resource": "arn:aws:secretsmanager:us-west-2:*:secret:eleanor/*"
    }
  ]
}
```

---

#### 3. HashiCorp Vault (Production)

**Usage:**
```python
from engine.security import VaultSecretsProvider

provider = VaultSecretsProvider(
    vault_addr="https://vault.example.com",
    mount_point="eleanor",
)
api_key = provider.get_secret("openai_key")
```

**Setup:**
```bash
# Enable KV v2 secrets engine
vault secrets enable -path=eleanor kv-v2

# Store secrets
vault kv put eleanor/openai_key value="sk-..."
vault kv put eleanor/anthropic_key value="sk-ant-..."

# Create policy
vault policy write eleanor-engine - <<EOF
path "eleanor/data/*" {
  capabilities = ["read", "list"]
}
EOF

# Create token
vault token create -policy=eleanor-engine
```

**Features:**
- ✅ Dynamic secrets generation
- ✅ Automatic rotation
- ✅ Fine-grained policies
- ✅ Comprehensive audit logs
- ✅ Multi-cloud support

---

## Credential Sanitization

### How It Works

**SecretsSanitizer** prevents credential leakage through:
1. **Pattern Matching**: Regex detection of common secret formats
2. **Key Redaction**: Automatic redaction of sensitive dictionary keys
3. **Recursive Scanning**: Deep inspection of nested structures

### Supported Patterns

| Pattern | Example | Replacement |
|---------|---------|-------------|
| OpenAI | `sk-1234...` | `[OPENAI_API_KEY]` |
| Anthropic | `sk-ant-...` | `[ANTHROPIC_API_KEY]` |
| AWS | `AKIA...` | `[AWS_ACCESS_KEY]` |
| GitHub | `ghp_...` | `[GITHUB_TOKEN]` |
| JWT | `eyJ...` | `[JWT_TOKEN]` |
| Bearer | `Bearer abc...` | `[BEARER_TOKEN]` |
| Generic | `password=...` | `password=[REDACTED]` |

### Usage

```python
from engine.security import SecretsSanitizer

sanitizer = SecretsSanitizer()

# Sanitize string
text = "My API key is sk-1234567890abcdefghijklmnopqrstuvwxyz12345678"
safe_text = sanitizer.sanitize_string(text)
# Result: "My API key is [OPENAI_API_KEY]"

# Sanitize dictionary
data = {
    "api_key": "sk-secret123",
    "username": "john",
}
safe_data = sanitizer.sanitize_dict(data)
# Result: {"api_key": "[REDACTED]", "username": "john"}
```

### Custom Patterns

```python
sanitizer = SecretsSanitizer()

# Add custom pattern
sanitizer.add_pattern(
    pattern=r"CUSTOM-\d{6}",
    replacement="[CUSTOM_ID]"
)

# Add sensitive key
sanitizer.add_sensitive_key("internal_token")
```

---

## Secure Audit Logging

### SecureAuditLogger

**Automatically sanitizes all audit events:**

```python
from engine.security import SecureAuditLogger

audit = SecureAuditLogger()

# Log access event
audit.log_access(
    user="john@example.com",
    resource="/api/data",
    action="read",
    allowed=True,
    metadata={"api_key": "sk-secret"},  # Automatically sanitized!
)

# Log secret access (never logs actual value)
audit.log_secret_access(
    secret_key="openai_key",
    accessor="engine_service",
    success=True,
)

# Log config change
audit.log_configuration_change(
    user="admin",
    config_key="max_tokens",
    old_value=1000,
    new_value=2000,
)
```

---

## Integration with Engine

### Engine Initialization

```python
from engine.engine import EleanorEngineV8
from engine.security import AWSSecretsProvider, SecretsSanitizer

# Initialize secrets provider
secrets = AWSSecretsProvider(region_name="us-west-2")

# Initialize engine with secrets
engine = EleanorEngineV8(
    config=config,
    secrets_provider=secrets,
)

# Engine automatically:
# 1. Uses secrets for LLM API keys
# 2. Sanitizes all logs and evidence
# 3. Prevents credential leaks
```

### Auto-Configuration

**Engine auto-detects provider based on environment:**

```python
# If AWS_SECRETS_MANAGER=true
engine = EleanorEngineV8()  # Uses AWSSecretsProvider

# If VAULT_ADDR is set
engine = EleanorEngineV8()  # Uses VaultSecretsProvider

# Otherwise
engine = EleanorEngineV8()  # Uses EnvironmentSecretsProvider
```

---

## Production Deployment

### AWS Deployment

**1. Create Secrets:**
```bash
# Store all LLM API keys
aws secretsmanager create-secret \
    --name eleanor/prod/openai_key \
    --secret-string "sk-..."

aws secretsmanager create-secret \
    --name eleanor/prod/anthropic_key \
    --secret-string "sk-ant-..."
```

**2. Configure IAM Role:**
```bash
# Attach policy to EC2/ECS role
aws iam attach-role-policy \
    --role-name eleanor-engine-role \
    --policy-arn arn:aws:iam::aws:policy/SecretsManagerReadWrite
```

**3. Configure Application:**
```yaml
# config/security.yaml
security:
  secrets:
    provider: "aws"
    aws:
      region: "us-west-2"
      prefix: "eleanor/prod/"
```

**4. Enable Rotation:**
```bash
# Enable automatic rotation (every 30 days)
aws secretsmanager rotate-secret \
    --secret-id eleanor/prod/openai_key \
    --rotation-lambda-arn arn:aws:lambda:us-west-2:...:function:rotate-secret \
    --rotation-rules AutomaticallyAfterDays=30
```

---

### Vault Deployment

**1. Enable Secrets Engine:**
```bash
vault secrets enable -path=eleanor kv-v2
```

**2. Store Secrets:**
```bash
vault kv put eleanor/prod/openai_key value="sk-..."
vault kv put eleanor/prod/anthropic_key value="sk-ant-..."
```

**3. Create Policy:**
```bash
vault policy write eleanor-engine - <<EOF
path "eleanor/data/prod/*" {
  capabilities = ["read", "list"]
}

path "eleanor/metadata/prod/*" {
  capabilities = ["read", "list"]
}
EOF
```

**4. Generate Token:**
```bash
# For service authentication
vault token create \
    -policy=eleanor-engine \
    -ttl=24h \
    -renewable
```

**5. Configure Application:**
```yaml
# config/security.yaml
security:
  secrets:
    provider: "vault"
    vault:
      addr: "https://vault.example.com"
      mount_point: "eleanor"
```

```bash
# Set token via environment
export VAULT_TOKEN="hvs.CAESIJ..."
```

---

## Security Testing

### Test Suite

```bash
# Run security tests
pytest tests/test_security.py -v

# Test coverage
pytest tests/test_security.py --cov=engine/security --cov-report=html
```

### Test Scenarios

1. **Pattern Detection**: All secret formats redacted
2. **Key Redaction**: Sensitive dictionary keys removed
3. **Nested Structures**: Deep sanitization works
4. **Custom Patterns**: Runtime pattern additions
5. **Audit Logging**: No secrets in audit trail
6. **Provider Fallback**: Graceful degradation

### Security Audit

**Manual verification:**

```bash
# Search logs for potential leaks
grep -r "sk-" logs/  # Should find no OpenAI keys
grep -r "AKIA" logs/  # Should find no AWS keys

# Check evidence recordings
jq '.context.api_key' evidence.jsonl  # Should be [REDACTED]
```

---

## Incident Response

### Credential Leak Detected

**1. Immediate Actions:**
```bash
# Rotate compromised secret
aws secretsmanager rotate-secret --secret-id eleanor/prod/openai_key

# Or with Vault
vault kv put eleanor/prod/openai_key value="new-sk-..."

# Revoke compromised key at provider
# (OpenAI dashboard, Anthropic console, etc.)
```

**2. Investigation:**
```bash
# Find all usages of leaked key
grep -r "sk-LEAKED_KEY" logs/ evidence.jsonl

# Check audit trail
jq 'select(.secret_key == "openai_key")' audit.jsonl

# Review access logs
aws secretsmanager list-secret-version-ids --secret-id eleanor/prod/openai_key
```

**3. Remediation:**
- Update sanitization patterns if needed
- Review and enhance logging
- Update security tests
- Document lessons learned

---

## Compliance

### Standards Met

- ✅ **GDPR**: PII sanitization, audit trails
- ✅ **SOC 2**: Access logging, encryption
- ✅ **HIPAA**: Secure credential storage (if applicable)
- ✅ **PCI DSS**: No plaintext secrets in logs

### Audit Requirements

**What we log:**
- ✅ Secret access (who, when, which key)
- ✅ Configuration changes
- ✅ Access control decisions
- ✅ Authentication events

**What we DON'T log:**
- ❌ Actual secret values
- ❌ Plaintext passwords
- ❌ Raw API keys
- ❌ Unredacted credentials

---

## Best Practices

### Development

1. **Never commit secrets to Git**
   ```bash
   # Use .env file (gitignored)
   echo "ELEANOR_OPENAI_API_KEY=sk-..." > .env
   ```

2. **Use environment provider**
   ```python
   provider = EnvironmentSecretsProvider()
   ```

3. **Test sanitization**
   ```python
   assert "sk-" not in sanitizer.sanitize_string(log_message)
   ```

### Production

1. **Always use AWS or Vault**
   ```yaml
   security:
     secrets:
       provider: "aws"  # Never "environment"
   ```

2. **Enable rotation**
   ```yaml
   security:
     rotation:
       enabled: true
       check_interval: 3600
   ```

3. **Monitor audit logs**
   ```bash
   # Alert on suspicious access
   jq 'select(.allowed == false)' audit.jsonl
   ```

4. **Review logs regularly**
   ```bash
   # Check for any credential patterns
   ./scripts/audit_logs_for_secrets.sh
   ```

---

## Troubleshooting

### "Required secret not found"

**Cause**: Secret missing from provider

**Solution**:
```bash
# Check secret exists
aws secretsmanager describe-secret --secret-id eleanor/prod/openai_key

# Or with Vault
vault kv get eleanor/prod/openai_key
```

### "Failed to authenticate with Vault"

**Cause**: Invalid or expired Vault token

**Solution**:
```bash
# Verify token
vault token lookup

# Renew if needed
vault token renew

# Or create new token
export VAULT_TOKEN=$(vault token create -policy=eleanor-engine -field=token)
```

### "AWS Secrets Manager access denied"

**Cause**: IAM role lacks permissions

**Solution**:
```bash
# Check IAM role
aws iam get-role --role-name eleanor-engine-role

# Verify policy attached
aws iam list-attached-role-policies --role-name eleanor-engine-role

# Add missing policy
aws iam attach-role-policy \
    --role-name eleanor-engine-role \
    --policy-arn arn:aws:iam::aws:policy/SecretsManagerReadWrite
```

---

## Related Issues

- #20: Secrets Management (this implementation)
- #19: Async Resource Management
- #17: Observability (audit log integration)
- #11: Configuration Management

---

**Security is critical. When in doubt, redact it out!** 🔒
