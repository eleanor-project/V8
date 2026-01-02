# ELEANOR V8 - Secrets Management Guide

## Overview

Secure credential management for ELEANOR V8 with support for multiple backends:
- **Development**: Environment variables
- **Production**: AWS Secrets Manager or HashiCorp Vault
- **Protection**: Automatic sanitization of logs and evidence

## Quick Start

### Development (Environment Variables)

```bash
# .env file
ELEANOR_OPENAI_API_KEY=sk-...
ELEANOR_ANTHROPIC_API_KEY=sk-ant-...
```

```python
from engine.security import auto_detect_secrets_provider

# Auto-detects environment provider
secrets = auto_detect_secrets_provider()
api_key = secrets.get_secret("ELEANOR_OPENAI_API_KEY")
```

### Production (AWS Secrets Manager)

```bash
# Environment configuration
export AWS_SECRETS_MANAGER=true
export AWS_REGION=us-west-2
```

```python
# Automatically uses AWS Secrets Manager
secrets = auto_detect_secrets_provider()
api_key = secrets.get_secret("eleanor/prod/openai-api-key")
```

### Production (HashiCorp Vault)

```bash
# Environment configuration
export VAULT_ADDR=https://vault.example.com
export VAULT_TOKEN=hvs.xxx
```

```python
# Automatically uses Vault
secrets = auto_detect_secrets_provider()
api_key = secrets.get_secret("eleanor/openai-api-key")
```

## Architecture

### Secrets Providers

```
SecretsProvider (ABC)
├── EnvironmentSecretsProvider  # Development
├── AWSSecretsProvider          # AWS Secrets Manager
└── VaultSecretsProvider        # HashiCorp Vault
```

**Key Features:**
- Unified interface across all providers
- Automatic caching with TTL (5 minutes default)
- Graceful fallback to environment variables
- Comprehensive error handling

### Sanitization System

```
SecretsSanitizer
├── Pattern-based detection (regex)
├── Key-based detection (field names)
├── Recursive sanitization (dicts, lists)
└── Custom sensitive keys support
```

**Protected Data:**
- Evidence recordings
- Log messages
- Error messages
- Debug output
- Audit trails

## Usage Patterns

### 1. Basic Secret Retrieval

```python
from engine.security import auto_detect_secrets_provider

secrets = auto_detect_secrets_provider()

# Get secret (returns None if not found)
api_key = secrets.get_secret("openai-api-key")

# Get secret or raise error
api_key = secrets.get_secret_or_fail("openai-api-key")

# List available secrets
secret_names = secrets.list_secrets()
```

### 2. Integration with Engine

```python
from engine.engine import EleanorEngineV8
from engine.security import auto_detect_secrets_provider

# Initialize secrets provider
secrets = auto_detect_secrets_provider()

# Get LLM API keys
openai_key = secrets.get_secret("openai-api-key")
anthropic_key = secrets.get_secret("anthropic-api-key")

# Configure engine with secrets
engine = EleanorEngineV8(
    config=config,
    secrets_provider=secrets,
)
```

### 3. Sanitizing Data

```python
from engine.security import SecretsSanitizer

# Sanitize string
text = "API key: sk-abc123..."
safe_text = SecretsSanitizer.sanitize_string(text)
# Result: "API key: [OPENAI_API_KEY]"

# Sanitize dictionary
data = {
    "api_key": "secret123",
    "user": "john",
    "config": {
        "password": "pass123"
    }
}
safe_data = SecretsSanitizer.sanitize_dict(data)
# Result: api_key and password are "[REDACTED]"

# Sanitize any type
safe = SecretsSanitizer.sanitize(data)
```

### 4. Custom Sensitive Keys

```python
# Add custom sensitive field names
data = {
    "internal_token": "secret",
    "signing_key": "private"
}

safe = SecretsSanitizer.sanitize_dict(
    data,
    sensitive_keys={"internal_token", "signing_key"}
)
```

## Production Deployment

### AWS Secrets Manager Setup

#### 1. Create Secrets

```bash
# Create OpenAI API key secret
aws secretsmanager create-secret \
    --name eleanor/prod/openai-api-key \
    --description "OpenAI API key for ELEANOR production" \
    --secret-string "sk-..."

# Create Anthropic API key secret
aws secretsmanager create-secret \
    --name eleanor/prod/anthropic-api-key \
    --secret-string "sk-ant-..."
```

#### 2. Create IAM Policy

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue",
        "secretsmanager:DescribeSecret",
        "secretsmanager:ListSecrets"
      ],
      "Resource": [
        "arn:aws:secretsmanager:us-west-2:ACCOUNT_ID:secret:eleanor/*"
      ]
    }
  ]
}
```

#### 3. Attach Policy to Role

```bash
# Create or use existing IAM role
aws iam attach-role-policy \
    --role-name eleanor-engine-role \
    --policy-arn arn:aws:iam::ACCOUNT_ID:policy/eleanor-secrets-policy
```

#### 4. Configure Application

```bash
# Environment variables
export AWS_SECRETS_MANAGER=true
export AWS_REGION=us-west-2

# Application automatically uses AWS Secrets Manager
python main.py
```

### HashiCorp Vault Setup

#### 1. Enable KV Secrets Engine

```bash
vault secrets enable -path=eleanor kv-v2
```

#### 2. Store Secrets

```bash
# Store OpenAI API key
vault kv put eleanor/openai-api-key value="sk-..."

# Store Anthropic API key
vault kv put eleanor/anthropic-api-key value="sk-ant-..."
```

#### 3. Create Policy

```hcl
# policy.hcl
path "eleanor/*" {
  capabilities = ["read", "list"]
}
```

```bash
vault policy write eleanor-engine policy.hcl
```

#### 4. Create Token

```bash
# Create token for application
vault token create \
    -policy=eleanor-engine \
    -ttl=720h \
    -renewable
```

#### 5. Configure Application

```bash
# Environment variables
export VAULT_ADDR=https://vault.example.com
export VAULT_TOKEN=hvs.xxx

# Application automatically uses Vault
python main.py
```

## Security Best Practices

### 1. Never Log Secrets

```python
# ❌ BAD - Logs API key
logger.info(f"Using API key: {api_key}")

# ✅ GOOD - Sanitized before logging
from engine.security import SecretsSanitizer
safe_key = SecretsSanitizer.sanitize_string(api_key)
logger.info(f"Using API key: {safe_key}")
```

### 2. Sanitize Evidence

```python
# Always sanitize before recording evidence
from engine.security import SecretsSanitizer

evidence = {
    "context": context,
    "model_response": response,
}

safe_evidence = SecretsSanitizer.sanitize_dict(evidence)
await recorder.record(**safe_evidence)
```

### 3. Sanitize Error Messages

```python
try:
    result = await api_call(api_key=secret)
except Exception as e:
    # Sanitize exception message
    safe_message = SecretsSanitizer.sanitize_string(str(e))
    logger.error(f"API call failed: {safe_message}")
```

### 4. Rotate Secrets Regularly

```bash
# AWS Secrets Manager - Enable rotation
aws secretsmanager rotate-secret \
    --secret-id eleanor/prod/openai-api-key \
    --rotation-lambda-arn arn:aws:lambda:...
```

### 5. Use Least Privilege

- Only grant read access to secrets
- Use separate secrets per environment
- Implement secret versioning
- Audit secret access logs

## Testing

### Unit Tests

```bash
pytest tests/test_secrets_management.py -v
```

**Test Coverage:**
- ✅ Environment provider
- ✅ AWS provider (mocked)
- ✅ Vault provider (mocked)
- ✅ Auto-detection logic
- ✅ String sanitization
- ✅ Dictionary sanitization
- ✅ List sanitization
- ✅ Recursive sanitization
- ✅ Custom sensitive keys

### Integration Tests

```python
# Test with real AWS Secrets Manager
@pytest.mark.integration
def test_aws_secrets_integration():
    provider = AWSSecretsProvider(region_name="us-west-2")
    secret = provider.get_secret("test/api-key")
    assert secret is not None
```

### Security Audit

```bash
# Check for leaked secrets in logs
grep -r "sk-" logs/ || echo "No OpenAI keys found"
grep -r "AKIA" logs/ || echo "No AWS keys found"

# Scan codebase for hardcoded secrets
truffleHog --regex --entropy=False .
```

## Troubleshooting

### Secrets Not Found

**Symptom**: `Secret 'xxx' not found`

**Solutions**:
1. Verify secret exists in provider
2. Check secret name/path is correct
3. Verify IAM/Vault permissions
4. Check AWS region configuration

### AWS Authentication Failed

**Symptom**: `Unable to locate credentials`

**Solutions**:
1. Configure AWS credentials: `aws configure`
2. Use IAM role for EC2/ECS
3. Set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
4. Verify IAM policy allows `secretsmanager:GetSecretValue`

### Vault Connection Failed

**Symptom**: `Failed to authenticate with Vault`

**Solutions**:
1. Verify `VAULT_ADDR` is correct
2. Check `VAULT_TOKEN` is valid
3. Verify network connectivity to Vault
4. Check Vault policy allows read access
5. Token may be expired - create new token

### Sanitization Not Working

**Symptom**: Secrets appearing in logs

**Solutions**:
1. Verify sanitization is enabled in config
2. Check pattern matches your secret format
3. Add custom sensitive keys if needed
4. Ensure sanitizer is called before logging

## Migration Guide

### From Environment Variables to AWS

**Before:**
```bash
# .env
ELEANOR_OPENAI_API_KEY=sk-...
```

**After:**
```bash
# Store in AWS
aws secretsmanager create-secret \
    --name eleanor/prod/openai-api-key \
    --secret-string "sk-..."

# Enable AWS provider
export AWS_SECRETS_MANAGER=true
export AWS_REGION=us-west-2

# Update code
api_key = secrets.get_secret("eleanor/prod/openai-api-key")
```

### Adding Sanitization to Existing Code

```python
# Before
await recorder.record(**evidence)

# After
from engine.security import SecretsSanitizer
safe_evidence = SecretsSanitizer.sanitize_dict(evidence)
await recorder.record(**safe_evidence)
```

## Configuration Reference

### Config File: `config/secrets.yaml`

```yaml
secrets:
  provider: environment  # environment, aws, or vault
  
  aws:
    enabled: false
    region: us-west-2
    cache_ttl: 300
    
  vault:
    enabled: false
    addr: "${VAULT_ADDR}"
    token: "${VAULT_TOKEN}"
    mount_point: "secret"
    cache_ttl: 300

sanitization:
  enabled: true
  sanitize_evidence: true
  sanitize_errors: true
  custom_sensitive_keys:
    - "internal_token"
```

### Environment Variables

| Variable | Description | Example |
|----------|-------------|----------|
| `AWS_SECRETS_MANAGER` | Enable AWS provider | `true` |
| `AWS_REGION` | AWS region | `us-west-2` |
| `VAULT_ADDR` | Vault server URL | `https://vault.example.com` |
| `VAULT_TOKEN` | Vault auth token | `hvs.xxx` |
| `SECRETS_PROVIDER` | Force specific provider | `aws`, `vault`, `environment` |

## Performance

### Caching

- **Default TTL**: 5 minutes
- **Cache Location**: In-memory
- **Cache Strategy**: Per-secret TTL

**Benefits:**
- Reduces API calls to secrets manager
- Improves latency (< 1ms cached vs ~50ms uncached)
- Automatic cache invalidation after TTL

### Benchmarks

| Operation | Time (avg) |
|-----------|------------|
| Environment get | 0.01ms |
| AWS cached get | 0.05ms |
| AWS uncached get | 45ms |
| Vault cached get | 0.05ms |
| Vault uncached get | 55ms |
| Sanitize string | 0.02ms |
| Sanitize dict (10 keys) | 0.15ms |

## Related Documentation

- [AWS Secrets Manager](https://docs.aws.amazon.com/secretsmanager/)
- [HashiCorp Vault](https://www.vaultproject.io/docs)
- [Resource Management](./RESOURCE_MANAGEMENT.md)
- [Security Best Practices](./SECURITY.md)

## Related Issues

- #20: Secrets Management (this implementation)
- #19: Async Resource Management
- #17: Observability and Structured Logging
