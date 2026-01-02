# AWS Secrets Manager Deployment Guide

## Overview

This guide covers deploying ELEANOR V8 with AWS Secrets Manager for production credential management.

---

## Prerequisites

- AWS account with admin access
- AWS CLI configured
- IAM role for your ECS/EC2 instances
- ELEANOR V8 with security module installed

---

## Step 1: Install Dependencies

```bash
# Install boto3 for AWS integration
pip install boto3

# Or install all security dependencies
pip install -r requirements-security.txt
```

---

## Step 2: Create Secrets

### Store LLM API Keys

```bash
# OpenAI API Key
aws secretsmanager create-secret \
    --name eleanor/prod/openai_key \
    --description "OpenAI API key for ELEANOR V8 production" \
    --secret-string "sk-proj-..."

# Anthropic API Key
aws secretsmanager create-secret \
    --name eleanor/prod/anthropic_key \
    --description "Anthropic API key for ELEANOR V8 production" \
    --secret-string "sk-ant-..."

# Optional: Other provider keys
aws secretsmanager create-secret \
    --name eleanor/prod/google_ai_key \
    --secret-string "AIza..."
```

### Verify Secrets

```bash
# List all secrets
aws secretsmanager list-secrets --query 'SecretList[?Name.starts_with(@, `eleanor/prod/`)]'

# Describe specific secret
aws secretsmanager describe-secret --secret-id eleanor/prod/openai_key
```

---

## Step 3: Configure IAM Permissions

### Create IAM Policy

**File**: `eleanor-secrets-policy.json`

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowReadELEANORSecrets",
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue",
        "secretsmanager:DescribeSecret",
        "secretsmanager:ListSecrets"
      ],
      "Resource": [
        "arn:aws:secretsmanager:*:*:secret:eleanor/prod/*"
      ]
    },
    {
      "Sid": "AllowDecryptSecrets",
      "Effect": "Allow",
      "Action": [
        "kms:Decrypt",
        "kms:DescribeKey"
      ],
      "Resource": [
        "arn:aws:kms:*:*:key/*"
      ],
      "Condition": {
        "StringEquals": {
          "kms:ViaService": "secretsmanager.us-west-2.amazonaws.com"
        }
      }
    }
  ]
}
```

### Create and Attach Policy

```bash
# Create policy
aws iam create-policy \
    --policy-name ELEANORSecretsManagerAccess \
    --policy-document file://eleanor-secrets-policy.json

# Attach to IAM role (ECS/EC2)
aws iam attach-role-policy \
    --role-name eleanor-engine-role \
    --policy-arn arn:aws:iam::YOUR_ACCOUNT_ID:policy/ELEANORSecretsManagerAccess

# Verify attachment
aws iam list-attached-role-policies --role-name eleanor-engine-role
```

---

## Step 4: Configure Application

### Update config/security.yaml

```yaml
security:
  secrets:
    provider: "aws"
    aws:
      region: "us-west-2"  # Your AWS region
      prefix: "eleanor/prod/"
      cache_ttl: 300  # 5 minutes
```

### Set Environment Variables

```bash
# Optional: Override config with env vars
export AWS_REGION="us-west-2"
export AWS_SECRETS_MANAGER="true"
```

---

## Step 5: Enable Automatic Rotation

### Create Rotation Lambda

**File**: `rotate_secret.py`

```python
import boto3
import json
import os

def lambda_handler(event, context):
    """
    Lambda function to rotate secrets.
    Called automatically by Secrets Manager.
    """
    service_client = boto3.client('secretsmanager')
    arn = event['SecretId']
    token = event['ClientRequestToken']
    step = event['Step']
    
    # Implement rotation logic based on step
    if step == "createSecret":
        # Generate new secret value
        # Call provider API to create new key
        pass
    
    elif step == "setSecret":
        # Store new secret in pending state
        pass
    
    elif step == "testSecret":
        # Test new secret works
        pass
    
    elif step == "finishSecret":
        # Mark new secret as current
        pass
```

### Deploy Lambda

```bash
# Package Lambda
zip rotation-function.zip rotate_secret.py

# Create Lambda function
aws lambda create-function \
    --function-name eleanor-secret-rotation \
    --runtime python3.11 \
    --role arn:aws:iam::YOUR_ACCOUNT_ID:role/lambda-rotation-role \
    --handler rotate_secret.lambda_handler \
    --zip-file fileb://rotation-function.zip
```

### Enable Rotation

```bash
# Enable rotation (every 30 days)
aws secretsmanager rotate-secret \
    --secret-id eleanor/prod/openai_key \
    --rotation-lambda-arn arn:aws:lambda:us-west-2:YOUR_ACCOUNT_ID:function:eleanor-secret-rotation \
    --rotation-rules AutomaticallyAfterDays=30

# Verify rotation enabled
aws secretsmanager describe-secret --secret-id eleanor/prod/openai_key \
    --query 'RotationEnabled'
```

---

## Step 6: Test Deployment

### Test Secret Retrieval

```python
from engine.security import AWSSecretsProvider

# Initialize provider
provider = AWSSecretsProvider(
    region_name="us-west-2",
    prefix="eleanor/prod/",
)

# Test retrieval
api_key = provider.get_secret("openai_key")
assert api_key is not None
assert api_key.startswith("sk-")

print("✅ Secret retrieval successful")
```

### Test Engine Integration

```python
from engine.engine import EleanorEngineV8
from engine.security import AWSSecretsProvider

# Initialize with AWS secrets
secrets = AWSSecretsProvider(region_name="us-west-2")
engine = EleanorEngineV8(secrets_provider=secrets)

# Test query (will use secrets for LLM access)
response = await engine.query("Test query")
assert response is not None

print("✅ Engine integration successful")
```

---

## Step 7: Monitoring

### Enable CloudTrail Logging

```bash
# Create trail for Secrets Manager
aws cloudtrail create-trail \
    --name eleanor-secrets-audit \
    --s3-bucket-name eleanor-audit-logs

# Start logging
aws cloudtrail start-logging --name eleanor-secrets-audit
```

### CloudWatch Alarms

```bash
# Alert on secret access failures
aws cloudwatch put-metric-alarm \
    --alarm-name eleanor-secret-access-failures \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 2 \
    --metric-name AccessDeniedExceptions \
    --namespace AWS/SecretsManager \
    --period 300 \
    --statistic Sum \
    --threshold 5 \
    --alarm-actions arn:aws:sns:us-west-2:YOUR_ACCOUNT_ID:eleanor-alerts
```

### Query CloudTrail Logs

```bash
# Find all secret access events
aws cloudtrail lookup-events \
    --lookup-attributes AttributeKey=EventName,AttributeValue=GetSecretValue \
    --max-results 50
```

---

## Step 8: Backup and Disaster Recovery

### Backup Secrets

```bash
# Export secrets (encrypted)
aws secretsmanager get-secret-value \
    --secret-id eleanor/prod/openai_key \
    --query SecretString \
    --output text > backup/openai_key.enc

# Encrypt backup
gpg --encrypt --recipient admin@example.com backup/openai_key.enc
```

### Cross-Region Replication

```bash
# Replicate to secondary region
aws secretsmanager replicate-secret-to-regions \
    --secret-id eleanor/prod/openai_key \
    --add-replica-regions Region=us-east-1
```

---

## Cost Optimization

### Secrets Manager Pricing

- **Storage**: $0.40 per secret per month
- **API Calls**: $0.05 per 10,000 calls

### Optimize Costs

```yaml
# Increase cache TTL to reduce API calls
security:
  secrets:
    aws:
      cache_ttl: 600  # 10 minutes (default: 5)
```

**Example Cost for 5 secrets:**
- Storage: 5 × $0.40 = $2.00/month
- API calls: 1M calls/month = $5.00/month
- **Total**: ~$7/month

---

## Troubleshooting

### "ResourceNotFoundException"

**Cause**: Secret doesn't exist or wrong name

```bash
# List all secrets
aws secretsmanager list-secrets

# Check prefix matches
aws secretsmanager get-secret-value --secret-id eleanor/prod/openai_key
```

### "AccessDeniedException"

**Cause**: IAM role lacks permissions

```bash
# Check role policies
aws iam list-attached-role-policies --role-name eleanor-engine-role

# Test policy simulation
aws iam simulate-principal-policy \
    --policy-source-arn arn:aws:iam::YOUR_ACCOUNT_ID:role/eleanor-engine-role \
    --action-names secretsmanager:GetSecretValue \
    --resource-arns arn:aws:secretsmanager:us-west-2:YOUR_ACCOUNT_ID:secret:eleanor/prod/openai_key
```

### "KMSAccessDeniedException"

**Cause**: Can't decrypt with KMS

```bash
# Check KMS key policy
aws kms get-key-policy \
    --key-id YOUR_KMS_KEY_ID \
    --policy-name default

# Grant decrypt permission
aws kms create-grant \
    --key-id YOUR_KMS_KEY_ID \
    --grantee-principal arn:aws:iam::YOUR_ACCOUNT_ID:role/eleanor-engine-role \
    --operations Decrypt
```

---

## Security Best Practices

### 1. Use KMS Encryption

```bash
# Create KMS key for secrets
aws kms create-key \
    --description "ELEANOR Secrets Encryption Key"

# Create secret with KMS key
aws secretsmanager create-secret \
    --name eleanor/prod/openai_key \
    --kms-key-id arn:aws:kms:us-west-2:YOUR_ACCOUNT_ID:key/YOUR_KEY_ID \
    --secret-string "sk-..."
```

### 2. Rotate Regularly

```bash
# Set rotation to 30 days
aws secretsmanager rotate-secret \
    --secret-id eleanor/prod/openai_key \
    --rotation-rules AutomaticallyAfterDays=30
```

### 3. Audit Access

```bash
# Enable CloudTrail logging
aws cloudtrail create-trail --name eleanor-audit

# Monitor access patterns
aws cloudtrail lookup-events --lookup-attributes AttributeKey=EventName,AttributeValue=GetSecretValue
```

### 4. Least Privilege

```json
{
  "Effect": "Allow",
  "Action": ["secretsmanager:GetSecretValue"],
  "Resource": ["arn:aws:secretsmanager:*:*:secret:eleanor/prod/*"],
  "Condition": {
    "StringEquals": {
      "aws:PrincipalTag/Environment": "production"
    }
  }
}
```

---

## Next Steps

1. ✅ Secrets stored in AWS Secrets Manager
2. ✅ IAM permissions configured
3. ✅ Application configured for AWS
4. ✅ Monitoring and alerting enabled
5. ⏭️ Enable automatic rotation
6. ⏭️ Set up cross-region replication
7. ⏭️ Configure backup procedures

---

**AWS Secrets Manager provides enterprise-grade secret management with minimal operational overhead.** 🔐
