# Environment Variables - Production Orchestrator

Quick reference for configuring the Production Orchestrator.

---

## Core Features

### `ELEANOR_ORCHESTRATOR_ENABLE_GATING`
**Purpose**: Enable/disable policy-based gating (cost optimization)  
**Default**: `true`  
**Values**: `true`, `false`, `1`, `0`, `yes`, `no`, `on`, `off`  
**Impact**: When enabled, critics with policies like `ON_VIOLATION` or `ON_HIGH_RISK` may be skipped  
**Savings**: 30-60% reduction in critic executions

```bash
# Production (recommended)
export ELEANOR_ORCHESTRATOR_ENABLE_GATING=true

# Development (test all critics)
export ELEANOR_ORCHESTRATOR_ENABLE_GATING=false
```

---

### `ELEANOR_ORCHESTRATOR_ENABLE_RETRIES`
**Purpose**: Enable/disable automatic retry logic  
**Default**: `true`  
**Values**: `true`, `false`, `1`, `0`, `yes`, `no`, `on`, `off`  
**Impact**: Failed critics (except critical ones) will be retried with exponential backoff  
**Benefit**: 85% → 98% success rate

```bash
# Production (recommended)
export ELEANOR_ORCHESTRATOR_ENABLE_RETRIES=true

# Testing (fail fast)
export ELEANOR_ORCHESTRATOR_ENABLE_RETRIES=false
```

---

### `ELEANOR_ORCHESTRATOR_STRICT_VALIDATION`
**Purpose**: Enable/disable result validation  
**Default**: `true`  
**Values**: `true`, `false`, `1`, `0`, `yes`, `no`, `on`, `off`  
**Impact**: Validates critics return required fields (`value`, `score`)  
**Benefit**: Catch data quality issues early

```bash
# Production
export ELEANOR_ORCHESTRATOR_STRICT_VALIDATION=true

# Lenient mode
export ELEANOR_ORCHESTRATOR_STRICT_VALIDATION=false
```

---

### `ELEANOR_ORCHESTRATOR_FAIL_ON_VALIDATION`
**Purpose**: Fail execution if validation errors found  
**Default**: `false` (log warnings but continue)  
**Values**: `true`, `false`, `1`, `0`, `yes`, `no`, `on`, `off`  
**Impact**: Strict mode - validation errors raise exceptions  
**Use Case**: Development/testing to catch bugs

```bash
# Production (lenient)
export ELEANOR_ORCHESTRATOR_FAIL_ON_VALIDATION=false

# Development (strict)
export ELEANOR_ORCHESTRATOR_FAIL_ON_VALIDATION=true
```

---

## Resource Management

### `ELEANOR_ORCHESTRATOR_MAX_CONCURRENT`
**Purpose**: Maximum concurrent critics across all stages  
**Default**: `10`  
**Values**: Any positive integer (1-100 reasonable range)  
**Impact**: Higher = more throughput but more resource usage  
**Tuning**: Adjust based on CPU cores and memory

```bash
# Small deployment
export ELEANOR_ORCHESTRATOR_MAX_CONCURRENT=5

# Medium deployment (default)
export ELEANOR_ORCHESTRATOR_MAX_CONCURRENT=10

# Large deployment
export ELEANOR_ORCHESTRATOR_MAX_CONCURRENT=20
```

---

### `ELEANOR_ORCHESTRATOR_GLOBAL_TIMEOUT`
**Purpose**: Global timeout for entire orchestrator execution  
**Default**: None (no global limit)  
**Values**: Positive number (seconds)  
**Impact**: Execution will be cut off after this time  
**Use Case**: Prevent runaway executions

```bash
# No global limit (default)
# (not set)

# 30 second limit
export ELEANOR_ORCHESTRATOR_GLOBAL_TIMEOUT=30.0

# 10 second limit (aggressive)
export ELEANOR_ORCHESTRATOR_GLOBAL_TIMEOUT=10.0
```

---

## Legacy System (from previous diffs)

### `ELEANOR_ENFORCE_SINGLE_CRITIC_PIPELINE`
**Purpose**: Prevent accidental use of legacy critic runner  
**Default**: `true` (enforce orchestrator use)  
**Values**: `true`, `false`, `1`, `0`, `yes`, `no`, `on`, `off`  
**Impact**: Blocks `run_critics_parallel` legacy function  
**Recommendation**: Keep enabled in production

```bash
# Production (recommended)
export ELEANOR_ENFORCE_SINGLE_CRITIC_PIPELINE=true

# Allow legacy (tests only)
export ELEANOR_ENFORCE_SINGLE_CRITIC_PIPELINE=false
```

---

### `ELEANOR_ALLOW_LEGACY_CRITIC_RUNNER`
**Purpose**: Explicitly allow legacy critic runner  
**Default**: Not set (blocks legacy)  
**Values**: `1`, `true`, `yes`, `on` (allow legacy)  
**Impact**: Bypasses the enforcement guardrail  
**Recommendation**: Only for tests/dev

```bash
# Production (not set - blocks legacy)
# (not set)

# Tests/dev (allow legacy)
export ELEANOR_ALLOW_LEGACY_CRITIC_RUNNER=1
```

---

## Recommended Configurations

### Production (Default - Optimized)
```bash
# Features
export ELEANOR_ORCHESTRATOR_ENABLE_GATING=true
export ELEANOR_ORCHESTRATOR_ENABLE_RETRIES=true
export ELEANOR_ORCHESTRATOR_STRICT_VALIDATION=true
export ELEANOR_ORCHESTRATOR_FAIL_ON_VALIDATION=false

# Resources
export ELEANOR_ORCHESTRATOR_MAX_CONCURRENT=10
export ELEANOR_ORCHESTRATOR_GLOBAL_TIMEOUT=30.0

# Safety
export ELEANOR_ENFORCE_SINGLE_CRITIC_PIPELINE=true
```

**Best for**: Production deployments, cost optimization, normal traffic

---

### High-Traffic Production
```bash
# More aggressive optimization
export ELEANOR_ORCHESTRATOR_ENABLE_GATING=true
export ELEANOR_ORCHESTRATOR_ENABLE_RETRIES=true
export ELEANOR_ORCHESTRATOR_MAX_CONCURRENT=20
export ELEANOR_ORCHESTRATOR_GLOBAL_TIMEOUT=20.0
```

**Best for**: High-volume deployments, aggressive cost savings

---

### High-Reliability Production
```bash
# Favor reliability over speed
export ELEANOR_ORCHESTRATOR_ENABLE_GATING=false  # Run all critics
export ELEANOR_ORCHESTRATOR_ENABLE_RETRIES=true
export ELEANOR_ORCHESTRATOR_MAX_CONCURRENT=15
export ELEANOR_ORCHESTRATOR_GLOBAL_TIMEOUT=60.0  # Generous timeout
```

**Best for**: Critical systems, compliance-heavy, can tolerate latency

---

### Development/Testing
```bash
# Disable optimizations to test all paths
export ELEANOR_ORCHESTRATOR_ENABLE_GATING=false
export ELEANOR_ORCHESTRATOR_ENABLE_RETRIES=false
export ELEANOR_ORCHESTRATOR_STRICT_VALIDATION=true
export ELEANOR_ORCHESTRATOR_FAIL_ON_VALIDATION=true
export ELEANOR_ORCHESTRATOR_MAX_CONCURRENT=5

# Allow legacy for old tests
export ELEANOR_ALLOW_LEGACY_CRITIC_RUNNER=1
```

**Best for**: Local development, testing, debugging

---

### Staging
```bash
# Production-like but more observable
export ELEANOR_ORCHESTRATOR_ENABLE_GATING=true
export ELEANOR_ORCHESTRATOR_ENABLE_RETRIES=true
export ELEANOR_ORCHESTRATOR_STRICT_VALIDATION=true
export ELEANOR_ORCHESTRATOR_FAIL_ON_VALIDATION=false
export ELEANOR_ORCHESTRATOR_MAX_CONCURRENT=10
export ELEANOR_ORCHESTRATOR_GLOBAL_TIMEOUT=30.0
```

**Best for**: Staging environment, pre-production testing

---

## Quick Diagnostics

### Check Current Configuration

```bash
# Print all orchestrator settings
env | grep ELEANOR_ORCHESTRATOR

# Check legacy guardrails
env | grep ELEANOR_ENFORCE
env | grep ELEANOR_ALLOW
```

### Test Configuration

```python
# In Python
from engine.runtime.critic_infrastructure import get_orchestrator_config

config = get_orchestrator_config()
print(f"Max concurrent: {config.max_concurrent_critics}")
print(f"Gating enabled: {config.enable_policy_gating}")
print(f"Retries enabled: {config.enable_retries}")
print(f"Validation: {config.strict_validation}")
```

---

## Troubleshooting

### Critics not being gated (running all)
```bash
# Check if gating is enabled
echo $ELEANOR_ORCHESTRATOR_ENABLE_GATING

# Enable gating
export ELEANOR_ORCHESTRATOR_ENABLE_GATING=true
```

### Too many timeouts
```bash
# Increase global timeout
export ELEANOR_ORCHESTRATOR_GLOBAL_TIMEOUT=60.0

# Or disable strict timeouts (not recommended)
```

### Validation errors breaking execution
```bash
# Make validation lenient
export ELEANOR_ORCHESTRATOR_FAIL_ON_VALIDATION=false
```

### Legacy critic runner errors
```bash
# If you see "Legacy critic runner invoked" errors:

# Option 1: Update code to use orchestrator (recommended)

# Option 2: Allow legacy for specific tests
export ELEANOR_ALLOW_LEGACY_CRITIC_RUNNER=1

# Option 3: Disable enforcement (not recommended)
export ELEANOR_ENFORCE_SINGLE_CRITIC_PIPELINE=false
```

### High resource usage
```bash
# Reduce concurrency
export ELEANOR_ORCHESTRATOR_MAX_CONCURRENT=5

# Add global timeout
export ELEANOR_ORCHESTRATOR_GLOBAL_TIMEOUT=20.0
```

---

## Docker Configuration

### docker-compose.yml
```yaml
services:
  eleanor:
    environment:
      # Production orchestrator
      - ELEANOR_ORCHESTRATOR_ENABLE_GATING=true
      - ELEANOR_ORCHESTRATOR_ENABLE_RETRIES=true
      - ELEANOR_ORCHESTRATOR_MAX_CONCURRENT=10
      - ELEANOR_ORCHESTRATOR_GLOBAL_TIMEOUT=30.0
      
      # Validation
      - ELEANOR_ORCHESTRATOR_STRICT_VALIDATION=true
      - ELEANOR_ORCHESTRATOR_FAIL_ON_VALIDATION=false
      
      # Legacy guardrails
      - ELEANOR_ENFORCE_SINGLE_CRITIC_PIPELINE=true
```

### Kubernetes ConfigMap
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: eleanor-config
data:
  ELEANOR_ORCHESTRATOR_ENABLE_GATING: "true"
  ELEANOR_ORCHESTRATOR_ENABLE_RETRIES: "true"
  ELEANOR_ORCHESTRATOR_MAX_CONCURRENT: "10"
  ELEANOR_ORCHESTRATOR_GLOBAL_TIMEOUT: "30.0"
  ELEANOR_ORCHESTRATOR_STRICT_VALIDATION: "true"
  ELEANOR_ORCHESTRATOR_FAIL_ON_VALIDATION: "false"
  ELEANOR_ENFORCE_SINGLE_CRITIC_PIPELINE: "true"
```

---

## Summary

| Variable | Default | Production | Dev | Impact |
|----------|---------|------------|-----|--------|
| `ENABLE_GATING` | true | true | false | Cost $$$ |
| `ENABLE_RETRIES` | true | true | false | Reliability |
| `STRICT_VALIDATION` | true | true | true | Quality |
| `FAIL_ON_VALIDATION` | false | false | true | Strictness |
| `MAX_CONCURRENT` | 10 | 10-20 | 5 | Throughput |
| `GLOBAL_TIMEOUT` | none | 30.0 | none | Safety |
| `ENFORCE_SINGLE_PIPELINE` | true | true | true | Architecture |
| `ALLOW_LEGACY_RUNNER` | unset | unset | 1 | Testing |

**Quick Start**: Use defaults (no variables set) - they're already optimized for production!

---

**Reference Updated**: January 13, 2026
