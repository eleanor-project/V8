# ELEANOR V8 Configuration Guide

## Overview

ELEANOR V8 uses a hierarchical configuration system with clear precedence and validation. Configuration can be provided through multiple sources with the following priority (highest to lowest):

1. **Command-line arguments** (runtime parameters)
2. **Environment variables** (with `ELEANOR_` prefix)
3. **Environment files** (`.env`, `.env.{environment}`)
4. **YAML configuration** (legacy support)
5. **Default values** (in code)

## Quick Start

### Development Setup

```bash
# Copy sample environment file
cp .env.sample .env

# Edit with your settings
nano .env

# Validate configuration
python scripts/validate_config.py
```

### Environment-Specific Configuration

```bash
# Development
export ENV=development
# Uses .env.development

# Staging
export ENV=staging
# Uses .env.staging

# Production
export ENV=production
# Uses .env.production
```

## Configuration Structure

### Engine Settings

```bash
ELEANOR_ENVIRONMENT=development
ELEANOR_DETAIL_LEVEL=2  # 1=minimal, 2=standard, 3=forensic
ELEANOR_ENABLE_REFLECTION=true
ELEANOR_ENABLE_DRIFT_CHECK=true
ELEANOR_ENABLE_PRECEDENT_ANALYSIS=true
```

### LLM Configuration

```bash
ELEANOR_LLM__PROVIDER=ollama  # ollama, openai, anthropic
ELEANOR_LLM__MODEL_NAME=llama3.2:3b
ELEANOR_LLM__BASE_URL=http://localhost:11434
ELEANOR_LLM__API_KEY=  # For cloud providers
ELEANOR_LLM__TIMEOUT=30.0
ELEANOR_LLM__MAX_RETRIES=3
ELEANOR_LLM__TEMPERATURE=0.7
```

### Router Configuration

```bash
ELEANOR_ROUTER__HEALTH_CHECK_INTERVAL=60
ELEANOR_ROUTER__FALLBACK_MODEL=llama3.2:3b
ELEANOR_ROUTER__ENABLE_COST_OPTIMIZATION=true
ELEANOR_ROUTER__MAX_LATENCY_MS=5000
```

### Precedent Configuration

```bash
ELEANOR_PRECEDENT__BACKEND=chroma  # none, chroma, qdrant, pinecone
ELEANOR_PRECEDENT__CONNECTION_STRING=http://chroma:8000
ELEANOR_PRECEDENT__CACHE_TTL=3600
ELEANOR_PRECEDENT__MAX_RESULTS=5
ELEANOR_PRECEDENT__SIMILARITY_THRESHOLD=0.7
```

### Evidence Recording

```bash
ELEANOR_EVIDENCE__ENABLED=true
ELEANOR_EVIDENCE__JSONL_PATH=evidence.jsonl
ELEANOR_EVIDENCE__BUFFER_SIZE=1000
ELEANOR_EVIDENCE__FLUSH_INTERVAL=5.0
ELEANOR_EVIDENCE__FAIL_ON_ERROR=false
ELEANOR_EVIDENCE__SANITIZE_SECRETS=true
```

### Performance

```bash
ELEANOR_PERFORMANCE__MAX_CONCURRENCY=12
ELEANOR_PERFORMANCE__TIMEOUT_SECONDS=10.0
ELEANOR_PERFORMANCE__ENABLE_ADAPTIVE_CONCURRENCY=true
ELEANOR_PERFORMANCE__TARGET_LATENCY_MS=500.0
```

### Caching

```bash
ELEANOR_CACHE__ENABLED=true
ELEANOR_CACHE__REDIS_URL=redis://redis:6379/0
ELEANOR_CACHE__PRECEDENT_TTL=3600
ELEANOR_CACHE__EMBEDDINGS_TTL=7200
ELEANOR_CACHE__CRITICS_TTL=1800
ELEANOR_CACHE__MAX_MEMORY_MB=500
```

### Security

```bash
ELEANOR_SECURITY__MAX_INPUT_SIZE_BYTES=100000
ELEANOR_SECURITY__MAX_CONTEXT_DEPTH=5
ELEANOR_SECURITY__ENABLE_PROMPT_INJECTION_DETECTION=true
ELEANOR_SECURITY__SECRET_PROVIDER=vault  # env, aws, vault
ELEANOR_SECURITY__SECRETS_CACHE_TTL=300
```

### API Rate Limiting

```bash
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
RATE_LIMIT_REDIS_URL=redis://localhost:6379/0
RATE_LIMIT_KEY_PREFIX=eleanor:rate_limit:
```

### Observability

```bash
ELEANOR_OBSERVABILITY__LOG_LEVEL=INFO
ELEANOR_OBSERVABILITY__ENABLE_STRUCTURED_LOGGING=true
ELEANOR_OBSERVABILITY__ENABLE_TRACING=true
ELEANOR_OBSERVABILITY__OTEL_ENDPOINT=http://otel-collector:4317
ELEANOR_OBSERVABILITY__JAEGER_ENDPOINT=http://jaeger:14268
ELEANOR_OBSERVABILITY__METRICS_PORT=9090
```

### Resilience

```bash
ELEANOR_RESILIENCE__ENABLE_CIRCUIT_BREAKERS=true
ELEANOR_RESILIENCE__CIRCUIT_BREAKER_THRESHOLD=5
ELEANOR_RESILIENCE__CIRCUIT_BREAKER_TIMEOUT=60
ELEANOR_RESILIENCE__ENABLE_GRACEFUL_DEGRADATION=true
ELEANOR_RESILIENCE__MAX_RETRY_ATTEMPTS=3
```

## Using Configuration in Code

### Basic Usage

```python
from engine.config import get_settings

# Get global settings instance
settings = get_settings()

print(f"Environment: {settings.environment}")
print(f"LLM Provider: {settings.llm.provider}")
print(f"Max Concurrency: {settings.performance.max_concurrency}")
```

### Using Config Manager

```python
from engine.config import ConfigManager

# Get singleton config manager
config = ConfigManager()

# Access settings
print(config.settings.llm.provider)

# Get by dot notation
max_concurrency = config.get("performance.max_concurrency", default=6)

# Validate configuration
validation = config.validate()
if not validation["valid"]:
    print(f"Errors: {validation['errors']}")
    print(f"Warnings: {validation['warnings']}")

# Reload configuration
config.reload(env_file=".env.production")
```

### Legacy Compatibility

```python
from engine.config import get_settings
from engine.engine import EleanorEngineV8, EngineConfig

# Get new settings
settings = get_settings()

# Convert to legacy format
legacy_config = settings.to_legacy_engine_config()

# Create engine with legacy config
engine = EleanorEngineV8(
    config=EngineConfig(**legacy_config)
)
```

## Environment-Specific Best Practices

### Development

- Use `.env.development`
- Enable debug logging (`LOG_LEVEL=DEBUG`)
- Disable caching for faster iteration
- Use local Ollama for LLM
- Forensic detail level (3) for debugging

### Staging

- Use `.env.staging`
- Enable all production features
- Use Redis caching
- Enable distributed tracing
- Enable circuit breakers
- Test with production-like data

### Production

- Use `.env.production`
- **Required settings:**
  - `SECRET_PROVIDER=vault` or `aws` (NOT `env`)
  - `ENABLE_CIRCUIT_BREAKERS=true`
  - `ENABLE_STRUCTURED_LOGGING=true`
  - `PRECEDENT__BACKEND` must be configured
  - `CACHE__ENABLED=true` with Redis
- Enable monitoring and tracing
- Use standard detail level (2)
- Configure proper timeouts and retries

## Validation

### Validate Current Configuration

```bash
python scripts/validate_config.py
```

### Validate Specific Environment

```bash
python scripts/validate_config.py --env production
```

### Validate All Environments

```bash
python scripts/validate_config.py --all --verbose
```

### JSON Output

```bash
python scripts/validate_config.py --json > config_validation.json
```

## Migration from Legacy Configuration

The new configuration system maintains backward compatibility:

```python
# Old way (still works)
from engine.engine import EngineConfig

config = EngineConfig(
    detail_level=2,
    max_concurrency=6,
    enable_reflection=True
)

# New way (recommended)
from engine.config import get_settings

settings = get_settings()
legacy_config = settings.to_legacy_engine_config()
```

## Troubleshooting

### Configuration Not Loading

1. Check environment file exists
2. Verify environment variable names (must start with `ELEANOR_`)
3. Check for syntax errors in `.env` file
4. Run validation script

### Nested Configuration

Use double underscores for nested settings:

```bash
# Correct
ELEANOR_LLM__PROVIDER=ollama

# Incorrect
ELEANOR_LLM.PROVIDER=ollama
```

### Production Warnings

The system will warn about:
- Using environment variables for secrets
- Missing precedent backend
- Disabled circuit breakers
- Disabled tracing

These warnings should be addressed before production deployment.

## See Also

- [Secrets Management](secrets-management.md)
- [Performance Tuning](performance.md)
- [Observability Setup](observability.md)
- [Circuit Breakers](resilience.md)
