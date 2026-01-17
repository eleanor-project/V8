# Configuration Options

ELEANOR V8 configuration is defined in `engine/config/settings.py` and
`engine/runtime/config.py`. Settings can be supplied via environment variables,
YAML config, or programmatic overrides.

## Major sections

- Environment: `environment`, `constitutional_config_path`
- Runtime: `detail_level`, `max_concurrency`, `timeout_seconds`
- LLM: provider, model name, API keys
- Router: fallback model, health check interval
- Precedent: backend selection, similarity thresholds
- Evidence: JSONL path, buffering, sanitization
- Performance: adaptive concurrency, cache sizing
- Observability: logging and tracing controls
- Resilience: circuit breakers, degradation behavior
- GPU: device preferences, batching, memory limits

## YAML files

Sample configuration files live in `config/`:

- `config/eleanor.yaml`
- `config/observability.yaml`
- `config/resource_management.yaml`
- `config/gpu.yaml`

## Programmatic override

```python
from engine.runtime.config import EngineConfig

config = EngineConfig(detail_level=3, timeout_seconds=15.0)
```

For full field definitions, see `engine/config/settings.py`.
