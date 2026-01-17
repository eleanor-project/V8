# Resilience

Resilience features keep the engine stable under partial failures and high load.
They include circuit breakers, retries, health checks, and graceful degradation.

## Where it lives

- `engine/resilience/` provides circuit breakers, retries, and recovery.
- `engine/resource/` and `engine/resource_manager.py` manage system limits.
- `engine/cache/` and `engine/observability/` support resilience decisions.

## Core mechanisms

- Circuit breakers: `engine/resilience/circuit_breaker.py`
- Retry policy: `engine/resilience/retry.py`
- Degradation strategy: `engine/resilience/degradation.py`
- Health checks: `engine/resilience/health.py`

## Runtime behavior

- Critical dependencies are monitored and can be marked degraded.
- Optional components (precedent, uncertainty, evidence recording) may fall
  back to defaults when degraded.
- Evidence of degradation is surfaced on `EngineResult`:
  - `degraded_components`
  - `is_degraded`

For operational guidance, see `docs/RESILIENCE.md` and `docs/RUNBOOKS.md`.
