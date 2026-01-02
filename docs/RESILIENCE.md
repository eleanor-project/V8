# ELEANOR V8 - Resilience & Graceful Degradation

## Overview

ELEANOR implements resilience patterns to handle failures gracefully:

1. **Circuit Breakers**: Prevent cascading failures
2. **Graceful Degradation**: Continue with reduced functionality
3. **Fallback Strategies**: Sensible defaults when components fail
4. **Health Monitoring**: Track component health

## Circuit Breaker Pattern

### States

- **CLOSED**: Normal operation (all requests pass through)
- **OPEN**: Too many failures (reject requests immediately)
- **HALF_OPEN**: Testing recovery (allow limited requests)

### Configuration

```python
from engine.resilience import CircuitBreaker

breaker = CircuitBreaker(
    failure_threshold=5,  # Open after 5 failures
    recovery_timeout=60,  # Wait 60s before testing recovery
    success_threshold=2   # Close after 2 successes in HALF_OPEN
)
```

### Usage

```python
try:
    result = await breaker.call(expensive_operation, arg1, arg2)
except CircuitBreakerOpenError:
    # Circuit is open, use fallback
    result = await fallback_operation()
```

## Graceful Degradation

### Degradation Levels

1. **Full Functionality**: All components operational
2. **Partial Degradation**: Some components unavailable, continue with subset
3. **Minimal Functionality**: Critical components only
4. **Complete Failure**: Cannot proceed

### Degradation Strategies

```python
from engine.resilience import DegradationStrategy

# Precedent store unavailable
fallback = await DegradationStrategy.precedent_fallback(error)
# Returns: {'cases': [], 'novel': True, 'degraded': True, ...}

# Router unavailable
fallback = await DegradationStrategy.router_fallback(error)
# Returns: {'model_name': 'llama3.2:3b', 'degraded': True, ...}

# Critic unavailable
fallback = await DegradationStrategy.critic_fallback('rights', error)
# Returns: {'violations': [], 'degraded': True, ...}
```

### Integration Example

```python
from engine.resilience import CircuitBreaker, DegradationStrategy

class ResilientEngine:
    def __init__(self, engine):
        self.engine = engine
        self.precedent_breaker = CircuitBreaker()
        self.router_breaker = CircuitBreaker()
        self.degradation = DegradationStrategy()
    
    async def run_with_resilience(self, text, context):
        degraded_components = []
        
        # Router with fallback
        try:
            router_result = await self.router_breaker.call(
                self.engine._select_model, text, context
            )
        except Exception as e:
            degraded_components.append('router')
            router_result = await self.degradation.router_fallback(e)
        
        # Precedent (optional)
        try:
            precedent = await self.precedent_breaker.call(
                self.engine._get_precedents, text
            )
        except Exception as e:
            degraded_components.append('precedent')
            precedent = await self.degradation.precedent_fallback(e)
        
        # Continue pipeline with available data
        result = await self.engine._complete_pipeline(
            router_result, precedent
        )
        
        result['degraded_components'] = degraded_components
        result['is_degraded'] = len(degraded_components) > 0
        
        return result
```

## Health Monitoring

```python
from engine.resilience import ComponentHealthChecker

health_checker = ComponentHealthChecker({
    'router': router_breaker,
    'precedent': precedent_breaker,
    'uncertainty': uncertainty_breaker,
})

# Get health status
status = health_checker.get_health_status()
print(status)
# {
#   'overall_health': 'healthy',  # or 'degraded', 'unhealthy'
#   'components': {
#     'router': {'state': 'closed', 'healthy': True},
#     'precedent': {'state': 'open', 'healthy': False},
#     ...
#   }
# }

# Check if healthy
if health_checker.is_healthy():
    print("All systems operational")

# Get unhealthy components
unhealthy = health_checker.get_unhealthy_components()
if unhealthy:
    print(f"Degraded components: {unhealthy}")
```

### API Endpoint

The API exposes circuit breaker status at:

```
GET /admin/resilience/health
```

Response includes per-component state, failure counts, and an overall health
summary (`healthy`, `degraded`, or `unhealthy`).

## Configuration

Configuration is controlled via `ELEANOR_RESILIENCE__*` settings:

```bash
ELEANOR_RESILIENCE__ENABLE_CIRCUIT_BREAKERS=true
ELEANOR_RESILIENCE__CIRCUIT_BREAKER_THRESHOLD=5
ELEANOR_RESILIENCE__CIRCUIT_BREAKER_TIMEOUT=60
ELEANOR_RESILIENCE__ENABLE_GRACEFUL_DEGRADATION=true
```

`config/resilience.yaml` remains a reference template.

## Failure Scenarios

### Scenario 1: Precedent Store Down

- Circuit breaker opens after 5 failures
- Fallback: Continue without precedent analysis
- Result marked as `degraded=True`
- System remains operational

### Scenario 2: Router Unavailable

- Circuit breaker opens after 3 failures
- Fallback: Use default model (llama3.2:3b)
- Result includes `degradation_reason`
- Pipeline completes with fallback model

### Result Signaling

All pipeline results include:
- `degraded_components`: list of components in degraded mode
- `is_degraded`: boolean indicating degraded operation

### Scenario 3: Single Critic Fails

- Other critics continue execution
- Failed critic returns empty result
- Aggregation continues with available critics
- Requires minimum 2 critics (configurable)

### Scenario 4: Complete LLM Failure

- All model calls fail
- Circuit breaker opens
- No fallback possible (critical component)
- Return error to user

## Best Practices

1. **Set Appropriate Thresholds**: Balance between sensitivity and stability
2. **Monitor Circuit States**: Alert on circuit breaker opens
3. **Test Degradation**: Regularly test fallback paths
4. **Document Behavior**: Clear communication about degraded mode
5. **Graceful Error Messages**: Inform users of degraded functionality

## Metrics

Track resilience metrics:

- Circuit breaker state changes
- Degradation frequency
- Time spent in degraded mode
- Recovery time
- Fallback usage rates
