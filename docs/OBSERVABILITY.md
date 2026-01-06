# ELEANOR V8 - Observability Guide

## Overview

ELEANOR provides comprehensive observability through:

1. **Structured Logging**: JSON logs with full context
2. **Distributed Tracing**: OpenTelemetry integration
3. **Metrics**: Cache, circuit breaker, and performance metrics

## Structured Logging

### Configuration

```python
from engine.observability import configure_logging, get_logger

# Configure logging
configure_logging(
    log_level="INFO",
    json_logs=True,  # JSON for production
    development_mode=False
)

# Get logger
logger = get_logger(__name__)
```

### Log Format

All logs follow a structured schema:

```json
{
  "timestamp": "2025-01-01T12:00:00.000Z",
  "level": "info",
  "logger": "engine.critics.rights",
  "event": "critic_evaluation_complete",
  "trace_id": "abc123-def456",
  "span_id": "span789",
  "component": "critic",
  "critic": "rights",
  "duration_ms": 125.3,
  "violations_count": 2,
  "severity": 0.75
}
```

### Key Events

- `engine_run_start`: Pipeline execution begins
- `router_selection_complete`: Model selected
- `critic_evaluation_complete`: Critic finished
- `precedent_alignment_complete`: Precedent analysis done
- `governance_review_triggered`: Review required
- `cache_hit` / `cache_miss`: Cache operations
- `circuit_breaker_opened` / `circuit_breaker_closed`: State changes

## Distributed Tracing

### Setup

```python
from engine.observability import configure_tracing, get_tracer, TraceContext

# Configure tracing
configure_tracing(
    service_name="eleanor-v8",
    jaeger_endpoint="localhost:6831",
    enabled=True
)

# Get tracer
tracer = get_tracer(__name__)
```

### Instrumentation

```python
# Instrument operations
with tracer.start_as_current_span(
    "engine.run",
    attributes={
        "engine.trace_id": trace_id,
        "input.length": len(text),
    }
) as span:
    result = await self._execute_pipeline(...)
    
    # Add result attributes
    span.set_attribute("critics.count", len(critic_results))
    span.set_attribute("pipeline.duration_ms", duration)
```

### Trace Context Propagation

```python
# Set trace context
TraceContext.set_trace_id(trace_id)
TraceContext.set_span_id(span_id)

# Get trace context (automatically propagated)
trace_id = TraceContext.get_trace_id()

# Bind to logger
logger = logger.bind(
    trace_id=TraceContext.get_trace_id(),
    span_id=TraceContext.get_span_id()
)
```

### Span Hierarchy

```
engine.run
├── router.select_model
├── critics.parallel_execution
│   ├── critic.rights.evaluate
│   ├── critic.risk.evaluate
│   ├── critic.fairness.evaluate
│   └── ...
├── precedent.retrieve
├── precedent.align
├── uncertainty.compute
└── aggregator.aggregate
```

## Metrics

### Cache Metrics

```python
stats = cache_manager.get_stats()
# Returns:
# {
#   'precedent': {'hits': 100, 'misses': 40, 'hit_rate': 0.71},
#   'embedding': {'hits': 200, 'misses': 50, 'hit_rate': 0.80},
#   ...
# }
```

### Circuit Breaker Metrics

```python
health = health_checker.get_health_status()
# Returns:
# {
#   'overall_health': 'healthy',
#   'components': {
#     'router': {'state': 'closed', 'healthy': True},
#     'precedent': {'state': 'closed', 'healthy': True},
#     ...
#   }
# }
```

### Concurrency Metrics

```python
stats = concurrency_manager.get_stats()
# Returns:
# {
#   'current_limit': 8,
#   'avg_latency_ms': 450,
#   'p95_latency_ms': 520,
#   'target_latency_ms': 500
# }
```

## Jaeger UI

View traces in Jaeger:

1. Start Jaeger: `docker run -p 6831:6831/udp -p 16686:16686 jaegertracing/all-in-one`
2. Open UI: http://localhost:16686
3. Select service: `eleanor-v8`
4. Search by trace ID or operation

## Query Examples

### Find Slow Requests

```
service="eleanor-v8" AND duration > 2s
```

### Find Failed Critic Evaluations

```
event="critic_evaluation_failed" AND critic="rights"
```

### Find Circuit Breaker Opens

```
event="circuit_breaker_opened"
```

### Correlate Logs and Traces

## Dependency Failure Alerting

1. `/admin/dependencies` now returns `total_failures`, `tracked_dependencies`, `status`, and `last_checked` alongside per-dependency failure metadata. If you surface this in Grafana it becomes a rich observability source instead of a simple table.
2. Grafana can consume these values either through the existing `/metrics` endpoint (which already exposes Prometheus format when `prometheus_client` is installed) or via a lightweight exporter. The backend now exposes `eleanor_dependency_failures_total{dependency="redis"}` counters and `eleanor_dependency_failure_last_timestamp_seconds{dependency="redis"}` gauges directly through the Prometheus client in `engine/utils/dependency_tracking`.
3. Build a Grafana panel that plots `eleanor_dependency_failures_total` and add an alert rule such as “Trigger when `sum(eleanor_dependency_failures_total) > 0` for 5 minutes.” Tie the alert to Slack/Teams so engineers know when a dependency starts failing repeatedly.
4. Within the UI dashboard, the new summary cards and alert suggestion (`Dependency Panel → Alert recommendation`) make it easy to decide whether to raise infra tickets or adjust Grafana thresholds; keep `/admin/dependencies` behind `ADMIN_ROLE` so only authorized dashboards can poll it.

Adding these pieces completes the observability loop: the backend emits the metrics, the dashboard shows the status, and Grafana escalates issues when failures appear.

```
trace_id="abc123-def456"
```
