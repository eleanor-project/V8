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
2. Grafana can consume these values either directly via the existing `/metrics` endpoint (Prometheus format) or by running `python scripts/dependency_prometheus_exporter.py` in a sidecar that polls `/admin/dependencies` and republishes the counters/gauges (`eleanor_dependency_failures_total` + `eleanor_dependency_failure_last_timestamp_seconds`). The exporter can be tuned with `DEP_EXPORT_*` env vars for endpoint, listen address/port, and scrape interval.
3. Build a Grafana panel that plots `eleanor_dependency_failures_total` and add an alert rule such as “Trigger when `sum(eleanor_dependency_failures_total) > 0` for 5 minutes.” Point the alert to your incident channel (Slack/Teams/PagerDuty) so engineers are notified without checking the dashboard.
4. Within the UI dashboard, the new summary cards and alert suggestion (`Dependency Panel → Alert recommendation`) make it easy to decide whether to raise infra tickets or adjust Grafana thresholds; keep `/admin/dependencies` behind `ADMIN_ROLE` so only authorized dashboards can poll it.
5. When a dependency failure alert fires, grab the matching `dependency_failed_to_load` log entry (it already includes structured `timestamp`, `dependency`, and `error` fields plus `trace_id`/`span_id` when tracing is enabled via `TraceContext`) and paste the `trace_id` into Jaeger or Grafana’s linked trace view for fast root-cause investigation.

## Grafana Notification Channels

1. Create Grafana notification channels for Teams and Slack by configuring webhooks in **Alerting → Notification channels**. Use the official Slack incoming webhook URL (e.g., `https://hooks.slack.com/services/…`) or the Teams connector URL from the channel’s “Incoming Webhook” action.
2. Assign both channels to the dependency failure alert you created; in the alert rule’s “Send to” list include the Slack channel first and the Teams channel second so every failure posts to both gateways.
3. Customize the alert message with placeholders like `{{ $labels.dependency }}` and `{{ $evalMatches | first | value }}` so the channel notifications convey which dependency failed and how many times.
4. If desired, add Grafana media types (for example, the Notify Webhook type) so the alert JSON can be parsed by downstream automation or chatops bots that create tickets.

With these channels in place, any `sum(eleanor_dependency_failures_total) > 0` alert automatically notifies your incident responders via Slack and Teams while the dashboard keeps showing the health snapshots.

Adding these pieces completes the observability loop: the backend emits the metrics, the dashboard shows the status, and Grafana escalates issues when failures appear.

```
trace_id="abc123-def456"
```
