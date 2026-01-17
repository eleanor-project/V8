# Metrics Reference

Metrics are emitted via Prometheus when `prometheus_client` is installed.
The implementation lives under `engine/observability/`.

## Business metrics

Defined in `engine/observability/business_metrics.py`:

- `eleanor_decisions_total`
- `eleanor_escalations_total`
- `eleanor_critic_agreement`
- `eleanor_critic_disagreement`
- `eleanor_uncertainty_score`
- `eleanor_severity_score`
- `eleanor_active_traces`
- `eleanor_degraded_components`

## Cost metrics

Defined in `engine/observability/cost_tracking.py`:

- LLM request counters and latency histograms
- Token usage tracking

## Operations

- Metrics are typically exposed via the `/metrics` endpoint.
- See `docs/OBSERVABILITY.md` and `docs/RUNBOOKS.md` for deployment guidance.
