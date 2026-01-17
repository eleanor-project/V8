# Uncertainty Engine

The uncertainty engine quantifies confidence in a decision by combining critic
agreement, precedent alignment, and model stability signals. The output is used
by governance to decide whether additional review is required.

## Where it lives

- `engine/uncertainty/uncertainty.py` provides `UncertaintyEngineV8`.
- `engine/runtime/models.py` exposes `EngineResult.uncertainty`.
- `engine/schemas/pipeline_types.py` defines the uncertainty result schema.

## Inputs

- Critic outputs keyed by critic name.
- Precedent alignment results (conflict and drift signals).
- The selected model name (used for stability heuristics).

## Outputs

The engine emits a structured payload with the following fields:

- `epistemic_uncertainty`
- `aleatoric_uncertainty`
- `critic_divergence`
- `precedent_conflict_uncertainty`
- `model_stability_uncertainty`
- `overall_uncertainty`
- `needs_escalation`
- `explanation`

## Escalation thresholds

The current implementation escalates when:

- `overall_uncertainty` is at or above 0.65, or
- `precedent_conflict_uncertainty` is at or above 0.75.

Tune thresholds by updating `engine/uncertainty/uncertainty.py` and aligning
with governance policy.
