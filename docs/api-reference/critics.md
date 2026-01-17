# Critics API

Critics live in `engine/critics/`. Each critic implements an async `evaluate()` method
and returns a `CriticResult` defined in `engine/schemas/pipeline_types.py`.

## Base classes

- `BaseCriticV8` in `engine/critics/base.py`
- `ConstitutionalCritic` in `engine/critics/base.py`

## Built-in critics

- Autonomy: `engine/critics/autonomy.py`
- Fairness: `engine/critics/fairness.py`
- Pragmatics: `engine/critics/pragmatics.py`
- Risk: `engine/critics/risk.py`
- Rights: `engine/critics/rights.py`
- Truth: `engine/critics/truth.py`

See `docs/CRITIC_INDEPENDENCE_AND_ESCALATION.md` for design constraints.
