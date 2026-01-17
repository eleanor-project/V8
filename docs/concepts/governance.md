# Governance

Governance enforces constitutional and policy gates before releasing a response.
The runtime routes critic findings and evidence through rules that determine
whether the system can answer or must escalate to human review.

## Where it lives

- `engine/governance/` contains runtime enforcement and explainable governance.
- `governance/` contains policies, review packets, and routing logic.
- `governance/*.rego` defines Open Policy Agent (OPA) policy gates.

## Governance flow

1. Critics evaluate the request and emit findings.
2. Aggregation produces a unified decision and severity score.
3. Governance evaluates constitutional rules and review triggers.
4. The engine marks the decision as allow, review_required, or escalate.

## Key components

- Review triggers: `governance/review_triggers.py`
- Review packets: `governance/review_packets.py`
- Constitutional config: `governance/constitutional.yaml`
- OPA enforcement: `engine/governance/opa_enforcer.py`
- Explainable governance: `engine/governance/explainable.py`

## Outputs

Governance decisions are surfaced on `EngineResult` fields:

- `human_review_required`
- `review_triggers`
- `governance_decision`
- `governance_metadata`

For deeper policy details, see `docs/GOVERNANCE.md` and
`docs/POLICY_CONSTITUTIONAL_GOVERNANCE.md`.
