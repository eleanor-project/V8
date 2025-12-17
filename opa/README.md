# OPA Constitutional Governance Policies

These policies enforce Eleanor's constitutional governance at deployment/runtime, independent of application code.

## What is enforced
- Execution only with an `ExecutableDecision` that is marked executable and carries an audit_record_id.
- Tier 2 escalations require the canonical human acknowledgment statement and linkage to all triggering escalation signals.
- Tier 3 escalations require the canonical human determination statement and linkage to all triggering escalation signals.
- If no escalation is present, execution may proceed.

## Files
- `policies/constitutional.rego` — main policy package `eleanor.governance`.

## How to use
Feed an input shaped like:
```json
{
  "action": "execute",
  "decision": {
    "executable": true,
    "execution_reason": "Required human review satisfied. Execution permitted.",
    "audit_record_id": "uuid",
    "aggregation_result": {
      "execution_gate": {
        "gated": true,
        "escalation_tier": "TIER_3"
      },
      "escalation_summary": {
        "triggering_signals": [
          {"critic_id": "privacy", "clause_id": "P1"}
        ]
      }
    },
    "human_action": {
      "action_type": "HUMAN_DETERMINATION",
      "statement": "I affirmatively determine the appropriate course of action in light of the identified constitutional risks.",
      "linked_escalations": [
        {"critic_id": "privacy", "clause_id": "P1"}
      ]
    }
  }
}
```

If policy denies, return a clear error such as:
`CONSTITUTIONAL_POLICY_VIOLATION: execution denied because required human determination for Tier 3 escalation is missing or invalid.`

## Suggested integration points
- Kubernetes admission via Gatekeeper to block workloads that bypass governance.
- API gateway or service mesh (Envoy + OPA) to guard `/decision/execute` calls.
- Workflow/job runners invoking OPA before side effects.
