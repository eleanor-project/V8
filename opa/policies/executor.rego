# rego.v1
package eleanor.executor

# Guards the executor service (side effects).
# Expects input:
# {
#   "http": { "path": "/execute", "method": "POST" },
#   "body": { "decision": ExecutableDecision }
# }

default allow := false

# allow when no denies
allow if {
  count(deny) == 0
}

# deny reasons (auditable)
deny contains "unsupported path" if {
  input.http.path != "/execute"
}

deny contains "unsupported method" if {
  input.http.method != "POST"
}

deny contains "missing decision payload" if {
  not input.body.decision
}

decision := input.body.decision

deny contains "decision.executable must be true" if {
  decision.executable != true
}

deny contains "missing audit_record_id" if {
  not decision.audit_record_id
}

deny contains "missing execution_reason" if {
  not decision.execution_reason
}

# Gate enforcement
gate := decision.aggregation_result.execution_gate
escalation_tier := gate.escalation_tier

gated if {
  gate.gated == true
}

deny contains "execution gated but missing human_action" if {
  gated
  not decision.human_action
}

# Tier 2 checks
deny contains "Tier 2 requires HUMAN_ACK" if {
  gated
  escalation_tier == "TIER_2"
  decision.human_action.action_type != "HUMAN_ACK"
}

deny contains "Tier 2 requires canonical acknowledgment statement" if {
  gated
  escalation_tier == "TIER_2"
  decision.human_action.statement != canonical_tier2_statement
}

# Tier 3 checks
deny contains "Tier 3 requires HUMAN_DETERMINATION" if {
  gated
  escalation_tier == "TIER_3"
  decision.human_action.action_type != "HUMAN_DETERMINATION"
}

deny contains "Tier 3 requires canonical determination statement" if {
  gated
  escalation_tier == "TIER_3"
  decision.human_action.statement != canonical_tier3_statement
}

# Linkage: human_action must reference all triggering escalation signals
deny contains "human_action.linked_escalations missing required clause references" if {
  gated
  required := triggering_clause_keys
  provided := linked_clause_keys
  not subset(required, provided)
}

triggering_clause_keys := { key |
  decision
  some i
  sig := decision.aggregation_result.escalation_summary.triggering_signals[i]
  key := sprintf("%s:%s", [sig.critic_id, sig.clause_id])
}

linked_clause_keys := { key |
  decision
  decision.human_action
  some i
  esc := decision.human_action.linked_escalations[i]
  key := sprintf("%s:%s", [esc.critic_id, esc.clause_id])
}

subset(a, b) if {
  not exists_missing(a, b)
}

exists_missing(a, b) if {
  some x
  x = a[_]
  not b[x]
}

canonical_tier2_statement := "I acknowledge the identified constitutional risks and accept responsibility for proceeding."
canonical_tier3_statement := "I affirmatively determine the appropriate course of action in light of the identified constitutional risks."
