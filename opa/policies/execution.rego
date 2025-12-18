# rego.v1
package eleanor.execution

# Rego v1 syntax

#
# Execution enforcement: mirrors runtime governance invariants.
# - Only ExecutableDecision objects may execute
# - Tier 2 requires HUMAN_ACK with the canonical acknowledgment statement
# - Tier 3 requires HUMAN_DETERMINATION with the canonical determination statement
# - Human action must reference all triggering escalation signals
# - No bypass/override flags permitted
#

default allow := false

allow if {
  input.action == "execute"
  count(deny) == 0
}

# ---------------------------------------------------------------------------
# Deny reasons (auditor friendly)
# ---------------------------------------------------------------------------

deny contains "unsupported action" if {
  not input.action
}

deny contains "unsupported action" if {
  input.action != "execute"
}

deny contains "missing decision payload" if {
  not input.decision
}

decision := input.decision

deny contains "decision.executable must be true" if {
  decision
  decision.executable != true
}

deny contains "missing audit_record_id" if {
  decision
  not decision.audit_record_id
}

deny contains "missing execution_reason" if {
  decision
  not decision.execution_reason
}

deny contains "override flags are prohibited" if {
  decision
  decision.override == true
}

deny contains "bypass flags are prohibited" if {
  decision
  decision.bypass_governance == true
}

# ---------------------------------------------------------------------------
# Gate enforcement
# ---------------------------------------------------------------------------

gate := decision.aggregation_result.execution_gate
escalation_tier := gate.escalation_tier

gated if {
  decision
  gate.gated == true
}

deny contains "execution gated but missing human_action" if {
  gated
  not decision.human_action
}

# Tier 2 requirements
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

# Tier 3 requirements
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

# Linkage: human action must reference all triggering escalation signals.
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

# ---------------------------------------------------------------------------
# Canonical statements (must match runtime enforcement)
# ---------------------------------------------------------------------------
canonical_tier2_statement := "I acknowledge the identified constitutional risks and accept responsibility for proceeding."
canonical_tier3_statement := "I affirmatively determine the appropriate course of action in light of the identified constitutional risks."
