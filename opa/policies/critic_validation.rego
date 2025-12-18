# rego.v1
package eleanor.critics

# Rego v1 syntax

#
# Validate critic outputs at the API boundary to ensure clause-aware escalation.
# Intended for /decision/evaluate (or similar) requests.
#
# Expected input:
# {
#   "http": { "path": "/decision/evaluate", "method": "POST" },
#   "body": {
#     "critic_evaluations": [ ... ],
#     "synthesis": "..."
#   }
# }
#

default allow := false

allow if {
  not is_evaluate
}

allow if {
  is_evaluate
  count(deny) == 0
}

is_evaluate if {
  input.http.path == "/decision/evaluate"
  input.http.method == "POST"
}

# Basic shape
deny contains "critic_evaluations must be provided" if {
  is_evaluate
  not input.body.critic_evaluations
}

# Each critic must have critic_id + charter_version
deny contains msg if {
  is_evaluate
  some i
  ev := input.body.critic_evaluations[i]
  ev.critic_id == ""
  msg := sprintf("critic_evaluations[%d].critic_id missing", [i])
}

deny contains msg if {
  is_evaluate
  some i
  ev := input.body.critic_evaluations[i]
  ev.charter_version == ""
  msg := sprintf("critic_evaluations[%d].charter_version missing", [i])
}

# If escalation exists, it MUST be clause-aware and tier-valid
deny contains msg if {
  is_evaluate
  some i
  ev := input.body.critic_evaluations[i]
  esc := ev.escalation
  esc
  esc.clause_id == ""
  msg := sprintf("critic_evaluations[%d].escalation.clause_id missing", [i])
}

deny contains msg if {
  is_evaluate
  some i
  ev := input.body.critic_evaluations[i]
  esc := ev.escalation
  esc
  not valid_tier(esc.tier)
  msg := sprintf("critic_evaluations[%d].escalation.tier invalid", [i])
}

# Escalation critic_id must match evaluation critic_id (prevents spoofing)
deny contains msg if {
  is_evaluate
  some i
  ev := input.body.critic_evaluations[i]
  esc := ev.escalation
  esc
  esc.critic_id != ev.critic_id
  msg := sprintf("critic_evaluations[%d] escalation critic_id mismatch", [i])
}

valid_tier(t) if {
  t == "TIER_2"
}

valid_tier(t) if {
  t == "TIER_3"
}
