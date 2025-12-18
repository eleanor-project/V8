# rego.v1
package eleanor.execution_test

import data.eleanor.execution

# Minimal “good” Tier 2 example
test_tier2_allows_with_valid_ack if {
  test_input := {
    "action": "execute",
    "decision": {
      "executable": true,
      "execution_reason": "ok",
      "audit_record_id": "abc",
      "aggregation_result": {
        "execution_gate": { "gated": true, "escalation_tier": "TIER_2" },
        "escalation_summary": {
          "triggering_signals": [
            { "critic_id": "privacy_identity", "clause_id": "P4" }
          ]
        }
      },
      "human_action": {
        "action_type": "HUMAN_ACK",
        "statement": "I acknowledge the identified constitutional risks and accept responsibility for proceeding.",
        "linked_escalations": [
          { "critic_id": "privacy_identity", "clause_id": "P4" }
        ]
      }
    }
  }

  execution.allow with input as test_input
}

# Tier 3 must reject when determination is missing
test_tier3_denies_without_determination if {
  test_input := {
    "action": "execute",
    "decision": {
      "executable": true,
      "execution_reason": "ok",
      "audit_record_id": "abc",
      "aggregation_result": {
        "execution_gate": { "gated": true, "escalation_tier": "TIER_3" },
        "escalation_summary": {
          "triggering_signals": [
            { "critic_id": "privacy_identity", "clause_id": "P1" }
          ]
        }
      },
      "human_action": {
        "action_type": "HUMAN_ACK",
        "statement": "I acknowledge the identified constitutional risks and accept responsibility for proceeding.",
        "linked_escalations": [
          { "critic_id": "privacy_identity", "clause_id": "P1" }
        ]
      }
    }
  }

  not execution.allow with input as test_input
  some d
  execution.deny[d] with input as test_input
}
