package eleanor.executor_test

import data.eleanor.executor

# Happy path: Tier 2 with valid ack
test_executor_allows_tier2 if {
  test_input := {
    "http": {"path": "/execute", "method": "POST"},
    "body": {
      "decision": {
        "executable": true,
        "execution_reason": "ok",
        "audit_record_id": "abc",
        "aggregation_result": {
          "execution_gate": {"gated": true, "escalation_tier": "TIER_2"},
          "escalation_summary": {
            "triggering_signals": [
              {"critic_id": "privacy_identity", "clause_id": "P4"}
            ]
          }
        },
        "human_action": {
          "action_type": "HUMAN_ACK",
          "statement": "I acknowledge the identified constitutional risks and accept responsibility for proceeding.",
          "linked_escalations": [
            {"critic_id": "privacy_identity", "clause_id": "P4"}
          ]
        }
      }
    }
  }

  executor.allow with input as test_input
}

# Tier 3 must deny without determination
test_executor_denies_missing_tier3_determination if {
  test_input := {
    "http": {"path": "/execute", "method": "POST"},
    "body": {
      "decision": {
        "executable": true,
        "execution_reason": "ok",
        "audit_record_id": "abc",
        "aggregation_result": {
          "execution_gate": {"gated": true, "escalation_tier": "TIER_3"},
          "escalation_summary": {
            "triggering_signals": [
              {"critic_id": "privacy_identity", "clause_id": "P1"}
            ]
          }
        },
        "human_action": {
          "action_type": "HUMAN_ACK",
          "statement": "I acknowledge the identified constitutional risks and accept responsibility for proceeding.",
          "linked_escalations": [
            {"critic_id": "privacy_identity", "clause_id": "P1"}
          ]
        }
      }
    }
  }

  not executor.allow with input as test_input
  some d
  executor.deny[d] with input as test_input
}
