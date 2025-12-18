# rego.v1
package eleanor.critics_test

import data.eleanor.critics

test_denies_escalation_without_clause_id if {
  test_input := {
    "http": { "path": "/decision/evaluate", "method": "POST" },
    "body": {
      "synthesis": "x",
      "critic_evaluations": [
        {
          "critic_id": "privacy_identity",
          "charter_version": "v1",
          "severity_score": 0.6,
          "concerns": [],
          "escalation": {
            "tier": "TIER_2",
            "critic_id": "privacy_identity",
            "clause_id": ""
          }
        }
      ]
    }
  }

  not critics.allow with input as test_input
  some d
  critics.deny[d] with input as test_input
}
