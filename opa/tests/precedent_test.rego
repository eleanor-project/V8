package eleanor.precedent_test

import data.eleanor.precedent

test_allow_propose_without_roles if {
  test_input := {
    "http": {"path": "/precedent/propose", "method": "POST"},
    "body": {"case_id": "x"}
  }
  precedent.allow with input as test_input
}

test_deny_invalid_path if {
  test_input := {
    "http": {"path": "/precedent/unknown", "method": "POST"},
    "body": {}
  }
  not precedent.allow with input as test_input
  some d
  precedent.deny[d] with input as test_input
}
