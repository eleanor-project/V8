package eleanor.decision_test

import rego.v1

# Basic sanity: uncertainty triggers escalation with a reason

test_escalate_on_uncertainty_needs_escalation if {
    input := {
        "aggregator": {"decision": "allow", "rights_impacted": []},
        "critics": {"risk": {"severity": 0.2}},
        "uncertainty": {"needs_escalation": true, "overall_uncertainty": 0.2},
        "precedent": {"novel": false, "alignment_score": 1.0},
        "context": {"domain": ""}
    }

    data.eleanor.decision.allow with input as input
    data.eleanor.decision.escalate with input as input

    some f
    f := data.eleanor.decision.failures[_] with input as input
    f.code == "UNCERTAINTY_ESCALATE"
}

# Engine hard-deny must deny and include reason

test_deny_on_engine_deny if {
    input := {
        "aggregator": {"decision": "deny", "rights_impacted": []},
        "critics": {"risk": {"severity": 0.1}},
        "uncertainty": {"needs_escalation": false, "overall_uncertainty": 0.1},
        "precedent": {"novel": false, "alignment_score": 1.0},
        "context": {"domain": ""}
    }

    not data.eleanor.decision.allow with input as input
    not data.eleanor.decision.escalate with input as input

    some f
    f := data.eleanor.decision.failures[_] with input as input
    f.code == "ENGINE_DENY"
}
