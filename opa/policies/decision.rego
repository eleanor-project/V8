package eleanor.decision

import rego.v1

# Decision policy for *outputs* (not side-effect execution).
#
# Query path (used by the API and OPAClientV8):
#   POST /v1/data/eleanor/decision  {"input": {...}}
#
# Output shape (expected by api.rest.main:resolve_final_decision):
#   {"allow": bool, "escalate": bool, "failures": [ ... ]}

default allow := true
default escalate := false
default failures := []

# ---- Inputs (normalize across endpoints) ------------------------------------

critics := input.aggregator.critics if {
    input.aggregator.critics
} else := input.critics

aggregator := input.aggregator if {
    input.aggregator
} else := {}

uncertainty := input.uncertainty if {
    input.uncertainty
} else := {}

precedent := input.precedent if {
    input.precedent
} else := {}

context := input.context if {
    input.context
} else := {}

# ---- Failures and reasons ---------------------------------------------------

# Collect deny + escalate reasons into a single list.
failures := array.concat(deny_reasons, escalate_reasons)

deny_reasons := [r | deny_reason[r]]
escalate_reasons := [r | escalate_reason[r]]

# ---- Allow / Escalate -------------------------------------------------------

# Hard deny if and only if we have at least one deny reason.
allow := count(deny_reasons) == 0

# Escalate requires at least one explicit reason.
escalate if {
    allow
    count(escalate_reasons) > 0
}

# ---- Deny conditions (hard gate) -------------------------------------------

deny_reason[{"code": "ENGINE_DENY", "severity": "critical", "message": msg}] if {
    aggregator.decision == "deny"
    msg := sprintf("Engine decision=%v", [aggregator.decision])
}

deny_reason[{"code": "CRITIC_CRITICAL", "severity": "critical", "message": msg}] if {
    max_critic_severity >= 0.90
    msg := sprintf("Max critic severity %.2f >= 0.90", [max_critic_severity])
}

deny_reason[{"code": "PRIVACY_CRITICAL", "severity": "critical", "message": msg}] if {
    # If any critic explicitly flags a privacy breach with high confidence, hard deny.
    some k
    c := critics[k]
    flags := object.get(c, "flags", [])
    flags[_] == "privacy_violation"
    sev := critic_severity(c)
    sev >= 0.75
    msg := sprintf("Privacy violation flagged by %v (severity %.2f)", [k, sev])
}

# ---- Escalation conditions (soft gate) -------------------------------------

escalate_reason[{"code": "UNCERTAINTY_ESCALATE", "severity": "high", "message": msg}] if {
    object.get(uncertainty, "needs_escalation", false)
    msg := "Uncertainty module requested escalation"
}

escalate_reason[{"code": "UNCERTAINTY_HIGH", "severity": "high", "message": msg}] if {
    u := to_number(object.get(uncertainty, "overall_uncertainty", 0))
    u >= 0.65
    msg := sprintf("Overall uncertainty %.2f >= 0.65", [u])
}

escalate_reason[{"code": "PRECEDENT_NOVEL", "severity": "medium", "message": msg}] if {
    object.get(precedent, "novel", false)
    msg := "Precedent engine marked the case as novel"
}

escalate_reason[{"code": "PRECEDENT_WEAK_ALIGNMENT", "severity": "medium", "message": msg}] if {
    s := to_number(object.get(precedent, "alignment_score", 1))
    s < 0.50
    msg := sprintf("Precedent alignment score %.2f < 0.50", [s])
}

escalate_reason[{"code": "HIGH_STAKES_DOMAIN", "severity": "medium", "message": msg}] if {
    d := lower(object.get(context, "domain", ""))
    high_stakes_domains[d]
    msg := sprintf("High-stakes domain=%v", [d])
}

escalate_reason[{"code": "RIGHTS_IMPACT", "severity": "medium", "message": msg}] if {
    rights := object.get(aggregator, "rights_impacted", [])
    count(rights) > 0
    msg := sprintf("Rights impacted: %v", [rights])
}

escalate_reason[{"code": "CRITIC_WARNING", "severity": "medium", "message": msg}] if {
    # Moderate critic severity + any rights impact => escalate.
    max_critic_severity >= 0.60
    rights := object.get(aggregator, "rights_impacted", [])
    count(rights) > 0
    msg := sprintf("Max critic severity %.2f with rights impact", [max_critic_severity])
}

# ---- Helpers ----------------------------------------------------------------

high_stakes_domains := {
    "medical": true,
    "healthcare": true,
    "legal": true,
    "finance": true,
    "banking": true,
    "insurance": true,
    "employment": true,
    "housing": true,
    "criminal_justice": true,
    "education": true,
}

critic_severity(c) := sev if {
    s := object.get(c, "severity", null)
    s != null
    sev := to_number(s)
} else := sev if {
    s := object.get(c, "severity_score", null)
    s != null
    sev := to_number(s)
} else := sev if {
    s := object.get(c, "score", null)
    s != null
    sev := to_number(s)
} else := 0.0

max_critic_severity := m if {
    vals := [critic_severity(critics[k]) | some k]
    count(vals) > 0
    m := max(vals)
} else := 0.0
