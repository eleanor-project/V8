package eleanor.safety
default allow := false

irreversible_threshold := 0.20

irreversible_risk if {
  input.risk_score.irreversible > irreversible_threshold
}

uncertainty_violation if {
  input.uncertainty_score > input.config.safety_uncertainty_limit
}

allow if {
  not irreversible_risk
  not uncertainty_violation
}
