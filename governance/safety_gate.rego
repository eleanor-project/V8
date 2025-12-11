package eleanor.safety
default allow = false

irreversible_threshold := 0.20

irreversible_risk {
  input.risk_score.irreversible > irreversible_threshold
}

uncertainty_violation {
  input.uncertainty_score > input.config.safety_uncertainty_limit
}

allow {
  not irreversible_risk
  not uncertainty_violation
}
