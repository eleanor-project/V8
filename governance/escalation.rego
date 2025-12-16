package eleanor.escalation
default escalate := false

escalate if {
  input.uncertainty_score > input.thresholds.uncertainty_global
}

escalate if {
  input.critic_dissent > input.thresholds.dissent_global
}

escalate if {
  input.dignity_violation_detected == true
}

escalate if {
  not input.precedent_conflict_resolved
}
