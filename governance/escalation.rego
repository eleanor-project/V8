package eleanor.escalation
default escalate = false

escalate {
  input.uncertainty_score > input.thresholds.uncertainty_global
}

escalate {
  input.critic_dissent > input.thresholds.dissent_global
}

escalate {
  input.dignity_violation_detected == true
}

escalate {
  not input.precedent_conflict_resolved
}
