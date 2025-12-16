package eleanor.pragmatics
default allow := false

cost_violation if {
  input.pragmatic_metrics.cost > input.config.cost_limit
}

latency_violation if {
  input.pragmatic_metrics.latency > input.config.latency_limit
}

allow if {
  not cost_violation
  not latency_violation
}
