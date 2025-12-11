package eleanor.pragmatics
default allow = false

cost_violation {
  input.pragmatic_metrics.cost > input.config.cost_limit
}

latency_violation {
  input.pragmatic_metrics.latency > input.config.latency_limit
}

allow {
  not cost_violation
  not latency_violation
}
