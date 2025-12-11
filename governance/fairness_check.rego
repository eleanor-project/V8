package eleanor.fairness
default allow = false

threshold := 0.10

disparity_violation {
  input.fairness_metrics.subgroup_disparity > threshold
}

allow {
  not disparity_violation
}
