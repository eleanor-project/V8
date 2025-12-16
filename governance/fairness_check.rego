package eleanor.fairness
default allow := false

threshold := 0.10

disparity_violation if {
  input.fairness_metrics.subgroup_disparity > threshold
}

allow if {
  not disparity_violation
}
