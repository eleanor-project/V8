package eleanor.truth
default allow = false

fabrications_detected {
  input.truth_metrics.fabrication_count > 0
}

missing_uncertainty_disclosure {
  input.uncertainty_score > 0.30
  not input.outputs.includes_uncertainty_statement
}

allow {
  not fabrications_detected
  not missing_uncertainty_disclosure
}
