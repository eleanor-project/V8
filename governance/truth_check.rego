package eleanor.truth
default allow := false

fabrications_detected if {
  input.truth_metrics.fabrication_count > 0
}

missing_uncertainty_disclosure if {
  input.uncertainty_score > 0.30
  not input.outputs.includes_uncertainty_statement
}

allow if {
  not fabrications_detected
  not missing_uncertainty_disclosure
}
