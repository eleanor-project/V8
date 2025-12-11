package eleanor.dignity
default allow = false
protected_classes := {"race", "gender", "religion", "disability", "ethnicity", "sexual_orientation"}

violation_detected {
  some c
  input.critic_outputs[c].value == "dignity_non_discrimination"
  input.critic_outputs[c].violation == true
}

protected_class_used {
  some field
  field := input.features_used[_]
  protected_classes[field]
}

allow {
  not violation_detected
  not protected_class_used
}
