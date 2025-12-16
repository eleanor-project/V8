package eleanor.dignity
default allow := false
protected_classes := {"race", "gender", "religion", "disability", "ethnicity", "sexual_orientation"}

violation_detected if {
  some c
  input.critic_outputs[c].value == "dignity_non_discrimination"
  input.critic_outputs[c].violation == true
}

protected_class_used if {
  some f
  input.features_used[_] == f
  protected_classes[f]
}

allow if {
  not violation_detected
  not protected_class_used
}
