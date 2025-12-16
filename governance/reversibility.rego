package eleanor.reversibility
default allow := false

reversibility_preferred if {
  input.options[_].reversible == true
}

irreversible_requires_precedent if {
  input.selected_option.reversible == false
  input.precedent_alignment_score < 0.85
}

allow if {
  reversibility_preferred
  not irreversible_requires_precedent
}
