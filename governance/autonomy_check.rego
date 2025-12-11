package eleanor.autonomy
default allow = false

unauthorized_data_use {
  some d
  input.data_usage[d].consent == false
  input.data_usage[d].used == true
}

manipulative_framing {
  input.manipulation_score > 0.25
}

allow {
  not unauthorized_data_use
  not manipulative_framing
}
