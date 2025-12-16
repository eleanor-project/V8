package eleanor.constitution

import data.eleanor.dignity
import data.eleanor.autonomy
import data.eleanor.truth
import data.eleanor.fairness
import data.eleanor.safety
import data.eleanor.reversibility
import data.eleanor.pragmatics

default allow := false

allow if {
  dignity.allow
  autonomy.allow
  truth.allow
  fairness.allow
  safety.allow
  reversibility.allow
  pragmatics.allow
}
