package eleanor.decision

# Minimal OPA decision policy for local/dev use.
# In production, replace with a richer combination of constitutional
# gates and escalation logic.

default allow := true
default escalate := false
failures := []
