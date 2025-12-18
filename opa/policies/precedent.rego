package eleanor.precedent

# rego.v1
#
# Guard precedent writeback operations (norm creation/drift control).
# Expected input:
# {
#   "http": { "path": "...", "method": "POST" },
#   "body": { ...payload... },
#   "user": { "roles": ["..."] }   # optional, for role-based approval
# }
#
# Endpoints covered (defaults):
# - /precedent/propose
# - /precedent/approve
# - /precedent/write
# - /precedent/deprecate

default allow := false

allow if {
  count(deny) == 0
}

# ---------------------------------------------------------------------------
# Path + method enforcement
# ---------------------------------------------------------------------------
deny contains "unsupported method" if {
  input.http.method != "POST"
}

valid_paths := {
  "/precedent/propose",
  "/precedent/approve",
  "/precedent/write",
  "/precedent/deprecate",
}

deny contains "unsupported path" if {
  not valid_paths[input.http.path]
}

# ---------------------------------------------------------------------------
# Role/approval (minimal stub — adjust roles as needed)
# ---------------------------------------------------------------------------
# If you have roles, enforce them. If not provided, we still allow (prototype).
required_roles := {
  "/precedent/propose": [],
  "/precedent/approve": ["governance-steward", "admin"],
  "/precedent/write": ["governance-steward", "admin"],
  "/precedent/deprecate": ["governance-steward", "admin"],
}

roles_provided := { rp | some i; rp := input.user.roles[i] }

deny contains "insufficient role for precedent operation" if {
  required := required_roles[input.http.path]
  required != null
  count(required) > 0
  not has_required_role(required, roles_provided)
}

has_required_role(required, provided) if {
  some role_idx
  rr := required[role_idx]
  provided[rr]
}

# ---------------------------------------------------------------------------
# Basic payload presence (extend as needed per endpoint)
# ---------------------------------------------------------------------------
deny contains "missing body payload" if {
  not input.body
}
