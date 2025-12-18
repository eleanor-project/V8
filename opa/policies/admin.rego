package eleanor.admin

# rego.v1
#
# Guard admin/governance endpoints to prevent unauthorized changes.
# Expected input:
# {
#   "http": { "path": "...", "method": "..." },
#   "user": { "roles": ["..."] }  # optional
# }
#

default allow := false

allow if {
  count(deny) == 0
}

deny contains "unsupported method" if {
  not allowed_methods[input.http.method]
}

allowed_methods := {"GET", "POST", "PUT", "PATCH", "DELETE"}

# Paths to guard (defaults)
guarded_paths := {
  "/admin",
}

deny contains "path not permitted for admin policy" if {
  not path_guarded
}

path_guarded if {
  some gp_idx
  gp := guarded_paths[gp_idx]
  startswith(input.http.path, gp)
}

path_guarded if {
  startswith(input.http.path, "/governance/")
}

# Roles (adjust as needed)
required_roles := ["admin", "governance-steward"]
roles_provided := { rp | some i; rp := input.user.roles[i] }

deny contains "insufficient role for admin operation" if {
  count(required_roles) > 0
  not has_required_role(required_roles, roles_provided)
}

has_required_role(required, provided) if {
  some role_idx
  rr := required[role_idx]
  provided[rr]
}
