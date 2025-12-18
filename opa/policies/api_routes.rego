# rego.v1
package eleanor.api

# Rego v1 syntax

#
# Route-level enforcement to prevent bypassing constitutional execution flow.
# Designed for gateway/ext_authz style inputs.
#
# Expected input:
# {
#   "http": { "path": "...", "method": "POST" | ... },
#   "body": { ...parsed JSON... }
# }
#

default allow := false

allow if {
  count(deny) == 0
}

# ---------------------------------------------------------------------------
# Execution route hardening
# ---------------------------------------------------------------------------

deny contains "/decision/execute must be POST" if {
  input.http.path == "/decision/execute"
  input.http.method != "POST"
}

deny contains "/decision/execute requires decision payload" if {
  input.http.path == "/decision/execute"
  not input.body.decision
}

# Block raw execution paths (defense-in-depth)
deny contains "raw execution endpoints are prohibited" if {
  endswith(input.http.path, "/execute_raw")
}

deny contains "direct /execute is prohibited; use /decision/execute" if {
  input.http.path == "/execute"
}
