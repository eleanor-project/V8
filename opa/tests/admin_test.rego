package eleanor.admin_test

import data.eleanor.admin

test_admin_allows_with_role if {
  test_input := {
    "http": {"path": "/admin/settings", "method": "POST"},
    "user": {"roles": ["admin"]}
  }
  admin.allow with input as test_input
}

test_admin_denies_without_role if {
  test_input := {
    "http": {"path": "/admin/settings", "method": "POST"},
    "user": {"roles": ["viewer"]}
  }
  not admin.allow with input as test_input
  some d
  admin.deny[d] with input as test_input
}
