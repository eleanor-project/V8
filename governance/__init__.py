"""ELEANOR Governance Package.

Patch Pack #3 adds:
  - governor.py: one-call governance entry point
  - audit.py: append-only governance event record builder
  - cli.py: dry-run CLI

Sanctity rule:
  Reviewers author precedent for future cases; they do not override runtime outcomes
  and do not directly interface with the critic ensemble.
"""
