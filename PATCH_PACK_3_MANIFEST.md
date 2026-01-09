# Patch Pack #3 — Governor + Audit + CLI

Adds a single-call governance entry, append-only event emitter, and a dry-run CLI.

## Files
- governance/__init__.py
- governance/governor.py
- governance/audit.py
- governance/cli.py
- governance/examples/precedent_candidates.json
- docs/CLI_DRY_RUN.md

## Notes
- Preserves the sanctity rule (reviewers author future precedent only).
- The CLI uses candidate JSON inputs for demos; production uses ledger/search.
