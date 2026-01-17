# Contributing

Thank you for improving ELEANOR V8. This guide covers the expected workflow and
quality bar for changes.

## Workflow

1. Create or pick an issue and clarify expected behavior.
2. Create a branch off `main`.
3. Implement changes with tests where appropriate.
4. Update documentation when behavior or APIs change.
5. Open a pull request with a summary and verification steps.

## Quality checklist

- Tests pass: `pytest` (see `docs/guides/testing.md`).
- Lint and format: `black` and `ruff`.
- Type checks: `mypy` or `pyright` where applicable.
- Docs updated: `docs/` and any relevant top-level docs.

## Where to look

- Runtime: `engine/`
- Governance: `governance/`
- API: `api/`
- Examples: `examples/`
- Tests: `tests/`

For broader setup guidance, see `docs/development/development-setup.md` and
`DEVELOPMENT.md` in the repo root.
