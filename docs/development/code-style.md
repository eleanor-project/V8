# Code Style

ELEANOR V8 follows standard Python tooling with repository configuration stored
in `pyproject.toml`.

## Formatting

- Use `black` for formatting.
- Line length is 100.

## Linting

- Use `ruff` for linting.
- Keep fixes consistent with existing patterns in `engine/`.

## Typing

- Type hints are expected for public APIs and core runtime paths.
- Use `mypy` or `pyright` to validate type checks.

## Style tips

- Keep functions focused and small.
- Prefer explicit naming for clarity in governance logic.
- Document non-obvious logic with concise comments.
