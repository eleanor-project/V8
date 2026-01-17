# Release Process

This repository uses `pyproject.toml` for versioning and release metadata.

## Steps

1. Ensure the main branch is green (tests and benchmarks if required).
2. Update `pyproject.toml` version.
3. Update `RELEASE_NOTES.md` with a summary of changes.
4. Regenerate any docs or artifacts tied to the release.
5. Tag the release in git and push tags.

## Suggested verification

```bash
pytest
pytest tests/benchmarks --benchmark-only
```

For deployment guidance, see `docs/guides/deployment.md`.
