# Repository Guidelines

## Project Structure & Module Organization
- Core runtime in `engine/` (critics, precedent engine, cache/observability/resilience); governance policies in `governance/`; API/event wiring in `api/` and `engine/events/`.
- Config defaults in `config/`; env overrides in `docker/` and `docker-compose*.yaml`.
- Tests live in `tests/`; docs and roadmap artifacts in `docs/`, `PRODUCTION_ROADMAP.md`, `SECURITY.md`, `PRODUCTION_READINESS_REVIEW*.md`.
- UI/demo assets in `ui/` and `Demo/`; scripts and utilities in `scripts/`, `start*.sh`, and `validate_eleanor_repo.py`.

## Build, Test, and Development Commands
- Python 3.10+; install deps: `pip install -r requirements.txt` (opt: `requirements-observability.txt`, `requirements-gpu.txt`, `requirements-security.txt`).
- Tests: `pytest`; coverage: `pytest --cov=engine --cov-report=html`.
- Quality: `ruff check .`, `ruff format .`, `mypy engine/ --strict`, `bandit -r engine/`.
- Local stack (optional): `docker-compose up`; app scripts: `./start.sh`, `./start_ui.sh`, `./start_eleanor.sh`.

## Coding Style & Naming Conventions
- PEP 8, 4-space indentation, type hints expected; keep units small and document non-obvious behavior.
- Naming: `snake_case` functions/vars, `PascalCase` classes, `UPPER_SNAKE` constants; prefer explicit imports.
- Use structured logging in `engine.observability`; keep audit paths intact (critics/auditors stay enabled).
- Config via YAML/JSON; override with env vars/compose, never hard-code secrets.

## Testing Guidelines
- Place new tests in `tests/` near touched modules; name `test_*` files/functions.
- Cover governance/routing/cache happy and failure paths; assert audit traces when relevant.
- Use fixtures/mocks; avoid network in unit tests; regenerate snapshots only for intentional changes.
- Ensure `pytest`, `ruff`, `mypy --strict`, and `bandit` pass; add coverage when altering routing/precedent logic.

## Commit & Pull Request Guidelines
- Commit subjects are short and imperative (e.g., “Update model registry defaults”); keep scope focused.
- PRs link issues/roadmap items, note impact, list tests run, and include UI screenshots when relevant; update docs when changing governance, routing, or security.
- Do not commit secrets or local artifacts; add transient files to `.gitignore`.

## Security & Configuration Tips
- See `SECURITY.md` and `requirements-secrets.txt`; use UTC-aware timestamps and preserve audit logging and critic escalation.
- Keep secrets in env vars or vault-backed mounts; validate inputs before model routing and sanitize outputs in `engine/security/`.
