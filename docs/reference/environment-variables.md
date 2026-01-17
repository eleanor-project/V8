# Environment Variables

Environment variables are the primary override mechanism for runtime settings.
Most configuration values accept `ELEANOR_` prefixed environment variables.

## Key behavior

- Prefix: `ELEANOR_`
- Nested fields use `__` (double underscore) as a delimiter.
- `.env` files are loaded by default (see `engine/config/settings.py`).

## Common variables

- `ELEANOR_ENVIRONMENT` - deployment environment
- `ELEANOR_CONFIG_PATH` - path to YAML config
- `ELEANOR_CONSTITUTIONAL_CONFIG_PATH` - constitutional YAML path
- `ELEANOR_REPLAY_LOG_PATH` - replay log path

## Reference lists

- `ENV_VARS_REFERENCE.md` provides orchestrator-specific variables.
- `engine/config/settings.py` is the authoritative source.
