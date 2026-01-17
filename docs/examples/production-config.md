# Production Configuration

ELEANOR V8 supports layered configuration with environment variables and YAML.

## YAML configuration

Use a base config file such as `config/eleanor.yaml`:

```bash
export ELEANOR_CONFIG_PATH=config/eleanor.yaml
```

## Environment overrides

Environment variables are prefixed with `ELEANOR_` and can override YAML
settings. See `engine/config/settings.py` and `ENV_VARS_REFERENCE.md` for the
full list.

## Related files

- `config/eleanor.yaml`
- `config/observability.yaml`
- `config/resource_management.yaml`
- `docs/guides/configuration.md`
