# Development Setup

This guide sets up a local environment for ELEANOR V8 development.

## Requirements

- Python 3.10+
- pip or a virtual environment manager

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Optional extras:

```bash
pip install -e .[api,observability,cache]
```

## Validation

```bash
pytest -q
```

For advanced configuration and GPU setup, see:

- `docs/guides/configuration.md`
- `docs/GPU_QUICK_START.md`
- `docs/GPU_ACCELERATION.md`
