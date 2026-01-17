# Testing

Run the full suite:

```bash
pytest
```

Coverage and quality tools:

```bash
pytest --cov=engine --cov-report=term-missing
ruff check .
ruff format .
```

Mocks live in `engine/mocks.py` and are used by many tests.
