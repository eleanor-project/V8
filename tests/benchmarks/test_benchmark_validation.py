"""Benchmarks for input validation."""

import pytest

from engine.validation import validate_input


@pytest.mark.performance
def test_validate_input_baseline(benchmark):
    text = "Sample input text " + " ".join(f"token{i}" for i in range(200))
    context = {"domain": "general", "metadata": {"user_id": "bench"}}

    result = benchmark(validate_input, text, context)

    assert result is not None
