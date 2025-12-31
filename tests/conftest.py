"""Pytest configuration and shared fixtures."""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock


@pytest.fixture
def mock_llm_adapter():
    """Mock LLM adapter for testing."""
    adapter = AsyncMock()
    adapter.generate = AsyncMock(return_value="Mock LLM response")
    return adapter


@pytest.fixture
def sample_context():
    """Sample context for testing."""
    return {
        "domain": "test",
        "user_id": "test_user",
        "priority": "medium",
    }


@pytest.fixture
def sample_critic_result():
    """Sample critic evaluation result."""
    return {
        "critic": "rights",
        "score": 0.3,
        "severity": "LOW",
        "violations": [
            {
                "rule_id": "privacy_001",
                "description": "Potential privacy concern",
                "confidence": 0.7,
            }
        ],
        "justification": "Test justification",
        "evaluated_rules": ["privacy_001", "privacy_002"],
    }


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
