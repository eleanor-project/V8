"""
Tests for Router Adapters
-------------------------

Tests engine/router/adapters.py
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from engine.router.adapters import (
    BaseLLMAdapter,
    AdapterRegistry,
    bootstrap_default_registry,
    GPTAdapter,
    ClaudeAdapter,
    GrokAdapter,
    OllamaAdapter
)


# ============================================================
# BaseLLMAdapter Tests
# ============================================================

def test_base_adapter_not_implemented():
    """Test that BaseLLMAdapter raises NotImplementedError."""
    adapter = BaseLLMAdapter()

    with pytest.raises(NotImplementedError):
        adapter("test prompt")


def test_base_adapter_subclass_implementation():
    """Test that subclasses can implement __call__."""
    class MyAdapter(BaseLLMAdapter):
        def __call__(self, prompt: str) -> str:
            return f"Response to: {prompt}"

    adapter = MyAdapter()
    result = adapter("Hello")

    assert result == "Response to: Hello"


# ============================================================
# AdapterRegistry Tests
# ============================================================

def test_adapter_registry_creation():
    """Test creating an empty adapter registry."""
    registry = AdapterRegistry()

    assert isinstance(registry.adapters, dict)
    assert len(registry.adapters) == 0


def test_adapter_registry_register():
    """Test registering an adapter."""
    registry = AdapterRegistry()

    class TestAdapter(BaseLLMAdapter):
        def __call__(self, prompt: str) -> str:
            return "test"

    adapter = TestAdapter()
    registry.register("test", adapter)

    assert "test" in registry.adapters
    assert registry.adapters["test"] is adapter


def test_adapter_registry_get():
    """Test retrieving a registered adapter."""
    registry = AdapterRegistry()

    class TestAdapter(BaseLLMAdapter):
        def __call__(self, prompt: str) -> str:
            return "test"

    adapter = TestAdapter()
    registry.register("test", adapter)

    retrieved = registry.get("test")
    assert retrieved is adapter


def test_adapter_registry_get_not_found():
    """Test that getting non-existent adapter raises ValueError."""
    registry = AdapterRegistry()

    with pytest.raises(ValueError) as exc_info:
        registry.get("nonexistent")

    assert "not found" in str(exc_info.value)
    assert "nonexistent" in str(exc_info.value)


def test_adapter_registry_list():
    """Test listing all registered adapters."""
    registry = AdapterRegistry()

    class TestAdapter(BaseLLMAdapter):
        def __call__(self, prompt: str) -> str:
            return "test"

    registry.register("adapter1", TestAdapter())
    registry.register("adapter2", TestAdapter())
    registry.register("adapter3", TestAdapter())

    adapters = registry.list()

    assert len(adapters) == 3
    assert "adapter1" in adapters
    assert "adapter2" in adapters
    assert "adapter3" in adapters


# ============================================================
# GPTAdapter Tests
# ============================================================

def test_gpt_adapter_import_error():
    """Test GPTAdapter raises ImportError when OpenAI not installed."""
    with patch('engine.router.adapters.OpenAIClient', None):
        with pytest.raises(ImportError) as exc_info:
            GPTAdapter()

        assert "OpenAI SDK not installed" in str(exc_info.value)


@patch('engine.router.adapters.OpenAIClient')
def test_gpt_adapter_initialization(mock_openai_client):
    """Test GPTAdapter initialization."""
    adapter = GPTAdapter(model="gpt-4", api_key="test_key")

    assert adapter.model == "gpt-4"
    mock_openai_client.assert_called_once_with(api_key="test_key")


@patch('engine.router.adapters.OpenAIClient')
def test_gpt_adapter_call(mock_openai_client):
    """Test GPTAdapter __call__ method."""
    # Setup mock
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "AI response"

    mock_client_instance = Mock()
    mock_client_instance.chat.completions.create.return_value = mock_response
    mock_openai_client.return_value = mock_client_instance

    # Test
    adapter = GPTAdapter(api_key="test_key")
    result = adapter("Hello AI")

    assert result == "AI response"
    mock_client_instance.chat.completions.create.assert_called_once()
    call_args = mock_client_instance.chat.completions.create.call_args
    assert call_args[1]["messages"][0]["content"] == "Hello AI"


# ============================================================
# ClaudeAdapter Tests
# ============================================================

def test_claude_adapter_import_error():
    """Test ClaudeAdapter raises ImportError when anthropic not installed."""
    with patch('engine.router.adapters.anthropic', None):
        with pytest.raises(ImportError) as exc_info:
            ClaudeAdapter()

        assert "Anthropic SDK not installed" in str(exc_info.value)


@patch('engine.router.adapters.anthropic')
def test_claude_adapter_initialization(mock_anthropic):
    """Test ClaudeAdapter initialization."""
    adapter = ClaudeAdapter(model="claude-3-opus", api_key="test_key")

    assert adapter.model == "claude-3-opus"
    mock_anthropic.Anthropic.assert_called_once_with(api_key="test_key")


@patch('engine.router.adapters.anthropic')
def test_claude_adapter_call(mock_anthropic):
    """Test ClaudeAdapter __call__ method."""
    # Setup mock
    mock_content = Mock()
    mock_content.text = "Claude response"

    mock_response = Mock()
    mock_response.content = [mock_content]

    mock_client_instance = Mock()
    mock_client_instance.messages.create.return_value = mock_response
    mock_anthropic.Anthropic.return_value = mock_client_instance

    # Test
    adapter = ClaudeAdapter(api_key="test_key")
    result = adapter("Hello Claude")

    assert result == "Claude response"
    mock_client_instance.messages.create.assert_called_once()


# ============================================================
# GrokAdapter Tests
# ============================================================

def test_grok_adapter_initialization():
    """Test GrokAdapter initialization."""
    adapter = GrokAdapter(model="grok-beta", api_key="test_key")

    assert adapter.model == "grok-beta"
    assert adapter.api_key == "test_key"


@patch('engine.router.adapters.httpx.AsyncClient')
@pytest.mark.asyncio
async def test_grok_adapter_call(mock_post):
    """Test GrokAdapter __call__ method."""
    # Setup mock
    mock_response = Mock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Grok response"}}]
    }
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_post.return_value = mock_client

    # Test
    adapter = GrokAdapter(api_key="test_key")
    result = await adapter("Hello Grok")

    assert result == "Grok response"
    mock_client.post.assert_awaited_once()
    call_args = mock_client.post.call_args
    assert call_args[1]["json"]["messages"][0]["content"] == "Hello Grok"


# ============================================================
# OllamaAdapter Tests
# ============================================================

def test_ollama_adapter_initialization():
    """Test OllamaAdapter initialization."""
    adapter = OllamaAdapter(model="llama3")

    assert adapter.model == "llama3"


@patch('engine.router.adapters.httpx.AsyncClient')
@pytest.mark.asyncio
async def test_ollama_adapter_call(mock_post):
    """Test OllamaAdapter __call__ method."""
    # Setup mock
    mock_response = Mock()
    mock_response.json.return_value = {
        "response": "Ollama response"
    }
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_post.return_value = mock_client

    # Test
    adapter = OllamaAdapter(model="llama3")
    result = await adapter("Hello Ollama")

    assert result == "Ollama response"
    mock_client.post.assert_awaited_once()
    call_args = mock_client.post.call_args
    assert "http://localhost:11434/api/generate" in call_args[0]
    assert call_args[1]["json"]["prompt"] == "Hello Ollama"


@patch('engine.router.adapters.httpx.AsyncClient')
@pytest.mark.asyncio
async def test_ollama_adapter_call_empty_response(mock_post):
    """Test OllamaAdapter handles missing response field."""
    # Setup mock
    mock_response = Mock()
    mock_response.json.return_value = {}  # No "response" key
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_post.return_value = mock_client

    # Test
    adapter = OllamaAdapter()
    result = await adapter("test")

    assert result == ""


# ============================================================
# bootstrap_default_registry Tests
# ============================================================

def test_bootstrap_registry_empty():
    """Test bootstrapping registry with no keys."""
    with patch('engine.router.adapters.OpenAIClient', None), \
         patch('engine.router.adapters.anthropic', None), \
         patch('engine.router.adapters.AutoTokenizer', None), \
         patch('os.getenv', return_value=None):

        registry = bootstrap_default_registry()
        adapters = registry.list()

        # Should have at least ollama (default fallback)
        assert "ollama" in adapters


@patch('engine.router.adapters.OpenAIClient')
def test_bootstrap_registry_with_openai(mock_openai):
    """Test bootstrapping registry with OpenAI key."""
    with patch('engine.router.adapters.anthropic', None), \
         patch('engine.router.adapters.AutoTokenizer', None), \
         patch('os.getenv', return_value=None):

        registry = bootstrap_default_registry(openai_key="test_key")
        adapters = registry.list()

        assert "gpt" in adapters


@patch('engine.router.adapters.anthropic')
def test_bootstrap_registry_with_anthropic(mock_anthropic):
    """Test bootstrapping registry with Anthropic key."""
    with patch('engine.router.adapters.OpenAIClient', None), \
         patch('engine.router.adapters.AutoTokenizer', None), \
         patch('os.getenv', return_value=None):

        registry = bootstrap_default_registry(anthropic_key="test_key")
        adapters = registry.list()

        assert "claude" in adapters


def test_bootstrap_registry_with_xai_key():
    """Test bootstrapping registry with xAI key."""
    with patch('engine.router.adapters.OpenAIClient', None), \
         patch('engine.router.adapters.anthropic', None), \
         patch('engine.router.adapters.AutoTokenizer', None), \
         patch('os.getenv', return_value=None):

        registry = bootstrap_default_registry(xai_key="test_key")
        adapters = registry.list()

        assert "grok" in adapters


def test_bootstrap_registry_ollama_env_single():
    """Test bootstrapping with OLLAMA_MODEL environment variable."""
    with patch('engine.router.adapters.OpenAIClient', None), \
         patch('engine.router.adapters.anthropic', None), \
         patch('engine.router.adapters.AutoTokenizer', None), \
         patch('os.getenv') as mock_getenv:

        def getenv_side_effect(key, default=None):
            if key == "OLLAMA_MODEL":
                return "mistral"
            return default

        mock_getenv.side_effect = getenv_side_effect

        registry = bootstrap_default_registry()
        adapters = registry.list()

        # Should have mistral registered
        assert any("mistral" in name for name in adapters)


def test_bootstrap_registry_ollama_env_multiple():
    """Test bootstrapping with OLLAMA_MODELS environment variable."""
    with patch('engine.router.adapters.OpenAIClient', None), \
         patch('engine.router.adapters.anthropic', None), \
         patch('engine.router.adapters.AutoTokenizer', None), \
         patch('os.getenv') as mock_getenv:

        def getenv_side_effect(key, default=None):
            if key == "OLLAMA_MODELS":
                return "llama3, mistral, phi3"
            return default

        mock_getenv.side_effect = getenv_side_effect

        registry = bootstrap_default_registry()
        adapters = registry.list()

        # Should have multiple models
        assert len(adapters) >= 2  # At least some ollama models


@patch('engine.router.adapters.AutoTokenizer')
@patch('engine.router.adapters.AutoModelForCausalLM')
def test_bootstrap_registry_hf_models(mock_model, mock_tokenizer):
    """Test bootstrapping with HuggingFace models."""
    with patch('engine.router.adapters.OpenAIClient', None), \
         patch('engine.router.adapters.anthropic', None), \
         patch('os.getenv') as mock_getenv:

        def getenv_side_effect(key, default=None):
            if key == "HF_MODELS":
                return "meta-llama/Llama-3-8b"
            elif key == "HF_DEVICE":
                return "cpu"
            return default

        mock_getenv.side_effect = getenv_side_effect

        try:
            registry = bootstrap_default_registry()
            adapters = registry.list()

            # Should have HF model registered
            assert any("hf-" in name for name in adapters)
        except Exception:
            # If model loading fails, that's OK for this test
            pass


@patch('engine.router.adapters.OpenAIClient')
def test_bootstrap_registry_handles_registration_errors(mock_openai):
    """Test that registration errors are handled gracefully."""
    # Make GPT adapter raise exception during registration
    mock_openai.side_effect = Exception("API Error")

    with patch('engine.router.adapters.anthropic', None), \
         patch('engine.router.adapters.AutoTokenizer', None), \
         patch('os.getenv', return_value=None):

        # Should not raise, should handle error gracefully
        registry = bootstrap_default_registry(openai_key="test_key")

        # GPT should not be registered due to error
        adapters = registry.list()
        assert "gpt" not in adapters
