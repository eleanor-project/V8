"""
Tests for Precedent Embeddings
------------------------------

Tests engine/precedent/embeddings.py
"""

import pytest
from unittest.mock import Mock, patch
from engine.precedent.embeddings import (
    BaseEmbeddingAdapter,
    EmbeddingRegistry,
    GPTEmbeddingAdapter,
    ClaudeEmbeddingAdapter,
    GrokEmbeddingAdapter,
    OllamaEmbeddingAdapter,
    bootstrap_embedding_registry,
)


# ============================================================
# BaseEmbeddingAdapter Tests
# ============================================================


def test_base_embedding_adapter_not_implemented():
    """Test that BaseEmbeddingAdapter raises NotImplementedError."""
    adapter = BaseEmbeddingAdapter()

    with pytest.raises(NotImplementedError):
        adapter.embed("test text")


def test_base_embedding_adapter_subclass():
    """Test that subclasses can implement embed."""

    class TestAdapter(BaseEmbeddingAdapter):
        def embed(self, text: str):
            return [0.1, 0.2, 0.3]

    adapter = TestAdapter()
    result = adapter.embed("hello")

    assert result == [0.1, 0.2, 0.3]


# ============================================================
# EmbeddingRegistry Tests
# ============================================================


def test_embedding_registry_creation():
    """Test creating an empty embedding registry."""
    registry = EmbeddingRegistry()

    assert isinstance(registry.adapters, dict)
    assert len(registry.adapters) == 0


def test_embedding_registry_register():
    """Test registering an embedding adapter."""
    registry = EmbeddingRegistry()

    class TestAdapter(BaseEmbeddingAdapter):
        def embed(self, text: str):
            return [1.0, 2.0]

    adapter = TestAdapter()
    registry.register("test", adapter)

    assert "test" in registry.adapters
    assert registry.adapters["test"] is adapter


def test_embedding_registry_get():
    """Test retrieving a registered adapter."""
    registry = EmbeddingRegistry()

    class TestAdapter(BaseEmbeddingAdapter):
        def embed(self, text: str):
            return [1.0, 2.0]

    adapter = TestAdapter()
    registry.register("test", adapter)

    retrieved = registry.get("test")
    assert retrieved is adapter


def test_embedding_registry_get_not_found():
    """Test that getting non-existent adapter raises ValueError."""
    registry = EmbeddingRegistry()

    with pytest.raises(ValueError) as exc_info:
        registry.get("nonexistent")

    assert "not found" in str(exc_info.value)
    assert "nonexistent" in str(exc_info.value)


def test_embedding_registry_list():
    """Test listing all registered adapters."""
    registry = EmbeddingRegistry()

    class TestAdapter(BaseEmbeddingAdapter):
        def embed(self, text: str):
            return [1.0]

    registry.register("adapter1", TestAdapter())
    registry.register("adapter2", TestAdapter())

    adapters = registry.list()

    assert len(adapters) == 2
    assert "adapter1" in adapters
    assert "adapter2" in adapters


# ============================================================
# GPTEmbeddingAdapter Tests
# ============================================================


def test_gpt_embedding_adapter_import_error():
    """Test GPTEmbeddingAdapter raises ImportError when OpenAI not installed."""
    with patch("engine.precedent.embeddings.OpenAIClient", None):
        with pytest.raises(ImportError) as exc_info:
            GPTEmbeddingAdapter()

        assert "OpenAI SDK not installed" in str(exc_info.value)


@patch("engine.precedent.embeddings.OpenAIClient")
def test_gpt_embedding_adapter_initialization(mock_openai):
    """Test GPTEmbeddingAdapter initialization."""
    adapter = GPTEmbeddingAdapter(model="text-embedding-3-large", api_key="test_key")

    assert adapter.model == "text-embedding-3-large"
    mock_openai.assert_called_once_with(api_key="test_key")


@patch("engine.precedent.embeddings.OpenAIClient")
def test_gpt_embedding_adapter_embed(mock_openai):
    """Test GPTEmbeddingAdapter embed method."""
    # Setup mock
    mock_embedding_data = Mock()
    mock_embedding_data.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

    mock_response = Mock()
    mock_response.data = [mock_embedding_data]

    mock_client_instance = Mock()
    mock_client_instance.embeddings.create.return_value = mock_response
    mock_openai.return_value = mock_client_instance

    # Test
    adapter = GPTEmbeddingAdapter(api_key="test_key")
    result = adapter.embed("test text")

    assert result == [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_client_instance.embeddings.create.assert_called_once()


# ============================================================
# GrokEmbeddingAdapter Tests
# ============================================================


def test_grok_embedding_adapter_initialization():
    """Test GrokEmbeddingAdapter initialization."""
    adapter = GrokEmbeddingAdapter(model="grok-embed", api_key="test_key")

    assert adapter.model == "grok-embed"
    assert adapter.api_key == "test_key"


@patch("engine.precedent.embeddings.requests.post")
def test_grok_embedding_adapter_embed(mock_post):
    """Test GrokEmbeddingAdapter embed method."""
    # Setup mock
    mock_response = Mock()
    mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
    mock_post.return_value = mock_response

    # Test
    adapter = GrokEmbeddingAdapter(api_key="test_key")
    result = adapter.embed("test text")

    assert result == [0.1, 0.2, 0.3]
    mock_post.assert_called_once()


# ============================================================
# OllamaEmbeddingAdapter Tests
# ============================================================


def test_ollama_embedding_adapter_initialization():
    """Test OllamaEmbeddingAdapter initialization."""
    adapter = OllamaEmbeddingAdapter(model="llama3")

    assert adapter.model == "llama3"


@patch("engine.precedent.embeddings.requests.post")
def test_ollama_embedding_adapter_embed(mock_post):
    """Test OllamaEmbeddingAdapter embed method."""
    # Setup mock
    mock_response = Mock()
    mock_response.json.return_value = {"embedding": [0.5, 0.6, 0.7, 0.8]}
    mock_post.return_value = mock_response

    # Test
    adapter = OllamaEmbeddingAdapter(model="llama3")
    result = adapter.embed("test text")

    assert result == [0.5, 0.6, 0.7, 0.8]
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert "http://localhost:11434/api/embeddings" in call_args[0]


# ============================================================
# ClaudeEmbeddingAdapter Tests
# ============================================================


@patch("engine.precedent.embeddings.os.getenv")
def test_claude_embedding_adapter_initialization(mock_getenv):
    """Test ClaudeEmbeddingAdapter initialization."""
    mock_getenv.return_value = None

    with patch("builtins.__import__", side_effect=ImportError):
        adapter = ClaudeEmbeddingAdapter(api_key="test_key", fallback_to_local=False)

        assert adapter.model == "voyage-large-2"
        assert adapter.api_key == "test_key"


@patch("engine.precedent.embeddings.os.getenv")
def test_claude_embedding_adapter_voyage_success(mock_getenv):
    """Test ClaudeEmbeddingAdapter with Voyage AI."""
    mock_getenv.return_value = None

    # Mock voyageai client
    mock_voyage_client = Mock()
    mock_result = Mock()
    mock_result.embeddings = [[0.1, 0.2, 0.3]]
    mock_voyage_client.embed.return_value = mock_result

    with patch("builtins.__import__") as mock_import:

        def import_side_effect(name, *args, **kwargs):
            if name == "voyageai":
                mock_voyageai = Mock()
                mock_voyageai.Client.return_value = mock_voyage_client
                return mock_voyageai
            raise ImportError()

        mock_import.side_effect = import_side_effect

        adapter = ClaudeEmbeddingAdapter(api_key="test_key")
        adapter._voyage_client = mock_voyage_client

        result = adapter.embed("test text")

        assert result == [0.1, 0.2, 0.3]


# ============================================================
# bootstrap_embedding_registry Tests
# ============================================================


def test_bootstrap_embedding_registry_empty():
    """Test bootstrapping embedding registry with no keys."""
    with patch("engine.precedent.embeddings.SentenceTransformer", None):
        registry = bootstrap_embedding_registry()
        adapters = registry.list()

        # Should at least have ollama
        assert "ollama" in adapters


@patch("engine.precedent.embeddings.OpenAIClient")
def test_bootstrap_embedding_registry_with_openai(mock_openai):
    """Test bootstrapping with OpenAI key."""
    with patch("engine.precedent.embeddings.SentenceTransformer", None):
        registry = bootstrap_embedding_registry(openai_key="test_key")
        adapters = registry.list()

        assert "gpt" in adapters


def test_bootstrap_embedding_registry_with_xai():
    """Test bootstrapping with xAI key."""
    with patch("engine.precedent.embeddings.SentenceTransformer", None):
        registry = bootstrap_embedding_registry(xai_key="test_key")
        adapters = registry.list()

        assert "grok" in adapters


@patch("engine.precedent.embeddings.SentenceTransformer")
def test_bootstrap_embedding_registry_with_hf(mock_st):
    """Test bootstrapping with HuggingFace."""
    registry = bootstrap_embedding_registry()
    adapters = registry.list()

    assert "hf" in adapters
    assert "ollama" in adapters


def test_bootstrap_embedding_registry_all_keys():
    """Test bootstrapping with all API keys."""
    with patch("engine.precedent.embeddings.OpenAIClient"), patch(
        "engine.precedent.embeddings.SentenceTransformer"
    ):
        registry = bootstrap_embedding_registry(
            openai_key="openai_key", anthropic_key="anthropic_key", xai_key="xai_key"
        )
        adapters = registry.list()

        # Should have multiple adapters
        assert len(adapters) >= 3
        assert "ollama" in adapters
