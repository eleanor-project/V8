"""
ELEANOR V8 — Embedding Adapter Registry
----------------------------------------

Provides a unified interface for embedding text using ANY backend.
Used by:
    - Precedent retrieval
    - Precedent drift detection
    - Precedent alignment scoring

Backends:
    - OpenAI GPT embeddings
    - Anthropic Claude embeddings
    - xAI Grok embeddings
    - HuggingFace (Llama / other)
    - Ollama local embeddings
"""

import json
import requests
from typing import List, Callable


# ============================================================
#  Base Embedding Interface
# ============================================================

class BaseEmbeddingAdapter:
    """All embedding adapters must return a vector list[float]."""

    def embed(self, text: str) -> List[float]:
        raise NotImplementedError


# ============================================================
#  OpenAI GPT Embeddings
# ============================================================

try:
    from openai import OpenAI
except:
    OpenAI = None


class GPTEmbeddingAdapter(BaseEmbeddingAdapter):
    """Uses ADA or text-embedding-3-large."""

    def __init__(self, model="text-embedding-3-large", api_key=None):
        if OpenAI is None:
            raise ImportError("OpenAI SDK not installed")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return resp.data[0].embedding


# ============================================================
#  Anthropic Claude Embeddings
# ============================================================

try:
    import anthropic
except:
    anthropic = None


class ClaudeEmbeddingAdapter(BaseEmbeddingAdapter):
    """
    Claude embedding adapter using Voyage AI embeddings (Anthropic partner) with
    a sentence-transformer fallback when Voyage is unavailable.
    """

    def __init__(self, model: str = "voyage-large-2", api_key: str = None,
                 fallback_to_local: bool = True):
        """
        Initialize Claude embedding adapter.

        Args:
            model: Voyage AI model name (voyage-large-2, voyage-code-2, etc.)
            api_key: Voyage AI API key (or set VOYAGE_API_KEY env var)
            fallback_to_local: If True, use local embeddings when Voyage unavailable
        """
        import os
        self.model = model
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        self.fallback_to_local = fallback_to_local
        self._local_model = None

        # Try to import voyageai client
        try:
            import voyageai
            self._voyage_client = voyageai.Client(api_key=self.api_key) if self.api_key else None
        except ImportError:
            self._voyage_client = None

        # Initialize local fallback if needed
        if self._voyage_client is None and self.fallback_to_local:
            self._init_local_fallback()

    def _init_local_fallback(self):
        """Initialize local sentence-transformer as fallback."""
        try:
            from sentence_transformers import SentenceTransformer
            # Use a high-quality model that's compatible with most use cases
            self._local_model = SentenceTransformer("all-mpnet-base-v2")
        except ImportError:
            raise ImportError(
                "Neither Voyage AI SDK nor sentence-transformers installed. "
                "Install one of: pip install voyageai OR pip install sentence-transformers"
            )

    def embed(self, text: str) -> List[float]:
        """
        Generate embeddings for the input text.

        Uses Voyage AI if available, otherwise falls back to local model.
        """
        # Try Voyage AI first
        if self._voyage_client is not None:
            try:
                result = self._voyage_client.embed(
                    texts=[text],
                    model=self.model,
                    input_type="document"
                )
                return result.embeddings[0]
            except Exception as e:
                if not self.fallback_to_local:
                    raise RuntimeError(f"Voyage AI embedding failed: {e}")
                # Fall through to local model

        # Use local fallback
        if self._local_model is not None:
            vec = self._local_model.encode(text)
            return vec.tolist()

        raise RuntimeError(
            "No embedding backend available. Configure VOYAGE_API_KEY or install sentence-transformers."
        )


# ============================================================
#  xAI Grok Embeddings
# ============================================================

class GrokEmbeddingAdapter(BaseEmbeddingAdapter):
    """Calls xAI embedding endpoint."""

    def __init__(self, model="grok-embed", api_key=None):
        self.model = model
        self.api_key = api_key

    def embed(self, text: str) -> List[float]:
        url = "https://api.x.ai/v1/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model, "input": text}

        resp = requests.post(url, headers=headers, json=payload)
        data = resp.json()
        return data["data"][0]["embedding"]


# ============================================================
#  HuggingFace Embeddings (Llama / Others)
# ============================================================

try:
    from sentence_transformers import SentenceTransformer
except:
    SentenceTransformer = None


class HFEmbeddingAdapter(BaseEmbeddingAdapter):
    """Uses any SentenceTransformer model for local embedding."""

    def __init__(self, model="all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers library not installed")
        self.model = SentenceTransformer(model)

    def embed(self, text: str) -> List[float]:
        vec = self.model.encode(text)
        return vec.tolist()


# ============================================================
#  Ollama Local Embeddings
# ============================================================

class OllamaEmbeddingAdapter(BaseEmbeddingAdapter):
    """Uses Ollama's /api/embeddings endpoint."""

    def __init__(self, model="llama3"):
        self.model = model

    def embed(self, text: str) -> List[float]:
        resp = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": self.model, "prompt": text}
        )
        data = resp.json()
        return data["embedding"]


# ============================================================
#  Unified Embedding Registry
# ============================================================

class EmbeddingRegistry:
    """
    Holds embedding backends and exposes lookup and list methods.
    """

    def __init__(self):
        self.adapters = {}

    def register(self, name: str, adapter: BaseEmbeddingAdapter):
        self.adapters[name] = adapter

    def get(self, name: str):
        if name not in self.adapters:
            raise ValueError(f"Embedding adapter '{name}' not found")
        return self.adapters[name]

    def list(self):
        return list(self.adapters.keys())


# ============================================================
#  Bootstrap Helper
# ============================================================

def bootstrap_embedding_registry(
    openai_key=None,
    anthropic_key=None,
    xai_key=None
) -> EmbeddingRegistry:

    reg = EmbeddingRegistry()

    # Cloud Embedding Backends
    if openai_key:
        reg.register("gpt", GPTEmbeddingAdapter(api_key=openai_key))
    if anthropic_key:
        reg.register("claude", ClaudeEmbeddingAdapter(api_key=anthropic_key))
    if xai_key:
        reg.register("grok", GrokEmbeddingAdapter(api_key=xai_key))

    # Local Backends
    if SentenceTransformer is not None:
        reg.register("hf", HFEmbeddingAdapter())
    reg.register("ollama", OllamaEmbeddingAdapter())

    return reg
