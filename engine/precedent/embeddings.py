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

import os
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, cast
import requests


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

if TYPE_CHECKING:
    from openai import OpenAI as OpenAIClient
else:
    OpenAIClient = None  # type: ignore[assignment]


class GPTEmbeddingAdapter(BaseEmbeddingAdapter):
    """Uses ADA or text-embedding-3-large."""

    def __init__(self, model: str = "text-embedding-3-large", api_key: Optional[str] = None):
        if OpenAIClient is None:
            raise ImportError("OpenAI SDK not installed")
        self.client = OpenAIClient(api_key=api_key)
        self.model = model

    def embed(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        embedding: Sequence[float] = resp.data[0].embedding
        return list(embedding)


# ============================================================
#  Anthropic Claude Embeddings
# ============================================================

try:
    import anthropic
except Exception:
    anthropic = None  # type: ignore[assignment]


class ClaudeEmbeddingAdapter(BaseEmbeddingAdapter):
    """
    Claude embedding adapter using Voyage AI embeddings (Anthropic partner) with
    a sentence-transformer fallback when Voyage is unavailable.
    """

    def __init__(
        self,
        model: str = "voyage-large-2",
        api_key: Optional[str] = None,
        fallback_to_local: bool = True,
        device: Optional[str] = None,
    ):
        """
        Initialize Claude embedding adapter.

        Args:
            model: Voyage AI model name (voyage-large-2, voyage-code-2, etc.)
            api_key: Voyage AI API key (or set VOYAGE_API_KEY env var)
            fallback_to_local: If True, use local embeddings when Voyage unavailable
        """
        self.model = model
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        self.fallback_to_local = fallback_to_local
        self.device = device
        self._local_model = None

        # Try to import voyageai client
        try:
            import voyageai  # type: ignore[import-not-found]
            self._voyage_client = voyageai.Client(api_key=self.api_key) if self.api_key else None
        except ImportError:
            self._voyage_client = None

        # Initialize local fallback if needed
        if self._voyage_client is None and self.fallback_to_local:
            self._init_local_fallback()

    def _init_local_fallback(self):
        """Initialize local sentence-transformer as fallback."""
        if SentenceTransformer is None:
            raise ImportError(
                "Neither Voyage AI SDK nor sentence-transformers installed. "
                "Install one of: pip install voyageai OR pip install sentence-transformers"
            )
        # Use a high-quality model that's compatible with most use cases
        if self.device:
            self._local_model = SentenceTransformer("all-mpnet-base-v2", device=self.device)
        else:
            self._local_model = SentenceTransformer("all-mpnet-base-v2")

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
                vec: Sequence[float] = result.embeddings[0]  # type: ignore[index]
                return list(vec)
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
        data: Any = resp.json()
        embedding = data["data"][0]["embedding"]
        return list(embedding)


# ============================================================
#  HuggingFace Embeddings (Llama / Others)
# ============================================================

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
except ImportError:
    SentenceTransformer = None


class HFEmbeddingAdapter(BaseEmbeddingAdapter):
    """Uses any SentenceTransformer model for local embedding."""

    def __init__(self, model="all-MiniLM-L6-v2", device: Optional[str] = None):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers library not installed")
        self.model = SentenceTransformer(model, device=device) if device else SentenceTransformer(model)

    def embed(self, text: str) -> List[float]:
        vec = self.model.encode(text)
        return cast(List[float], vec.tolist())


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
        data: Any = resp.json()
        embedding = data["embedding"]
        return list(embedding)


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
    xai_key=None,
    device: Optional[str] = None,
) -> EmbeddingRegistry:

    reg = EmbeddingRegistry()

    # Cloud Embedding Backends
    if openai_key:
        reg.register("gpt", GPTEmbeddingAdapter(api_key=openai_key))
    if anthropic_key:
        reg.register("claude", ClaudeEmbeddingAdapter(api_key=anthropic_key, device=device))
    if xai_key:
        reg.register("grok", GrokEmbeddingAdapter(api_key=xai_key))

    # Local Backends
    if SentenceTransformer is not None:
        reg.register("hf", HFEmbeddingAdapter(device=device))
    reg.register("ollama", OllamaEmbeddingAdapter())

    return reg
