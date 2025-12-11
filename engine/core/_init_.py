"""
ELEANOR V8 — Engine Builder (FULL ENV + OPA + EMBEDDINGS + DUAL BACKENDS)
--------------------------------------------------------------------------

This file builds the entire ELEANOR V8 governance engine, integrating:

  • LLM-integrated critics
  • Multi-model router
  • Orchestrator
  • Aggregator
  • Embedding registry (GPT, Claude, Grok, HF, Ollama)
  • Precedent stores (Weaviate / pgvector / memory)
  • Evidence recorder
  • Real OPA governance evaluation
  • Full environment-variable driven configuration

This is the fully production-ready engine bootstrapper.
"""

import os
import json

# Core Engine
from engine.core.engine import EleanorEngineV8

# Router + Orchestrator + Critics
from engine.router.router import RouterV8
from engine.orchestrator.orchestrator import OrchestratorV8
from engine.critics.rights import RightsCriticV8
from engine.critics.fairness import FairnessCriticV8
from engine.critics.truth import TruthCriticV8
from engine.critics.risk import RiskCriticV8
from engine.critics.pragmatics import PragmaticsCriticV8

# Precedent: retrieval + stores + embeddings
from engine.precedent.retrieval import PrecedentRetrievalV8
from engine.precedent.stores import (
    WeaviatePrecedentStore,
    PGVectorPrecedentStore
)
from engine.precedent.embeddings import bootstrap_embedding_registry

# Evidence recorder
from engine.evidence.recorder import EvidenceRecorderV8

# Governance: OPA Client
from engine.governance.opa_client import OPAClientV8

# Optional SDKs
try:
    import weaviate
except ImportError:
    weaviate = None


# ========================================================================
#  Engine Builder
# ========================================================================

def build_eleanor_engine_v8(
    llm_fn,
    constitutional_config,
    router_adapters,
    router_policy,
):
    """
    Build a fully operational ELEANOR V8 engine.

    All configuration is driven by environment variables:

        PRECEDENT_BACKEND = weaviate | pgvector | memory
        EMBEDDING_BACKEND = openai | claude | grok | hf | ollama

        OPENAI_KEY=...
        ANTHROPIC_KEY=...
        XAI_KEY=...

        WEAVIATE_URL=http://weaviate:8080
        WEAVIATE_CLASS_NAME=Precedent

        PG_CONN_STRING=postgresql://postgres:postgres@pgvector:5432/eleanor

        OPA_URL=http://opa:8181
        OPA_POLICY_PATH=v1/data/eleanor/decision

        EVIDENCE_STORAGE=jsonl | memory
        EVIDENCE_PATH=./audit
    """

    # --------------------------------------------------------------
    # ENVIRONMENT VARIABLES (with defaults)
    # --------------------------------------------------------------

    precedent_backend = os.getenv("PRECEDENT_BACKEND", "weaviate").lower()
    embedding_backend = os.getenv("EMBEDDING_BACKEND", "openai")

    openai_key = os.getenv("OPENAI_KEY")
    anthropic_key = os.getenv("ANTHROPIC_KEY")
    xai_key = os.getenv("XAI_KEY")

    weaviate_url = os.getenv("WEAVIATE_URL", "http://weaviate:8080")
    weaviate_class = os.getenv("WEAVIATE_CLASS_NAME", "Precedent")

    pg_conn_string = os.getenv("PG_CONN_STRING")

    opa_url = os.getenv("OPA_URL", "http://opa:8181")
    opa_policy_path = os.getenv("OPA_POLICY_PATH", "v1/data/eleanor/decision")

    evidence_mode = os.getenv("EVIDENCE_STORAGE", "jsonl")
    evidence_path = os.getenv("EVIDENCE_PATH", "./audit")

    # --------------------------------------------------------------
    # Router (model selection)
    # --------------------------------------------------------------
    router = RouterV8(
        adapters=router_adapters,
        routing_policy=router_policy
    )

    # --------------------------------------------------------------
    # Critics (LLM-integrated)
    # --------------------------------------------------------------
    critics = {
        "rights": RightsCriticV8(llm_fn),
        "fairness": FairnessCriticV8(llm_fn),
        "truth": TruthCriticV8(llm_fn),
        "risk": RiskCriticV8(llm_fn),
        "pragmatics": PragmaticsCriticV8(llm_fn),
    }

    orchestrator = OrchestratorV8(critics)

    # --------------------------------------------------------------
    # Embedding system
    # --------------------------------------------------------------
    embedding_registry = bootstrap_embedding_registry(
        openai_key=openai_key,
        anthropic_key=anthropic_key,
        xai_key=xai_key
    )

    if embedding_backend not in embedding_registry.list():
        raise ValueError(
            f"Embedding backend '{embedding_backend}' not found. "
            f"Available: {embedding_registry.list()}"
        )

    embed_fn = embedding_registry.get(embedding_backend).embed

    # --------------------------------------------------------------
    # Precedent Store Backend
    # --------------------------------------------------------------
    if precedent_backend == "weaviate":
        if weaviate is None:
            raise ImportError("Weaviate client not installed")

        client = weaviate.Client(url=weaviate_url)

        store = WeaviatePrecedentStore(
            client=client,
            class_name=weaviate_class,
            embed_fn=embed_fn
        )

    elif precedent_backend == "pgvector":
        if not pg_conn_string:
            raise ValueError("PG_CONN_STRING must be set for pgvector backend")

        store = PGVectorPrecedentStore(
            conn_string=pg_conn_string,
            table_name="precedent",
            embed_fn=embed_fn
        )

    elif precedent_backend == "memory":
        class MemoryStore:
            def __init__(self): self.items = []
            def add(self, text, metadata=None):
                self.items.append({"text": text, "metadata": metadata or {}})
            def search(self, q, top_k=5):
                return self.items[:top_k]
        store = MemoryStore()

    else:
        raise ValueError(
            f"Unknown PRECEDENT_BACKEND='{precedent_backend}'. "
            "Use: weaviate | pgvector | memory"
        )

    precedent_retriever = PrecedentRetrievalV8(store_client=store)

    # --------------------------------------------------------------
    # Evidence Recorder
    # --------------------------------------------------------------
    recorder = EvidenceRecorderV8(
        storage_mode=evidence_mode,
        path=evidence_path
    )

    # --------------------------------------------------------------
    # OPA Governance Layer (REAL)
    # --------------------------------------------------------------
    opa_client = OPAClientV8(
        base_url=opa_url,
        policy_path=opa_policy_path
    )

    opa_callback = opa_client.evaluate

    # --------------------------------------------------------------
    # Final Engine Assembly
    # --------------------------------------------------------------
    engine = EleanorEngineV8(
        constitutional_config=constitutional_config,
        router=router,
        orchestrator=orchestrator,
        precedent_retriever=precedent_retriever,
        evidence_recorder=recorder,
        opa_governance_callback=opa_callback
    )

    return engine
