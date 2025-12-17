"""
Shared bootstrap utilities for API surfaces (REST + WebSocket).

Responsibilities:
- load constitutional config
- construct the async ELEANOR V8 engine with sensible defaults
- bind critics to specific model adapters when requested
- helper to evaluate OPA callbacks (sync or async)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
import random

import yaml

from engine.core import build_eleanor_engine_v8
from engine.logging_config import get_logger

logger = get_logger(__name__)
GOVERNANCE_SCHEMA_VERSION = "v1"


# -------------------------------------------------------------
# Configuration helpers
# -------------------------------------------------------------

def load_constitutional_config(path_override: Optional[str] = None) -> Dict[str, Any]:
    """Load constitutional configuration from YAML."""
    config_path = path_override or os.getenv("CONSTITUTIONAL_CONFIG_PATH", "governance/constitutional.yaml")
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Constitutional config not found: {config_path}")
    if not path.is_file():
        raise ValueError(f"Constitutional config path is not a file: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_critic_bindings(raw: Optional[str]) -> Dict[str, str]:
    """Parse critic->adapter bindings from JSON or comma-delimited pairs."""
    if not raw:
        return {}
    raw = raw.strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        logger.warning("Failed to parse CRITIC_MODEL_BINDINGS as JSON; falling back to comma parsing")

    bindings: Dict[str, str] = {}
    for chunk in raw.split(","):
        if "=" in chunk:
            critic, adapter = chunk.split("=", 1)
            bindings[critic.strip()] = adapter.strip()
    return bindings


def bind_critic_models(engine, bindings: Optional[Dict[str, str]] = None) -> None:
    """
    Bind critics to specific router adapters so each critic can run on a
    dedicated model. Bindings can be provided explicitly or via the
    CRITIC_MODEL_BINDINGS env var (JSON or critic=adapter,crit...).

    If no explicit bindings are provided, we fall back to a sensible default:
    - CRITIC_DEFAULT_ADAPTER env var, if set
    - OLLAMA_CRITIC_MODEL / OLLAMA_MODEL env var (auto-prefix with ollama-)
    - first Ollama adapter discovered on the router (ollama-*), if present
    """
    mapping = bindings or _parse_critic_bindings(os.getenv("CRITIC_MODEL_BINDINGS"))
    if not mapping:
        default_adapter = os.getenv("CRITIC_DEFAULT_ADAPTER")

        if not default_adapter:
            ollama_model = os.getenv("OLLAMA_CRITIC_MODEL") or os.getenv("OLLAMA_MODEL")
            if ollama_model:
                default_adapter = f"ollama-{ollama_model}" if ollama_model != "llama3" else "ollama"

        if not default_adapter:
            router = getattr(engine, "router", None)
            adapters = getattr(router, "adapters", {}) if router else {}
            for name in adapters.keys():
                if name.startswith("ollama"):
                    default_adapter = name
                    break

        if default_adapter:
            critics = getattr(engine, "critics", {}) or {}
            mapping = {critic: default_adapter for critic in critics.keys()}
        else:
            return

    router = getattr(engine, "router", None)
    adapters = getattr(router, "adapters", {}) if router else {}

    for critic, adapter_name in mapping.items():
        adapter = adapters.get(adapter_name)
        if adapter is None:
            logger.warning(f"Critic binding skipped: adapter '{adapter_name}' not found for critic '{critic}'")
            continue
        engine.critic_models[critic] = adapter
        logger.info(f"Critic '{critic}' bound to adapter '{adapter_name}'")


def build_engine(constitution: Optional[Dict[str, Any]] = None) -> Any:
    """
    Construct the async engine using the shared builder and apply critic
    bindings. The builder will auto-discover adapters based on available
    SDKs and keys, and fall back to a safe echo adapter if none are present.
    """
    seed = os.getenv("ELEANOR_SEED")
    if seed is not None:
        try:
            seed_val = int(seed)
            random.seed(seed_val)
            try:
                import numpy as np  # type: ignore
                np.random.seed(seed_val)
            except Exception:
                logger.debug("NumPy not available for deterministic seeding; continuing without it")
        except ValueError:
            logger.warning("ELEANOR_SEED is not an int; skipping deterministic seeding")

    constitution = constitution or load_constitutional_config()
    evidence_path = os.getenv("EVIDENCE_PATH") or os.getenv("ELEANOR_EVIDENCE_PATH") or "evidence.jsonl"

    engine = build_eleanor_engine_v8(
        constitutional_config=constitution,
        evidence_path=evidence_path,
    )

    bind_critic_models(engine)
    return engine


# -------------------------------------------------------------
# Governance helper
# -------------------------------------------------------------

async def evaluate_opa(callback, payload: Dict[str, Any], fallback_strategy: Optional[str] = None) -> Dict[str, Any]:
    """
    Call OPA callback regardless of sync/async implementation.
    fallback_strategy: "escalate" (default), "deny", or "allow" on failure.
    """
    fallback = (fallback_strategy or os.getenv("OPA_FAIL_STRATEGY", "escalate")).lower()

    if callback is None:
        if fallback == "allow":
            return {"allow": True, "escalate": False, "failures": [{"policy": "opa_missing", "reason": "No OPA configured"}]}
        if fallback == "deny":
            return {"allow": False, "escalate": False, "failures": [{"policy": "opa_missing", "reason": "No OPA configured"}]}
        return {"allow": False, "escalate": True, "failures": [{"policy": "opa_missing", "reason": "No OPA configured"}]}

    try:
        result = callback(payload)
        if hasattr(result, "__await__"):
            result = await result
        return result or {"allow": False, "failures": [{"policy": "empty", "reason": "No result returned"}]}
    except Exception as exc:
        logger.warning(f"OPA evaluation failed: {exc}")
        failure = {"policy": "opa_exception", "reason": str(exc)}
        if fallback == "allow":
            return {"allow": True, "escalate": False, "failures": [failure]}
        if fallback == "deny":
            return {"allow": False, "escalate": False, "failures": [failure]}
        return {"allow": False, "escalate": True, "failures": [failure]}


__all__ = ["build_engine", "bind_critic_models", "load_constitutional_config", "evaluate_opa", "GOVERNANCE_SCHEMA_VERSION"]
