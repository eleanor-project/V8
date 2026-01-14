from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from api.middleware.auth import require_role
from api.rest.deps import get_engine, get_secret_provider
from api.rest.admin_write import require_admin_write_enabled

from engine.logging_config import get_logger
from engine.utils.critic_names import canonical_critic_name
from engine.router.adapters import GPTAdapter, ClaudeAdapter, GrokAdapter, LlamaHFAdapter, OllamaAdapter
from engine.security.secrets import get_llm_api_key

logger = get_logger(__name__)
router = APIRouter(tags=["Admin"])

ADMIN_ROLE = os.getenv("ADMIN_ROLE", "admin")


def _resolve_environment() -> str:
    return os.getenv("ELEANOR_ENVIRONMENT") or os.getenv("ELEANOR_ENV") or "development"


@router.get("/critics/bindings")
@require_role(ADMIN_ROLE)
async def get_critic_bindings(engine=Depends(get_engine)):
    r = getattr(engine, "router", None)
    adapters = getattr(r, "adapters", {}) if r else {}
    return {
        "bindings": getattr(engine, "critic_models", {}),
        "critics": list(getattr(engine, "critics", {}).keys()),
        "available_adapters": list(adapters.keys()),
    }


class CriticBinding(BaseModel):
    critic: str
    adapter: str


@router.post("/critics/bindings")
@require_role(ADMIN_ROLE)
async def set_critic_binding(
    binding: CriticBinding,
    _write_enabled: None = Depends(require_admin_write_enabled),
    engine=Depends(get_engine),
):
    r = getattr(engine, "router", None)
    adapters = getattr(r, "adapters", {}) if r else {}
    adapter_fn = adapters.get(binding.adapter)
    if adapter_fn is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Adapter '{binding.adapter}' not registered",
        )
    critic_key = canonical_critic_name(binding.critic)
    engine.critic_models[critic_key] = adapter_fn
    return {"status": "ok", "binding": {critic_key: binding.adapter}}


class AdapterRegistration(BaseModel):
    name: str
    type: str = "ollama"  # ollama | hf | openai | claude | grok
    model: Optional[str] = None
    device: Optional[str] = None
    api_key: Optional[str] = None


@router.post("/router/adapters")
@require_role(ADMIN_ROLE)
async def register_adapter(
    adapter: AdapterRegistration,
    _write_enabled: None = Depends(require_admin_write_enabled),
    engine=Depends(get_engine),
    secret_provider=Depends(get_secret_provider),
):
    if not getattr(engine, "router", None):
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Router not available")

    r = engine.router
    adapters = getattr(r, "adapters", {}) if r else {}
    factory = adapter.type.lower()

    try:
        if factory == "ollama":
            adapters[adapter.name] = OllamaAdapter(model=adapter.model or adapter.name)
        elif factory == "hf":
            adapters[adapter.name] = LlamaHFAdapter(
                model_path=adapter.model or adapter.name, device=adapter.device or "cpu"
            )
        elif factory == "openai":
            api_key = adapter.api_key
            if api_key is None and secret_provider is not None:
                api_key = await get_llm_api_key("openai", secret_provider)
            if api_key is None and _resolve_environment() != "production":
                api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
            if api_key is None:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OpenAI API key not available")
            adapters[adapter.name] = GPTAdapter(model=adapter.model or "gpt-4o-mini", api_key=api_key)
        elif factory == "claude":
            api_key = adapter.api_key
            if api_key is None and secret_provider is not None:
                api_key = await get_llm_api_key("anthropic", secret_provider)
            if api_key is None and _resolve_environment() != "production":
                api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_KEY")
            if api_key is None:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Anthropic API key not available")
            adapters[adapter.name] = ClaudeAdapter(model=adapter.model or "claude-3-5-sonnet-20241022", api_key=api_key)
        elif factory == "grok":
            api_key = adapter.api_key
            if api_key is None and secret_provider is not None:
                api_key = await get_llm_api_key("xai", secret_provider)
            if api_key is None and _resolve_environment() != "production":
                api_key = os.getenv("XAI_API_KEY") or os.getenv("XAI_KEY")
            if api_key is None:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="xAI API key not available")
            adapters[adapter.name] = GrokAdapter(model=adapter.model or "grok-beta", api_key=api_key)
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown adapter type '{adapter.type}'")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to register adapter: {exc}")

    policy = getattr(r, "policy", {}) or {}
    fallback = policy.get("fallback_order") or []
    if adapter.name not in fallback and adapter.name != policy.get("primary"):
        fallback.append(adapter.name)
    policy["fallback_order"] = fallback
    r.policy = policy

    return {
        "status": "ok",
        "registered": adapter.name,
        "type": adapter.type,
        "available_adapters": list(adapters.keys()),
        "policy": r.policy,
    }
