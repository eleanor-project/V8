from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:  # pragma: no cover
    from engine.config.settings import EleanorSettings


class EngineConfig(BaseModel):
    detail_level: int = 2
    max_concurrency: int = 6
    timeout_seconds: float = 10.0
    enable_adaptive_concurrency: bool = False
    target_latency_ms: float = 500.0
    enable_circuit_breakers: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    enable_graceful_degradation: bool = True
    shutdown_timeout_seconds: float = 30.0

    enable_reflection: bool = True
    enable_drift_check: bool = True
    enable_precedent_analysis: bool = True

    # Traffic Light governance hook (external governor; sanctity-preserving)
    enable_traffic_light_governance: bool = True
    traffic_light_router_config_path: str = 'governance/router_config.yaml'
    governance_events_jsonl_path: str | None = 'governance_events.jsonl'

    jsonl_evidence_path: Optional[str] = "evidence.jsonl"

    @classmethod
    def from_settings(cls, settings: "EleanorSettings") -> "EngineConfig":
        return cls(**settings.to_legacy_engine_config())


def load_config_from_yaml(path: str) -> EngineConfig:
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for YAML config support.")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return EngineConfig(**data)
