from typing import Dict, Any
from .base import CriticOutput
from engine.schemas.escalation import EscalationSignal


def parse_critic_response(raw: Dict[str, Any], critic_name: str) -> CriticOutput:
    """
    Normalize/validate raw JSON from the LLM into CriticOutput.
    """
    escalation_payload = raw.get("escalation")
    escalation = EscalationSignal(**escalation_payload) if isinstance(escalation_payload, dict) else None

    return CriticOutput(
        critic=critic_name,
        concern=raw.get("concern", "").strip(),
        severity=float(raw.get("severity", 0.0)),
        principle=raw.get("principle", "").strip(),
        uncertainty=raw.get("uncertainty"),
        rationale=raw.get("rationale", "").strip(),
        precedent_refs=raw.get("precedent_refs") or [],
        escalation=escalation,
    )
