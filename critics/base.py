from pydantic import BaseModel
from typing import Optional, List
from engine.schemas.escalation import EscalationSignal


class CriticOutput(BaseModel):
    critic: str
    concern: str
    severity: float  # magnitude of ethical impact, not probability
    principle: str
    uncertainty: Optional[str]
    rationale: str
    precedent_refs: Optional[List[str]] = None  # e.g. ["UDHR Art. 1: dignity"]
    escalation: Optional[EscalationSignal] = None  # critic-initiated escalation signal, if any
