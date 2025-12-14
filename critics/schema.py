from pydantic import BaseModel
from typing import Optional


class CriticOutput(BaseModel):
    critic: str
    concern: str
    severity: float  # impact magnitude, not probability
    principle: str
    uncertainty: Optional[str]
    rationale: str
    precedent: Optional[str] = None  # normative grounding (e.g., UDHR Article)
