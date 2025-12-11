from pydantic import BaseModel, Field
from typing import Any, Dict

class DetectorSignal(BaseModel):
    violation: bool = False
    severity: str = "S0"
    description: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    mitigation: str | None = None
