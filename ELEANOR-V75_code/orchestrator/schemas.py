# orchestrator/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List

class EvaluationRequest(BaseModel):
    input: str = Field(..., description="The user scenario or model output to be evaluated.")

class CriticAssessment(BaseModel):
    claim: str
    evidence: str
    constitutional_principle: str
    confidence: float
    mitigation: str

class Precedent(BaseModel):
    id: int
    citation: str
    input_text: str
    similarity_score: Optional[float] = None

class EvaluationResponse(BaseModel):
    input: str
    rights_critic: CriticAssessment
    fairness_critic: CriticAssessment
    risk_critic: CriticAssessment
    truth_critic: CriticAssessment
    pragmatics_critic: CriticAssessment
    final_decision: str
    grounding_critic: str
    precedents_referenced: List[str]
    saved_as_precedent: Optional[str] = None
    similar_precedents: List[Precedent] = []
