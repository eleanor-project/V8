from typing import Any, Dict, List, Optional, cast

from pydantic import BaseModel, Field

from engine.schemas.pipeline_types import (
    AggregationOutput,
    PrecedentAlignmentResult,
    UncertaintyResult,
    ViolationEntry,
)


class EngineCriticFinding(BaseModel):
    critic: str
    violations: List[ViolationEntry]
    duration_ms: Optional[float] = None
    evaluated_rules: Optional[List[str]] = None


class EngineModelInfo(BaseModel):
    model_name: str
    model_version: Optional[str] = None
    router_selection_reason: Optional[str] = None
    cost_estimate: Optional[Dict[str, Any]] = None
    health_score: Optional[float] = None


def _default_uncertainty_result() -> UncertaintyResult:
    return cast(UncertaintyResult, {})


def _default_precedent_alignment() -> PrecedentAlignmentResult:
    return cast(PrecedentAlignmentResult, {})


class EngineForensicData(BaseModel):
    detector_metadata: Dict[str, Any] = Field(default_factory=dict)
    uncertainty_graph: UncertaintyResult = Field(default_factory=_default_uncertainty_result)
    precedent_alignment: PrecedentAlignmentResult = Field(
        default_factory=_default_precedent_alignment
    )
    router_diagnostics: Dict[str, Any] = Field(default_factory=dict)
    timings: Dict[str, float] = Field(default_factory=dict)
    evidence_references: List[Dict[str, Any]] = Field(default_factory=list)


class EngineResult(BaseModel):
    output_text: Optional[str] = None
    trace_id: str
    model_info: Optional[EngineModelInfo] = None
    critic_findings: Optional[Dict[str, EngineCriticFinding]] = None
    aggregated: Optional[AggregationOutput] = None
    uncertainty: Optional[UncertaintyResult] = None
    precedent_alignment: Optional[PrecedentAlignmentResult] = None
    evidence_count: Optional[int] = None
    degraded_components: Optional[List[str]] = None
    is_degraded: Optional[bool] = None
    forensic: Optional[EngineForensicData] = None
