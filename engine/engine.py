"""
ELEANOR V8 — Enterprise Constitutional Engine Runtime
Dual API (run + run_stream)
Dependency Injection Ready
Full Evidence Recorder Integration
Precedent Alignment + Uncertainty Engine Hooks
Forensic Detail-Level Output Mode
"""

import asyncio
import inspect
import json
import logging
import uuid
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, AsyncGenerator, Callable, cast

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from engine.factory import DependencyFactory, EngineDependencies
from engine.exceptions import (
    AggregationError,
    CriticEvaluationError,
    DetectorExecutionError,
    EleanorV8Exception,
    EvidenceRecordingError,
    GovernanceEvaluationError,
    PrecedentRetrievalError,
    RouterSelectionError,
    UncertaintyComputationError,
)
from engine.protocols import (
    AggregatorProtocol,
    CriticProtocol,
    DetectorEngineProtocol,
    EvidenceRecorderProtocol,
    PrecedentEngineProtocol,
    PrecedentRetrieverProtocol,
    ReviewTriggerEvaluatorProtocol,
    RouterProtocol,
    UncertaintyEngineProtocol,
)
from engine.schemas.pipeline_types import (
    AggregationOutput,
    CriticResult,
    CriticResultsMap,
    PrecedentAlignmentResult,
    PrecedentRetrievalResult,
    UncertaintyResult,
    ViolationEntry,
)
from engine.utils.critic_names import canonicalize_critic_map

# Governance
from governance.review_triggers import Case
from governance.review_packets import build_review_packet
from replay_store import store_review_packet


# ---------------------------------------------------------
# Dependency Resolution
# ---------------------------------------------------------

def _resolve_router_backend(router_backend: Optional[Any]) -> RouterProtocol:
    if router_backend is None:
        return DependencyFactory.create_router()
    if isinstance(router_backend, str):
        return DependencyFactory.create_router(backend=router_backend)
    if inspect.isclass(router_backend):
        return cast(RouterProtocol, router_backend())
    if callable(router_backend):
        return cast(RouterProtocol, router_backend())
    return cast(RouterProtocol, router_backend)


def _resolve_dependencies(
    *,
    config: "EngineConfig",
    dependencies: Optional[EngineDependencies],
    evidence_recorder: Optional[EvidenceRecorderProtocol],
    detector_engine: Optional[DetectorEngineProtocol],
    precedent_engine: Optional[PrecedentEngineProtocol],
    precedent_retriever: Optional[PrecedentRetrieverProtocol],
    uncertainty_engine: Optional[UncertaintyEngineProtocol],
    aggregator: Optional[AggregatorProtocol],
    router_backend: Optional[Any],
    critics: Optional[Dict[str, CriticProtocol]],
    critic_models: Optional[Dict[str, Any]],
    review_trigger_evaluator: Optional[ReviewTriggerEvaluatorProtocol],
) -> EngineDependencies:
    if dependencies is not None:
        return dependencies

    router_backend_name = router_backend if isinstance(router_backend, str) else None

    deps = DependencyFactory.create_all_dependencies(
        router_backend=router_backend_name,
        jsonl_evidence_path=config.jsonl_evidence_path,
        critics=critics,
        critic_models=critic_models,
        enable_precedent=config.enable_precedent_analysis,
        enable_uncertainty=config.enable_reflection,
    )

    if router_backend is not None and not isinstance(router_backend, str):
        deps.router = _resolve_router_backend(router_backend)
    if detector_engine is not None:
        deps.detector_engine = detector_engine
    if evidence_recorder is not None:
        deps.evidence_recorder = evidence_recorder
    if precedent_engine is not None:
        deps.precedent_engine = precedent_engine
    if precedent_retriever is not None:
        deps.precedent_retriever = precedent_retriever
    if uncertainty_engine is not None:
        deps.uncertainty_engine = uncertainty_engine
    if aggregator is not None:
        deps.aggregator = aggregator
    if review_trigger_evaluator is not None:
        deps.review_trigger_evaluator = review_trigger_evaluator

    return deps


# ---------------------------------------------------------
# Engine Configuration
# ---------------------------------------------------------

class EngineConfig(BaseModel):
    detail_level: int = 2
    max_concurrency: int = 6
    timeout_seconds: float = 10.0

    enable_reflection: bool = True
    enable_drift_check: bool = True
    enable_precedent_analysis: bool = True

    jsonl_evidence_path: Optional[str] = "evidence.jsonl"


# ---------------------------------------------------------
# Output Models
# ---------------------------------------------------------

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


class EngineForensicData(BaseModel):
    detector_metadata: Dict[str, Any] = Field(default_factory=dict)
    uncertainty_graph: UncertaintyResult = Field(default_factory=dict)  # type: ignore[arg-type]
    precedent_alignment: PrecedentAlignmentResult = Field(default_factory=dict)  # type: ignore[arg-type]
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
    forensic: Optional[EngineForensicData] = None


# ---------------------------------------------------------
# ELEANOR ENGINE V8
# ---------------------------------------------------------

class EleanorEngineV8:

    def __init__(
        self,
        *,
        config: Optional[EngineConfig] = None,
        evidence_recorder: Optional[EvidenceRecorderProtocol] = None,
        detector_engine: Optional[DetectorEngineProtocol] = None,
        precedent_engine: Optional[PrecedentEngineProtocol] = None,
        precedent_retriever: Optional[PrecedentRetrieverProtocol] = None,
        uncertainty_engine: Optional[UncertaintyEngineProtocol] = None,
        aggregator: Optional[AggregatorProtocol] = None,
        router_backend: Optional[Any] = None,
        critics: Optional[Dict[str, CriticProtocol]] = None,
        critic_models: Optional[Dict[str, Any]] = None,
        review_trigger_evaluator: Optional[ReviewTriggerEvaluatorProtocol] = None,
        dependencies: Optional[EngineDependencies] = None,
        error_monitor: Optional[Callable[[Exception, Dict[str, Any]], None]] = None,
    ):
        self.config = config or EngineConfig()
        self.instance_id = str(uuid.uuid4())

        deps = _resolve_dependencies(
            config=self.config,
            dependencies=dependencies,
            evidence_recorder=evidence_recorder,
            detector_engine=detector_engine,
            precedent_engine=precedent_engine,
            precedent_retriever=precedent_retriever,
            uncertainty_engine=uncertainty_engine,
            aggregator=aggregator,
            router_backend=router_backend,
            critics=critics,
            critic_models=critic_models,
            review_trigger_evaluator=review_trigger_evaluator,
        )

        self.router = deps.router
        self.critics = canonicalize_critic_map(deps.critics or {})
        self.critic_models = deps.critic_models or {}
        self.detector_engine = deps.detector_engine
        self.recorder = deps.evidence_recorder
        self.precedent_engine = deps.precedent_engine
        self.precedent_retriever = deps.precedent_retriever
        self.uncertainty_engine = deps.uncertainty_engine
        self.aggregator = deps.aggregator
        self.review_trigger_evaluator = deps.review_trigger_evaluator
        self.error_monitor = error_monitor

        # Concurrency
        self.semaphore = asyncio.Semaphore(self.config.max_concurrency)

        # Diagnostics
        self.timings: Dict[str, float] = {}
        self.router_diagnostics: Dict[str, Any] = {}

        print(f"[ELEANOR ENGINE] Initialized V8 engine {self.instance_id}")

    # -----------------------------------------------------
    # ERROR HANDLING
    # -----------------------------------------------------
    def _emit_error(
        self,
        exc: Exception,
        *,
        stage: str,
        trace_id: Optional[str] = None,
        critic: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "stage": stage,
            "trace_id": trace_id,
            "critic": critic,
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
        if context is not None:
            payload["context_keys"] = list(context.keys())
        if isinstance(exc, EleanorV8Exception):
            payload.update(exc.details)
        if extra:
            payload.update(extra)

        logger.error("engine_error", extra=payload, exc_info=True)

        if self.error_monitor:
            try:
                self.error_monitor(exc, payload)
            except Exception:
                logger.debug("Error monitor hook failed", exc_info=True)

    def _build_critic_error_result(
        self,
        critic_name: str,
        error: Exception,
        duration_ms: Optional[float] = None,
    ) -> CriticResult:
        return {
            "critic": critic_name,
            "severity": 0.0,
            "score": 0.0,
            "violations": [],
            "justification": f"critic_error:{type(error).__name__}",
            "duration_ms": duration_ms if duration_ms is not None else 0.0,
            "error": str(error),
        }

    def _build_aggregation_fallback(
        self,
        model_response: str,
        precedent_data: Optional[PrecedentAlignmentResult],
        uncertainty_data: Optional[UncertaintyResult],
        error: Exception,
    ) -> AggregationOutput:
        return {
            "decision": "requires_human_review",
            "final_output": model_response,
            "score": {"average_severity": 0.0, "total_severity": 0.0},
            "rights_impacted": [],
            "dissent": None,
            "precedent": precedent_data or {},
            "uncertainty": uncertainty_data or {},
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }


    # -----------------------------------------------------
    # MODEL ROUTING
    # -----------------------------------------------------
    async def _run_detectors(self, text: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.detector_engine:
            return None

        start = asyncio.get_event_loop().time()
        try:
            signals = await self.detector_engine.detect_all(text, context)
            summary = self.detector_engine.aggregate_signals(signals)
        except Exception as exc:
            raise DetectorExecutionError(
                "Detector execution failed",
                details={"error": str(exc)},
            ) from exc
        end = asyncio.get_event_loop().time()
        self.timings["detectors_ms"] = (end - start) * 1000

        converted_signals = {
            name: (sig.model_dump() if hasattr(sig, "model_dump") else sig)
            for name, sig in signals.items()
        }
        return {
            **summary,
            "signals": converted_signals,
        }

    async def _select_model(self, text: str, context: dict) -> Dict[str, Any]:
        start = asyncio.get_event_loop().time()

        try:
            call = self.router.route(text=text, context=context)
            router_result = await call if inspect.isawaitable(call) else call
        except RouterSelectionError:
            end = asyncio.get_event_loop().time()
            self.timings["router_selection_ms"] = (end - start) * 1000
            raise
        except Exception as exc:
            end = asyncio.get_event_loop().time()
            self.timings["router_selection_ms"] = (end - start) * 1000
            raise RouterSelectionError(
                "Router failed to select a model",
                details={"error": str(exc)},
            ) from exc

        if not router_result or router_result.get("response_text") is None:
            end = asyncio.get_event_loop().time()
            self.timings["router_selection_ms"] = (end - start) * 1000
            raise RouterSelectionError(
                "Router returned no response",
                details={"router_result": router_result},
            )

        end = asyncio.get_event_loop().time()
        self.timings["router_selection_ms"] = (end - start) * 1000
        self.router_diagnostics = router_result.get("diagnostics", {})

        model_info = {
            "model_name": router_result.get("model_name"),
            "model_version": router_result.get("model_version"),
            "router_selection_reason": router_result.get("reason"),
            "health_score": router_result.get("health_score"),
            "cost_estimate": router_result.get("cost"),
        }

        return {
            "model_info": model_info,
            "response_text": router_result.get("response_text") or "",
        }


    # -----------------------------------------------------
    # CRITIC EXECUTION
    # -----------------------------------------------------
    async def _run_single_critic(
        self,
        name: str,
        critic_ref: Any,
        model_response: str,
        input_text: str,
        context: dict,
        trace_id: str,
    ) -> CriticResult:
        async with self.semaphore:
            start = asyncio.get_event_loop().time()

            # Instantiate critic
            critic = critic_ref if not inspect.isclass(critic_ref) else critic_ref()

            # Choose model for this critic: explicit binding overrides router output
            bound_adapter = None if context.get("force_model_output") else self.critic_models.get(name)
            model_adapter: Any = None

            if bound_adapter is None:
                class _StaticModelResponse:
                    def __init__(self, response: str):
                        self.response = response

                    async def generate(self, prompt: str, context: Optional[Dict[str, Any]] = None):
                        return self.response

                model_adapter = _StaticModelResponse(model_response)
            else:
                if hasattr(bound_adapter, "generate"):
                    model_adapter = bound_adapter
                elif callable(bound_adapter):
                    class _BoundCallable:
                        def __init__(self, fn):
                            self.fn = fn
                        async def generate(self, prompt: str, context: Optional[Dict[str, Any]] = None):
                            res = self.fn(prompt, context=context) if "context" in inspect.signature(self.fn).parameters else self.fn(prompt)
                            return await res if inspect.isawaitable(res) else res
                    model_adapter = _BoundCallable(bound_adapter)
                else:
                    class _StaticModelFallback:
                        def __init__(self, response: str):
                            self.response = response
                        async def generate(self, prompt: str, context: Optional[Dict[str, Any]] = None):
                            return self.response
                    model_adapter = _StaticModelFallback(model_response)

            try:
                evaluation = critic.evaluate(model_adapter, input_text=input_text, context=context)
                evaluation_result = await evaluation if inspect.isawaitable(evaluation) else evaluation
            except Exception as exc:
                end = asyncio.get_event_loop().time()
                duration_ms = (end - start) * 1000
                failure_result = self._build_critic_error_result(
                    critic_name=name,
                    error=exc,
                    duration_ms=duration_ms,
                )
                try:
                    await self.recorder.record(
                        critic=name,
                        rule_id=str(name),
                        severity="INFO",
                        violation_description=str(failure_result.get("justification")),
                        confidence=0.0,
                        mitigation=None,
                        redundancy_group=None,
                        detector_metadata={"error": str(exc)},
                        context=context,
                        raw_text=model_response,
                        trace_id=trace_id,
                    )
                except Exception as record_exc:
                    error = EvidenceRecordingError(
                        "Evidence recording failed",
                        details={"error": str(record_exc), "critic": name},
                    )
                    self._emit_error(error, stage="evidence", trace_id=trace_id, critic=name)

                raise CriticEvaluationError(
                    critic_name=name,
                    message=str(exc),
                    trace_id=trace_id,
                    details={
                        "duration_ms": duration_ms,
                        "error_type": type(exc).__name__,
                        "result": failure_result,
                    },
                ) from exc

            end = asyncio.get_event_loop().time()

            evaluation_result = cast(CriticResult, evaluation_result or {})
            evaluation_result["critic"] = name
            evaluation_result["duration_ms"] = (end - start) * 1000

            # Evidence logging (best-effort)
            try:
                severity_score = float(evaluation_result.get("score", 0.0))
                severity_label = critic.severity(severity_score) if hasattr(critic, "severity") else "INFO"
                violations_list = evaluation_result.get("violations") or []
                violation_description = evaluation_result.get("justification") or (
                    violations_list[0] if violations_list else f"{name} check"
                )

                await self.recorder.record(
                    critic=name,
                    rule_id=str(evaluation_result.get("principle") or name),
                    severity=severity_label,
                    violation_description=str(violation_description),
                    confidence=float(evaluation_result.get("score", 0.0)),
                    mitigation=None,
                    redundancy_group=None,
                    detector_metadata=evaluation_result.get("evidence") or {},
                    context=context,
                    raw_text=model_response,
                    trace_id=trace_id,
                )
            except Exception:
                error = EvidenceRecordingError(
                    "Evidence recording failed",
                    details={"critic": name},
                )
                self._emit_error(error, stage="evidence", trace_id=trace_id, critic=name)

            return cast(CriticResult, evaluation_result)


    async def _run_critics_parallel(
        self,
        model_response: str,
        context: dict,
        trace_id: str,
        input_text: Optional[str] = None,
    ) -> CriticResultsMap:
        input_text = context.get("input_text_override") or input_text or ""
        if not isinstance(input_text, str):
            input_text = str(input_text)
        critic_items = list(self.critics.items())
        tasks = [
            asyncio.create_task(
                self._run_single_critic(name, critic_ref, model_response, input_text, context, trace_id)
            )
            for name, critic_ref in critic_items
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        output: CriticResultsMap = {}
        for (critic_name, _), result in zip(critic_items, results):
            if isinstance(result, CriticEvaluationError):
                self._emit_error(
                    result,
                    stage="critic",
                    trace_id=trace_id,
                    critic=critic_name,
                    context=context,
                )
                fallback = result.details.get("result") if isinstance(result.details, dict) else None
                output[critic_name] = fallback or self._build_critic_error_result(critic_name, result)
                continue
            if isinstance(result, Exception):
                error = CriticEvaluationError(
                    critic_name=critic_name,
                    message=str(result),
                    trace_id=trace_id,
                    details={"error_type": type(result).__name__},
                )
                self._emit_error(
                    error,
                    stage="critic",
                    trace_id=trace_id,
                    critic=critic_name,
                    context=context,
                )
                output[critic_name] = self._build_critic_error_result(critic_name, result)
                continue
            if not isinstance(result, dict):
                output[critic_name] = self._build_critic_error_result(
                    critic_name,
                    Exception("critic_result_invalid"),
                )
                continue
            critic_result = cast(CriticResult, result)
            output[critic_result.get("critic", critic_name)] = critic_result
        return output


    # -----------------------------------------------------
    # PRECEDENT + UNCERTAINTY + AGGREGATION
    # -----------------------------------------------------

    async def _run_precedent_alignment(
        self,
        critic_results: CriticResultsMap,
        trace_id: str,
        text: str = "",
    ) -> Optional[PrecedentAlignmentResult]:
        if not self.precedent_engine:
            return None
        start = asyncio.get_event_loop().time()
        cases: List[Dict[str, Any]] = []
        query_embedding: List[float] = []
        retrieval_meta: Optional[PrecedentRetrievalResult] = None

        if self.precedent_retriever:
            try:
                retrieval_meta = self.precedent_retriever.retrieve(text, list(critic_results.values()))
                cases = cast(
                    List[Dict[str, Any]],
                    retrieval_meta.get("precedent_cases")
                    or retrieval_meta.get("cases")
                    or [],
                )
                query_embedding = cast(
                    List[float],
                    retrieval_meta.get("query_embedding") or [],
                )
            except Exception as exc:
                raise PrecedentRetrievalError(
                    "Precedent retrieval failed",
                    details={"error": str(exc), "trace_id": trace_id},
                ) from exc

        analyze_fn = getattr(self.precedent_engine, "analyze", None)
        try:
            if analyze_fn:
                out = analyze_fn(
                    critics=critic_results,
                    precedent_cases=cases,
                    query_embedding=query_embedding,
                )
            else:
                out = None
        except Exception as exc:
            raise PrecedentRetrievalError(
                "Precedent alignment failed",
                details={"error": str(exc), "trace_id": trace_id},
            ) from exc

        out_result: Optional[PrecedentAlignmentResult] = cast(
            Optional[PrecedentAlignmentResult],
            out,
        )
        if retrieval_meta:
            out_result = cast(
                PrecedentAlignmentResult,
                {**(out_result or {}), "retrieval": retrieval_meta},
            )

        end = asyncio.get_event_loop().time()
        self.timings["precedent_alignment_ms"] = (end - start) * 1000
        return out_result


    async def _run_uncertainty_engine(
        self,
        precedent_alignment: Optional[PrecedentAlignmentResult],
        critic_results: CriticResultsMap,
        model_name: str = "unknown-model",
    ) -> Optional[UncertaintyResult]:
        if not self.uncertainty_engine:
            return None
        start = asyncio.get_event_loop().time()
        compute_fn = getattr(self.uncertainty_engine, "compute", None) or getattr(self.uncertainty_engine, "evaluate", None)
        if not compute_fn:
            return None

        try:
            out = compute_fn(
                critics=critic_results,
                model_used=model_name,
                precedent_alignment=precedent_alignment or {},
            )
            if inspect.isawaitable(out):
                out = await out
        except Exception as exc:
            raise UncertaintyComputationError(
                "Uncertainty computation failed",
                details={"error": str(exc)},
            ) from exc
        end = asyncio.get_event_loop().time()
        self.timings["uncertainty_engine_ms"] = (end - start) * 1000
        return cast(UncertaintyResult, out)


    async def _aggregate_results(
        self,
        critic_results: CriticResultsMap,
        model_response: str,
        precedent_data: Optional[PrecedentAlignmentResult] = None,
        uncertainty_data: Optional[UncertaintyResult] = None,
    ) -> AggregationOutput:
        if not self.aggregator:
            raise AggregationError("AggregatorV8 not available")
        start = asyncio.get_event_loop().time()
        try:
            agg_result = self.aggregator.aggregate(
                critics=critic_results,
                precedent=precedent_data or {},
                uncertainty=uncertainty_data or {},
                model_output=model_response,
            )
            out = await agg_result if inspect.isawaitable(agg_result) else agg_result
        except Exception as exc:
            raise AggregationError(
                "Aggregation failed",
                details={"error": str(exc)},
            ) from exc

        end = asyncio.get_event_loop().time()
        self.timings["aggregation_ms"] = (end - start) * 1000
        return cast(AggregationOutput, out)

    # -----------------------------------------------------
    # GOVERNANCE HOOKS
    # -----------------------------------------------------
    def _calculate_critic_disagreement(self, critic_outputs: CriticResultsMap) -> float:
        severities: List[float] = []
        for critic_data in critic_outputs.values():
            if not isinstance(critic_data, dict):
                continue
            val: Any = critic_data.get("severity")
            if val is None:
                val = critic_data.get("score")
            try:
                severities.append(float(val))
            except (TypeError, ValueError):
                continue

        if len(severities) < 2:
            return 0.0

        mean = sum(severities) / len(severities)
        variance = sum((s - mean) ** 2 for s in severities) / len(severities)
        return min(1.0, variance / 1.56)

    def _collect_citations(self, critic_outputs: CriticResultsMap) -> Dict[str, Any]:
        citations = {}
        for critic_name, critic_data in critic_outputs.items():
            if isinstance(critic_data, dict) and "precedent_refs" in critic_data:
                citations[critic_name] = critic_data["precedent_refs"]
        return citations

    def _collect_uncertainty_flags(self, uncertainty_data: Optional[UncertaintyResult]) -> List[str]:
        flags: List[str] = []
        if not uncertainty_data:
            return flags

        if uncertainty_data.get("needs_escalation"):
            flags.append("needs_escalation")

        overall = uncertainty_data.get("overall_uncertainty")
        try:
            if overall is not None and float(overall) >= 0.65:
                flags.append("high_overall_uncertainty")
        except (TypeError, ValueError):
            pass

        return flags

    def _build_case_for_review(
        self,
        trace_id: str,
        context: Dict[str, Any],
        aggregated: AggregationOutput,
        critic_results: CriticResultsMap,
        precedent_data: Optional[PrecedentAlignmentResult],
        uncertainty_data: Optional[UncertaintyResult],
    ) -> Case:
        aggregated = aggregated or {}
        critic_results = critic_results or {}

        severity_raw: Any = (aggregated.get("score") or {}).get("average_severity", 0.0)
        try:
            severity = float(severity_raw)
        except (TypeError, ValueError):
            severity = 0.0

        critic_severities: List[float] = []
        for critic_data in critic_results.values():
            if not isinstance(critic_data, dict):
                continue
            val: Any = critic_data.get("severity")
            if val is None:
                val = critic_data.get("score")
            try:
                critic_severities.append(float(val))
            except (TypeError, ValueError):
                continue
        if not severity and critic_severities:
            severity = max(critic_severities)

        uncertainty_flags = self._collect_uncertainty_flags(uncertainty_data)
        case_uncertainty = SimpleNamespace(flags=uncertainty_flags)

        case_obj = Case(
            severity=severity,
            critic_disagreement=self._calculate_critic_disagreement(critic_results),
            novel_precedent=bool((precedent_data or {}).get("novel", False)),
            rights_impacted=aggregated.get("rights_impacted", []),
            uncertainty_flags=uncertainty_flags,
            uncertainty=case_uncertainty,
        )

        # Attach additional fields used by packet builders
        for key, value in {
            "id": trace_id,
            "domain": context.get("domain", "general"),
            "critic_outputs": critic_results,
            "aggregator_summary": aggregated.get("final_output", "") or "",
            "dissent": aggregated.get("dissent"),
            "citations": self._collect_citations(critic_results),
        }.items():
            setattr(case_obj, key, value)

        return case_obj

    def _run_governance_review_gate(self, case: Case):
        try:
            review_decision = self.review_trigger_evaluator.evaluate(case)

            if review_decision.get("review_required"):
                review_packet = build_review_packet(case, review_decision)
                store_review_packet(review_packet)

                setattr(
                    case,
                    "governance_flags",
                    {
                        "human_review_required": True,
                        "review_triggers": review_decision.get("triggers", []),
                    },
                )
            else:
                setattr(
                    case,
                    "governance_flags",
                    {
                        "human_review_required": False,
                    },
                )
        except Exception as review_exc:
            setattr(
                case,
                "governance_flags",
                {
                    "human_review_required": False,
                    "error": str(review_exc),
                },
            )
            raise GovernanceEvaluationError(
                "Governance review gate failed",
                details={"error": str(review_exc)},
            ) from review_exc
    # -----------------------------------------------------
    # FULL STRUCTURED OUTPUT MODE — run()
    # -----------------------------------------------------
    async def run(
        self,
        text: str,
        context: Optional[dict] = None,
        *,
        detail_level: Optional[int] = None,
        trace_id: Optional[str] = None,
    ) -> EngineResult:

        context = context or {}
        trace_id = trace_id or str(uuid.uuid4())
        level = detail_level or self.config.detail_level

        pipeline_start = asyncio.get_event_loop().time()

        try:
            detector_payload = await self._run_detectors(text, context)
        except DetectorExecutionError as exc:
            self._emit_error(exc, stage="detectors", trace_id=trace_id, context=context)
            detector_payload = None
        except Exception as exc:
            detector_error = DetectorExecutionError(
                "Detector execution failed",
                details={"error": str(exc)},
            )
            self._emit_error(
                detector_error,
                stage="detectors",
                trace_id=trace_id,
                context=context,
            )
            detector_payload = None
        if detector_payload:
            context = {**context, "detectors": detector_payload}

        # Step 1: Model Routing (or model_output override)
        if context.get("skip_router"):
            raw_output = context.get("model_output")
            if raw_output is None:
                raise ValueError("model_output is required when skip_router is true")
            if isinstance(raw_output, str):
                model_response = raw_output
            else:
                model_response = json.dumps(raw_output, ensure_ascii=True, default=str)
            meta = context.get("model_metadata") or {}
            model_info = {
                "model_name": meta.get("model_id") or meta.get("model_name") or "external",
                "model_version": meta.get("model_version") or "",
                "router_selection_reason": "model_output_override",
                "health_score": None,
                "cost_estimate": None,
            }
            self.router_diagnostics = {"skipped": True, "reason": "model_output_override"}
        else:
            try:
                router_data = await self._select_model(text, context)
                model_info = router_data["model_info"]
                model_response = router_data["response_text"]
            except RouterSelectionError as exc:
                self._emit_error(exc, stage="router", trace_id=trace_id, context=context)
                model_info = {
                    "model_name": "router_error",
                    "model_version": None,
                    "router_selection_reason": "router_failure",
                    "health_score": 0.0,
                    "cost_estimate": None,
                }
                model_response = ""
                diagnostics = exc.details.get("diagnostics") if isinstance(exc, EleanorV8Exception) else None
                self.router_diagnostics = diagnostics or {"error": str(exc)}

        model_name_value = str(model_info.get("model_name") or "unknown-model")
        model_info["model_name"] = model_name_value
        engine_model_info = EngineModelInfo(
            model_name=model_name_value,
            model_version=cast(Optional[str], model_info.get("model_version")),
            router_selection_reason=cast(
                Optional[str],
                model_info.get("router_selection_reason"),
            ),
            cost_estimate=cast(Optional[Dict[str, Any]], model_info.get("cost_estimate")),
            health_score=cast(Optional[float], model_info.get("health_score")),
        )

        # Step 2: Critics (parallel)
        critic_results = await self._run_critics_parallel(
            model_response=model_response,
            input_text=text,
            context=context,
            trace_id=trace_id,
        )

        # Step 3: Precedent Alignment
        precedent_data = None
        if self.config.enable_precedent_analysis:
            try:
                precedent_data = await self._run_precedent_alignment(
                    critic_results=critic_results,
                    trace_id=trace_id,
                    text=text,
                )
            except PrecedentRetrievalError as exc:
                self._emit_error(exc, stage="precedent", trace_id=trace_id, context=context)
                precedent_data = None

        # Step 4: Uncertainty Modeling
        uncertainty_data = None
        if self.config.enable_reflection and self.uncertainty_engine:
            try:
                uncertainty_data = await self._run_uncertainty_engine(
                    precedent_alignment=precedent_data,
                    critic_results=critic_results,
                    model_name=engine_model_info.model_name,
                )
            except UncertaintyComputationError as exc:
                self._emit_error(exc, stage="uncertainty", trace_id=trace_id, context=context)
                uncertainty_data = None

        # Step 5: Aggregation / Constitutional Fusion
        try:
            aggregated = await self._aggregate_results(
                critic_results=critic_results,
                model_response=model_response,
                precedent_data=precedent_data,
                uncertainty_data=uncertainty_data,
            )
        except AggregationError as exc:
            self._emit_error(exc, stage="aggregation", trace_id=trace_id, context=context)
            aggregated = self._build_aggregation_fallback(
                model_response,
                precedent_data,
                uncertainty_data,
                exc,
            )

        # --- Governance: Human Review Gate (Non-Blocking) ---
        try:
            case = self._build_case_for_review(
                trace_id=trace_id,
                context=context,
                aggregated=aggregated,
                critic_results=critic_results,
                precedent_data=precedent_data,
                uncertainty_data=uncertainty_data,
            )
            self._run_governance_review_gate(case)
        except GovernanceEvaluationError as review_exc:
            self._emit_error(review_exc, stage="governance", trace_id=trace_id, context=context)
        except Exception as review_exc:
            governance_error = GovernanceEvaluationError(
                "Governance review gate failed",
                details={"error": str(review_exc)},
            )
            self._emit_error(
                governance_error,
                stage="governance",
                trace_id=trace_id,
                context=context,
            )

        # Timing
        pipeline_end = asyncio.get_event_loop().time()
        self.timings["total_pipeline_ms"] = (pipeline_end - pipeline_start) * 1000

        # Evidence buffer
        buffer = getattr(self.recorder, "buffer", None)
        evidence_count = len(buffer) if buffer else None

        # Base kwargs
        critic_findings = {
            k: EngineCriticFinding(
                critic=k,
                violations=list(v.get("violations", [])),
                duration_ms=v.get("duration_ms"),
                evaluated_rules=cast(Optional[List[str]], v.get("evaluated_rules")),
            )
            for k, v in critic_results.items()
        }
        base_result = EngineResult(
            trace_id=trace_id,
            output_text=aggregated.get("final_output") or model_response,
            model_info=engine_model_info,
            critic_findings=critic_findings,
            aggregated=aggregated,
            uncertainty=uncertainty_data,
            precedent_alignment=precedent_data,
            evidence_count=evidence_count,
        )

        # ---------------------------
        # DETAIL LEVEL 1
        # ---------------------------
        if level == 1:
            return EngineResult(
                trace_id=trace_id,
                output_text=aggregated.get("final_output") or model_response,
                model_info=engine_model_info,
            )

        # ---------------------------
        # DETAIL LEVEL 2
        # ---------------------------
        if level == 2:
            return base_result

        # ---------------------------
        # DETAIL LEVEL 3 — FORENSIC
        # ---------------------------
        forensic_data = None
        if level == 3:
            forensic_buffer = buffer[-200:] if buffer else []

            forensic_data = EngineForensicData(
                detector_metadata=detector_payload or {},
                uncertainty_graph=uncertainty_data or {},
                precedent_alignment=precedent_data or {},
                router_diagnostics=self.router_diagnostics,
                timings=self.timings,
                evidence_references=[
                    r.dict() if hasattr(r, "dict") else r for r in forensic_buffer
                ],
            )

            return EngineResult(
                trace_id=base_result.trace_id,
                output_text=base_result.output_text,
                model_info=base_result.model_info,
                critic_findings=base_result.critic_findings,
                aggregated=base_result.aggregated,
                uncertainty=base_result.uncertainty,
                precedent_alignment=base_result.precedent_alignment,
                evidence_count=base_result.evidence_count,
                forensic=forensic_data,
            )

        raise ValueError(f"Invalid detail_level: {level}")


    # -----------------------------------------------------
    # STREAMING OUTPUT MODE — run_stream()
    # -----------------------------------------------------
    async def run_stream(
        self,
        text: str,
        context: Optional[dict] = None,
        *,
        detail_level: Optional[int] = None,
        trace_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:

        context = context or {}
        trace_id = trace_id or str(uuid.uuid4())
        level = detail_level or self.config.detail_level

        pipeline_start = asyncio.get_event_loop().time()

        try:
            detector_payload = await self._run_detectors(text, context)
        except DetectorExecutionError as exc:
            self._emit_error(exc, stage="detectors", trace_id=trace_id, context=context)
            detector_payload = None
        except Exception as exc:
            detector_error = DetectorExecutionError(
                "Detector execution failed",
                details={"error": str(exc)},
            )
            self._emit_error(
                detector_error,
                stage="detectors",
                trace_id=trace_id,
                context=context,
            )
            detector_payload = None
        if detector_payload:
            context = {**context, "detectors": detector_payload}
            yield {
                "event": "detectors_complete",
                "trace_id": trace_id,
                "data": {"summary": {k: v for k, v in detector_payload.items() if k != "signals"}},
            }

        # Step 1: Model Routing (or model_output override)
        if context.get("skip_router"):
            raw_output = context.get("model_output")
            if raw_output is None:
                raise ValueError("model_output is required when skip_router is true")
            if isinstance(raw_output, str):
                model_response = raw_output
            else:
                model_response = json.dumps(raw_output, ensure_ascii=True, default=str)
            meta = context.get("model_metadata") or {}
            model_info = {
                "model_name": meta.get("model_id") or meta.get("model_name") or "external",
                "model_version": meta.get("model_version") or "",
                "router_selection_reason": "model_output_override",
                "health_score": None,
                "cost_estimate": None,
            }
            self.router_diagnostics = {"skipped": True, "reason": "model_output_override"}
        else:
            try:
                router_data = await self._select_model(text, context)
                model_info = router_data["model_info"]
                model_response = router_data["response_text"]
            except RouterSelectionError as exc:
                self._emit_error(exc, stage="router", trace_id=trace_id, context=context)
                model_info = {
                    "model_name": "router_error",
                    "model_version": None,
                    "router_selection_reason": "router_failure",
                    "health_score": 0.0,
                    "cost_estimate": None,
                }
                model_response = ""
                diagnostics = exc.details.get("diagnostics") if isinstance(exc, EleanorV8Exception) else None
                self.router_diagnostics = diagnostics or {"error": str(exc)}

        model_name_value = str(model_info.get("model_name") or "unknown-model")
        model_info["model_name"] = model_name_value
        engine_model_info = EngineModelInfo(
            model_name=model_name_value,
            model_version=cast(Optional[str], model_info.get("model_version")),
            router_selection_reason=cast(
                Optional[str],
                model_info.get("router_selection_reason"),
            ),
            cost_estimate=cast(Optional[Dict[str, Any]], model_info.get("cost_estimate")),
            health_score=cast(Optional[float], model_info.get("health_score")),
        )

        yield {
            "event": "router_selected",
            "trace_id": trace_id,
            "model_info": model_info,
            "timings": {"router_selection_ms": self.timings.get("router_selection_ms")},
        }

        yield {
            "event": "model_response",
            "trace_id": trace_id,
            "text": model_response,
        }

        # Step 2: Critics
        yield {
            "event": "critics_start",
            "trace_id": trace_id,
            "critics": list(self.critics.keys()),
        }

        critic_results: CriticResultsMap = {}

        critic_input_text = context.get("input_text_override") or text
        if not isinstance(critic_input_text, str):
            critic_input_text = str(critic_input_text)

        async def _run_and_return(name, critic_ref):
            return await self._run_single_critic(name, critic_ref, model_response, critic_input_text, context, trace_id)

        task_map: Dict[asyncio.Future[Any], str] = {}
        for name, critic_ref in self.critics.items():
            task = asyncio.create_task(_run_and_return(name, critic_ref))
            task_map[task] = name

        for future in asyncio.as_completed(task_map):
            critic_name = task_map[future]
            try:
                res = await future
            except CriticEvaluationError as exc:
                self._emit_error(
                    exc,
                    stage="critic",
                    trace_id=trace_id,
                    critic=critic_name,
                    context=context,
                )
                fallback = exc.details.get("result") if isinstance(exc.details, dict) else None
                res = fallback or self._build_critic_error_result(critic_name, exc)
            except Exception as exc:
                critic_error = CriticEvaluationError(
                    critic_name=critic_name,
                    message=str(exc),
                    trace_id=trace_id,
                    details={"error_type": type(exc).__name__},
                )
                self._emit_error(
                    critic_error,
                    stage="critic",
                    trace_id=trace_id,
                    critic=critic_name,
                    context=context,
                )
                res = self._build_critic_error_result(critic_name, exc)
            res = cast(CriticResult, res)
            critic_results[res.get("critic", critic_name)] = res

            yield {
                "event": "critic_result",
                "trace_id": trace_id,
                "critic": res.get("critic", critic_name),
                "violations": list(res.get("violations", [])),
                "duration_ms": res.get("duration_ms"),
                "evaluated_rules": res.get("evaluated_rules"),
            }

        yield {
            "event": "critics_complete",
            "trace_id": trace_id,
        }

        # Step 3: Precedent
        precedent_data = None
        if self.config.enable_precedent_analysis and self.precedent_engine:
            try:
                precedent_data = await self._run_precedent_alignment(critic_results, trace_id, text)
                yield {
                    "event": "precedent_alignment",
                    "trace_id": trace_id,
                    "data": precedent_data,
                }
            except PrecedentRetrievalError as exc:
                self._emit_error(exc, stage="precedent", trace_id=trace_id, context=context)
                precedent_data = None
                yield {
                    "event": "precedent_alignment",
                    "trace_id": trace_id,
                    "data": {"error": str(exc)},
                }

        # Step 4: Uncertainty
        uncertainty_data = None
        if self.config.enable_reflection and self.uncertainty_engine:
            try:
                uncertainty_data = await self._run_uncertainty_engine(
                    precedent_alignment=precedent_data,
                    critic_results=critic_results,
                    model_name=engine_model_info.model_name,
                )
                yield {
                    "event": "uncertainty",
                    "trace_id": trace_id,
                    "data": uncertainty_data,
                }
            except UncertaintyComputationError as exc:
                self._emit_error(exc, stage="uncertainty", trace_id=trace_id, context=context)
                uncertainty_data = None
                yield {
                    "event": "uncertainty",
                    "trace_id": trace_id,
                    "data": {"error": str(exc)},
                }

        # Step 5: Aggregation
        try:
            aggregated = await self._aggregate_results(critic_results, model_response, precedent_data, uncertainty_data)
        except AggregationError as exc:
            self._emit_error(exc, stage="aggregation", trace_id=trace_id, context=context)
            aggregated = self._build_aggregation_fallback(
                model_response,
                precedent_data,
                uncertainty_data,
                exc,
            )

        yield {
            "event": "aggregation",
            "trace_id": trace_id,
            "data": aggregated,
        }

        # --- Governance: Human Review Gate (Non-Blocking) ---
        try:
            case = self._build_case_for_review(
                trace_id=trace_id,
                context=context,
                aggregated=aggregated,
                critic_results=critic_results,
                precedent_data=precedent_data,
                uncertainty_data=uncertainty_data,
            )
            self._run_governance_review_gate(case)
        except GovernanceEvaluationError as review_exc:
            self._emit_error(review_exc, stage="governance", trace_id=trace_id, context=context)
        except Exception as review_exc:
            governance_error = GovernanceEvaluationError(
                "Governance review gate failed",
                details={"error": str(review_exc)},
            )
            self._emit_error(
                governance_error,
                stage="governance",
                trace_id=trace_id,
                context=context,
            )

        # Final Output
        final_output = aggregated.get("final_output", "") if isinstance(aggregated, dict) else ""
        if not final_output:
            final_output = model_response

        pipeline_end = asyncio.get_event_loop().time()
        self.timings["total_pipeline_ms"] = (pipeline_end - pipeline_start) * 1000

        buffer = getattr(self.recorder, "buffer", None)
        forensic_evidence = buffer[-200:] if buffer else []

        base_final = {
            "event": "final_output",
            "trace_id": trace_id,
            "output_text": final_output,
        }

        if level == 1:
            yield base_final

        elif level == 2:
            yield {
                **base_final,
                "critic_findings": critic_results,
                "precedent_alignment": precedent_data,
                "uncertainty": uncertainty_data,
            }

        elif level == 3:
            yield {
                **base_final,
                "critic_findings": critic_results,
                "precedent_alignment": precedent_data,
                "uncertainty": uncertainty_data,
                "router_diagnostics": self.router_diagnostics,
                "timings": self.timings,
                "forensic_evidence": [
                    r.dict() if hasattr(r, "dict") else r for r in forensic_evidence
                ],
            }

        else:
            raise ValueError(f"Invalid detail_level: {level}")


    # -----------------------------------------------------
    # CONFIG LOADER
    # -----------------------------------------------------
    @staticmethod
    def load_config_from_yaml(path: str) -> EngineConfig:
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML config support.")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return EngineConfig(**data)


# ---------------------------------------------------------
# ENGINE FACTORY
# ---------------------------------------------------------

def create_engine(
    config: Optional[EngineConfig] = None,
    *,
    evidence_recorder: Optional[EvidenceRecorderProtocol] = None,
    detector_engine: Optional[DetectorEngineProtocol] = None,
    precedent_engine: Optional[PrecedentEngineProtocol] = None,
    precedent_retriever: Optional[PrecedentRetrieverProtocol] = None,
    uncertainty_engine: Optional[UncertaintyEngineProtocol] = None,
    aggregator: Optional[AggregatorProtocol] = None,
    router_backend: Optional[Any] = None,
    critics: Optional[Dict[str, CriticProtocol]] = None,
    critic_models: Optional[Dict[str, Any]] = None,
    review_trigger_evaluator: Optional[ReviewTriggerEvaluatorProtocol] = None,
    dependencies: Optional[EngineDependencies] = None,
    error_monitor: Optional[Callable[[Exception, Dict[str, Any]], None]] = None,
) -> EleanorEngineV8:

    engine = EleanorEngineV8(
        config=config,
        evidence_recorder=evidence_recorder,
        detector_engine=detector_engine,
        precedent_engine=precedent_engine,
        precedent_retriever=precedent_retriever,
        uncertainty_engine=uncertainty_engine,
        aggregator=aggregator,
        router_backend=router_backend,
        critics=critics,
        critic_models=critic_models,
        review_trigger_evaluator=review_trigger_evaluator,
        dependencies=dependencies,
        error_monitor=error_monitor,
    )

    print(f"[ELEANOR ENGINE] create_engine() → Engine instance ready: {engine.instance_id}")
    return engine


__all__ = [
    "EleanorEngineV8",
    "EngineConfig",
    "EngineResult",
    "EngineCriticFinding",
    "EngineModelInfo",
    "EngineForensicData",
    "create_engine",
]

print("[ELEANOR ENGINE] V8 Enterprise Engine module loaded successfully.")
