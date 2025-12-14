"""
ELEANOR V8 — Enterprise Constitutional Engine Runtime
Dual API (run + run_stream)
Dynamic Router Auto-Discovery
Full Evidence Recorder Integration
Precedent Alignment + Uncertainty Engine Hooks
Forensic Detail-Level Output Mode
"""

import asyncio
import importlib
import inspect
import uuid
from typing import Any, Dict, List, Optional, AsyncGenerator

from pydantic import BaseModel, Field

# Critics (V8 implementations)
from engine.critics.rights import RightsCriticV8
from engine.critics.risk import RiskCriticV8
from engine.critics.fairness import FairnessCriticV8
from engine.critics.pragmatics import PragmaticsCriticV8
from engine.critics.truth import TruthCriticV8
from engine.critics.autonomy import AutonomyCriticV8

# Detector Engine
from engine.detectors.engine import DetectorEngineV8

# Evidence Recorder
from engine.recorder.evidence_recorder import EvidenceRecorder

# Precedent Engine
try:
    from engine.precedent.alignment import PrecedentAlignmentEngineV8
    from engine.precedent.retrieval import PrecedentRetrievalV8
except Exception:
    PrecedentAlignmentEngineV8 = None
    PrecedentRetrievalV8 = None

# Uncertainty Engine
try:
    from engine.uncertainty.uncertainty import UncertaintyEngineV8
except Exception:
    UncertaintyEngineV8 = None

# Aggregator
try:
    from engine.aggregator.aggregator import AggregatorV8
except Exception:
    AggregatorV8 = None


# ---------------------------------------------------------
# Dynamic Router Loader
# ---------------------------------------------------------

def load_router_backend():
    """
    Prefer the packaged RouterV8; fall back to dynamic discovery to keep
    backward compatibility with alternative router modules.
    """
    try:
        from engine.router.router import RouterV8
        return RouterV8
    except Exception:
        pass

    candidates = [
        "engine.router",
        "engine.router.router",
        "orchestrator.router",
        "router",
    ]
    for mod in candidates:
        try:
            module = importlib.import_module(mod)
            for name, obj in inspect.getmembers(module):
                lowered = name.lower()
                if inspect.isclass(obj) and (lowered.endswith("router") or lowered.endswith("routerv8")):
                    return obj
        except Exception:
            continue
    raise ImportError("No valid router backend found.")


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
    violations: List[Dict[str, Any]]
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
    uncertainty_graph: Dict[str, Any] = Field(default_factory=dict)
    precedent_alignment: Dict[str, Any] = Field(default_factory=dict)
    router_diagnostics: Dict[str, Any] = Field(default_factory=dict)
    timings: Dict[str, float] = Field(default_factory=dict)
    evidence_references: List[Dict[str, Any]] = Field(default_factory=list)


class EngineResult(BaseModel):
    output_text: Optional[str] = None
    trace_id: str
    model_info: Optional[EngineModelInfo] = None
    critic_findings: Optional[Dict[str, EngineCriticFinding]] = None
    aggregated: Optional[Dict[str, Any]] = None
    uncertainty: Optional[Dict[str, Any]] = None
    precedent_alignment: Optional[Dict[str, Any]] = None
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
        evidence_recorder: Optional[EvidenceRecorder] = None,
        detector_engine: Optional[DetectorEngineV8] = None,
        precedent_engine: Optional[Any] = None,
        precedent_retriever: Optional[Any] = None,
        uncertainty_engine: Optional[Any] = None,
        aggregator: Optional[Any] = None,
        router_backend: Optional[Any] = None,
        critics: Optional[Dict[str, Any]] = None,
        critic_models: Optional[Dict[str, Any]] = None,
    ):
        self.config = config or EngineConfig()
        self.instance_id = str(uuid.uuid4())

        # Router
        if router_backend is None:
            RouterClass = load_router_backend()
            self.router = RouterClass()
        else:
            self.router = router_backend() if inspect.isclass(router_backend) else router_backend

        # Critics
        self.critics = critics or {
            "rights": RightsCriticV8,
            "autonomy": AutonomyCriticV8,
            "risk": RiskCriticV8,
            "fairness": FairnessCriticV8,
            "pragmatics": PragmaticsCriticV8,
            "truth": TruthCriticV8,
        }
        self.critic_models = critic_models or {}

        # Detector Engine
        self.detector_engine = detector_engine or DetectorEngineV8(detectors={})

        # Recorder
        self.recorder = evidence_recorder or EvidenceRecorder(
            jsonl_path=self.config.jsonl_evidence_path
        )

        # Precedent
        if precedent_engine:
            self.precedent_engine = precedent_engine
        elif PrecedentAlignmentEngineV8:
            self.precedent_engine = PrecedentAlignmentEngineV8()
        else:
            self.precedent_engine = None

        # Precedent Retriever (optional)
        self.precedent_retriever = precedent_retriever
        if self.precedent_retriever is None and PrecedentRetrievalV8:
            class _NullStore:
                def search(self, q: str, top_k: int = 5):
                    return []
            self.precedent_retriever = PrecedentRetrievalV8(store_client=_NullStore())

        # Uncertainty
        if uncertainty_engine:
            self.uncertainty_engine = uncertainty_engine
        elif UncertaintyEngineV8:
            self.uncertainty_engine = UncertaintyEngineV8()
        else:
            self.uncertainty_engine = None

        # Aggregator
        if aggregator:
            self.aggregator = aggregator
        elif AggregatorV8:
            self.aggregator = AggregatorV8()
        else:
            self.aggregator = None

        # Concurrency
        self.semaphore = asyncio.Semaphore(self.config.max_concurrency)

        # Diagnostics
        self.timings: Dict[str, float] = {}
        self.router_diagnostics: Dict[str, Any] = {}

        print(f"[ELEANOR ENGINE] Initialized V8 engine {self.instance_id}")


    # -----------------------------------------------------
    # MODEL ROUTING
    # -----------------------------------------------------
    async def _select_model(self, text: str, context: dict) -> Dict[str, Any]:
        start = asyncio.get_event_loop().time()

        call = self.router.route(text=text, context=context)
        router_result = await call if inspect.isawaitable(call) else call

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
    ):
        async with self.semaphore:
            start = asyncio.get_event_loop().time()

            # Instantiate critic
            critic = critic_ref if not inspect.isclass(critic_ref) else critic_ref()

            # Choose model for this critic: explicit binding overrides router output
            bound_adapter = self.critic_models.get(name)

            if bound_adapter is None:
                class _StaticModel:
                    def __init__(self, response: str):
                        self.response = response

                    async def generate(self, prompt: str, context: Optional[Dict[str, Any]] = None):
                        return self.response

                model_adapter = _StaticModel(model_response)
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
                    class _StaticModel:
                        def __init__(self, response: str):
                            self.response = response
                        async def generate(self, prompt: str, context: Optional[Dict[str, Any]] = None):
                            return self.response
                    model_adapter = _StaticModel(model_response)

            try:
                evaluation = critic.evaluate(model_adapter, input_text=input_text, context=context)
                evaluation_result = await evaluation if inspect.isawaitable(evaluation) else evaluation
            except Exception as exc:
                evaluation_result = {
                    "severity": 0.0,
                    "violations": [],
                    "justification": f"critic_error:{exc}",
                }

            end = asyncio.get_event_loop().time()

            evaluation_result = evaluation_result or {}
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
                pass

            return evaluation_result


    async def _run_critics_parallel(
        self,
        model_response: str,
        input_text: str,
        context: dict,
        trace_id: str,
    ):
        tasks = [
            asyncio.create_task(
                self._run_single_critic(name, critic_ref, model_response, input_text, context, trace_id)
            )
            for name, critic_ref in self.critics.items()
        ]
        results = await asyncio.gather(*tasks)
        return {r.get("critic", f"critic_{idx}"): r for idx, r in enumerate(results)}


    # -----------------------------------------------------
    # PRECEDENT + UNCERTAINTY + AGGREGATION
    # -----------------------------------------------------

    async def _run_precedent_alignment(self, critic_results, trace_id: str, text: str):
        if not self.precedent_engine:
            return None
        start = asyncio.get_event_loop().time()
        cases = []
        query_embedding = []
        retrieval_meta = None

        if self.precedent_retriever:
            try:
                retrieval_meta = self.precedent_retriever.retrieve(text, list(critic_results.values()))
                cases = retrieval_meta.get("precedent_cases") or retrieval_meta.get("cases") or []
                query_embedding = retrieval_meta.get("query_embedding") or []
            except Exception as exc:
                retrieval_meta = {"error": str(exc)}

        analyze_fn = getattr(self.precedent_engine, "analyze", None)
        if analyze_fn:
            out = analyze_fn(
                critics=critic_results,
                precedent_cases=cases,
                query_embedding=query_embedding,
            )
        else:
            out = None

        if retrieval_meta:
            out = {**(out or {}), "retrieval": retrieval_meta}

        end = asyncio.get_event_loop().time()
        self.timings["precedent_alignment_ms"] = (end - start) * 1000
        return out


    async def _run_uncertainty_engine(self, precedent_alignment, critic_results, model_name: str):
        if not self.uncertainty_engine:
            return None
        start = asyncio.get_event_loop().time()
        compute_fn = getattr(self.uncertainty_engine, "compute", None) or getattr(self.uncertainty_engine, "evaluate", None)
        if not compute_fn:
            return None

        out = compute_fn(
            critics=critic_results,
            model_used=model_name,
            precedent_alignment=precedent_alignment or {},
        )
        if inspect.isawaitable(out):
            out = await out
        end = asyncio.get_event_loop().time()
        self.timings["uncertainty_engine_ms"] = (end - start) * 1000
        return out


    async def _aggregate_results(self, critic_results, model_response: str, precedent_data: Optional[Dict[str, Any]], uncertainty_data: Optional[Dict[str, Any]]):
        if not self.aggregator:
            raise RuntimeError("AggregatorV8 not available")
        start = asyncio.get_event_loop().time()
        agg_result = self.aggregator.aggregate(
            critics=critic_results,
            precedent=precedent_data or {},
            uncertainty=uncertainty_data or {},
            model_output=model_response,
        )
        out = await agg_result if inspect.isawaitable(agg_result) else agg_result
        end = asyncio.get_event_loop().time()
        self.timings["aggregation_ms"] = (end - start) * 1000
        return out
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

        # Step 1: Model Routing
        router_data = await self._select_model(text, context)
        model_info = router_data["model_info"]
        model_response = router_data["response_text"]

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
            precedent_data = await self._run_precedent_alignment(
                critic_results=critic_results,
                trace_id=trace_id,
                text=text,
            )

        # Step 4: Uncertainty Modeling
        uncertainty_data = None
        if self.config.enable_reflection and self.uncertainty_engine:
            uncertainty_data = await self._run_uncertainty_engine(
                precedent_alignment=precedent_data,
                critic_results=critic_results,
                model_name=model_info.get("model_name") or "unknown-model",
            )

        # Step 5: Aggregation / Constitutional Fusion
        aggregated = await self._aggregate_results(
            critic_results=critic_results,
            model_response=model_response,
            precedent_data=precedent_data,
            uncertainty_data=uncertainty_data,
        )

        # Step 6: Human Review Trigger Evaluation (async, non-blocking)
        # This DOES NOT block user response, only blocks precedent promotion
        try:
            from governance.stewardship import should_review, create_and_emit_review_packet

            case_data = {
                "severity": aggregated.get("max_severity", 0.0),
                "critic_outputs": critic_results,
                "novel_precedent": precedent_data.get("novel", False) if precedent_data else False,
                "rights_impacted": aggregated.get("rights_impacted", []),
                "uncertainty_flags": list(uncertainty_data.get("flags", [])) if uncertainty_data else [],
            }

            review_decision = should_review(case_data)

            if review_decision["review_required"]:
                # Extract citations from critic results
                citations = {}
                for critic_name, critic_data in critic_results.items():
                    if "precedent_refs" in critic_data:
                        citations[critic_name] = critic_data["precedent_refs"]

                # Emit review packet (async, non-blocking)
                create_and_emit_review_packet(
                    case_id=trace_id,
                    domain=context.get("domain", "general"),
                    severity=case_data["severity"],
                    uncertainty_flags=case_data["uncertainty_flags"],
                    critic_outputs=critic_results,
                    aggregator_summary=aggregated.get("final_output", ""),
                    dissent=aggregated.get("dissent", None),
                    citations=citations,
                    triggers=review_decision["triggers"],
                )
        except Exception as review_exc:
            # Review hook failure should NOT break the pipeline
            print(f"[ELEANOR ENGINE] Review trigger failed (non-fatal): {review_exc}")

        # Timing
        pipeline_end = asyncio.get_event_loop().time()
        self.timings["total_pipeline_ms"] = (pipeline_end - pipeline_start) * 1000

        # Evidence buffer
        buffer = getattr(self.recorder, "buffer", None)
        evidence_count = len(buffer) if buffer else None

        # Base kwargs
        result_kwargs = {
            "trace_id": trace_id,
            "output_text": aggregated.get("final_output") or model_response,
            "model_info": EngineModelInfo(**model_info),
            "critic_findings": {
                k: EngineCriticFinding(
                    critic=k,
                    violations=v["violations"],
                    duration_ms=v["duration_ms"],
                    evaluated_rules=v.get("evaluated_rules"),
                )
                for k, v in critic_results.items()
            },
            "aggregated": aggregated,
            "uncertainty": uncertainty_data,
            "precedent_alignment": precedent_data,
            "evidence_count": evidence_count,
        }

        # ---------------------------
        # DETAIL LEVEL 1
        # ---------------------------
        if level == 1:
            return EngineResult(
                trace_id=trace_id,
                output_text=aggregated.get("final_output") or model_response,
                model_info=EngineModelInfo(**model_info),
            )

        # ---------------------------
        # DETAIL LEVEL 2
        # ---------------------------
        if level == 2:
            return EngineResult(**result_kwargs)

        # ---------------------------
        # DETAIL LEVEL 3 — FORENSIC
        # ---------------------------
        forensic_data = None
        if level == 3:
            forensic_buffer = buffer[-200:] if buffer else []

            forensic_data = EngineForensicData(
                detector_metadata=aggregated.get("critic_details", {}) if isinstance(aggregated.get("critic_details"), dict) else {},
                uncertainty_graph=uncertainty_data or {},
                precedent_alignment=precedent_data or {},
                router_diagnostics=self.router_diagnostics,
                timings=self.timings,
                evidence_references=[
                    r.dict() if hasattr(r, "dict") else r for r in forensic_buffer
                ],
            )

            return EngineResult(
                **result_kwargs,
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

        # Step 1: Model Routing
        router_data = await self._select_model(text, context)

        yield {
            "event": "router_selected",
            "trace_id": trace_id,
            "model_info": router_data["model_info"],
            "timings": {"router_selection_ms": self.timings.get("router_selection_ms")},
        }

        model_response = router_data["response_text"]

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

        critic_results = {}

        async def _run_and_return(name, critic_ref):
            return await self._run_single_critic(name, critic_ref, model_response, text, context, trace_id)

        tasks = [
            asyncio.create_task(_run_and_return(name, critic_ref))
            for name, critic_ref in self.critics.items()
        ]

        for task in asyncio.as_completed(tasks):
            res = await task
            critic_results[res["critic"]] = res

            yield {
                "event": "critic_result",
                "trace_id": trace_id,
                "critic": res["critic"],
                "violations": res["violations"],
                "duration_ms": res["duration_ms"],
                "evaluated_rules": res.get("evaluated_rules"),
            }

        yield {
            "event": "critics_complete",
            "trace_id": trace_id,
        }

        # Step 3: Precedent
        precedent_data = None
        if self.config.enable_precedent_analysis and self.precedent_engine:
            precedent_data = await self._run_precedent_alignment(critic_results, trace_id, text)
            yield {
                "event": "precedent_alignment",
                "trace_id": trace_id,
                "data": precedent_data,
            }

        # Step 4: Uncertainty
        uncertainty_data = None
        if self.config.enable_reflection and self.uncertainty_engine:
            uncertainty_data = await self._run_uncertainty_engine(
                precedent_alignment=precedent_data,
                critic_results=critic_results,
                model_name=router_data["model_info"].get("model_name") or "unknown-model",
            )
            yield {
                "event": "uncertainty",
                "trace_id": trace_id,
                "data": uncertainty_data,
            }

        # Step 5: Aggregation
        aggregated = await self._aggregate_results(critic_results, model_response, precedent_data, uncertainty_data)

        yield {
            "event": "aggregation",
            "trace_id": trace_id,
            "data": aggregated,
        }

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
    evidence_recorder: Optional[EvidenceRecorder] = None,
    detector_engine: Optional[DetectorEngineV8] = None,
    precedent_engine: Optional[Any] = None,
    precedent_retriever: Optional[Any] = None,
    uncertainty_engine: Optional[Any] = None,
    aggregator: Optional[Any] = None,
    router_backend: Optional[Any] = None,
    critics: Optional[Dict[str, Any]] = None,
    critic_models: Optional[Dict[str, Any]] = None,
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
