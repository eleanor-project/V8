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

# Critics
from engine.critics.rights import RightsCritic
from engine.critics.risk import RiskCritic
from engine.critics.fairness import FairnessCritic
from engine.critics.pragmatics import PragmaticsCritic
from engine.critics.truth import TruthCritic

# Detector Engine
from engine.detectors.engine import DetectorEngineV8

# Evidence Recorder
from engine.recorder.evidence_recorder import EvidenceRecorder

# Precedent Engine
try:
    from engine.precedent.alignment import PrecedentAlignmentEngineV8
except Exception:
    PrecedentAlignmentEngineV8 = None

# Uncertainty Engine
try:
    from engine.uncertainty.engine import UncertaintyEngineV8
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
                if inspect.isclass(obj) and name.lower().endswith("router"):
                    return obj
        except Exception:
            pass
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
        uncertainty_engine: Optional[Any] = None,
        aggregator: Optional[Any] = None,
        router_backend: Optional[Any] = None,
        critics: Optional[Dict[str, Any]] = None,
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
            "rights": RightsCritic,
            "risk": RiskCritic,
            "fairness": FairnessCritic,
            "pragmatics": PragmaticsCritic,
            "truth": TruthCritic,
        }

        # Detector Engine
        self.detector_engine = detector_engine or DetectorEngineV8(detectors={})

        # Recorder
        self.recorder = evidence_recorder or EvidenceRecorder(
            jsonl_evidence_path=self.config.jsonl_evidence_path
        )

        # Precedent
        if precedent_engine:
            self.precedent_engine = precedent_engine
        elif PrecedentAlignmentEngineV8:
            self.precedent_engine = PrecedentAlignmentEngineV8()
        else:
            self.precedent_engine = None

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
            "response_text": router_result["response_text"],
        }


    # -----------------------------------------------------
    # CRITIC EXECUTION
    # -----------------------------------------------------
    async def _run_single_critic(self, name: str, critic_ref: Any, model_response: str, context: dict, trace_id: str):
        async with self.semaphore:
            start = asyncio.get_event_loop().time()

            critic = critic_ref if not inspect.isclass(critic_ref) else critic_ref(
                rules=None,
                detectors=self.detector_engine.detectors,
                prompt_pack=None,
            )

            evaluation = critic.evaluate(model_response, context=context, trace_id=trace_id)
            violations, evaluated_rules = await evaluation if inspect.isawaitable(evaluation) else evaluation

            end = asyncio.get_event_loop().time()

            # Evidence logging
            for v in violations:
                await self.recorder.record(
                    critic=name,
                    rule_id=v.rule_id,
                    severity=v.severity,
                    violation_description=v.claim,
                    confidence=v.confidence,
                    mitigation=v.mitigation,
                    redundancy_group=v.redundancy_group,
                    detector_metadata=v.evidence,
                    context=context,
                    raw_text=model_response,
                    trace_id=trace_id,
                )

            return {
                "critic": name,
                "violations": [v.dict() for v in violations],
                "duration_ms": (end - start) * 1000,
                "evaluated_rules": evaluated_rules,
            }


    async def _run_critics_parallel(self, model_response: str, context: dict, trace_id: str):
        tasks = [
            asyncio.create_task(
                self._run_single_critic(name, critic_ref, model_response, context, trace_id)
            )
            for name, critic_ref in self.critics.items()
        ]
        results = await asyncio.gather(*tasks)
        return {r["critic"]: r for r in results}


    # -----------------------------------------------------
    # PRECEDENT + UNCERTAINTY + AGGREGATION
    # -----------------------------------------------------

    async def _run_precedent_alignment(self, critic_results, trace_id: str):
        if not self.precedent_engine:
            return None
        start = asyncio.get_event_loop().time()
        out = await self.precedent_engine.align(critic_results=critic_results, trace_id=trace_id)
        end = asyncio.get_event_loop().time()
        self.timings["precedent_alignment_ms"] = (end - start) * 1000
        return out


    async def _run_uncertainty_engine(self, aggregated, critic_results):
        if not self.uncertainty_engine:
            return None
        start = asyncio.get_event_loop().time()
        out = await self.uncertainty_engine.evaluate(aggregated=aggregated, critic_results=critic_results)
        end = asyncio.get_event_loop().time()
        self.timings["uncertainty_engine_ms"] = (end - start) * 1000
        return out


    async def _aggregate_results(self, critic_results, model_response: str):
        if not self.aggregator:
            raise RuntimeError("AggregatorV8 not available")
        start = asyncio.get_event_loop().time()
        out = await self.aggregator.aggregate(critic_results=critic_results, model_response=model_response)
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
            context=context,
            trace_id=trace_id,
        )

        # Step 3: Precedent Alignment
        precedent_data = None
        if self.config.enable_precedent_analysis:
            precedent_data = await self._run_precedent_alignment(
                critic_results=critic_results,
                trace_id=trace_id,
            )

        # Step 4: Aggregation / Constitutional Fusion
        aggregated = await self._aggregate_results(
            critic_results=critic_results,
            model_response=model_response,
        )

        # Step 5: Uncertainty Modeling
        uncertainty_data = None
        if self.config.enable_reflection and self.uncertainty_engine:
            uncertainty_data = await self._run_uncertainty_engine(
                aggregated,
                critic_results,
            )

        # Timing
        pipeline_end = asyncio.get_event_loop().time()
        self.timings["total_pipeline_ms"] = (pipeline_end - pipeline_start) * 1000

        # Evidence buffer
        buffer = getattr(self.recorder, "buffer", None)
        evidence_count = len(buffer) if buffer else None

        # Base kwargs
        result_kwargs = {
            "trace_id": trace_id,
            "output_text": aggregated.get("final_output"),
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
                output_text=aggregated.get("final_output"),
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
            return await self._run_single_critic(name, critic_ref, model_response, context, trace_id)

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
            precedent_data = await self._run_precedent_alignment(critic_results, trace_id)
            yield {
                "event": "precedent_alignment",
                "trace_id": trace_id,
                "data": precedent_data,
            }

        # Step 4: Aggregation
        aggregated = await self._aggregate_results(critic_results, model_response)

        yield {
            "event": "aggregation",
            "trace_id": trace_id,
            "data": aggregated,
        }

        # Step 5: Uncertainty
        uncertainty_data = None
        if self.config.enable_reflection and self.uncertainty_engine:
            uncertainty_data = await self._run_uncertainty_engine(aggregated, critic_results)
            yield {
                "event": "uncertainty",
                "trace_id": trace_id,
                "data": uncertainty_data,
            }

        # Final Output
        final_output = aggregated.get("final_output", "")

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
    uncertainty_engine: Optional[Any] = None,
    aggregator: Optional[Any] = None,
    router_backend: Optional[Any] = None,
    critics: Optional[Dict[str, Any]] = None,
) -> EleanorEngineV8:

    engine = EleanorEngineV8(
        config=config,
        evidence_recorder=evidence_recorder,
        detector_engine=detector_engine,
        precedent_engine=precedent_engine,
        uncertainty_engine=uncertainty_engine,
        aggregator=aggregator,
        router_backend=router_backend,
        critics=critics,
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
