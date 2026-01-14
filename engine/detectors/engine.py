"""
ELEANOR V8 - Detector Engine
-----------------------------

Orchestrates all detectors in parallel, similar to the Orchestrator for critics.

This engine:
1. Loads all available detectors dynamically
2. Runs them in parallel with timeout
3. Aggregates signals
4. Routes to appropriate critics
"""

import asyncio
import importlib
import logging
from typing import Any, Dict, List, Optional

from .base import Detector
from .signals import DetectorSignal

logger = logging.getLogger(__name__)


class DetectorEngineV8:
    """
    Orchestrates detector execution and signal aggregation.
    """

    def __init__(
        self, detectors: Optional[Dict[str, Detector]] = None, timeout_seconds: float = 2.0
    ):
        self.detectors: Dict[str, Detector] = detectors or {}
        self.timeout = timeout_seconds
        if not self.detectors:
            self._load_detectors()

    def _load_detectors(self):
        """Dynamically load all detector modules."""
        # List of detector module names
        detector_modules = [
            "autonomy",
            "coercion",
            "dehumanization",
            "discrimination",
            "disparate_impact",
            "disparate_treatment",
            "hallucination",
            "privacy",
            "physical_safety",
            "psychological_harm",
            "factual_accuracy",
            "evidence_grounding",
            "feasibility",
            "resource_burden",
            "time_constraints",
            "irreversible_harm",
            "cascading_failure",
            "operational_risk",
            "environmental_impact",
            "omission",
            "contradiction",
            "embedding_bias",
            "procedural_fairness",
            "structural_disadvantage",
            "cascading_pragmatic_failure",
        ]

        for module_name in detector_modules:
            try:
                # Try to import the detector module
                module_path = f"engine.detectors.{module_name}.detector"
                module = importlib.import_module(module_path)

                # Find detector class (should be [Name]Detector)
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, Detector)
                        and attr is not Detector
                    ):
                        detector = attr()
                        detector_name = getattr(detector, "name", attr_name)
                        self.detectors[detector_name] = detector
                        break
            except Exception as e:
                # Log but don't crash if a detector fails to load
                logger.warning(
                    "detector_load_failed",
                    extra={"detector_module": module_name, "error": str(e)},
                )

    async def detect_all(self, text: str, context: Dict[str, Any]) -> Dict[str, DetectorSignal]:
        """
        Run all detectors in parallel.

        Returns:
            Dict mapping detector name to signal
        """
        tasks = []
        for name, detector in self.detectors.items():
            task = asyncio.create_task(self._run_detector_safe(name, detector, text, context))
            tasks.append((name, task))

        results = {}
        for name, task in tasks:
            try:
                signal = await task
                results[name] = signal
            except Exception as e:
                # Return empty signal on failure
                results[name] = DetectorSignal(
                    detector_name=name,
                    severity=0.0,  # type: ignore[arg-type]  # coerced by validator
                    violations=[],
                    evidence={"error": str(e)},
                    flags=["DETECTOR_FAILURE"],
                )

        return results

    async def _run_detector_safe(
        self, name: str, detector: Detector, text: str, context: Dict[str, Any]
    ) -> DetectorSignal:
        """Run a single detector with timeout and error handling."""
        try:
            signal = await asyncio.wait_for(detector.detect(text, context), timeout=self.timeout)
            return signal
        except asyncio.TimeoutError:
            return DetectorSignal(
                detector_name=name,
                severity=0.0,  # type: ignore[arg-type]  # coerced by validator
                violations=[],
                evidence={"error": "timeout"},
                flags=["TIMEOUT"],
            )
        except Exception as e:
            return DetectorSignal(
                detector_name=name,
                severity=0.0,  # type: ignore[arg-type]  # coerced by validator
                violations=[],
                evidence={"error": str(e)},
                flags=["ERROR"],
            )

    def aggregate_signals(self, signals: Dict[str, DetectorSignal]) -> Dict[str, Any]:
        """
        Aggregate detector signals for critic consumption.

        Returns summary statistics and routing recommendations.
        """
        total_violations = 0
        max_severity = 0.0
        critical_flags = []

        by_severity: Dict[str, List[str]] = {
            "critical": [],  # severity >= 0.8
            "high": [],  # severity >= 0.6
            "medium": [],  # severity >= 0.3
            "low": [],  # severity < 0.3
        }

        for name, signal in signals.items():
            total_violations += len(signal.violations)
            severity_score = float(signal.severity)
            max_severity = max(max_severity, severity_score)

            # Categorize by severity
            if severity_score >= 0.8:
                by_severity["critical"].append(name)
            elif severity_score >= 0.6:
                by_severity["high"].append(name)
            elif severity_score >= 0.3:
                by_severity["medium"].append(name)
            else:
                by_severity["low"].append(name)

            # Collect critical flags
            for flag in signal.flags:
                if flag.startswith("CRITICAL_"):
                    critical_flags.append(flag)

        return {
            "total_detectors": len(signals),
            "total_violations": total_violations,
            "max_severity": max_severity,
            "by_severity": by_severity,
            "critical_flags": list(set(critical_flags)),
            "signals": signals,
        }
