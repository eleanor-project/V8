import importlib
import re

import pytest

from engine.detectors.signals import DetectorSignal, SeverityLevel


@pytest.mark.parametrize(
    "module_path,class_name",
    [
        ("engine.detectors.cascading_failure.detector", "CascadingFailureDetector"),
        ("engine.detectors.cascading_pragmatic_failure.detector", "CascadingPragmaticFailureDetector"),
        ("engine.detectors.contradiction.detector", "ContradictionDetector"),
        ("engine.detectors.disparate_impact.detector", "DisparateImpactDetector"),
        ("engine.detectors.disparate_treatment.detector", "DisparateTreatmentDetector"),
        ("engine.detectors.embedding_bias.detector", "EmbeddingBiasDetector"),
        ("engine.detectors.environmental_impact.detector", "EnvironmentalImpactDetector"),
        ("engine.detectors.evidence_grounding.detector", "EvidenceGroundingDetector"),
        ("engine.detectors.factual_accuracy.detector", "FactualAccuracyDetector"),
        ("engine.detectors.omission.detector", "OmissionDetector"),
        ("engine.detectors.operational_risk.detector", "OperationalRiskDetector"),
        ("engine.detectors.procedural_fairness.detector", "ProceduralFairnessDetector"),
        ("engine.detectors.resource_burden.detector", "ResourceBurdenDetector"),
        ("engine.detectors.structural_disadvantage.detector", "StructuralDisadvantageDetector"),
        ("engine.detectors.time_constraints.detector", "TimeConstraintsDetector"),
    ],
)
@pytest.mark.asyncio
async def test_detector_regex_and_keyword_branches(monkeypatch, module_path, class_name):
    module = importlib.import_module(module_path)
    detector_cls = getattr(module, class_name)
    detector = detector_cls()

    original_patterns = list(module.DETECTION_PATTERNS)
    dummy_pattern = module.DetectionPattern(
        category="dummy_branch",
        patterns=["nomatch"],
        keywords=["keyword"],
        severity_weight=0.95,
        description="dummy",
    )
    patterns = [original_patterns[0], dummy_pattern]
    monkeypatch.setattr(module, "DETECTION_PATTERNS", patterns)

    compiled = {dp.category: [re.compile("nomatch")] for dp in patterns}
    compiled[patterns[0].category] = [re.compile("match")]
    detector._compiled_patterns = compiled

    signal = await detector.detect("match keyword", {})
    assert signal.violations
    assert any(flag.startswith("HIGH_SEVERITY_") for flag in signal.flags)


def test_detector_signal_properties():
    signal = DetectorSignal(detector_name="test", severity=0.7, violations=["a"])
    assert signal.violation is True
    assert signal.severity_label == "S3"
    assert signal.metadata == signal.evidence
    assert signal.confidence_score == signal.confidence
    assert SeverityLevel(0.2) < "S2"
