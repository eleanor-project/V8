"""
Tests for Small Engine Components
---------------------------------

Tests for:
- engine/critics/severity.py
- engine/version.py
- engine/orchestrator/orchestrator.py
- engine/recorder/package.py
"""

import pytest
import asyncio
from engine.critics.severity import (
    GLOBAL_SEVERITY_SCALE,
    MANDATORY_ESCALATION_THRESHOLD,
    normalize_severity,
    severity_label
)
from engine.version import (
    ELEANOR_VERSION,
    BUILD_NAME,
    CRITIC_SUITE,
    CORE_HASH
)
from engine.orchestrator.orchestrator import OrchestratorV8
from engine.recorder.package import EvidencePackageV8


# ============================================================
# Severity Tests
# ============================================================

def test_severity_scale_constants():
    """Test that severity scale constants are defined correctly."""
    assert GLOBAL_SEVERITY_SCALE["info"] == 0.0
    assert GLOBAL_SEVERITY_SCALE["low"] == 0.25
    assert GLOBAL_SEVERITY_SCALE["moderate"] == 0.5
    assert GLOBAL_SEVERITY_SCALE["high"] == 0.75
    assert GLOBAL_SEVERITY_SCALE["critical"] == 1.0
    assert MANDATORY_ESCALATION_THRESHOLD == 0.75


def test_normalize_severity_valid_inputs():
    """Test severity normalization with valid inputs."""
    assert normalize_severity("info") == 0.0
    assert normalize_severity("low") == 0.25
    assert normalize_severity("moderate") == 0.5
    assert normalize_severity("high") == 0.75
    assert normalize_severity("critical") == 1.0


def test_normalize_severity_case_insensitive():
    """Test that severity normalization is case-insensitive."""
    assert normalize_severity("INFO") == 0.0
    assert normalize_severity("Low") == 0.25
    assert normalize_severity("MODERATE") == 0.5
    assert normalize_severity("High") == 0.75
    assert normalize_severity("CRITICAL") == 1.0


def test_normalize_severity_with_whitespace():
    """Test that severity normalization handles whitespace."""
    assert normalize_severity("  info  ") == 0.0
    assert normalize_severity(" low ") == 0.25


def test_normalize_severity_invalid_input():
    """Test that invalid severity defaults to moderate."""
    assert normalize_severity("unknown") == 0.5
    assert normalize_severity("invalid") == 0.5
    assert normalize_severity("") == 0.5


def test_severity_label_exact_matches():
    """Test severity label for exact values."""
    assert severity_label(0.0) == "info"
    assert severity_label(0.25) == "low"
    assert severity_label(0.5) == "moderate"
    assert severity_label(0.75) == "high"
    assert severity_label(1.0) == "critical"


def test_severity_label_approximations():
    """Test severity label for values between exact points."""
    # Closest to info
    assert severity_label(0.1) == "info"

    # Closest to low
    assert severity_label(0.3) == "low"

    # Closest to moderate
    assert severity_label(0.4) == "moderate"
    assert severity_label(0.6) == "moderate"

    # Closest to high
    assert severity_label(0.7) == "high"
    assert severity_label(0.8) == "high"

    # Closest to critical
    assert severity_label(0.9) == "critical"


# ============================================================
# Version Tests
# ============================================================

def test_version_constants_exist():
    """Test that all version constants are defined."""
    assert ELEANOR_VERSION is not None
    assert BUILD_NAME is not None
    assert CRITIC_SUITE is not None
    assert CORE_HASH is not None


def test_version_format():
    """Test that version follows expected format."""
    # Should be in format X.Y.Z
    parts = ELEANOR_VERSION.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)


# ============================================================
# Orchestrator Tests
# ============================================================

@pytest.mark.asyncio
async def test_orchestrator_runs_all_critics():
    """Test that orchestrator runs all provided critics."""
    async def mock_critic_a(input_data):
        return {"value": "A", "score": 1}

    async def mock_critic_b(input_data):
        return {"value": "B", "score": 2}

    critics = {
        "critic_a": mock_critic_a,
        "critic_b": mock_critic_b
    }

    orchestrator = OrchestratorV8(critics)
    results = await orchestrator.run_all("test_input")

    assert "critic_a" in results
    assert "critic_b" in results
    assert results["critic_a"]["value"] == "A"
    assert results["critic_b"]["value"] == "B"


@pytest.mark.asyncio
async def test_orchestrator_handles_critic_failure():
    """Test that orchestrator handles critic failures gracefully."""
    async def failing_critic(input_data):
        raise ValueError("Test failure")

    async def working_critic(input_data):
        return {"value": "success", "score": 1}

    critics = {
        "failing": failing_critic,
        "working": working_critic
    }

    orchestrator = OrchestratorV8(critics)
    results = await orchestrator.run_all("test_input")

    # Working critic should succeed
    assert results["working"]["value"] == "success"

    # Failing critic should return error template
    assert results["failing"]["value"] is None
    assert results["failing"]["details"]["critic_failed"] is True
    assert "Test failure" in results["failing"]["details"]["error"]


@pytest.mark.asyncio
async def test_orchestrator_timeout():
    """Test that orchestrator enforces timeouts."""
    async def slow_critic(input_data):
        await asyncio.sleep(5.0)  # Longer than timeout
        return {"value": "should_not_reach"}

    critics = {"slow": slow_critic}
    orchestrator = OrchestratorV8(critics, timeout_seconds=0.1)

    results = await orchestrator.run_all("test_input")

    # Should have timed out and returned failure template
    assert results["slow"]["value"] is None
    assert results["slow"]["details"]["critic_failed"] is True


def test_orchestrator_sync_wrapper():
    """Test that sync wrapper works correctly."""
    async def mock_critic(input_data):
        return {"value": "test", "score": 1}

    critics = {"test": mock_critic}
    orchestrator = OrchestratorV8(critics)

    # Use sync wrapper
    results = orchestrator.run("test_input")

    assert "test" in results
    assert results["test"]["value"] == "test"


@pytest.mark.asyncio
async def test_orchestrator_passes_input_to_critics():
    """Test that orchestrator passes input correctly to critics."""
    received_input = []

    async def capturing_critic(input_data):
        received_input.append(input_data)
        return {"value": "ok"}

    critics = {"capture": capturing_critic}
    orchestrator = OrchestratorV8(critics)

    test_input = {"test": "data", "value": 123}
    await orchestrator.run_all(test_input)

    assert len(received_input) == 1
    assert received_input[0] == test_input


# ============================================================
# Evidence Package Tests
# ============================================================

def test_evidence_package_build_structure():
    """Test that evidence package builds complete structure."""
    package = EvidencePackageV8()

    input_snapshot = {"prompt": "test"}
    router_result = {"used_adapter": "test_model", "attempts": []}
    critic_outputs = {"rights": {"score": 1}}
    deliberation_state = {
        "priority_violations": [],
        "values_respected": ["privacy"],
        "values_violated": []
    }
    uncertainty_state = {
        "uncertainty_score": 0.1,
        "dissent_score": 0.0,
        "entropy_estimate": 0.05,
        "stability": "stable",
        "requires_escalation": False,
        "escalation_reasons": []
    }
    precedent_result = {
        "alignment_score": 0.95,
        "top_case": None,
        "precedent_cases": []
    }

    evidence = package.build(
        input_snapshot,
        router_result,
        critic_outputs,
        deliberation_state,
        uncertainty_state,
        precedent_result
    )

    # Check structure
    assert "timestamp" in evidence
    assert "trace_id" in evidence
    assert evidence["trace_id"].startswith("trace_")
    assert "input_snapshot" in evidence
    assert "model_used" in evidence
    assert "critic_outputs" in evidence
    assert "priority_violations" in evidence
    assert "values_respected" in evidence
    assert "values_violated" in evidence
    assert "uncertainty" in evidence
    assert "precedent" in evidence
    assert "governance_ready_payload" in evidence


def test_evidence_package_unique_trace_ids():
    """Test that each package gets a unique trace ID."""
    package = EvidencePackageV8()

    evidence1 = package.build({}, {}, {}, {}, {}, {})
    evidence2 = package.build({}, {}, {}, {}, {}, {})

    assert evidence1["trace_id"] != evidence2["trace_id"]


def test_evidence_package_includes_all_inputs():
    """Test that evidence package includes all input data."""
    package = EvidencePackageV8()

    input_snapshot = {"prompt": "test_prompt"}
    critic_outputs = {"rights": {"score": 1}, "fairness": {"score": 2}}

    evidence = package.build(
        input_snapshot,
        {"used_adapter": "gpt-4"},
        critic_outputs,
        {},
        {},
        {}
    )

    assert evidence["input_snapshot"] == input_snapshot
    assert evidence["model_used"] == "gpt-4"
    assert evidence["critic_outputs"] == critic_outputs


def test_evidence_package_uncertainty_structure():
    """Test that uncertainty is properly packaged."""
    package = EvidencePackageV8()

    uncertainty_state = {
        "uncertainty_score": 0.5,
        "dissent_score": 0.3,
        "entropy_estimate": 0.2,
        "stability": "unstable",
        "requires_escalation": True,
        "escalation_reasons": ["high_dissent"]
    }

    evidence = package.build({}, {}, {}, {}, uncertainty_state, {})

    assert evidence["uncertainty"]["uncertainty_score"] == 0.5
    assert evidence["uncertainty"]["dissent"] == 0.3
    assert evidence["uncertainty"]["entropy"] == 0.2
    assert evidence["uncertainty"]["stability"] == "unstable"
    assert evidence["uncertainty"]["requires_escalation"] is True
    assert "high_dissent" in evidence["uncertainty"]["escalation_reasons"]


def test_evidence_package_precedent_structure():
    """Test that precedent is properly packaged."""
    package = EvidencePackageV8()

    precedent_result = {
        "alignment_score": 0.85,
        "top_case": {"case_id": "123"},
        "precedent_cases": [{"case_id": "123"}, {"case_id": "456"}]
    }

    evidence = package.build({}, {}, {}, {}, {}, precedent_result)

    assert evidence["precedent"]["alignment_score"] == 0.85
    assert evidence["precedent"]["top_case"]["case_id"] == "123"
    assert len(evidence["precedent"]["precedent_cases"]) == 2
