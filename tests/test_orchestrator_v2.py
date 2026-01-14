"""
Tests for Enhanced Orchestrator V2
----------------------------------

Tests the orchestrator's core execution logic, hooks, and error handling.
"""

import pytest
import asyncio
from typing import Dict, Any

from engine.orchestrator.orchestrator_v2 import (
    OrchestratorV2,
    OrchestratorHooks,
    CriticInput,
    CriticExecutionStatus,
    CriticExecutionResult,
)


# ============================================================
# Test Fixtures
# ============================================================


@pytest.fixture
def sample_input():
    """Create sample critic input."""
    return CriticInput(
        model_response="The model's response about fairness and rights.",
        input_text="Should we allow this action?",
        context={"domain": "healthcare", "risk_tier": "high"},
        trace_id="test-trace-123",
    )


@pytest.fixture
def simple_critic():
    """Create a simple critic that always succeeds."""
    async def critic_fn(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "value": "fairness",
            "score": 0.8,
            "severity": 0.3,
            "violation": False,
            "violations": [],
            "justification": "This action is fair.",
        }
    return critic_fn


@pytest.fixture
def slow_critic():
    """Create a critic that takes 2 seconds."""
    async def critic_fn(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(2.0)
        return {
            "value": "rights",
            "score": 0.9,
            "severity": 0.2,
            "violation": False,
            "violations": [],
        }
    return critic_fn


@pytest.fixture
def failing_critic():
    """Create a critic that always fails."""
    async def critic_fn(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        raise ValueError("Critic failed intentionally")
    return critic_fn


# ============================================================
# Basic Execution Tests
# ============================================================


@pytest.mark.asyncio
async def test_orchestrator_single_critic_success(sample_input, simple_critic):
    """Test orchestrator with a single successful critic."""
    orchestrator = OrchestratorV2(
        critics={"fairness": simple_critic},
        timeout_seconds=5.0,
    )
    
    results = await orchestrator.run_all(sample_input)
    
    assert "fairness" in results
    assert results["fairness"]["value"] == "fairness"
    assert results["fairness"]["score"] == 0.8
    assert results["fairness"]["critic"] == "fairness"
    assert "duration_ms" in results["fairness"]
    assert results["fairness"]["execution_status"] == "success"


@pytest.mark.asyncio
async def test_orchestrator_multiple_critics(sample_input, simple_critic):
    """Test orchestrator with multiple critics."""
    async def rights_critic(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "value": "rights",
            "score": 0.7,
            "severity": 0.4,
            "violation": True,
            "violations": ["privacy_concern"],
        }
    
    orchestrator = OrchestratorV2(
        critics={
            "fairness": simple_critic,
            "rights": rights_critic,
        },
        timeout_seconds=5.0,
    )
    
    results = await orchestrator.run_all(sample_input)
    
    assert len(results) == 2
    assert "fairness" in results
    assert "rights" in results
    assert results["rights"]["violation"] is True
    assert "privacy_concern" in results["rights"]["violations"]


@pytest.mark.asyncio
async def test_orchestrator_empty_critics(sample_input):
    """Test orchestrator with no critics."""
    orchestrator = OrchestratorV2(critics={}, timeout_seconds=5.0)
    
    results = await orchestrator.run_all(sample_input)
    
    assert results == {}


# ============================================================
# Timeout Tests
# ============================================================


@pytest.mark.asyncio
async def test_orchestrator_critic_timeout(sample_input, slow_critic):
    """Test that slow critics timeout properly."""
    orchestrator = OrchestratorV2(
        critics={"slow_critic": slow_critic},
        timeout_seconds=0.5,  # 500ms timeout, critic takes 2s
    )
    
    results = await orchestrator.run_all(sample_input)
    
    assert "slow_critic" in results
    assert results["slow_critic"]["execution_status"] == "timeout"
    assert "timed out" in results["slow_critic"]["details"]["error"].lower()
    assert results["slow_critic"]["score"] == 0.0
    assert results["slow_critic"]["violation"] is False


@pytest.mark.asyncio
async def test_orchestrator_timeout_isolation(sample_input, simple_critic, slow_critic):
    """Test that one slow critic doesn't block others."""
    orchestrator = OrchestratorV2(
        critics={
            "fast": simple_critic,
            "slow": slow_critic,
        },
        timeout_seconds=0.5,
    )
    
    results = await orchestrator.run_all(sample_input)
    
    # Fast critic should succeed
    assert results["fast"]["execution_status"] == "success"
    assert results["fast"]["score"] == 0.8
    
    # Slow critic should timeout
    assert results["slow"]["execution_status"] == "timeout"


# ============================================================
# Error Handling Tests
# ============================================================


@pytest.mark.asyncio
async def test_orchestrator_critic_failure(sample_input, failing_critic):
    """Test that failing critics are handled gracefully."""
    orchestrator = OrchestratorV2(
        critics={"failing": failing_critic},
        timeout_seconds=5.0,
    )
    
    results = await orchestrator.run_all(sample_input)
    
    assert "failing" in results
    assert results["failing"]["execution_status"] == "error"
    assert "ValueError" in results["failing"]["details"]["error"]
    assert results["failing"]["score"] == 0.0
    assert results["failing"]["details"]["critic_failed"] is True


@pytest.mark.asyncio
async def test_orchestrator_failure_isolation(sample_input, simple_critic, failing_critic):
    """Test that one failing critic doesn't crash others."""
    orchestrator = OrchestratorV2(
        critics={
            "good": simple_critic,
            "bad": failing_critic,
        },
        timeout_seconds=5.0,
    )
    
    results = await orchestrator.run_all(sample_input)
    
    # Good critic should succeed
    assert results["good"]["execution_status"] == "success"
    assert results["good"]["score"] == 0.8
    
    # Bad critic should fail gracefully
    assert results["bad"]["execution_status"] == "error"
    assert results["bad"]["score"] == 0.0


# ============================================================
# Hook Tests
# ============================================================


@pytest.mark.asyncio
async def test_pre_execution_hook_cache(sample_input, simple_critic):
    """Test that pre-execution hook can return cached results."""
    cache_hit_count = 0
    
    async def pre_hook(critic_name: str, input_snapshot: CriticInput):
        nonlocal cache_hit_count
        cache_hit_count += 1
        # Return cached result
        return {
            "value": "cached",
            "score": 0.99,
            "severity": 0.0,
            "violation": False,
            "violations": [],
            "from_cache": True,
        }
    
    hooks = OrchestratorHooks(pre_execution=pre_hook)
    
    orchestrator = OrchestratorV2(
        critics={"test": simple_critic},
        timeout_seconds=5.0,
        hooks=hooks,
    )
    
    results = await orchestrator.run_all(sample_input)
    
    assert cache_hit_count == 1
    assert results["test"]["value"] == "cached"
    assert results["test"]["from_cache"] is True
    assert results["test"]["score"] == 0.99


@pytest.mark.asyncio
async def test_post_execution_hook(sample_input, simple_critic):
    """Test that post-execution hook is called after success."""
    post_hook_calls = []
    
    async def post_hook(critic_name: str, result: Dict[str, Any], duration_ms: float):
        post_hook_calls.append({
            "critic": critic_name,
            "result": result,
            "duration": duration_ms,
        })
    
    hooks = OrchestratorHooks(post_execution=post_hook)
    
    orchestrator = OrchestratorV2(
        critics={"test": simple_critic},
        timeout_seconds=5.0,
        hooks=hooks,
    )
    
    results = await orchestrator.run_all(sample_input)
    
    assert len(post_hook_calls) == 1
    assert post_hook_calls[0]["critic"] == "test"
    assert post_hook_calls[0]["result"]["value"] == "fairness"
    assert post_hook_calls[0]["duration"] > 0


@pytest.mark.asyncio
async def test_on_failure_hook(sample_input, failing_critic):
    """Test that on-failure hook is called on errors."""
    failure_hook_calls = []
    
    async def failure_hook(critic_name: str, error: Exception, duration_ms: float):
        failure_hook_calls.append({
            "critic": critic_name,
            "error_type": type(error).__name__,
            "duration": duration_ms,
        })
        # Return fallback result
        return {
            "value": "fallback",
            "score": 0.0,
            "severity": 0.0,
            "violation": False,
            "violations": [],
            "degraded": True,
        }
    
    hooks = OrchestratorHooks(on_failure=failure_hook)
    
    orchestrator = OrchestratorV2(
        critics={"test": failing_critic},
        timeout_seconds=5.0,
        hooks=hooks,
    )
    
    results = await orchestrator.run_all(sample_input)
    
    assert len(failure_hook_calls) == 1
    assert failure_hook_calls[0]["critic"] == "test"
    assert failure_hook_calls[0]["error_type"] == "ValueError"
    assert results["test"]["value"] == "fallback"
    assert results["test"]["degraded"] is True


@pytest.mark.asyncio
async def test_hook_failure_doesnt_crash(sample_input, simple_critic):
    """Test that hook failures don't crash critic execution."""
    async def broken_post_hook(critic_name: str, result: Dict[str, Any], duration_ms: float):
        raise RuntimeError("Hook is broken!")
    
    hooks = OrchestratorHooks(post_execution=broken_post_hook)
    
    orchestrator = OrchestratorV2(
        critics={"test": simple_critic},
        timeout_seconds=5.0,
        hooks=hooks,
    )
    
    # Should not raise exception
    results = await orchestrator.run_all(sample_input)
    
    # Critic should still succeed despite hook failure
    assert results["test"]["execution_status"] == "success"
    assert results["test"]["score"] == 0.8


# ============================================================
# Performance Tests
# ============================================================


@pytest.mark.asyncio
async def test_orchestrator_parallelism(sample_input):
    """Test that critics run in parallel, not sequentially."""
    async def delayed_critic(delay: float):
        async def critic_fn(input_dict: Dict[str, Any]) -> Dict[str, Any]:
            await asyncio.sleep(delay)
            return {"value": "test", "score": 1.0, "violation": False, "violations": []}
        return critic_fn
    
    orchestrator = OrchestratorV2(
        critics={
            "critic1": await delayed_critic(0.1),
            "critic2": await delayed_critic(0.1),
            "critic3": await delayed_critic(0.1),
        },
        timeout_seconds=5.0,
    )
    
    start = asyncio.get_event_loop().time()
    results = await orchestrator.run_all(sample_input)
    elapsed = asyncio.get_event_loop().time() - start
    
    # If sequential: 0.3s, if parallel: ~0.1s
    assert elapsed < 0.2  # Should be close to 0.1s
    assert len(results) == 3


@pytest.mark.asyncio
async def test_orchestrator_timing_accuracy(sample_input, simple_critic):
    """Test that duration_ms is accurately measured."""
    orchestrator = OrchestratorV2(
        critics={"test": simple_critic},
        timeout_seconds=5.0,
    )
    
    results = await orchestrator.run_all(sample_input)
    
    duration_ms = results["test"]["duration_ms"]
    assert duration_ms > 0
    assert duration_ms < 100  # Simple critic should be very fast


# ============================================================
# Sync Wrapper Tests
# ============================================================


def test_orchestrator_sync_wrapper(sample_input, simple_critic):
    """Test that sync wrapper works correctly."""
    orchestrator = OrchestratorV2(
        critics={"test": simple_critic},
        timeout_seconds=5.0,
    )
    
    # This should not raise an exception
    results = orchestrator.run(sample_input)
    
    assert "test" in results
    assert results["test"]["execution_status"] == "success"


# ============================================================
# Integration Tests
# ============================================================


@pytest.mark.asyncio
async def test_orchestrator_complete_workflow(sample_input):
    """Test complete workflow with multiple critics and hooks."""
    # Track all hook calls
    hooks_called = {
        "pre": [],
        "post": [],
        "fail": [],
    }
    
    async def pre_hook(critic_name: str, input_snapshot: CriticInput):
        hooks_called["pre"].append(critic_name)
        return None  # No cache
    
    async def post_hook(critic_name: str, result: Dict[str, Any], duration_ms: float):
        hooks_called["post"].append(critic_name)
    
    async def fail_hook(critic_name: str, error: Exception, duration_ms: float):
        hooks_called["fail"].append(critic_name)
        return None
    
    hooks = OrchestratorHooks(
        pre_execution=pre_hook,
        post_execution=post_hook,
        on_failure=fail_hook,
    )
    
    # Create mixed critics
    async def good_critic(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {"value": "good", "score": 0.8, "violation": False, "violations": []}
    
    async def bad_critic(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        raise ValueError("Bad critic")
    
    orchestrator = OrchestratorV2(
        critics={
            "good1": good_critic,
            "bad": bad_critic,
            "good2": good_critic,
        },
        timeout_seconds=5.0,
        hooks=hooks,
    )
    
    results = await orchestrator.run_all(sample_input)
    
    # All critics should have pre-hook called
    assert len(hooks_called["pre"]) == 3
    assert set(hooks_called["pre"]) == {"good1", "good2", "bad"}
    
    # Only successful critics should have post-hook called
    assert len(hooks_called["post"]) == 2
    assert "bad" not in hooks_called["post"]
    
    # Only failed critic should have fail-hook called
    assert hooks_called["fail"] == ["bad"]
    
    # All critics should return results
    assert len(results) == 3
    assert results["good1"]["execution_status"] == "success"
    assert results["good2"]["execution_status"] == "success"
    assert results["bad"]["execution_status"] == "error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
