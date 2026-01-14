"""
Tests for Production-Grade Orchestrator
---------------------------------------

Tests advanced features:
- Staged execution with dependencies
- Policy-based gating
- Resource management
- Retry strategies
- Result validation
- Priority scheduling
"""

import pytest
import asyncio
from typing import Dict, Any

from engine.orchestrator.orchestrator_production import (
    ProductionOrchestrator,
    CriticConfig,
    OrchestratorConfig,
    CriticInput,
    ExecutionStage,
    ExecutionPolicy,
    CriticExecutionStatus,
)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def sample_input():
    """Create sample input."""
    return CriticInput(
        model_response="Test response",
        input_text="Test input",
        context={"domain": "test", "risk_tier": "medium"},
        trace_id="test-123",
    )


@pytest.fixture
def simple_config():
    """Simple orchestrator config."""
    return OrchestratorConfig(
        max_concurrent_critics=5,
        enable_policy_gating=True,
    )


# ============================================================
# Staged Execution Tests
# ============================================================


@pytest.mark.asyncio
async def test_staged_execution_order(sample_input, simple_config):
    """Test that critics execute in correct stage order."""
    execution_order = []
    
    async def make_critic(stage: ExecutionStage, name: str):
        async def critic(input_dict: Dict[str, Any]) -> Dict[str, Any]:
            execution_order.append((stage.value, name))
            return {"value": name, "score": 1.0, "violation": False, "violations": []}
        return critic
    
    critics = {
        "pre": CriticConfig(
            name="pre",
            callable=await make_critic(ExecutionStage.PRE_VALIDATION, "pre"),
            stage=ExecutionStage.PRE_VALIDATION,
        ),
        "core": CriticConfig(
            name="core",
            callable=await make_critic(ExecutionStage.CORE_ANALYSIS, "core"),
            stage=ExecutionStage.CORE_ANALYSIS,
        ),
        "post": CriticConfig(
            name="post",
            callable=await make_critic(ExecutionStage.POST_PROCESSING, "post"),
            stage=ExecutionStage.POST_PROCESSING,
        ),
    }
    
    orchestrator = ProductionOrchestrator(critics, simple_config)
    await orchestrator.run_all(sample_input)
    
    # Verify execution order follows stages
    stages_executed = [stage for stage, _ in execution_order]
    assert stages_executed == [
        ExecutionStage.PRE_VALIDATION.value,
        ExecutionStage.CORE_ANALYSIS.value,
        ExecutionStage.POST_PROCESSING.value,
    ]


@pytest.mark.asyncio
async def test_dependency_resolution(sample_input, simple_config):
    """Test that dependencies are respected."""
    execution_order = []
    
    async def tracking_critic(name: str):
        async def critic(input_dict: Dict[str, Any]) -> Dict[str, Any]:
            execution_order.append(name)
            # Check dependencies were executed first
            prior = input_dict.get("prior_results", {})
            return {"value": name, "score": 1.0, "violation": False, "violations": [], "prior": list(prior.keys())}
        return critic
    
    critics = {
        "a": CriticConfig(name="a", callable=await tracking_critic("a")),
        "b": CriticConfig(name="b", callable=await tracking_critic("b"), depends_on=["a"]),
        "c": CriticConfig(name="c", callable=await tracking_critic("c"), depends_on=["a", "b"]),
    }
    
    orchestrator = ProductionOrchestrator(critics, simple_config)
    results = await orchestrator.run_all(sample_input)
    
    # Verify order
    assert execution_order.index("a") < execution_order.index("b")
    assert execution_order.index("b") < execution_order.index("c")
    
    # Verify dependencies were available
    assert "a" in results["b"]["prior"]
    assert "a" in results["c"]["prior"]
    assert "b" in results["c"]["prior"]


@pytest.mark.asyncio
async def test_circular_dependency_detection(sample_input, simple_config):
    """Test that circular dependencies are detected."""
    async def dummy_critic(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {"value": "test", "score": 1.0, "violation": False, "violations": []}
    
    critics = {
        "a": CriticConfig(name="a", callable=dummy_critic, depends_on=["b"]),
        "b": CriticConfig(name="b", callable=dummy_critic, depends_on=["a"]),
    }
    
    with pytest.raises(ValueError, match="Circular dependency"):
        ProductionOrchestrator(critics, simple_config)


# ============================================================
# Policy Gating Tests
# ============================================================


@pytest.mark.asyncio
async def test_always_policy(sample_input, simple_config):
    """Test ALWAYS policy executes unconditionally."""
    async def critic(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {"value": "test", "score": 1.0, "violation": False, "violations": []}
    
    critics = {
        "always": CriticConfig(
            name="always",
            callable=critic,
            execution_policy=ExecutionPolicy.ALWAYS,
        ),
    }
    
    orchestrator = ProductionOrchestrator(critics, simple_config)
    results = await orchestrator.run_all(sample_input)
    
    assert results["always"]["execution_status"] == "success"


@pytest.mark.asyncio
async def test_on_violation_policy_gates(sample_input, simple_config):
    """Test ON_VIOLATION policy gates when no violations."""
    async def critic(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {"value": "test", "score": 1.0, "violation": False, "violations": []}
    
    critics = {
        "check": CriticConfig(
            name="check",
            callable=critic,
            execution_policy=ExecutionPolicy.ON_VIOLATION,
        ),
    }
    
    orchestrator = ProductionOrchestrator(critics, simple_config)
    results = await orchestrator.run_all(sample_input)
    
    assert results["check"]["execution_status"] == "gated"
    assert "no_prior_violations" in results["check"]["gating_reason"]


@pytest.mark.asyncio
async def test_on_violation_policy_executes(sample_input, simple_config):
    """Test ON_VIOLATION policy executes when violations found."""
    async def violating_critic(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {"value": "test", "score": 1.0, "violation": True, "violations": ["issue"]}
    
    async def checking_critic(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {"value": "check", "score": 0.5, "violation": False, "violations": []}
    
    critics = {
        "first": CriticConfig(name="first", callable=violating_critic, priority=1),
        "second": CriticConfig(
            name="second",
            callable=checking_critic,
            execution_policy=ExecutionPolicy.ON_VIOLATION,
            priority=2,
        ),
    }
    
    orchestrator = ProductionOrchestrator(critics, simple_config)
    results = await orchestrator.run_all(sample_input)
    
    assert results["first"]["violation"] is True
    assert results["second"]["execution_status"] == "success"


@pytest.mark.asyncio
async def test_on_high_risk_policy(sample_input, simple_config):
    """Test ON_HIGH_RISK policy respects risk tier."""
    async def critic(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {"value": "test", "score": 1.0, "violation": False, "violations": []}
    
    critics = {
        "risk_check": CriticConfig(
            name="risk_check",
            callable=critic,
            execution_policy=ExecutionPolicy.ON_HIGH_RISK,
        ),
    }
    
    orchestrator = ProductionOrchestrator(critics, simple_config)
    
    # Medium risk - should be gated
    results = await orchestrator.run_all(sample_input)
    assert results["risk_check"]["execution_status"] == "gated"
    
    # High risk - should execute
    high_risk_input = CriticInput(
        model_response="Test",
        input_text="Test",
        context={"risk_tier": "high"},
        trace_id="test-456",
    )
    results = await orchestrator.run_all(high_risk_input)
    assert results["risk_check"]["execution_status"] == "success"


@pytest.mark.asyncio
async def test_conditional_policy(sample_input, simple_config):
    """Test CONDITIONAL policy with custom function."""
    async def critic(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {"value": "test", "score": 1.0, "violation": False, "violations": []}
    
    def custom_condition(input_dict: Dict[str, Any]) -> bool:
        # Only execute if domain is "special"
        return input_dict.get("context", {}).get("domain") == "special"
    
    critics = {
        "conditional": CriticConfig(
            name="conditional",
            callable=critic,
            execution_policy=ExecutionPolicy.CONDITIONAL,
            policy_condition=custom_condition,
        ),
    }
    
    orchestrator = ProductionOrchestrator(critics, simple_config)
    
    # Normal domain - gated
    results = await orchestrator.run_all(sample_input)
    assert results["conditional"]["execution_status"] == "gated"
    
    # Special domain - executes
    special_input = CriticInput(
        model_response="Test",
        input_text="Test",
        context={"domain": "special"},
        trace_id="test-789",
    )
    results = await orchestrator.run_all(special_input)
    assert results["conditional"]["execution_status"] == "success"


# ============================================================
# Priority Scheduling Tests
# ============================================================


@pytest.mark.asyncio
async def test_priority_ordering(sample_input, simple_config):
    """Test that critics execute in priority order within stage."""
    execution_order = []
    
    async def tracking_critic(name: str):
        async def critic(input_dict: Dict[str, Any]) -> Dict[str, Any]:
            execution_order.append(name)
            await asyncio.sleep(0.01)  # Small delay
            return {"value": name, "score": 1.0, "violation": False, "violations": []}
        return critic
    
    critics = {
        "low": CriticConfig(name="low", callable=await tracking_critic("low"), priority=10),
        "high": CriticConfig(name="high", callable=await tracking_critic("high"), priority=1),
        "medium": CriticConfig(name="medium", callable=await tracking_critic("medium"), priority=5),
    }
    
    orchestrator = ProductionOrchestrator(critics, simple_config)
    await orchestrator.run_all(sample_input)
    
    # High priority (1) should execute before medium (5) and low (10)
    assert execution_order.index("high") < execution_order.index("medium")
    assert execution_order.index("medium") < execution_order.index("low")


# ============================================================
# Retry Strategy Tests
# ============================================================


@pytest.mark.asyncio
async def test_retry_on_timeout(sample_input):
    """Test retry strategy for timeouts."""
    attempts = []
    
    async def flaky_critic(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        attempts.append(1)
        if len(attempts) < 3:
            await asyncio.sleep(2.0)  # Timeout
        return {"value": "success", "score": 1.0, "violation": False, "violations": []}
    
    critics = {
        "flaky": CriticConfig(
            name="flaky",
            callable=flaky_critic,
            timeout_seconds=0.5,
            max_retries=3,
            retry_on_timeout=True,
            retry_backoff_base=1.5,
        ),
    }
    
    config = OrchestratorConfig(enable_retries=True)
    orchestrator = ProductionOrchestrator(critics, config)
    results = await orchestrator.run_all(sample_input)
    
    assert len(attempts) == 3  # Initial + 2 retries
    assert results["flaky"]["execution_status"] == "success"
    assert results["flaky"]["retry_count"] == 2


@pytest.mark.asyncio
async def test_max_retries_exhausted(sample_input):
    """Test that retries are limited."""
    attempts = []
    
    async def always_fails(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        attempts.append(1)
        await asyncio.sleep(1.0)  # Always timeout
        return {}
    
    critics = {
        "fails": CriticConfig(
            name="fails",
            callable=always_fails,
            timeout_seconds=0.1,
            max_retries=2,
            retry_on_timeout=True,
            retry_backoff_base=1.1,
        ),
    }
    
    config = OrchestratorConfig(enable_retries=True)
    orchestrator = ProductionOrchestrator(critics, config)
    results = await orchestrator.run_all(sample_input)
    
    assert len(attempts) == 3  # Initial + 2 retries
    assert results["fails"]["execution_status"] == "error"


# ============================================================
# Result Validation Tests
# ============================================================


@pytest.mark.asyncio
async def test_required_fields_validation(sample_input):
    """Test validation of required output fields."""
    async def incomplete_critic(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {"score": 1.0}  # Missing 'value' field
    
    critics = {
        "incomplete": CriticConfig(
            name="incomplete",
            callable=incomplete_critic,
            required_output_fields={"value", "score"},
        ),
    }
    
    config = OrchestratorConfig(strict_validation=True, fail_on_validation_error=False)
    orchestrator = ProductionOrchestrator(critics, config)
    results = await orchestrator.run_all(sample_input)
    
    assert "validation_errors" in results["incomplete"]
    assert any("missing_required_field: value" in err for err in results["incomplete"]["validation_errors"])


@pytest.mark.asyncio
async def test_custom_validation(sample_input):
    """Test custom validation function."""
    async def critic(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {"value": "test", "score": -1.0}  # Invalid score
    
    def validate_score(output: Dict[str, Any]) -> bool:
        score = output.get("score", 0)
        return 0 <= score <= 1.0
    
    critics = {
        "validated": CriticConfig(
            name="validated",
            callable=critic,
            validate_output=validate_score,
        ),
    }
    
    config = OrchestratorConfig(strict_validation=True, fail_on_validation_error=False)
    orchestrator = ProductionOrchestrator(critics, config)
    results = await orchestrator.run_all(sample_input)
    
    assert "validation_errors" in results["validated"]
    assert any("custom_validation_failed" in err for err in results["validated"]["validation_errors"])


# ============================================================
# Resource Management Tests
# ============================================================


@pytest.mark.asyncio
async def test_global_concurrency_limit(sample_input):
    """Test global concurrency limit."""
    concurrent_count = 0
    max_concurrent = 0
    lock = asyncio.Lock()
    
    async def concurrent_critic(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal concurrent_count, max_concurrent
        async with lock:
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
        
        await asyncio.sleep(0.1)
        
        async with lock:
            concurrent_count -= 1
        
        return {"value": "test", "score": 1.0, "violation": False, "violations": []}
    
    # Create many critics
    critics = {
        f"critic_{i}": CriticConfig(
            name=f"critic_{i}",
            callable=concurrent_critic,
        )
        for i in range(20)
    }
    
    config = OrchestratorConfig(max_concurrent_critics=3)
    orchestrator = ProductionOrchestrator(critics, config)
    await orchestrator.run_all(sample_input)
    
    # Should never exceed limit
    assert max_concurrent <= 3


@pytest.mark.asyncio
async def test_per_critic_concurrency_limit(sample_input):
    """Test per-critic concurrency limit."""
    concurrent_count = 0
    max_concurrent = 0
    lock = asyncio.Lock()
    
    async def concurrent_critic(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal concurrent_count, max_concurrent
        async with lock:
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
        
        await asyncio.sleep(0.05)
        
        async with lock:
            concurrent_count -= 1
        
        return {"value": "test", "score": 1.0, "violation": False, "violations": []}
    
    critics = {
        "limited": CriticConfig(
            name="limited",
            callable=concurrent_critic,
            max_concurrent=2,  # Limit this specific critic
        ),
    }
    
    config = OrchestratorConfig(max_concurrent_critics=10)
    orchestrator = ProductionOrchestrator(critics, config)
    
    # Run multiple times concurrently
    tasks = [orchestrator.run_all(sample_input) for _ in range(10)]
    await asyncio.gather(*tasks)
    
    # Should respect per-critic limit
    assert max_concurrent <= 2


@pytest.mark.asyncio
async def test_rate_limiting(sample_input):
    """Test rate limiting."""
    execution_times = []
    
    async def rate_limited_critic(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        execution_times.append(asyncio.get_event_loop().time())
        return {"value": "test", "score": 1.0, "violation": False, "violations": []}
    
    critics = {
        "limited": CriticConfig(
            name="limited",
            callable=rate_limited_critic,
            rate_limit_per_second=2.0,  # Max 2 per second
        ),
    }
    
    config = OrchestratorConfig()
    orchestrator = ProductionOrchestrator(critics, config)
    
    # Execute 5 times rapidly
    start = asyncio.get_event_loop().time()
    tasks = [orchestrator.run_all(sample_input) for _ in range(5)]
    await asyncio.gather(*tasks)
    total_time = asyncio.get_event_loop().time() - start
    
    # Should take at least 2 seconds for 5 executions at 2/sec
    assert total_time >= 2.0


# ============================================================
# Metrics Tests
# ============================================================


@pytest.mark.asyncio
async def test_execution_metrics(sample_input, simple_config):
    """Test that metrics are tracked."""
    async def success_critic(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {"value": "test", "score": 1.0, "violation": False, "violations": []}
    
    async def failing_critic(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        raise ValueError("Intentional failure")
    
    critics = {
        "success": CriticConfig(name="success", callable=success_critic),
        "failure": CriticConfig(name="failure", callable=failing_critic),
    }
    
    orchestrator = ProductionOrchestrator(critics, simple_config)
    await orchestrator.run_all(sample_input)
    
    metrics = orchestrator.metrics
    assert metrics.total_executions == 2
    assert metrics.successful == 1
    assert metrics.failed == 1
    assert metrics.total_duration_ms > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
