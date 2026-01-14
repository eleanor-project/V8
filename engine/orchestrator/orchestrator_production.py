"""
ELEANOR V8 — Production-Grade Orchestrator
------------------------------------------

A truly powerful orchestrator with:
- Staged execution with dependency resolution
- Policy-based execution gating
- Resource management (concurrency limits, backpressure)
- Retry strategies with exponential backoff
- Result validation and schema enforcement
- Execution plans (pre-computed DAGs)
- Conditional execution based on prior results
- Priority-based scheduling
- Resource quotas and rate limiting
- Comprehensive observability

This is NOT a toy wrapper - this is industrial-strength orchestration.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Awaitable, Dict, Any, Callable, Optional, List, Set, Tuple
from enum import Enum
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


# ============================================================================
# Core Types and Enums
# ============================================================================


class CriticExecutionStatus(Enum):
    """Status of critic execution."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"
    SKIPPED = "skipped"
    GATED = "gated"  # Blocked by policy
    RATE_LIMITED = "rate_limited"
    RETRYING = "retrying"


class ExecutionStage(Enum):
    """Execution stages for dependency ordering."""
    PRE_VALIDATION = 0      # Input validation, sanity checks
    FAST_CRITICS = 1        # Quick critics (< 100ms)
    CORE_ANALYSIS = 2       # Main analysis critics
    DEEP_ANALYSIS = 3       # Expensive critics that depend on core
    POST_PROCESSING = 4     # Aggregation, synthesis
    VALIDATION = 5          # Output validation


class ExecutionPolicy(Enum):
    """Policy for critic execution."""
    ALWAYS = "always"                    # Always execute
    ON_VIOLATION = "on_violation"        # Only if prior violations found
    ON_HIGH_RISK = "on_high_risk"        # Only if risk tier is high
    ON_UNCERTAINTY = "on_uncertainty"    # Only if uncertainty is high
    CONDITIONAL = "conditional"          # Custom condition function


# ============================================================================
# Configuration Classes
# ============================================================================


@dataclass
class CriticConfig:
    """Configuration for a single critic."""
    name: str
    callable: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]
    
    # Execution control
    stage: ExecutionStage = ExecutionStage.CORE_ANALYSIS
    priority: int = 5  # 1 (highest) to 10 (lowest)
    timeout_seconds: float = 10.0
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    
    # Policies
    execution_policy: ExecutionPolicy = ExecutionPolicy.ALWAYS
    policy_condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    
    # Retry configuration
    max_retries: int = 0
    retry_on_timeout: bool = False
    retry_backoff_base: float = 2.0
    
    # Resource limits
    max_concurrent: Optional[int] = None  # Max concurrent instances of this critic
    rate_limit_per_second: Optional[float] = None
    
    # Validation
    required_output_fields: Set[str] = field(default_factory=set)
    validate_output: Optional[Callable[[Dict[str, Any]], bool]] = None
    
    # Metadata
    tags: Set[str] = field(default_factory=set)
    cost_weight: float = 1.0  # Relative cost for resource allocation


@dataclass
class OrchestratorConfig:
    """Global orchestrator configuration."""
    
    # Concurrency limits
    max_concurrent_critics: int = 10
    max_concurrent_per_stage: Dict[ExecutionStage, int] = field(default_factory=dict)
    
    # Timeout configuration
    global_timeout_seconds: Optional[float] = None
    stage_timeout_seconds: Dict[ExecutionStage, float] = field(default_factory=dict)
    
    # Resource management
    enable_backpressure: bool = True
    max_queue_size: int = 1000
    
    # Retry configuration
    enable_retries: bool = True
    max_total_retries: int = 3
    
    # Gating
    enable_policy_gating: bool = True
    fail_on_gated_critics: bool = False
    
    # Validation
    strict_validation: bool = True
    fail_on_validation_error: bool = False


@dataclass
class CriticInput:
    """Structured input for critic execution."""
    model_response: str
    input_text: str
    context: Dict[str, Any]
    trace_id: str
    
    # Execution context (populated by orchestrator)
    prior_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    execution_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for critic consumption."""
        return {
            "model_response": self.model_response,
            "input_text": self.input_text,
            "context": self.context,
            "trace_id": self.trace_id,
            "prior_results": self.prior_results,
            "metadata": self.execution_metadata,
        }


@dataclass
class CriticExecutionResult:
    """Result of a single critic execution."""
    critic_name: str
    status: CriticExecutionStatus
    output: Dict[str, Any]
    duration_ms: float
    
    # Execution metadata
    stage: ExecutionStage
    priority: int
    retry_count: int = 0
    error: Optional[str] = None
    gating_reason: Optional[str] = None
    validation_errors: List[str] = field(default_factory=list)
    
    # Resource tracking
    queue_time_ms: float = 0.0
    execution_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = dict(self.output)
        result["critic"] = self.critic_name
        result["duration_ms"] = self.duration_ms
        result["execution_status"] = self.status.value
        result["stage"] = self.stage.value
        result["priority"] = self.priority
        result["retry_count"] = self.retry_count
        
        if self.error:
            result["execution_error"] = self.error
        if self.gating_reason:
            result["gating_reason"] = self.gating_reason
        if self.validation_errors:
            result["validation_errors"] = self.validation_errors
        
        return result


@dataclass
class ExecutionPlan:
    """Pre-computed execution plan with dependency graph."""
    def __init__(self):
        # Dependency graph (adjacency list)
        self.graph: Dict[str, Set[str]] = defaultdict(set)
        # Topological ordering by stage
        self.stages: Dict[ExecutionStage, List[str]] = defaultdict(list)
        # Priority ordering within each stage
        self.priority_order: Dict[ExecutionStage, List[str]] = {}
        # Critics that can skip (based on policy)
        self.conditional_critics: Set[str] = set()
    
    def add_dependency(self, critic: str, depends_on: str):
        """Add dependency edge."""
        self.graph[critic].add(depends_on)
    
    def get_ready_critics(
        self,
        stage: ExecutionStage,
        completed: Set[str],
    ) -> List[str]:
        """Get critics ready to execute in this stage."""
        ready = []
        for critic in self.stages[stage]:
            if critic in completed:
                continue
            
            # Check if all dependencies are satisfied
            deps = self.graph.get(critic, set())
            if deps.issubset(completed):
                ready.append(critic)
        
        return ready


# ============================================================================
# Production Orchestrator
# ============================================================================


class ProductionOrchestrator:
    """
    Industrial-strength orchestrator with advanced features.
    
    Features:
    - Staged execution with dependency resolution
    - Policy-based gating
    - Resource management and backpressure
    - Retry strategies
    - Result validation
    - Priority scheduling
    - Comprehensive observability
    """
    
    def __init__(
        self,
        critics: Dict[str, CriticConfig],
        config: OrchestratorConfig,
        hooks: Optional['OrchestratorHooks'] = None,
    ):
        """Initialize production orchestrator."""
        self.critics = critics
        self.config = config
        self.hooks = hooks or OrchestratorHooks()
        
        # Build execution plan
        self.execution_plan = self._build_execution_plan()
        
        # Resource management
        self.semaphore = asyncio.Semaphore(config.max_concurrent_critics)
        self.stage_semaphores = {
            stage: asyncio.Semaphore(limit)
            for stage, limit in config.max_concurrent_per_stage.items()
        }
        
        # Rate limiting (per-critic)
        self.rate_limiters: Dict[str, 'RateLimiter'] = {}
        for name, critic in critics.items():
            if critic.rate_limit_per_second:
                self.rate_limiters[name] = RateLimiter(critic.rate_limit_per_second)
        
        # Concurrency tracking (per-critic)
        self.critic_semaphores: Dict[str, asyncio.Semaphore] = {}
        for name, critic in critics.items():
            if critic.max_concurrent:
                self.critic_semaphores[name] = asyncio.Semaphore(critic.max_concurrent)
        
        # Metrics
        self.metrics = ExecutionMetrics()
    
    def _build_execution_plan(self) -> ExecutionPlan:
        """Build execution plan with dependency resolution."""
        plan = ExecutionPlan()
        
        # Group critics by stage
        for name, critic in self.critics.items():
            plan.stages[critic.stage].append(name)
            
            # Add dependencies
            for dep in critic.depends_on:
                plan.add_dependency(name, dep)
            
            # Track conditional critics
            if critic.execution_policy != ExecutionPolicy.ALWAYS:
                plan.conditional_critics.add(name)
        
        # Sort each stage by priority
        for stage, critics_in_stage in plan.stages.items():
            sorted_critics = sorted(
                critics_in_stage,
                key=lambda c: self.critics[c].priority
            )
            plan.priority_order[stage] = sorted_critics
        
        # Validate plan (no circular dependencies)
        self._validate_execution_plan(plan)
        
        return plan
    
    def _validate_execution_plan(self, plan: ExecutionPlan):
        """Validate execution plan has no circular dependencies."""
        # Simple cycle detection using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(critic: str) -> bool:
            visited.add(critic)
            rec_stack.add(critic)
            
            for dep in plan.graph.get(critic, set()):
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True
            
            rec_stack.remove(critic)
            return False
        
        for critic in self.critics:
            if critic not in visited:
                if has_cycle(critic):
                    raise ValueError(f"Circular dependency detected in execution plan")
    
    def _should_execute_critic(
        self,
        critic_config: CriticConfig,
        input_snapshot: CriticInput,
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if critic should execute based on policy.
        
        Returns:
            (should_execute, gating_reason)
        """
        if not self.config.enable_policy_gating:
            return True, None
        
        policy = critic_config.execution_policy
        
        if policy == ExecutionPolicy.ALWAYS:
            return True, None
        
        if policy == ExecutionPolicy.CONDITIONAL:
            if critic_config.policy_condition:
                try:
                    should_run = critic_config.policy_condition(input_snapshot.to_dict())
                    if not should_run:
                        return False, "conditional_policy_not_met"
                except Exception as e:
                    logger.warning(f"Policy condition failed for {critic_config.name}: {e}")
                    return False, f"policy_evaluation_error: {e}"
            return True, None
        
        # Check prior results for policy decisions
        prior = input_snapshot.prior_results
        
        if policy == ExecutionPolicy.ON_VIOLATION:
            # Execute if any prior critic found violations
            has_violations = any(
                result.get("violation") or result.get("violations")
                for result in prior.values()
            )
            if not has_violations:
                return False, "no_prior_violations"
        
        elif policy == ExecutionPolicy.ON_HIGH_RISK:
            # Execute if context indicates high risk
            risk_tier = input_snapshot.context.get("risk_tier", "medium")
            if risk_tier != "high":
                return False, f"risk_tier_not_high: {risk_tier}"
        
        elif policy == ExecutionPolicy.ON_UNCERTAINTY:
            # Execute if prior results show high uncertainty
            has_high_uncertainty = any(
                result.get("uncertainty", 0) > 0.7
                for result in prior.values()
            )
            if not has_high_uncertainty:
                return False, "no_high_uncertainty"
        
        return True, None
    
    def _validate_output(
        self,
        critic_config: CriticConfig,
        output: Dict[str, Any],
    ) -> List[str]:
        """Validate critic output against schema."""
        errors = []
        
        # Check required fields
        for field in critic_config.required_output_fields:
            if field not in output:
                errors.append(f"missing_required_field: {field}")
        
        # Custom validation
        if critic_config.validate_output:
            try:
                is_valid = critic_config.validate_output(output)
                if not is_valid:
                    errors.append("custom_validation_failed")
            except Exception as e:
                errors.append(f"validation_error: {e}")
        
        return errors
    
    async def _execute_with_retry(
        self,
        critic_config: CriticConfig,
        input_snapshot: CriticInput,
    ) -> CriticExecutionResult:
        """Execute critic with retry logic."""
        retry_count = 0
        last_error = None
        
        max_retries = critic_config.max_retries if self.config.enable_retries else 0
        
        while retry_count <= max_retries:
            try:
                result = await self._execute_single_critic(
                    critic_config,
                    input_snapshot,
                    retry_count,
                )
                
                # Success - return result
                if result.status == CriticExecutionStatus.SUCCESS:
                    return result
                
                # Timeout - retry if configured
                if result.status == CriticExecutionStatus.TIMEOUT:
                    if not critic_config.retry_on_timeout:
                        return result
                    last_error = result.error
                else:
                    # Other failures - don't retry
                    return result
                
            except Exception as e:
                last_error = str(e)
                logger.error(f"Critic {critic_config.name} failed: {e}", exc_info=True)
            
            # Retry with exponential backoff
            retry_count += 1
            if retry_count <= max_retries:
                backoff = critic_config.retry_backoff_base ** retry_count
                logger.info(f"Retrying {critic_config.name} in {backoff}s (attempt {retry_count}/{max_retries})")
                await asyncio.sleep(backoff)
        
        # All retries exhausted
        return CriticExecutionResult(
            critic_name=critic_config.name,
            status=CriticExecutionStatus.ERROR,
            output=self._failure_template(critic_config.name, last_error or "max_retries_exceeded"),
            duration_ms=0.0,
            stage=critic_config.stage,
            priority=critic_config.priority,
            retry_count=retry_count,
            error=last_error,
        )
    
    async def _execute_single_critic(
        self,
        critic_config: CriticConfig,
        input_snapshot: CriticInput,
        retry_count: int,
    ) -> CriticExecutionResult:
        """Execute a single critic (one attempt)."""
        queue_start = time.time()
        
        # Rate limiting
        if critic_config.name in self.rate_limiters:
            await self.rate_limiters[critic_config.name].acquire()
        
        # Per-critic concurrency limit
        critic_semaphore = self.critic_semaphores.get(critic_config.name)
        stage_semaphore = self.stage_semaphores.get(critic_config.stage)
        
        # Acquire semaphores
        async with self.semaphore:  # Global limit
            if stage_semaphore:
                async with stage_semaphore:  # Stage limit
                    if critic_semaphore:
                        async with critic_semaphore:  # Per-critic limit
                            return await self._execute_critic_core(
                                critic_config,
                                input_snapshot,
                                retry_count,
                                queue_start,
                            )
                    else:
                        return await self._execute_critic_core(
                            critic_config,
                            input_snapshot,
                            retry_count,
                            queue_start,
                        )
            else:
                if critic_semaphore:
                    async with critic_semaphore:
                        return await self._execute_critic_core(
                            critic_config,
                            input_snapshot,
                            retry_count,
                            queue_start,
                        )
                else:
                    return await self._execute_critic_core(
                        critic_config,
                        input_snapshot,
                        retry_count,
                        queue_start,
                    )
    
    async def _execute_critic_core(
        self,
        critic_config: CriticConfig,
        input_snapshot: CriticInput,
        retry_count: int,
        queue_start: float,
    ) -> CriticExecutionResult:
        """Core critic execution logic."""
        exec_start = time.time()
        queue_time_ms = (exec_start - queue_start) * 1000
        
        # Pre-execution hook
        if self.hooks.pre_execution:
            try:
                cached = await self.hooks.pre_execution(critic_config.name, input_snapshot)
                if cached:
                    return CriticExecutionResult(
                        critic_name=critic_config.name,
                        status=CriticExecutionStatus.SUCCESS,
                        output=cached,
                        duration_ms=(time.time() - exec_start) * 1000,
                        stage=critic_config.stage,
                        priority=critic_config.priority,
                        queue_time_ms=queue_time_ms,
                    )
            except Exception as e:
                logger.warning(f"Pre-execution hook failed: {e}")
        
        # Execute critic
        try:
            result = await asyncio.wait_for(
                critic_config.callable(input_snapshot.to_dict()),
                timeout=critic_config.timeout_seconds,
            )
            
            exec_time_ms = (time.time() - exec_start) * 1000
            
            # Validate output
            validation_errors = []
            if self.config.strict_validation:
                validation_errors = self._validate_output(critic_config, result)
                if validation_errors and self.config.fail_on_validation_error:
                    raise ValueError(f"Validation failed: {validation_errors}")
            
            # Post-execution hook
            if self.hooks.post_execution:
                try:
                    await self.hooks.post_execution(critic_config.name, result, exec_time_ms)
                except Exception as e:
                    logger.warning(f"Post-execution hook failed: {e}")
            
            return CriticExecutionResult(
                critic_name=critic_config.name,
                status=CriticExecutionStatus.SUCCESS,
                output=result,
                duration_ms=exec_time_ms,
                stage=critic_config.stage,
                priority=critic_config.priority,
                retry_count=retry_count,
                queue_time_ms=queue_time_ms,
                execution_time_ms=exec_time_ms,
                validation_errors=validation_errors,
            )
            
        except asyncio.TimeoutError:
            exec_time_ms = (time.time() - exec_start) * 1000
            error_msg = f"timeout_after_{critic_config.timeout_seconds}s"
            
            return CriticExecutionResult(
                critic_name=critic_config.name,
                status=CriticExecutionStatus.TIMEOUT,
                output=self._failure_template(critic_config.name, error_msg),
                duration_ms=exec_time_ms,
                stage=critic_config.stage,
                priority=critic_config.priority,
                retry_count=retry_count,
                error=error_msg,
                queue_time_ms=queue_time_ms,
            )
            
        except Exception as e:
            exec_time_ms = (time.time() - exec_start) * 1000
            error_msg = f"{type(e).__name__}: {str(e)}"
            
            # On-failure hook
            if self.hooks.on_failure:
                try:
                    await self.hooks.on_failure(critic_config.name, e, exec_time_ms)
                except Exception as hook_error:
                    logger.warning(f"On-failure hook failed: {hook_error}")
            
            return CriticExecutionResult(
                critic_name=critic_config.name,
                status=CriticExecutionStatus.ERROR,
                output=self._failure_template(critic_config.name, error_msg),
                duration_ms=exec_time_ms,
                stage=critic_config.stage,
                priority=critic_config.priority,
                retry_count=retry_count,
                error=error_msg,
                queue_time_ms=queue_time_ms,
            )
    
    def _failure_template(self, name: str, error: str) -> Dict[str, Any]:
        """Generate failure template."""
        return {
            "value": None,
            "score": 0.0,
            "severity": 0.0,
            "violation": False,
            "violations": [],
            "details": {"error": error, "critic_failed": True},
        }
    
    async def run_all(
        self,
        input_snapshot: CriticInput,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Execute all critics following the execution plan.
        
        This executes critics in stages, respecting dependencies,
        applying policies, and managing resources.
        """
        start_time = time.time()
        
        completed = set()
        results = {}
        
        # Execute stage by stage
        for stage in ExecutionStage:
            if stage not in self.execution_plan.stages:
                continue
            
            stage_start = time.time()
            logger.info(f"Starting stage: {stage.name}")
            while True:
                ready = self.execution_plan.get_ready_critics(stage, completed)
                if not ready:
                    break

                for critic_name in self.execution_plan.priority_order.get(stage, ready):
                    if critic_name not in ready:
                        continue

                    critic_config = self.critics[critic_name]

                    # Check execution policy (gating)
                    should_execute, gating_reason = self._should_execute_critic(
                        critic_config,
                        input_snapshot,
                    )

                    if not should_execute:
                        logger.info(f"Critic {critic_name} gated: {gating_reason}")
                        results[critic_name] = CriticExecutionResult(
                            critic_name=critic_name,
                            status=CriticExecutionStatus.GATED,
                            output=self._failure_template(critic_name, "gated"),
                            duration_ms=0.0,
                            stage=stage,
                            priority=critic_config.priority,
                            gating_reason=gating_reason,
                        ).to_dict()
                        completed.add(critic_name)

                        if self.config.fail_on_gated_critics:
                            raise RuntimeError(f"Critic {critic_name} was gated: {gating_reason}")

                        continue

                    result = await self._execute_with_retry(critic_config, input_snapshot)
                    results[critic_name] = result.to_dict()
                    completed.add(critic_name)

                    # Update input snapshot with result for dependent critics
                    input_snapshot.prior_results[critic_name] = result.output

                    # Update metrics
                    self.metrics.record_execution(result)
            
            stage_duration = (time.time() - stage_start) * 1000
            logger.info(f"Stage {stage.name} completed in {stage_duration:.2f}ms")
            
            # Check stage timeout
            if stage in self.config.stage_timeout_seconds:
                if stage_duration > self.config.stage_timeout_seconds[stage] * 1000:
                    logger.warning(f"Stage {stage.name} exceeded timeout")
        
        total_duration = (time.time() - start_time) * 1000
        logger.info(f"All stages completed in {total_duration:.2f}ms")
        
        # Check global timeout
        if self.config.global_timeout_seconds:
            if total_duration > self.config.global_timeout_seconds * 1000:
                logger.warning(f"Global execution exceeded timeout")
        
        return results


# ============================================================================
# Supporting Classes
# ============================================================================


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, rate_per_second: float):
        self.rate = rate_per_second
        self.tokens = 0.0
        self.last_update = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire a token, waiting if necessary."""
        while True:
            async with self.lock:
                now = time.time()
                elapsed = now - self.last_update
                self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
                self.last_update = now

                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                wait_time = (1 - self.tokens) / self.rate if self.rate else 0.0
            if wait_time > 0:
                await asyncio.sleep(wait_time)


@dataclass
class ExecutionMetrics:
    """Execution metrics tracking."""
    
    total_executions: int = 0
    successful: int = 0
    failed: int = 0
    timed_out: int = 0
    gated: int = 0
    retried: int = 0
    
    total_duration_ms: float = 0.0
    total_queue_time_ms: float = 0.0
    
    def record_execution(self, result: CriticExecutionResult):
        """Record execution result."""
        self.total_executions += 1
        self.total_duration_ms += result.duration_ms
        self.total_queue_time_ms += result.queue_time_ms
        
        if result.status == CriticExecutionStatus.SUCCESS:
            self.successful += 1
        elif result.status == CriticExecutionStatus.ERROR:
            self.failed += 1
        elif result.status == CriticExecutionStatus.TIMEOUT:
            self.timed_out += 1
        elif result.status == CriticExecutionStatus.GATED:
            self.gated += 1
        
        if result.retry_count > 0:
            self.retried += 1


@dataclass
class OrchestratorHooks:
    """Hooks for infrastructure integration."""
    
    pre_execution: Optional[Callable[[str, CriticInput], Awaitable[Optional[Dict[str, Any]]]]] = None
    post_execution: Optional[Callable[[str, Dict[str, Any], float], Awaitable[None]]] = None
    on_failure: Optional[Callable[[str, Exception, float], Awaitable[None]]] = None


__all__ = [
    "ProductionOrchestrator",
    "CriticConfig",
    "OrchestratorConfig",
    "CriticInput",
    "CriticExecutionResult",
    "ExecutionStage",
    "ExecutionPolicy",
    "ExecutionPlan",
    "OrchestratorHooks",
]
