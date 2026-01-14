"""
ELEANOR V8 — Enhanced Orchestrator V2
------------------------------------

Enhanced orchestrator with infrastructure hooks for production use.

Changes from V1:
- Structured input/output schemas
- Pre/post execution hooks for infrastructure
- Better error metadata
- Timing information per critic
- Support for async callbacks

The Orchestrator manages the critical execution path:
1. Load and run all critics in parallel
2. Enforce timeouts per critic
3. Ensure one critic failure does not collapse the engine
4. Normalize outputs into the schema the Aggregator expects
5. Return a complete `critic_outputs` dictionary with metadata

This is the backbone of the deliberation pipeline.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Awaitable, Dict, Any, Callable, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class CriticExecutionStatus(Enum):
    """Status of critic execution."""
    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class CriticInput:
    """Structured input for critic execution."""
    model_response: str
    input_text: str
    context: Dict[str, Any]
    trace_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for critic consumption."""
        return {
            "model_response": self.model_response,
            "input_text": self.input_text,
            "context": self.context,
            "trace_id": self.trace_id,
        }


@dataclass
class CriticExecutionResult:
    """Result of a single critic execution."""
    critic_name: str
    status: CriticExecutionStatus
    output: Dict[str, Any]
    duration_ms: float
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format expected by pipeline."""
        result = dict(self.output)
        result["critic"] = self.critic_name
        result["duration_ms"] = self.duration_ms
        result["execution_status"] = self.status.value
        if self.error:
            result["execution_error"] = self.error
        return result


@dataclass
class OrchestratorHooks:
    """Hooks for infrastructure integration."""
    
    # Called before executing a critic (for caching, circuit breaker checks)
    pre_execution: Optional[Callable[[str, CriticInput], Awaitable[Optional[Dict[str, Any]]]]] = None
    
    # Called after successful execution (for evidence recording, events)
    post_execution: Optional[Callable[[str, Dict[str, Any], float], Awaitable[None]]] = None
    
    # Called on execution failure (for error logging, degradation)
    on_failure: Optional[Callable[[str, Exception, float], Awaitable[Optional[Dict[str, Any]]]]] = None


class OrchestratorV2:
    """
    Enhanced orchestrator with infrastructure hooks.
    
    Responsibilities:
    - Parallel critic execution with timeout enforcement
    - Failure isolation (one critic failure doesn't crash pipeline)
    - Timing measurement per critic
    - Hook-based integration with infrastructure
    
    Does NOT handle:
    - Caching (use pre_execution hook)
    - Circuit breakers (use pre_execution hook)
    - Evidence recording (use post_execution hook)
    - Event emission (use post_execution hook)
    - Degradation (use on_failure hook)
    """
    
    def __init__(
        self,
        critics: Dict[str, Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]],
        timeout_seconds: float = 10.0,
        hooks: Optional[OrchestratorHooks] = None,
    ):
        """
        Initialize orchestrator.
        
        Args:
            critics: Dictionary of {critic_name: async_critic_callable}
            timeout_seconds: Maximum time allowed per critic
            hooks: Optional infrastructure hooks
        """
        self.critics = critics
        self.timeout = timeout_seconds
        self.hooks = hooks or OrchestratorHooks()
        
    def _critic_failure_template(
        self, 
        name: str, 
        error: str,
        status: CriticExecutionStatus = CriticExecutionStatus.ERROR
    ) -> Dict[str, Any]:
        """Generate standardized failure output."""
        return {
            "value": None,
            "score": 0.0,
            "severity": 0.0,
            "violation": False,
            "violations": [],
            "details": {
                "error": error,
                "critic_failed": True,
                "execution_status": status.value,
            },
        }
    
    async def _execute_critic(
        self,
        name: str,
        critic_fn: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]],
        input_snapshot: CriticInput,
    ) -> CriticExecutionResult:
        """
        Execute a single critic with full error handling and timing.
        
        Returns:
            CriticExecutionResult with status, output, and timing
        """
        start_time = asyncio.get_event_loop().time()
        
        # Pre-execution hook (cache check, circuit breaker)
        if self.hooks.pre_execution:
            try:
                cached_result = await self.hooks.pre_execution(name, input_snapshot)
                if cached_result is not None:
                    duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
                    logger.debug(f"Critic '{name}' returned from cache")
                    return CriticExecutionResult(
                        critic_name=name,
                        status=CriticExecutionStatus.SUCCESS,
                        output=cached_result,
                        duration_ms=duration_ms,
                    )
            except Exception as hook_error:
                logger.warning(
                    f"Pre-execution hook failed for critic '{name}': {hook_error}",
                    exc_info=True
                )
                # Continue with execution despite hook failure
        
        # Execute critic with timeout
        try:
            input_dict = input_snapshot.to_dict()
            result = await asyncio.wait_for(
                critic_fn(input_dict),
                timeout=self.timeout
            )
            
            duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Validate result is a dictionary
            if not isinstance(result, dict):
                raise ValueError(f"Critic returned non-dict: {type(result)}")
            
            # Post-execution hook (evidence recording, events)
            if self.hooks.post_execution:
                try:
                    await self.hooks.post_execution(name, result, duration_ms)
                except Exception as hook_error:
                    logger.warning(
                        f"Post-execution hook failed for critic '{name}': {hook_error}",
                        exc_info=True
                    )
            
            return CriticExecutionResult(
                critic_name=name,
                status=CriticExecutionStatus.SUCCESS,
                output=result,
                duration_ms=duration_ms,
            )
            
        except asyncio.TimeoutError:
            duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            error_msg = f"Critic timed out after {self.timeout}s"
            logger.warning(f"Critic '{name}' timed out after {self.timeout}s")
            
            # On-failure hook
            fallback_output = None
            if self.hooks.on_failure:
                try:
                    fallback_output = await self.hooks.on_failure(
                        name, 
                        asyncio.TimeoutError(error_msg),
                        duration_ms
                    )
                except Exception as hook_error:
                    logger.warning(
                        f"On-failure hook failed for critic '{name}': {hook_error}",
                        exc_info=True
                    )
            
            output = fallback_output or self._critic_failure_template(
                name, error_msg, CriticExecutionStatus.TIMEOUT
            )
            
            return CriticExecutionResult(
                critic_name=name,
                status=CriticExecutionStatus.TIMEOUT,
                output=output,
                duration_ms=duration_ms,
                error=error_msg,
            )
            
        except Exception as e:
            duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(
                f"Critic '{name}' failed with error: {error_msg}",
                exc_info=True
            )
            
            # On-failure hook
            fallback_output = None
            if self.hooks.on_failure:
                try:
                    fallback_output = await self.hooks.on_failure(name, e, duration_ms)
                except Exception as hook_error:
                    logger.warning(
                        f"On-failure hook failed for critic '{name}': {hook_error}",
                        exc_info=True
                    )
            
            output = fallback_output or self._critic_failure_template(
                name, error_msg, CriticExecutionStatus.ERROR
            )
            
            return CriticExecutionResult(
                critic_name=name,
                status=CriticExecutionStatus.ERROR,
                output=output,
                duration_ms=duration_ms,
                error=error_msg,
            )
    
    async def run_all(
        self, 
        input_snapshot: CriticInput
    ) -> Dict[str, Dict[str, Any]]:
        """
        Execute all critics concurrently with full error isolation.
        
        Args:
            input_snapshot: Structured input containing model response, context, etc.
            
        Returns:
            Dictionary mapping critic_name -> critic_output with metadata
        """
        if not self.critics:
            logger.warning("No critics configured in orchestrator")
            return {}
        
        # Create tasks for all critics
        tasks = [
            asyncio.create_task(
                self._execute_critic(name, critic_fn, input_snapshot),
                name=f"critic_{name}"
            )
            for name, critic_fn in self.critics.items()
        ]
        
        # Wait for all critics to complete (with isolation)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build output dictionary
        critic_outputs = {}
        for result in results:
            if isinstance(result, Exception):
                # Should never happen since _execute_critic catches everything,
                # but we defensively handle it
                logger.error(f"Unhandled exception in critic execution: {result}", exc_info=result)
                continue
            
            if isinstance(result, CriticExecutionResult):
                critic_outputs[result.critic_name] = result.to_dict()
            else:
                logger.error(f"Unexpected result type: {type(result)}")
        
        return critic_outputs
    
    def run(self, input_snapshot: CriticInput) -> Dict[str, Dict[str, Any]]:
        """
        Synchronous wrapper for async execution.
        
        Useful for non-async contexts, testing, and legacy integrations.
        """
        return asyncio.run(self.run_all(input_snapshot))


__all__ = [
    "OrchestratorV2",
    "OrchestratorHooks", 
    "CriticInput",
    "CriticExecutionResult",
    "CriticExecutionStatus",
]
