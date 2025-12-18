"""
ELEANOR V8 — Router
------------------------------------

The Router directs requests to the correct model adapter.

Responsibilities:

1. Select the primary model adapter
2. Fall back to secondary adapters on failure
3. Enforce routing policies (capacity, availability, reliability)
4. Provide a unified interface for model inference
5. Gather model metadata for evidence bundles
6. Expose health status for orchestration & monitoring
7. Circuit breaker protection for resilient LLM calls

The Router DOES NOT:
- perform critic reasoning (critics do that)
- aggregate judgments (Aggregator does that)
- enforce governance (OPA does that)

It ONLY handles model selection + call orchestration.
"""

from typing import Any, Awaitable, Callable, Dict, List, Optional, Union, cast
import inspect
import traceback
import time
import asyncio
import logging

from engine.utils.circuit_breaker import CircuitBreakerRegistry, CircuitBreakerOpen

logger = logging.getLogger(__name__)


class AdapterError(Exception):
    """Custom exception for adapter-level failures."""
    pass


class RouterV8:

    def __init__(
        self,
        adapters: Optional[Dict[str, Callable]] = None,
        routing_policy: Optional[Dict[str, Any]] = None,
    ):
        """
        adapters: dict of adapter_name -> adapter_callable(text, context=None)
        routing_policy:
            {
                "primary": "gpt",
                "fallback_order": ["claude", "grok", "llama"],
                "max_retries": 2,
                "health": {adapter_name: status},
                "circuit_breaker": {...}
            }
        """
        self.adapters = adapters or {"default": self._default_adapter}

        primary = None
        if routing_policy:
            primary = routing_policy.get("primary")
        primary = primary or next(iter(self.adapters.keys()))

        self.policy = routing_policy or {
            "primary": primary,
            "fallback_order": [name for name in self.adapters.keys() if name != primary],
            "max_retries": 1,
            "circuit_breaker": {"enabled": False},
            "adapter_costs": {},
            "max_cost": None,
            "adapter_latency": {},
            "latency_budget_ms": None,
        }
        self.health = {name: True for name in self.adapters.keys()}
        self.max_retries = self.policy.get("max_retries", 2)

        # Initialize circuit breakers for each adapter
        self._circuit_breakers = CircuitBreakerRegistry()
        cb_config = self.policy.get("circuit_breaker", {})
        self._cb_enabled = cb_config.get("enabled", False)
        self._cb_failure_threshold = cb_config.get("failure_threshold", 5)
        self._cb_recovery_timeout = cb_config.get("recovery_timeout", 30.0)

        # Pre-create circuit breakers for all adapters
        for adapter_name in self.adapters.keys():
            self._get_circuit_breaker(adapter_name)

    @staticmethod
    def _default_adapter(input_text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fallback adapter that simply echoes the input."""
        return {
            "response_text": input_text,
            "model_name": "mock-local",
            "model_version": "0.0",
            "reason": "default_adapter_echo",
        }

    # ---------------------------------------------------------------
    # Output normalization
    # ---------------------------------------------------------------
    def _normalize_output(self, adapter_name: str, response: Any) -> Dict[str, Any]:
        """
        Normalize adapter outputs into a common router schema.

        Accepts:
          • plain string → treated as response_text
          • dict with keys: response_text | text | model_output
        """
        model_name = adapter_name
        model_version = None
        cost = None
        diagnostics = {}

        if isinstance(response, dict):
            response_text = (
                response.get("response_text")
                or response.get("text")
                or response.get("model_output")
                or response.get("output")
                or response.get("response")
            )
            model_name = response.get("model_name", model_name)
            model_version = response.get("model_version")
            cost = response.get("cost")
            diagnostics = response.get("diagnostics", {})
        else:
            response_text = str(response) if response is not None else None

        return {
            "response_text": response_text,
            "model_name": model_name,
            "model_version": model_version,
            "cost": cost,
            "diagnostics": diagnostics,
        }

    # ---------------------------------------------------------------
    # Circuit Breaker Management
    # ---------------------------------------------------------------

    def _get_circuit_breaker(self, adapter_name: str):
        """Get or create circuit breaker for an adapter."""
        return self._circuit_breakers.get_or_create(
            name=f"adapter_{adapter_name}",
            failure_threshold=self._cb_failure_threshold,
            recovery_timeout=self._cb_recovery_timeout
        )

    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers."""
        return self._circuit_breakers.get_all_status()

    # ---------------------------------------------------------------
    # Health Checking
    # ---------------------------------------------------------------

    def _check_health(self, adapter_name: str) -> bool:
        """
        Check adapter health including circuit breaker state.
        """
        # Basic health check
        basic_health = self.health.get(adapter_name, True)

        # Circuit breaker health check
        if self._cb_enabled:
            breaker = self._get_circuit_breaker(adapter_name)
            cb_status = breaker.get_status()
            cb_healthy = cb_status["state"] != "open"
            return basic_health and cb_healthy

        return basic_health

    # ---------------------------------------------------------------
    # Core model call logic (safe execution with circuit breaker)
    # ---------------------------------------------------------------

    async def _call_adapter(self, adapter_name: str, input_text: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        adapter = self.adapters.get(adapter_name)

        if adapter is None:
            raise AdapterError(f"Adapter '{adapter_name}' is not registered.")

        start_time = time.time()

        # Use circuit breaker if enabled
        if self._cb_enabled:
            breaker = self._get_circuit_breaker(adapter_name)

            try:
                # Wrap adapter call in circuit breaker (synchronous adapters only)
                response = breaker.call_sync(adapter, input_text)
                if inspect.isawaitable(response):
                    response = await response

                normalized = self._normalize_output(adapter_name, response)
                duration_ms = (time.time() - start_time) * 1000
                logger.debug(f"Adapter {adapter_name} succeeded in {duration_ms:.2f}ms")

                return {
                    "adapter": adapter_name,
                    "output": normalized,
                    "success": normalized["response_text"] is not None,
                    "duration_ms": duration_ms,
                    "circuit_breaker": breaker.get_status()
                }

            except CircuitBreakerOpen as e:
                logger.warning(f"Circuit breaker open for adapter {adapter_name}")
                return {
                    "adapter": adapter_name,
                    "success": False,
                    "error": str(e),
                    "circuit_breaker_open": True,
                    "recovery_time": e.recovery_time
                }

            except Exception as e:
                self.health[adapter_name] = False
                duration_ms = (time.time() - start_time) * 1000

                logger.error(f"Adapter {adapter_name} failed after {duration_ms:.2f}ms: {e}")

                return {
                    "adapter": adapter_name,
                    "success": False,
                    "error": str(e),
                    "trace": traceback.format_exc(),
                    "duration_ms": duration_ms,
                    "circuit_breaker": breaker.get_status()
                }

        # Fallback without circuit breaker
        try:
            if inspect.iscoroutinefunction(adapter):
                response = await adapter(input_text, context=context)
            else:
                sig = inspect.signature(adapter)
                if "context" in sig.parameters and len(sig.parameters) >= 2:
                    response = adapter(input_text, context=context)
                else:
                    response = adapter(input_text)

                if inspect.isawaitable(response):
                    response = await response

            normalized = self._normalize_output(adapter_name, response)
            duration_ms = (time.time() - start_time) * 1000
            return {
                "adapter": adapter_name,
                "output": normalized,
                "success": normalized["response_text"] is not None,
                "duration_ms": duration_ms
            }
        except Exception as e:
            self.health[adapter_name] = False
            duration_ms = (time.time() - start_time) * 1000
            return {
                "adapter": adapter_name,
                "success": False,
                "error": str(e),
                "trace": traceback.format_exc(),
                "duration_ms": duration_ms
            }

    # ---------------------------------------------------------------
    # Routing with fallback logic
    # ---------------------------------------------------------------

    async def _route_async(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Async implementation of routing logic.
        """
        context = context or {}
        primary = self.policy.get("primary")
        fallbacks = self.policy.get("fallback_order", [])
        adapter_costs = self.policy.get("adapter_costs", {}) or {}
        max_cost = self.policy.get("max_cost")
        adapter_latency = self.policy.get("adapter_latency", {}) or {}
        latency_budget = self.policy.get("latency_budget_ms")

        attempts = []

        def within_cost(name: str) -> bool:
            if max_cost is None:
                return True
            cost = adapter_costs.get(name)
            return cost is None or cost <= max_cost

        def within_latency(name: str) -> bool:
            if latency_budget is None:
                return True
            lat = adapter_latency.get(name)
            return lat is None or lat <= latency_budget

        candidates = []
        ordered = [primary] + fallbacks if primary else fallbacks
        for name in ordered:
            if name and self._check_health(name) and within_cost(name) and within_latency(name):
                candidates.append(name)

        if adapter_costs or adapter_latency:
            candidates = sorted(
                candidates,
                key=lambda n: (
                    adapter_costs.get(n, float("inf")),
                    adapter_latency.get(n, float("inf")),
                ),
            )

        for adapter_name in candidates:
            for _ in range(self.max_retries):
                result = await self._call_adapter(adapter_name, text, context)
                attempts.append(result)

                if result["success"]:
                    out = result["output"]
                    return {
                        "success": True,
                        "response_text": out["response_text"],
                        "model_output": out["response_text"],
                        "model_name": out["model_name"] or adapter_name,
                        "model_version": out["model_version"],
                        "reason": out.get("reason") or f"policy_selected:{adapter_name}",
                        "health_score": self.health.get(adapter_name, True),
                        "cost": adapter_costs.get(adapter_name, out.get("cost")),
                        "used_adapter": adapter_name,
                        "diagnostics": {
                            "attempts": attempts,
                            "adapter_used": adapter_name,
                        },
                    }

        # If no model succeeded
        return {
            "success": False,
            "response_text": None,
            "model_output": None,
            "model_name": None,
            "model_version": None,
            "reason": "All model adapters failed to produce output.",
            "health_score": 0.0,
            "cost": None,
            "used_adapter": None,
            "diagnostics": {"attempts": attempts},
        }

    def route(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> Union[Dict[str, Any], Awaitable[Dict[str, Any]]]:
        """
        Route request; works in both sync and async contexts.
        Returns dict if called synchronously, or coroutine if awaited in an event loop.
        """
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                return self._route_async(text, context)
        except RuntimeError:
            loop = None

        # No running loop; execute synchronously
        return asyncio.run(self._route_async(text, context))

    # ---------------------------------------------------------------
    # Public convenience method
    # ---------------------------------------------------------------

    async def generate(self, input_text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Public interface — semantic alias to route().
        """
        result = await self._route_async(input_text, context)  # re-use async path directly
        return cast(Dict[str, Any], result)
