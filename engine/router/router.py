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

from typing import Dict, Any, Callable, Optional, List
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

    def __init__(self, adapters: Dict[str, Callable], routing_policy: Dict[str, Any]):
        """
        adapters: dict of adapter_name -> adapter_callable(input_text) -> dict
        routing_policy:
            {
                "primary": "gpt",
                "fallback_order": ["claude", "grok", "llama"],
                "max_retries": 2,
                "health": {adapter_name: status},
                "circuit_breaker": {
                    "enabled": True,
                    "failure_threshold": 5,
                    "recovery_timeout": 30.0
                }
            }
        """
        self.adapters = adapters
        self.policy = routing_policy
        self.health = {name: True for name in adapters.keys()}
        self.max_retries = routing_policy.get("max_retries", 2)

        # Initialize circuit breakers for each adapter
        self._circuit_breakers = CircuitBreakerRegistry()
        cb_config = routing_policy.get("circuit_breaker", {})
        self._cb_enabled = cb_config.get("enabled", True)
        self._cb_failure_threshold = cb_config.get("failure_threshold", 5)
        self._cb_recovery_timeout = cb_config.get("recovery_timeout", 30.0)

        # Pre-create circuit breakers for all adapters
        for adapter_name in adapters.keys():
            self._get_circuit_breaker(adapter_name)

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

    def _call_adapter(self, adapter_name: str, input_text: str) -> Dict[str, Any]:
        adapter = self.adapters.get(adapter_name)

        if adapter is None:
            raise AdapterError(f"Adapter '{adapter_name}' is not registered.")

        start_time = time.time()

        # Use circuit breaker if enabled
        if self._cb_enabled:
            breaker = self._get_circuit_breaker(adapter_name)

            try:
                # Wrap adapter call in circuit breaker
                response = breaker.call_sync(adapter, input_text)

                if not isinstance(response, dict):
                    raise AdapterError("Adapter returned non-dict output.")

                duration_ms = (time.time() - start_time) * 1000
                logger.debug(
                    f"Adapter {adapter_name} succeeded in {duration_ms:.2f}ms"
                )

                return {
                    "adapter": adapter_name,
                    "output": response,
                    "success": True,
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

                logger.error(
                    f"Adapter {adapter_name} failed after {duration_ms:.2f}ms: {e}"
                )

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
            response = adapter(input_text)
            if not isinstance(response, dict):
                raise AdapterError("Adapter returned non-dict output.")

            duration_ms = (time.time() - start_time) * 1000
            return {
                "adapter": adapter_name,
                "output": response,
                "success": True,
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

    def route(self, input_text: str) -> Dict[str, Any]:
        """
        Route the request to the primary adapter.
        If it fails, apply fallback logic based on routing_policy.
        """
        primary = self.policy.get("primary")
        fallbacks = self.policy.get("fallback_order", [])

        attempts = []

        # Try primary first
        if primary:
            result = self._call_adapter(primary, input_text)
            attempts.append(result)

            if result["success"]:
                return {
                    "model_output": result["output"],
                    "used_adapter": primary,
                    "attempts": attempts
                }

        # Try fallbacks
        for fb in fallbacks:
            if not self._check_health(fb):
                continue  # Skip unhealthy models

            for _ in range(self.max_retries):
                result = self._call_adapter(fb, input_text)
                attempts.append(result)

                if result["success"]:
                    return {
                        "model_output": result["output"],
                        "used_adapter": fb,
                        "attempts": attempts
                    }

        # If no model succeeded
        return {
            "model_output": None,
            "used_adapter": None,
            "attempts": attempts,
            "error": "All model adapters failed to produce output."
        }

    # ---------------------------------------------------------------
    # Public convenience method
    # ---------------------------------------------------------------

    def generate(self, input_text: str) -> Dict[str, Any]:
        """
        Public interface — semantic alias to route().
        """
        return self.route(input_text)
