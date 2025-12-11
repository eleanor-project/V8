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

The Router DOES NOT:
- perform critic reasoning (critics do that)
- aggregate judgments (Aggregator does that)
- enforce governance (OPA does that)

It ONLY handles model selection + call orchestration.
"""

from typing import Dict, Any, Callable, Optional, List
import traceback
import time


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
                "health": {adapter_name: status}
            }
        """
        self.adapters = adapters
        self.policy = routing_policy
        self.health = {name: True for name in adapters.keys()}
        self.max_retries = routing_policy.get("max_retries", 2)

    # ---------------------------------------------------------------
    # Health Checking
    # ---------------------------------------------------------------

    def _check_health(self, adapter_name: str) -> bool:
        """
        Placeholder health check — replace with real calls later.
        """
        return self.health.get(adapter_name, True)

    # ---------------------------------------------------------------
    # Core model call logic (safe execution)
    # ---------------------------------------------------------------

    def _call_adapter(self, adapter_name: str, input_text: str) -> Dict[str, Any]:
        adapter = self.adapters.get(adapter_name)

        if adapter is None:
            raise AdapterError(f"Adapter '{adapter_name}' is not registered.")

        try:
            response = adapter(input_text)
            if not isinstance(response, dict):
                raise AdapterError("Adapter returned non-dict output.")
            return {
                "adapter": adapter_name,
                "output": response,
                "success": True
            }
        except Exception as e:
            self.health[adapter_name] = False  # Mark unhealthy
            return {
                "adapter": adapter_name,
                "success": False,
                "error": str(e),
                "trace": traceback.format_exc()
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
