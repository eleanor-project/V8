"""
ELEANOR V8 — Component Health Monitoring

Health check system for monitoring circuit breaker states.
"""

import logging
from typing import Dict, Any, List
from .circuit_breaker import CircuitBreaker, CircuitState

logger = logging.getLogger(__name__)


class ComponentHealthChecker:
    """
    Monitor health status of all components with circuit breakers.

    Provides health check endpoint and overall system health assessment.
    """

    def __init__(self, breakers: Dict[str, CircuitBreaker]):
        """
        Initialize health checker.

        Args:
            breakers: Dictionary mapping component names to circuit breakers
        """
        self.breakers = breakers

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of all components.

        Returns:
            Dictionary with component health and overall status
        """
        components = {}

        for name, breaker in self.breakers.items():
            state = breaker.get_state()
            components[name] = {
                "state": state["state"],
                "healthy": state["healthy"],
                "failure_count": state["failure_count"],
                "last_failure": state["last_failure"],
            }

        overall_health = self._calculate_overall_health()

        return {
            "overall_health": overall_health,
            "components": components,
            "healthy_count": sum(1 for c in components.values() if c["healthy"]),
            "total_count": len(components),
        }

    def _calculate_overall_health(self) -> str:
        """
        Calculate overall system health.

        Returns:
            'healthy', 'degraded', or 'unhealthy'
        """
        if not self.breakers:
            return "unknown"

        open_count = sum(1 for b in self.breakers.values() if b.state == CircuitState.OPEN)

        if open_count == 0:
            return "healthy"
        elif open_count < len(self.breakers) * 0.3:
            return "degraded"
        else:
            return "unhealthy"

    def get_unhealthy_components(self) -> List[str]:
        """
        Get list of unhealthy component names.

        Returns:
            List of component names with open circuit breakers
        """
        return [
            name for name, breaker in self.breakers.items() if breaker.state == CircuitState.OPEN
        ]

    def is_healthy(self) -> bool:
        """
        Check if system is healthy.

        Returns:
            True if all circuits are closed
        """
        return self._calculate_overall_health() == "healthy"


__all__ = ["ComponentHealthChecker"]
