"""
OPA enforcement helper for local prototypes and configurable gateways.

Supports:
- Execution gating via eleanor.execution (allow/deny)
- Route hardening via eleanor.api (deny list)
- Critic output validation via eleanor.critics (allow/deny)

The enforcer is configurable via env vars so the same code path can run
locally (prototype) or behind a gateway without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


@dataclass
class OPAEnforcerConfig:
    enabled: bool = _bool_env("ELEANOR_OPA_ENABLED", True)
    fail_open: bool = _bool_env("ELEANOR_OPA_FAIL_OPEN", False)
    base_url: str = (os.getenv("ELEANOR_OPA_BASE_URL") or os.getenv("OPA_URL") or "http://localhost:8181")
    execution_allow_path: str = os.getenv(
        "ELEANOR_OPA_EXEC_ALLOW", "v1/data/eleanor/execution/allow"
    )
    execution_deny_path: str = os.getenv("ELEANOR_OPA_EXEC_DENY", "v1/data/eleanor/execution/deny")
    route_deny_path: str = os.getenv("ELEANOR_OPA_ROUTE_DENY", "v1/data/eleanor/api/deny")
    critics_allow_path: str = os.getenv(
        "ELEANOR_OPA_CRITICS_ALLOW", "v1/data/eleanor/critics/allow"
    )
    critics_deny_path: str = os.getenv("ELEANOR_OPA_CRITICS_DENY", "v1/data/eleanor/critics/deny")
    timeout_seconds: float = float(os.getenv("ELEANOR_OPA_TIMEOUT_SECONDS", "3.0"))


class OPAEnforcer:
    """
    Thin helper around the OPA policies we ship in opa/policies.
    Intended for local prototypes and easy gateway integration.
    """

    def __init__(self, config: Optional[OPAEnforcerConfig] = None):
        self.config = config or OPAEnforcerConfig()
        self.base_url = self.config.base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def check_execution(self, decision: Dict[str, Any], action: str = "execute") -> Dict[str, Any]:
        """
        Evaluate the execution gate via eleanor.execution.{allow,deny}.

        Returns:
        {
            "allowed": bool,
            "deny_reasons": List[str],
            "error": Optional[str],
            "source": "opa" | "disabled" | "fail_open"
        }
        """
        if not self.config.enabled:
            return {"allowed": True, "deny_reasons": [], "error": None, "source": "disabled"}

        payload = {"action": action, "decision": decision}
        allow, allow_err = self._query_bool(self.config.execution_allow_path, payload)
        deny, deny_err = self._query_list(self.config.execution_deny_path, payload)

        error = allow_err or deny_err
        if error:
            if self.config.fail_open:
                return {
                    "allowed": True,
                    "deny_reasons": [],
                    "error": error,
                    "source": "fail_open",
                }
            return {"allowed": False, "deny_reasons": [], "error": error, "source": "opa"}

        allowed = bool(allow) and not deny
        return {
            "allowed": allowed,
            "deny_reasons": deny or [],
            "error": None,
            "source": "opa",
        }

    def check_route(
        self, path: str, method: str, body: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate routing rules via eleanor.api.deny (defense-in-depth).
        """
        if not self.config.enabled:
            return {"allowed": True, "deny_reasons": [], "error": None, "source": "disabled"}

        payload = {"http": {"path": path, "method": method}, "body": body or {}}
        deny, err = self._query_list(self.config.route_deny_path, payload)
        if err:
            if self.config.fail_open:
                return {"allowed": True, "deny_reasons": [], "error": err, "source": "fail_open"}
            return {"allowed": False, "deny_reasons": [], "error": err, "source": "opa"}

        return {"allowed": not deny, "deny_reasons": deny or [], "error": None, "source": "opa"}

    def validate_critics(self, critic_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate critic outputs via eleanor.critics.{allow,deny}.

        critic_payload typically includes:
        {
            "http": {"path": "/decision/evaluate", "method": "POST"},
            "body": {"critic_evaluations": [...], "synthesis": "..."}
        }
        """
        if not self.config.enabled:
            return {"allowed": True, "deny_reasons": [], "error": None, "source": "disabled"}

        allow, allow_err = self._query_bool(self.config.critics_allow_path, critic_payload)
        deny, deny_err = self._query_list(self.config.critics_deny_path, critic_payload)

        error = allow_err or deny_err
        if error:
            if self.config.fail_open:
                return {
                    "allowed": True,
                    "deny_reasons": [],
                    "error": error,
                    "source": "fail_open",
                }
            return {"allowed": False, "deny_reasons": [], "error": error, "source": "opa"}

        return {
            "allowed": bool(allow) and not deny,
            "deny_reasons": deny or [],
            "error": None,
            "source": "opa",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _query_bool(
        self, policy_path: str, payload: Dict[str, Any]
    ) -> Tuple[Optional[bool], Optional[str]]:
        """
        Evaluate a boolean result policy (e.g., .../allow).
        """
        result, err = self._post(policy_path, payload)
        if err:
            return None, err
        return bool(result), None

    def _query_list(
        self, policy_path: str, payload: Dict[str, Any]
    ) -> Tuple[Optional[List[str]], Optional[str]]:
        """
        Evaluate a deny-set policy (returns a list of strings).
        """
        result, err = self._post(policy_path, payload)
        if err:
            return None, err
        if result is None:
            return [], None
        if isinstance(result, list):
            return result, None
        # If the policy returns a single string, normalize to list.
        if isinstance(result, str):
            return [result], None
        return [], None

    def _post(self, policy_path: str, payload: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
        path = policy_path.strip("/")
        url = f"{self.base_url}/{path}"
        try:
            resp = requests.post(url, json={"input": payload}, timeout=self.config.timeout_seconds)
            resp.raise_for_status()
            data = resp.json()
            return data.get("result"), None
        except Exception as exc:  # broad on purpose: network/JSON/etc
            return None, str(exc)
