"""
ELEANOR V8 — OPA Governance Client
------------------------------------

Responsible for:
    - Sending evidence payloads to Open Policy Agent
    - Receiving allow/deny/escalate results
    - Handling errors gracefully
    - Running policy module evaluations
"""

import json
import os
import requests
from typing import Any, Dict


def _get_timeout() -> float:
    raw = os.getenv("OPA_TIMEOUT_SECONDS", "5")
    try:
        return float(raw)
    except ValueError:
        return 5.0


class OPAClientV8:

    def __init__(
        self,
        base_url: str = "http://localhost:8181",
        policy_path: str = "v1/data/eleanor/decision"
    ):
        """
        Args:
            base_url:
                OPA server address. Default: local instance.
            policy_path:
                The OPA-referenced path for policy evaluation.

                Example:
                    /v1/data/eleanor/decision
        """
        self.base_url = base_url.rstrip("/")
        self.policy_path = policy_path.strip("/")

    # ----------------------------------------------------------
    # Health Check
    # ----------------------------------------------------------
    def health(self) -> bool:
        timeout = _get_timeout()
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False

    # ----------------------------------------------------------
    # Policy Evaluation (Primary Method)
    # ----------------------------------------------------------
    def evaluate(self, evidence_payload: Dict[str, Any], policy_path: str | None = None) -> Dict[str, Any]:
        """
        Executes OPA policy evaluation.

        Returns schema:
        {
            "allow": bool,
            "escalate": bool,
            "failures": [
                {"policy": ..., "reason": ...}
            ]
        }
        """

        path = policy_path.strip("/") if policy_path else self.policy_path
        url = f"{self.base_url}/{path}"

        timeout = _get_timeout()
        try:
            resp = requests.post(url, json={"input": evidence_payload}, timeout=timeout)
        except Exception as e:
            return {
                "allow": False,
                "escalate": True,
                "failures": [{
                    "policy": "opa_client_error",
                    "reason": str(e)
                }]
            }

        if resp.status_code != 200:
            return {
                "allow": False,
                "escalate": True,
                "failures": [{
                    "policy": "opa_http_error",
                    "reason": f"HTTP {resp.status_code}: {resp.text}"
                }]
            }

        try:
            data = resp.json()
        except ValueError:
            return {
                "allow": False,
                "escalate": True,
                "failures": [{
                    "policy": "invalid_json",
                    "reason": resp.text
                }]
            }

        # Expecting OPA schema:
        # {
        #   "result": {
        #       "allow": bool,
        #       "escalate": bool,
        #       "failures": [...]
        #   }
        # }
        result = data.get("result", {})

        return {
            "allow": result.get("allow", False),
            "escalate": result.get("escalate", False),
            "failures": result.get("failures", [])
        }
