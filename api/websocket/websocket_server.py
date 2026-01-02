"""
ELEANOR V8 — WebSocket Streaming Server
-----------------------------------------

This module provides a live streaming API for:
    - critic execution events
    - aggregation updates
    - uncertainty signals
    - precedent retrieval
    - governance evaluation
    - final deliberation result

Clients connect to:
    ws://localhost:8000/ws/deliberate

And receive JSON messages during pipeline execution.
"""

import asyncio
import json
import os
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.bootstrap import build_engine, evaluate_opa
from api.middleware.auth import decode_token, get_auth_config
from engine.exceptions import InputValidationError
from engine.logging_config import get_logger
from engine.execution.human_review import enforce_human_review
from engine.schemas.escalation import AggregationResult, HumanAction, ExecutableDecision
from engine.utils.critic_names import canonicalize_critic_map

engine = build_engine()
logger = get_logger(__name__)

# -------------------------------------
# Router
# -------------------------------------

ws_router = APIRouter()


# -------------------------------------
# Governance resolution helper
# -------------------------------------

def resolve_final_decision(aggregator_decision: Optional[str], opa_result: Dict[str, Any]) -> str:
    if not opa_result.get("allow", True):
        return "deny"
    if opa_result.get("escalate"):
        return "escalate"
    return aggregator_decision or "allow"


def map_assessment_label(decision: Optional[str]) -> str:
    if not decision:
        return "requires_human_review"
    mapping = {
        "allow": "aligned",
        "constrained_allow": "aligned_with_constraints",
        "deny": "misaligned",
        "escalate": "requires_human_review",
    }
    normalized = str(decision).lower()
    return mapping.get(normalized, normalized)


# -------------------------------------
# Utility: Send JSON safely
# -------------------------------------
async def ws_send(ws: WebSocket, event_type: str, payload: dict):
    await ws.send_text(json.dumps({
        "event": event_type,
        "data": payload
    }))


async def require_ws_auth(ws: WebSocket) -> bool:
    config = get_auth_config()
    if not config.enabled:
        return True

    auth_header = ws.headers.get("authorization")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        await ws_send(ws, "error", {"message": "Authorization header required"})
        await ws.close(code=1008)
        return False

    token = auth_header.split(" ", 1)[1].strip()
    try:
        payload = decode_token(token, config)
    except Exception as exc:
        detail = getattr(exc, "detail", "Unauthorized")
        await ws_send(ws, "error", {"message": str(detail)})
        await ws.close(code=1008)
        return False

    required_role = os.getenv("WS_REQUIRED_ROLE")
    required_scope = os.getenv("WS_REQUIRED_SCOPE")
    if required_role and not payload.has_role(required_role):
        await ws_send(ws, "error", {"message": f"Role '{required_role}' required"})
        await ws.close(code=1008)
        return False
    if required_scope and not payload.has_scope(required_scope):
        await ws_send(ws, "error", {"message": f"Scope '{required_scope}' required"})
        await ws.close(code=1008)
        return False

    return True


def resolve_execution_decision(
    aggregated: Dict[str, Any],
    human_action: Optional[HumanAction],
) -> Optional[ExecutableDecision]:
    aggregation_payload = aggregated.get("aggregation_result") if isinstance(aggregated, dict) else None
    if not aggregation_payload:
        return None
    aggregation_result = AggregationResult.model_validate(aggregation_payload)
    return enforce_human_review(aggregation_result=aggregation_result, human_action=human_action)


def apply_execution_gate(final_decision: str, execution_decision: Optional[ExecutableDecision]) -> str:
    if execution_decision and not execution_decision.executable and final_decision != "deny":
        return "escalate"
    return final_decision


# -------------------------------------
# WebSocket Endpoint
# -------------------------------------
@ws_router.websocket("/ws/deliberate")
async def ws_deliberate(ws: WebSocket):
    await ws.accept()
    if not await require_ws_auth(ws):
        return
    await ws_send(ws, "connection", {"status": "connected"})

    try:
        incoming = await ws.receive_text()
        incoming_json = json.loads(incoming)

        user_text = incoming_json.get("input")
        context = incoming_json.get("context") or {}
        trace_id = incoming_json.get("trace_id") or str(uuid.uuid4())
        human_action = incoming_json.get("human_action")
        try:
            human_action_payload = HumanAction.model_validate(human_action) if human_action else None
        except Exception as exc:
            await ws_send(ws, "error", {"message": str(exc)})
            await ws.close()
            return

        if not user_text:
            await ws_send(ws, "error", {"message": "Missing 'input' field"})
            await ws.close()
            return

        critic_outputs: Dict[str, Any] = {}
        aggregated: Dict[str, Any] = {}
        precedent_data: Dict[str, Any] = {}
        uncertainty_data: Dict[str, Any] = {}
        model_info: Dict[str, Any] = {}
        model_output: Any = None

        async for event in engine.run_stream(user_text, context=context, trace_id=trace_id, detail_level=3):
            event_type = event.get("event")
            payload = {k: v for k, v in event.items() if k != "event"}
            await ws_send(ws, event_type, payload)

            if event_type == "router_selected":
                model_info = payload.get("model_info") or {}
            elif event_type == "model_response":
                model_output = payload.get("text")
            elif event_type == "critic_result":
                name = payload.get("critic")
                if name:
                    critic_outputs[name] = payload
            elif event_type == "precedent_alignment":
                precedent_data = payload.get("data") or {}
            elif event_type == "uncertainty":
                uncertainty_data = payload.get("data") or {}
            elif event_type == "aggregation":
                aggregated = payload.get("data") or {}
                critic_outputs = aggregated.get("critics", critic_outputs)

        critic_outputs = canonicalize_critic_map(critic_outputs)

        governance_payload = {
            "critics": critic_outputs,
            "aggregator": aggregated,
            "precedent": precedent_data,
            "uncertainty": uncertainty_data,
            "model_used": model_info.get("model_name"),
            "input": user_text,
            "trace_id": trace_id,
        }

        governance_result = await evaluate_opa(
            getattr(engine, "opa_callback", None),
            governance_payload,
        )

        final_decision = resolve_final_decision(aggregated.get("decision"), governance_result)
        execution_decision = None
        try:
            execution_decision = resolve_execution_decision(aggregated, human_action_payload)
        except Exception as exc:
            await ws_send(ws, "error", {"message": str(exc)})
            await ws.close()
            return
        final_decision = apply_execution_gate(final_decision, execution_decision)
        final_assessment = map_assessment_label(final_decision)

        final = {
            "trace_id": trace_id,
            "input": user_text,
            "model_info": model_info,
            "model_output": model_output,
            "critic_outputs": critic_outputs,
            "aggregation": aggregated,
            "uncertainty": uncertainty_data,
            "precedent": precedent_data,
            "governance": governance_result,
            "final_decision": final_assessment,
            "execution_decision": execution_decision.model_dump(mode="json") if execution_decision else None,
        }

        await ws_send(ws, "governance", governance_result)
        await ws_send(ws, "final", final)
        await ws.close()

    except WebSocketDisconnect:
        # client closed connection
        return

    except InputValidationError as exc:
        await ws_send(ws, "error", {"message": exc.message, "details": exc.details})
        await ws.close()

    except Exception as e:
        await ws_send(ws, "error", {"message": str(e)})
        await ws.close()
