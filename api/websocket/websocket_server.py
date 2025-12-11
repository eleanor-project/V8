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
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.bootstrap import build_engine, evaluate_opa
from engine.logging_config import get_logger

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


# -------------------------------------
# Utility: Send JSON safely
# -------------------------------------
async def ws_send(ws: WebSocket, event_type: str, payload: dict):
    await ws.send_text(json.dumps({
        "event": event_type,
        "data": payload
    }))


# -------------------------------------
# WebSocket Endpoint
# -------------------------------------
@ws_router.websocket("/ws/deliberate")
async def ws_deliberate(ws: WebSocket):
    await ws.accept()
    await ws_send(ws, "connection", {"status": "connected"})

    try:
        incoming = await ws.receive_text()
        incoming_json = json.loads(incoming)

        user_text = incoming_json.get("input")
        context = incoming_json.get("context") or {}
        trace_id = incoming_json.get("trace_id") or str(uuid.uuid4())

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
            "final_decision": final_decision,
        }

        await ws_send(ws, "governance", governance_result)
        await ws_send(ws, "final", final)
        await ws.close()

    except WebSocketDisconnect:
        # client closed connection
        return

    except Exception as e:
        await ws_send(ws, "error", {"message": str(e)})
        await ws.close()
