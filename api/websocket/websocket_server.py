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

import json
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from engine.core import build_eleanor_engine_v8

# -------------------------------------
# Temporary build dependencies
# -------------------------------------

import yaml
with open("governance/constitutional.yaml") as f:
    CONSTITUTION = yaml.safe_load(f)


class DummyStore:
    def search(self, q, top_k): return []


def opa_stub(payload):
    return {"allow": True, "failures": [], "escalate": False}


def llm_adapter(prompt: str) -> str:
    """Replace this with GPT, Claude, or Grok adapter."""
    return json.dumps({
        "score": 1.0, "violation": False,
        "details": {"notes": "placeholder critic"}
    })


router_adapters = {
    "primary": llm_adapter,
    "backup": llm_adapter
}

router_policy = {
    "primary": "primary",
    "fallback_order": ["backup"],
    "max_retries": 2
}

precedent_store = DummyStore()

engine = build_eleanor_engine_v8(
    llm_fn=llm_adapter,
    constitutional_config=CONSTITUTION,
    router_adapters=router_adapters,
    router_policy=router_policy,
    precedent_store=precedent_store,
    opa_callback=opa_stub
)

# -------------------------------------
# Router
# -------------------------------------

ws_router = APIRouter()


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
        # First message must contain input text
        incoming = await ws.receive_text()
        incoming_json = json.loads(incoming)

        user_text = incoming_json.get("input")
        if not user_text:
            await ws_send(ws, "error", {"message": "Missing 'input' field"})
            await ws.close()
            return

        # -------------------------------------
        # Begin streaming pipeline
        # -------------------------------------

        # STEP 1 — Router
        router_result = engine.router.generate(user_text)
        await ws_send(ws, "router", router_result)
        await asyncio.sleep(0.05)

        # STEP 2 — Critics (stream critic results as they finish)
        critic_outputs = {}
        tasks = []

        async def run_single_critic(name, fn):
            result = await fn(user_text)
            critic_outputs[name] = result
            await ws_send(ws, f"critic_{name}", result)

        for name, fn in engine.orchestrator.critics.items():
            tasks.append(asyncio.create_task(run_single_critic(name, fn)))

        await asyncio.gather(*tasks)
        await asyncio.sleep(0.05)

        # STEP 3 — Aggregator
        deliberation_state = engine.aggregator.aggregate(
            critic_outputs=critic_outputs,
            input_snapshot=user_text,
            precedent_result=None
        )
        await ws_send(ws, "aggregator", deliberation_state)
        await asyncio.sleep(0.05)

        # STEP 4 — Precedent
        precedent_result = engine.precedent.retrieve(
            query_text=user_text,
            critic_outputs=deliberation_state["critic_outputs"]
        )
        await ws_send(ws, "precedent", precedent_result)
        await asyncio.sleep(0.05)

        # STEP 5 — Uncertainty
        deliberation_state["precedent_alignment_score"] = precedent_result["alignment_score"]
        uncertainty_state = engine.uncertainty.compute(deliberation_state)
        await ws_send(ws, "uncertainty", uncertainty_state)
        await asyncio.sleep(0.05)

        # STEP 6 — Governance
        evidence_bundle = engine.evidence_packager.build(
            input_snapshot=user_text,
            router_result=router_result,
            critic_outputs=critic_outputs,
            deliberation_state=deliberation_state,
            uncertainty_state=uncertainty_state,
            precedent_result=precedent_result
        )

        governance_result = engine.opa(evidence_bundle["governance_ready_payload"])
        await ws_send(ws, "governance", governance_result)
        await asyncio.sleep(0.05)

        # STEP 7 — Record Evidence
        engine.recorder.record(evidence_bundle)
        await ws_send(ws, "evidence_recorded", {"trace_id": evidence_bundle["trace_id"]})

        # FINAL RESULT
        final = {
            "input": user_text,
            "model_output": router_result.get("model_output"),
            "critic_outputs": critic_outputs,
            "priority_violations": deliberation_state["priority_violations"],
            "uncertainty": uncertainty_state,
            "precedent": precedent_result,
            "governance": governance_result,
            "trace_id": evidence_bundle["trace_id"]
        }

        await ws_send(ws, "final", final)
        await ws.close()

    except WebSocketDisconnect:
        # client closed connection
        return

    except Exception as e:
        await ws_send(ws, "error", {"message": str(e)})
        await ws.close()
