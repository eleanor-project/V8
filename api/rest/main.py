"""
ELEANOR V8 — FastAPI REST Server
------------------------------------

This service exposes the unified deliberation engine via:

1. POST /deliberate
    → Runs a full V8 deliberation pipeline

2. GET /trace/{trace_id}
    → Retrieves an audit record (from evidence recorder)

3. POST /governance/preview
    → Runs governance (OPA) evaluation without full deliberation

4. GET /health
    → Basic system-integrity check
"""

import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Engine Builder
from engine.core import build_eleanor_engine_v8

# ---------------------------------------------
# Dependencies you will plug into the builder
# ---------------------------------------------

# 1. Load constitutional.yaml at startup
import yaml
with open("governance/constitutional.yaml", "r") as f:
    CONSTITUTION = yaml.safe_load(f)

# 2. Dummy precedent store until real DB connected
class DummyStore:
    def search(self, q, top_k): return []

# 3. Dummy OPA evaluator (replace with real HTTP call later)
def opa_eval_stub(payload):
    return {"allow": True, "failures": [], "escalate": False}

# 4. Dummy LLM adapter (replace with GPT, Claude, etc.)
def llm_adapter(prompt: str) -> str:
    return json.dumps({
        "score": 1.0,
        "violation": False,
        "details": {"notes": "Placeholder LLM response"}
    })


# Router adapters
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

# Build engine on startup
engine = build_eleanor_engine_v8(
    llm_fn=llm_adapter,
    constitutional_config=CONSTITUTION,
    router_adapters=router_adapters,
    router_policy=router_policy,
    precedent_store=precedent_store,
    opa_callback=opa_eval_stub,
)


# ---------------------------------------------
# FASTAPI APP
# ---------------------------------------------
app = FastAPI(
    title="ELEANOR V8 — Governance API",
    description="Responsible multi-critic constitutional deliberation engine",
    version="8.0.0"
)

# Allow local testing from any domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------
# Health Endpoint
# -----------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "engine": "ELEANOR V8", "critics": list(engine.orchestrator.critics.keys())}


# -----------------------------------------------------
# Main Deliberation Endpoint
# -----------------------------------------------------
@app.post("/deliberate")
def deliberate_payload(payload: dict):
    """
    Expect:
    {
        "input": "Should I approve this loan?"
    }
    """
    user_input = payload.get("input")
    if not user_input:
        raise HTTPException(status_code=400, detail="Missing 'input' field")
    try:
        result = engine.deliberate(user_input)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------
# Retrieve Evidence Bundle by Trace ID
# -----------------------------------------------------
@app.get("/trace/{trace_id}")
def get_trace(trace_id: str):
    """
    Only works in memory or JSONL mode.
    """
    if engine.recorder.storage_mode == "memory":
        for item in engine.recorder.get_memory_log():
            if item.get("trace_id") == trace_id:
                return item
        raise HTTPException(status_code=404, detail="Trace ID not found")

    elif engine.recorder.storage_mode == "jsonl":
        try:
            with open(engine.recorder.file_path, "r") as f:
                for line in f:
                    item = json.loads(line)
                    if item.get("trace_id") == trace_id:
                        return item
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Audit log missing")

        raise HTTPException(status_code=404, detail="Trace ID not found")

    else:
        raise HTTPException(status_code=400, detail="Unsupported storage mode")


# -----------------------------------------------------
# Governance Preview Endpoint
# -----------------------------------------------------
@app.post("/governance/preview")
def governance_preview(payload: dict):
    """
    Run governance evaluation on a mock evidence bundle.
    This is useful for debugging OPA policies.
    """
    try:
        result = engine.opa(payload)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
