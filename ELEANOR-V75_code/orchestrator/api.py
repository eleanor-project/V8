# file: orchestrator/api.py

from fastapi import FastAPI
from .engine import run_all_critics
from .decision import aggregate_decision

app = FastAPI(title="ELEANOR Constitutional Orchestrator")

@app.post("/evaluate")
def evaluate(payload: dict):
    text = payload["text"]
    outputs = run_all_critics(text)
    decision = aggregate_decision(outputs)
    return decision
