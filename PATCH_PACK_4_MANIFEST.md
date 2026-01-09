# V8 Governance Routing — Patch Pack 4

## Purpose
Wire the Traffic Light governance router into the *actual V8 runtime* (run + run_stream):
- Every governed run emits an append-only governance event
- Every response carries reviewer-safe governance metadata (incl. applied precedent IDs)
- Reviewers remain precedent authors only (sanctity rule preserved)

## Contents
- `engine/integrations/traffic_light_governance.py` — Hook + adapters
- `engine/integrations/__init__.py`
- `governance/audit_sink.py` — JSONL append sink
- `patches/engine_engine_py.patch` — Minimal wiring changes for `engine/engine.py`
- `docs/PATCH_PACK_4_APPLY.md` — How to apply

## New EngineConfig fields
- `enable_traffic_light_governance` (default True)
- `traffic_light_router_config_path` (default `governance/router_config.yaml`)
- `governance_events_jsonl_path` (default `governance_events.jsonl`)

