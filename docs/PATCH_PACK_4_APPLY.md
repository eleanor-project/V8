# Patch Pack 4 — Engine Wiring Hooks (Traffic Light Governance)

## What this pack adds
- A runtime hook that:
  - Computes a Traffic Light route using (a) precedent coverage and (b) uncertainty divergence
  - Emits append-only governance events to JSONL
  - Attaches reviewer-safe governance metadata to the engine's aggregated output

## Why this exists
This is the *entry point* you and Jim were talking about: a lightweight signal system that
can route edge cases to Eleanor governance without letting reviewers talk to the critic ensemble
or reverse runtime decisions.

## Files included
- `engine/integrations/traffic_light_governance.py`
- `governance/audit_sink.py`
- `patches/engine_engine_py.patch`

## Apply steps
1. Copy the new files into your repo root (preserving paths):
   - `engine/integrations/*`
   - `governance/audit_sink.py`

2. Apply the engine patch:
   - `git apply patches/engine_engine_py.patch`

3. Ensure Patch Packs 1–3 are already applied (the `governance/*` python routing modules and `governance/router_config.yaml`).

## Runtime configuration
- Enable/disable via env var:
  - `ELEANOR_TRAFFIC_LIGHT_ENABLED=true|false` (default: enabled)

- Engine config additions (defaults are safe):
  - `enable_traffic_light_governance: bool = True`
  - `traffic_light_router_config_path: str = "governance/router_config.yaml"`
  - `governance_events_jsonl_path: Optional[str] = "governance_events.jsonl"`

## Output changes
- The engine attaches a `governance_meta` object under `aggregated`.
  - Contains: `route`, `outcome`, `reason`, `applied_precedent_ids`, `event_id`
  - Does **not** include critic ensemble internals.

