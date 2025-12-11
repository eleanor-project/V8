# Architecture Overview

## Runtime Flow
1. **Routing** — `RouterV8` selects an adapter (async-capable, circuit-breaker ready) and returns `response_text` + diagnostics. Adapter registry (`engine/router/adapters.py`) boots any available cloud/local adapters from installed SDKs + API keys.
2. **Critic Ensemble** — `EleanorEngineV8` runs Rights, Autonomy, Fairness, Truth, Risk, Pragmatics in parallel with a shared semaphore. Critics consume the routed output via a static `generate` shim to keep signatures consistent.
3. **Precedent Alignment** — optional retrieval via `PrecedentRetrievalV8` and alignment via `PrecedentAlignmentEngineV8` (handles missing embeddings gracefully; returns novel-case bundle when empty).
4. **Uncertainty Modeling** — `UncertaintyEngineV8` blends critic divergence, precedent conflict, and model stability into overall uncertainty + escalation flag.
5. **Aggregation** — `AggregatorV8` applies lexicographic priorities `[rights, autonomy, fairness, truth, risk, pragmatics]`, weights by precedent/uncertainty, and emits decision + fused justification.
6. **Evidence** — `EvidenceRecorder` logs per-critic findings to buffer/JSONL for later audit; forensic mode exposes diagnostics/timings.

## Public Interfaces
- **Async Engine**: `engine/engine.py` with `run`/`run_stream`.
- **Builder**: `engine/core/__init__.py` exposes `build_eleanor_engine_v8` for API/websocket bootstrap, wiring adapters (registry-backed), precedent store (weaviate/pgvector/memory), embeddings, detectors, evidence path, and governance (OPA client or injected callback).
- **Router**: `engine/router/router.py` now accepts missing adapters (ships with a safe default) and normalizes string/dict adapter outputs into a single schema.

## Design Notes
- Autonomy critic added to satisfy V8 spec; aggregator priority updated accordingly.
- Precedent alignment and uncertainty tolerate absent stores to keep local development frictionless.
- Aggregator now accepts optional precedent/uncertainty/model_output parameters and defaults to safe fallbacks to avoid runtime crashes.
