# V8 Specification

## Core Pipeline
- **Router**: async `RouterV8` with adapter fallback, context-aware calls, and diagnostic metadata. Returns `response_text`, model metadata, and attempt trace. Default registry boots available adapters (OpenAI/Anthropic/xAI/local HF/Ollama) based on installed SDKs + keys.
- **Critics**: parallel execution of Rights, Autonomy, Fairness, Truth, Risk, Pragmatics. Each critic emits severity (0–3), violations, justification, evidence bundle, and flags.
- **Precedent**: optional retrieval via `PrecedentRetrievalV8` and alignment via `PrecedentAlignmentEngineV8` (conflict, drift, support strength). Novel cases return neutral alignment. Precedent stores: Weaviate, pgvector, or in-memory with embedding registry for similarity.
- **Uncertainty**: `UncertaintyEngineV8` computes epistemic/aleatoric uncertainty (critic divergence, precedent conflict, model stability) and escalation flags.
- **Aggregator**: lexicographic fusion with priority order `[rights, autonomy, fairness, truth, risk, pragmatics]`, applying precedent and uncertainty weighting. Outputs decision (`allow|constrained_allow|deny|escalate`), scores, and final output text.
- **Evidence**: `EvidenceRecorder` buffers JSONL-ready evidence per critic, including severity label, principle, justification, and detector metadata.
- **Governance**: OPA client wiring available through the engine builder; `opa_callback` can be injected or defaults to `OPAClientV8.evaluate`.

## Engine (async)
- Primary entrypoints: `engine/engine.py` → `EleanorEngineV8.run` and `run_stream`.
- Configuration: `EngineConfig` toggles precedent analysis, reflection (uncertainty), and evidence jsonl path.
- Router auto-discovery with default echo adapter for local use; supports injected adapters/policy.
- Supports forensic mode (detail_level 3) with timings, router diagnostics, uncertainty graph, and evidence references.

## Critics (V8)
- **Rights**: discrimination, coercion, dignity attacks, privacy.
- **Autonomy**: consent bypass, coercion, manipulation, surveillance pressure.
- **Fairness**: disparate impact/treatment patterns, protected class cues.
- **Truth**: factual accuracy patterns, evidence grounding.
- **Risk**: safety domains, irreversibility, vulnerable populations.
- **Pragmatics**: feasibility, resource burden, operational constraints.

## Precedent & Uncertainty
- Alignment tolerates missing embeddings and returns novel-case bundle when no precedents exist.
- Uncertainty uses critic severity variance + precedent conflict + model stability heuristics.

## Governance Interfaces
- `engine/core/__init__.py` exposes `build_eleanor_engine_v8` for API/websocket bootstraps, wiring router adapters, precedent store, and evidence path into the async engine.

## Expected Decisions
- Hard block: rights/autonomy violations with severity ≥ 2.5.
- Escalate: uncertainty ≥ 0.6 with moderate average severity.
- Constrained allow: average severity ≥ 1.0 without hard block.
- Allow: otherwise, with precedent/uncertainty adjustments applied.
