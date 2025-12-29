# Architecture Overview (Patched for V8.0.1)

This document describes the **runtime architecture**, public interfaces, and high-level design principles of the Eleanor V8 engine. It reflects the corrected governance framing: Eleanor provides **interpretive constitutional analysis**, not command-and-control or behavioral enforcement.

---

## Runtime Flow

1. **Routing**  
   `RouterV8` selects an appropriate model adapter using async routing with diagnostic tracing, circuit-breaker resilience, and fallback logic.  
   It returns:
   - `response_text`
   - model metadata
   - routing diagnostics

   Adapters autoload from environment-detected SDKs and API keys (OpenAI/Anthropic/xAI/HF/Ollama).  
   Router output is normalized into a unified schema.

2. **Critic Ensemble (Constitutional Deliberation Layer)**  
   `EleanorEngineV8` runs the critic suite in parallel:

   - Rights  
   - Autonomy & Agency  
   - Fairness  
   - Truth  
   - Risk  
   - Pragmatics  

   Each critic evaluates the model-generated text **interpretively**, not prescriptively, producing:
   - severity (0–3)  
   - violations  
   - justification  
   - constitutional references  
   - optional detector + precedent context  

   Critics do **not** block, deny, or override anything; they provide **constitutional readings**.

3. **Precedent Alignment (Advisory Layer)**  
   Using `PrecedentRetrievalV8` and `PrecedentAlignmentEngineV8`, the system retrieves relevant prior cases (if stores exist) and computes:
   - alignment score  
   - support strength  
   - conflict detection  
   - drift signals  

   **Precedent is advisory** and **cannot override rights, autonomy, or any higher-order critic domain**.  
   Missing stores or missing embeddings are tolerated gracefully.

4. **Uncertainty Modeling (Interpretive Confidence Layer)**  
   `UncertaintyEngineV8` quantifies:
   - epistemic uncertainty (critic divergence)  
   - aleatoric uncertainty (model instability, precedent conflict)  

   It produces an interpretive escalation flag:  
   - `requires_human_review` when uncertainty is too high to provide a reliable constitutional assessment.

5. **Aggregator (Lexicographic Constitutional Fusion)**  
   `AggregatorV8` fuses critic signals using strict constitutional priority:

   ```
   rights 
   > autonomy & agency 
   > fairness 
   > truth 
   > risk 
   > operations
   ```

   The aggregator produces an *interpretive constitutional assessment*, not permission or denial.  
   Interpretive statuses:

   - `aligned`  
   - `aligned_with_constraints`  
   - `misaligned`  
   - `requires_human_review`  

   Precedent and uncertainty inform the analysis but never supersede higher-order rights or dignity principles.

6. **Evidence Recording (Audit Layer)**  
   `EvidenceRecorder` captures:
   - critic outputs  
   - detector signals (if enabled)  
   - precedent bundle  
   - uncertainty metadata  
   - constitutional reasoning  

   Evidence is written to a buffer, JSONL file, or database depending on configuration.  
   Forensic mode exposes extended diagnostics and timing breakdowns.

---

## Public Interfaces

### **Async Engine**
`engine/engine.py` exposes:
- `run(text, context)`  
- `run_stream(text, context)`  

Both return structured constitutional assessments with trace IDs, critic outputs, uncertainty, and final aggregated interpretation.

### **Engine Builder**
`engine/core/__init__.py` exposes:
- `build_eleanor_engine_v8`

This bootstraps:
- router and adapters  
- detector engine (optional)  
- precedent store  
- evidence recorder  
- OPA client or injectable governance callback  

Builders tolerate missing adapters or stores for frictionless local development.

### **Router**
`engine/router/router.py` provides:
- automatic adapter selection  
- fallback to safe local/echo adapter  
- normalized response schema  
- resilience patterns (circuit breaker, retry-with-backoff)  

---

## Design Notes

- **Autonomy critic updated** to **Autonomy & Agency Critic** for clarity and alignment with constitutional dignity frameworks.
- Precedent and uncertainty layers operate in **advisory** capacity; they **inform** but do not **control** critic judgments.
- Aggregator strictly enforces **lexicographic constitutional priority**, ensuring that no downstream factor can override rights or dignity considerations.
- All components degrade gracefully—missing embeddings, missing stores, or missing adapters never cause pipeline failure.
- The architecture is explicitly **interpretive**, not enforcement-oriented. Eleanor evaluates model reasoning; it does not “allow” or “deny” actions.

---
