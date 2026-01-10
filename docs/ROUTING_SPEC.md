# Traffic-Light Routing Spec (ELEANOR Front Door)

This spec defines **when** a request enters ELEANOR’s governance court (**GREEN / AMBER / RED**).

## Design goals
- Route **only** risky / novel / uncertain requests into deeper governance.
- Keep low-risk requests fast.
- Preserve sanctity: **reviewers author future precedent**; they do **not** alter runtime critic decisions.

## Routing signals (multi-signal by design)
**Signal A — Risk Domain**
- **High:** medical, legal, finance, minors, privacy/PII, discrimination, self-harm, weapons, regulated compliance
- **Medium:** HR, education guidance, consumer product safety, sensitive interpersonal
- **Low:** general knowledge, creative writing, non-regulated how-tos

**Signal B — Precedent Coverage**
- `coverage_score ∈ [0,1]` from hybrid precedent match (vector + keyword + triggers)

**Signal C — Uncertainty**
Choose one:
- **Telemetry (preferred where available):** sentence-level aggregates (avg/min logprob, entropy spikes)
- **Proxy (works everywhere):** multi-sample divergence (2–3 short candidates) or cross-model agreement

## Default thresholds (starting point)
Coverage thresholds (tune after audit data):
- High risk: `coverage >= 0.75` → likely GREEN
- Medium risk: `coverage >= 0.65` → likely GREEN
- Low risk: `coverage >= 0.55` → likely GREEN

Uncertainty proxy (example bands):
- `divergence >= 0.35` → high uncertainty
- `0.20–0.35` → medium
- `< 0.20` → low

## Router decision logic
**Hard RED triggers (immediate):**
- Explicit policy violation triggers
- Prohibited content categories for the deployment
- Matches a **hard** precedent with outcome `refuse`

**AMBER triggers:**
- High risk AND `coverage < 0.75`
- High risk AND `uncertainty_high`
- Medium risk AND (`coverage < 0.65` AND `uncertainty_medium_or_high`)
- Any prompt implying a rights conflict AND coverage is not strong

**GREEN:**
- Otherwise (still audited)

## Post-route workflow
- Router outputs `GREEN/AMBER/RED`
- If AMBER/RED → ELEANOR produces a **Constraints Bundle** (see `governance/schemas/constraints.bundle.schema.json`)
- The model generates the user-facing response **only after** constraints exist.

## Why this is “model off the hook”
The model is not punished for escalating. ELEANOR decides; the model communicates.
