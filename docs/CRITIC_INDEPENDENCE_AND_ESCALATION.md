# Critic Independence, Dissent Preservation, and Escalation (V8)

This document codifies the rules we agreed in conversation: critics stay epistemically isolated; dissent is preserved; critics can unilaterally gate automation; and autonomy escalation clauses are mapped to explicit human-review tiers.

## Independence and Dissent (Spec Language)
- **Epistemic isolation:** Critics SHALL NOT see peer outputs, aggregator reasoning, downstream outcomes, or human preferences during primary evaluation. This is non-configurable at runtime.
- **Non-consensus principle:** Disagreement is expected and valued. No critic may modify primary evaluation to achieve consensus or coherence.
- **Post-hoc boundary:** Cross-critic visibility occurs only after all primary evaluations are sealed. Critics SHALL NOT revise primary judgments post-aggregation.
- **Dissent integrity:** Any concern above a critic’s threshold must surface verbatim in the aggregator output; dissent cannot be averaged away or suppressed.
- **Controlled dissent review (rare):** If invoked, it is logged/versioned, allows clarification only, never coercion; original evaluations remain preserved.

## Critic-Initiated Human Review Escalation (Spec Language)
- **Unilateral authority:** Each critic may trigger human review when its chartered conditions are met; no aggregator or peer approval required.
- **Scope:** Escalation gates automatic execution/finality and requires human involvement; it does **not** modify critic or aggregator reasoning.
- **Signaling:** The system records triggering critic, severity/rationale, charter clause, and required escalation tier. The signal is binding on execution.
- **Aggregator obligations:** Surface the escalation explicitly, preserve dissent verbatim, continue synthesis without suppression.

## Escalation Tiers (for reference)
- **Tier 2 — Mandatory Human Acknowledgment:** Aggregator output may proceed only after a human explicitly acknowledges the risk (accepting responsibility). No change to reasoning.
- **Tier 3 — Mandatory Human Determination:** Automatic execution prohibited. A human must actively decide to proceed/modify/halt. Logged as human-authored.

## Autonomy Critic — Escalation Clauses and Tiers (Canon)
| Clause ID | Definition | Tier | Why |
| --- | --- | --- | --- |
| A1 — Meaningful Consent Failure | Material effect without a reasonable opportunity for informed, voluntary consent/refusal. | Tier 2 (Human Acknowledgment) | Serious, often UX/scale related; requires ownership but not automatic halt. |
| A2 — Coercive or Manipulative Influence | Pressure/deception/asymmetric information constraining free choice. | Tier 3 (Human Determination) | Legitimacy breach; automation cannot decide acceptability. |
| A3 — Irreversible Autonomy Impact | Non-reversible limitation on future choice/agency/self-determination. | Tier 3 (Human Determination) | Irreversibility demands explicit human decision. |

Hard rule: a critic’s escalation cannot be vetoed or averaged out. It blocks automatic finality until the required human action occurs.

## Cross-Critic Clause Matrix
All critics now carry rights-anchored clauses and tiers (see `governance/critic_escalation.py` and `docs/ESCALATION_Tiers_Human_Review_Doctrine.md`). Clauses are executable contracts: when a critic triggers one, it emits an `EscalationSignal` that gates automation until the mandated human action occurs.

## Enforcement Notes
- Escalation signals are encoded in `engine/schemas/escalation.py` and surfaced via `engine/aggregator/escalation.py`.
- Human review enforcement lives in `engine/execution/human_review.py` and is guarded by CI in `.github/workflows/constitutional-ci.yml`.
- Policy intent is mirrored in `POLICY_CONSTITUTIONAL_GOVERNANCE.md` so governance drift is caught in review.
