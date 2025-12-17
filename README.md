# ELEANOR V8 Engine Repository

ELEANOR V8 enforces constitutional governance at runtime: critic-declared escalation gates execution, `enforce_human_review` must approve every decision, and `execute_decision` refuses anything without an audit trail. CI (`.github/workflows/constitutional-ci.yml`) blocks merges if these guardrails drift.

Governance and escalation references:
- `docs/CRITIC_INDEPENDENCE_AND_ESCALATION.md` — critic isolation, dissent preservation, critic-initiated escalation.
- `docs/ESCALATION_Tiers_Human_Review_Doctrine.md` — canonical escalation tiers, cross-critic clause matrix, human review duties.
- `POLICY_CONSTITUTIONAL_GOVERNANCE.md` — CI invariants and why governance failures are treated as merge blockers.
