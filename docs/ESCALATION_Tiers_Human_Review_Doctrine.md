# ELEANOR V8 — Escalation Tiers & Human Review Doctrine (v8.1)

Version v8.1 is enforced in code via `engine/schemas/escalation.py`, `engine/aggregator/escalation.py`, `engine/execution/human_review.py`, and CI at `.github/workflows/constitutional-ci.yml`.

## 1. Purpose
Defines escalation tiers, triggers, and human review requirements so automation never exceeds legitimate authority, minority concerns cannot be silently suppressed, and humans explicitly own risk when proceeding past constitutional warnings. Escalation is governance, not error.

## 2. Core Principles
### 2.1 Separation of Roles
- Critics detect constitutional risk and trigger escalation.
- Aggregator synthesizes reasoning and preserves dissent.
- Humans exercise authority where legitimacy requires it.
No layer usurps another.

### 2.2 Execution Gating, Not Reasoning Override
- Escalation gates automatic execution/finality, not analysis.
- Aggregator outputs are never altered by escalation.
- Critics do not decide outcomes.
- Humans do not rewrite machine reasoning.

### 2.3 Unilateral Escalation Authority
Any single critic may trigger escalation when charter clauses are met. Escalation does not require consensus, cannot be vetoed, and is binding on execution pathways.

## 3. Escalation Tiers (Canonical)
🟡 **Tier 2 — Mandatory Human Acknowledgment**  
Automation may proceed only with explicit human acknowledgment of risk/ambiguity/moral cost. No silent continuation; attribution required.  
Human action: “I understand the identified risk and accept responsibility for proceeding.”

🔴 **Tier 3 — Mandatory Human Determination**  
Automation lacks legitimacy to decide alone. Automatic execution is prohibited; a human must actively proceed/modify/halt. Logged as human-authored.  
Human action: “I affirmatively determine the appropriate course of action.”

## 4. Cross-Critic Escalation Matrix
| Critic | Clause | Trigger | Tier |
| --- | --- | --- | --- |
| Autonomy | A1 | Meaningful consent failure | Tier 2 |
|  | A2 | Coercive / manipulative influence | Tier 3 |
|  | A3 | Irreversible loss of agency | Tier 3 |
| Dignity | D1 | Instrumentalization of persons | Tier 2 |
|  | D2 | Degrading / dehumanizing outcome | Tier 3 |
|  | D3 | Harm without voice | Tier 2 |
| Privacy & Identity | P1 | Non-consensual identity inference | Tier 3 |
|  | P2 | Persistent / linked identity | Tier 3 |
|  | P3 | Context collapse | Tier 2 |
| Fairness | F1 | Protected class impact | Tier 3 |
|  | F2 | Structural bias amplification | Tier 2 |
|  | F3 | Opaque differential treatment | Tier 2 |
| Due Process | DP1 | No contestability | Tier 3 |
|  | DP2 | No attribution | Tier 3 |
|  | DP3 | Unreviewable automation | Tier 2 |
| Precedent | PR1 | No precedent + high impact | Tier 3 |
|  | PR2 | Conflicting precedent | Tier 2 |
|  | PR3 | Uncontrolled norm creation | Tier 3 |
| Uncertainty | U1 | Epistemic insufficiency | Tier 2 |
|  | U2 | High impact × high uncertainty | Tier 3 |
|  | U3 | Competence boundary exceeded | Tier 3 |

## 5. Intentional Tier Convergence
Different moral failure modes may share a tier; tiers reflect legitimacy and responsibility, not fine-grained severity scores.

## 6. Human Review Responsibilities
- Human reviewers must not edit critic or aggregator outputs.
- All acknowledgments/determinations are time-stamped, attributed, and linked to the aggregator record.
- Proceeding after escalation = acceptance of responsibility.

## 7. Audit & Traceability Guarantees
For every escalation the system preserves: triggering critic + clause, tier, aggregator synthesis, human action, timestamps, attribution. Records are immutable and auditable.

## 8. Enforcement Linkage
- Schemas and gates are implemented in `engine/schemas/escalation.py`, `engine/aggregator/escalation.py`, and `engine/execution/human_review.py`.
- CI (`.github/workflows/constitutional-ci.yml`) blocks merges if escalation, human review, or execution guardrails are altered.
- Policy intent is captured in `POLICY_CONSTITUTIONAL_GOVERNANCE.md` so governance failures are visible to reviewers.

## 9. Closing Principle
> Automation may recommend. Critics may warn. Only humans may legitimize irreversible risk.

See also: `docs/CRITIC_INDEPENDENCE_AND_ESCALATION.md` for independence/dissent rules and autonomy clause rationale.***
