# ELEANOR V8.0 — Constitutional Critics & Escalation Governance Handbook

**Version:** v8.0 (2025-12-17)
**Status:** Canonical / Binding
**Scope:** Critics → Aggregation → Human Review → Execution → CI → Deployment (OPA)

---

## Table of Contents

1. Purpose
2. System Roles and Non-Negotiable Boundaries
3. Escalation Tiers
4. Dissent Preservation Mechanism
5. Critic Mandates and Boundaries Matrix
6. Clause Reference Matrix
7. Critic Charter Sections

   * 7.1 Autonomy
   * 7.2 Dignity
   * 7.3 Privacy & Identity
   * 7.4 Fairness & Non-Discrimination
   * 7.5 Due Process & Accountability
   * 7.6 Precedent & Legitimacy
   * 7.7 Uncertainty
8. Implementation Discipline

   * 8.1 Clause-Aware Escalation Signals
   * 8.2 Routing and Execution Gating
   * 8.3 CI and Policy Checks
   * 8.4 Deployment Enforcement (OPA)
9. Amendment Notice (v8.1)
10. Appendix

* A. Canonical Human Review Statements
* B. Glossary

---

## 1. Purpose

This handbook defines the **constitutional critics** and the **escalation governance doctrine** for ELEANOR V8.1.

It exists to ensure that:

* constitutional concerns cannot be silently suppressed,
* automation never exceeds legitimate authority,
* humans explicitly own risk when proceeding past constitutional warnings,
* and execution is gated in a way that remains enforceable under real operational pressure.

**Escalation is not an error state.**
Escalation is a **governance safeguard**.

---

## 2. System Roles and Non-Negotiable Boundaries

### 2.1 Critics

Critics:

* evaluate constitutional risk within their domain,
* emit structured concerns, and
* emit clause-aware escalation signals when thresholds are met.

Critics **do not**:

* approve actions,
* override other critics,
* or decide final outcomes.

Think: **circuit breakers**, not judges.

---

### 2.2 Aggregator

The Aggregator:

* preserves all critic outputs verbatim,
* synthesizes reasoning, and
* computes execution gating based on escalation tier.

The Aggregator **shall not**:

* suppress escalation,
* average it away,
* reinterpret it,
* or down-rank it due to consensus or convenience.

---

### 2.3 Human Review

Humans:

* satisfy execution gates (Tier 2 or Tier 3),
* accept responsibility, and
* provide attributable decisions where legitimacy requires it.

Humans **shall not**:

* edit critic outputs,
* rewrite aggregator reasoning,
* or treat escalation as “optional.”

---

### 2.4 Execution Layer

Execution:

* may proceed **only** with an `ExecutableDecision` that has passed human review enforcement.

Execution **shall not**:

* occur directly from critic output, or
* occur directly from aggregation output.

---

### 2.5 External Enforcement (OPA)

OPA may enforce the same invariants **independently of application code**, ensuring governance is not bypassed by:

* misconfiguration,
* hot-patching,
* “just this once” exceptions,
* or compromised services.

---

## 3. Escalation Tiers

### 🟡 Tier 2 — Mandatory Human Acknowledgment

Tier 2 applies when automation may proceed **only** if a human explicitly acknowledges the constitutional risk and accepts responsibility.

Characteristics:

* no silent continuation,
* full audit logging required,
* action is allowed only with attributable ownership.

Human action: **acknowledgment** (see Appendix A).

---

### 🔴 Tier 3 — Mandatory Human Determination

Tier 3 applies when automation lacks legitimacy to decide alone.

Characteristics:

* automatic execution is prohibited,
* a human must decide whether to proceed, modify, or halt,
* decision is logged as human-authored and attributable.

Human action: **determination** (see Appendix A).

---

### 3.1 Intentional Tier Convergence

Multiple clauses may intentionally map to the same tier.

This reflects that distinct moral failure modes can require the same form of human accountability without implying identical severity, cause, or remedy.

This is a feature. Not an oversight.

---

## 4. Dissent Preservation Mechanism

**Unilateral escalation authority:** Any single critic may trigger escalation.
Escalation:

* does **not** require consensus,
* cannot be vetoed, and
* is binding on execution.

When escalation occurs, the system shall:

* record critic ID, clause ID, tier, rationale, and doctrine reference,
* surface escalation explicitly in aggregator output,
* preserve dissent verbatim,
* gate execution automatically.

Escalation **does not**:

* override reasoning,
* modify evaluations,
* imply correctness of the dissenting critic.

---

## 5. Critic Mandates and Boundaries Matrix

| Critic                        | Owns                                                                                          | Must NOT                               | Intentional Overlap                                                    |
| ----------------------------- | --------------------------------------------------------------------------------------------- | -------------------------------------- | ---------------------------------------------------------------------- |
| Autonomy                      | consent, agency, coercion, reversibility                                                      | dignity/fairness/outcome judgment      | overlaps with Privacy only in consent framing; logic remains distinct  |
| Dignity                       | intrinsic worth, non-degradation, moral presence                                              | fairness math; consent checks          | overlaps with Due Process: “voice” vs “appeal”                         |
| Privacy & Identity            | inference, persistence/linkage, contextual integrity, secondary use                           | fairness/dignity consequence scoring   | overlaps with Fairness when identity inference causes disparate impact |
| Fairness & Non-Discrimination | disparate impact, protected class outcomes, bias amplification, opaque differential treatment | intent/virtue; dignity framing         | overlaps with Due Process: explainability vs contestability            |
| Due Process & Accountability  | contestability, attribution, reviewability                                                    | fairness/dignity re-argument           | overlaps with Uncertainty: auditability vs epistemic insufficiency     |
| Precedent & Legitimacy        | norm creation, precedent voids/conflicts                                                      | present harm scoring; consent analysis | overlaps with Due Process on legitimacy documentation                  |
| Uncertainty                   | missing context, competence bounds, high-impact unknowns                                      | moral judgment                         | overlaps with Due Process: “can we know” vs “can we review”            |

---

## 6. Clause Reference Matrix

### Autonomy

* **A1 (Tier 2):** Meaningful consent failure
* **A2 (Tier 3):** Coercive or manipulative influence
* **A3 (Tier 3):** Irreversible autonomy impact

### Dignity

* **D1 (Tier 2):** Instrumentalization of persons
* **D2 (Tier 3):** Degrading / dehumanizing outcome
* **D3 (Tier 2):** Asymmetric harm without voice

### Privacy & Identity

* **P1 (Tier 3):** Non-consensual identity inference
* **P2 (Tier 3):** Identity persistence or linkage across contexts
* **P3 (Tier 2):** Context collapse
* **P4 (Tier 2):** Secondary use expansion without renewed consent

### Fairness & Non-Discrimination

* **F1 (Tier 3):** Protected class impact
* **F2 (Tier 2):** Structural bias amplification
* **F3 (Tier 2):** Opaque differential treatment

### Due Process & Accountability

* **DP1 (Tier 3):** Lack of contestability
* **DP2 (Tier 3):** Decision without attribution
* **DP3 (Tier 2):** Unreviewable automation

### Precedent & Legitimacy

* **PR1 (Tier 3):** Precedent void in high-impact domain
* **PR2 (Tier 2):** Precedent conflict
* **PR3 (Tier 3):** Uncontrolled precedent creation

### Uncertainty

* **U1 (Tier 2):** Epistemic insufficiency
* **U2 (Tier 3):** High impact × high uncertainty
* **U3 (Tier 3):** Competence boundary exceeded

---

## 7. Critic Charter Sections

### 7.1 Autonomy Critic

**Mandate:** Meaningful consent, agency, coercion resistance, reversibility of choice.
**Must NOT:** Judge dignity, fairness, or outcomes.

| Clause | Trigger                                                                                  | Tier   |
| ------ | ---------------------------------------------------------------------------------------- | ------ |
| A1     | action materially affects an individual without informed, voluntary consent/refusal      | Tier 2 |
| A2     | coercion, manipulation, deception, or asymmetric pressure meaningfully constrains choice | Tier 3 |
| A3     | non-reversible limitation on future choice or self-determination                         | Tier 3 |

**Interpretation:** Autonomy judges whether **choice itself** was respected—not whether the outcome is good.

---

### 7.2 Dignity Critic

**Mandate:** Intrinsic worth, non-instrumentalization, non-degradation, moral presence.
**Must NOT:** Do fairness math or consent analysis.

| Clause | Trigger                                                                          | Tier   |
| ------ | -------------------------------------------------------------------------------- | ------ |
| D1     | persons treated primarily as means rather than ends                              | Tier 2 |
| D2     | humiliation, stigmatization, demeaning or dehumanizing treatment risk            | Tier 3 |
| D3     | harm imposed without meaningful participation, representation, or moral presence | Tier 2 |

**Interpretation:** Dignity failures become dangerous when they become ordinary. Escalation prevents quiet normalization.

---

### 7.3 Privacy & Identity Critic

**Mandate:** Identity inference controls, persistence/linkage limits, contextual integrity, scope control (secondary use).
**Must NOT:** Decide fairness or dignity consequences.

| Clause | Trigger                                                             | Tier   |
| ------ | ------------------------------------------------------------------- | ------ |
| P1     | identity inferred without explicit consent                          | Tier 3 |
| P2     | persistent identity linkage across sessions/domains                 | Tier 3 |
| P3     | context collapse: cross-domain use violates contextual integrity    | Tier 2 |
| P4     | secondary use beyond original authorization without renewed consent | Tier 2 |

**Interpretation:** Capability ≠ permission. P4 exists to stop scope creep from becoming “normal.”

---

### 7.4 Fairness & Non-Discrimination Critic

**Mandate:** Disparate impact, protected class outcomes, structural bias amplification, explainable disparity.
**Must NOT:** Judge intent or dignity.

| Clause | Trigger                                                        | Tier   |
| ------ | -------------------------------------------------------------- | ------ |
| F1     | differential impact affecting protected classes                | Tier 3 |
| F2     | feedback loops or design amplify structural inequity           | Tier 2 |
| F3     | materially different outcomes cannot be meaningfully explained | Tier 2 |

**Interpretation:** This critic blocks “math laundering.” Dashboards don’t absolve moral responsibility.

---

### 7.5 Due Process & Accountability Critic

**Mandate:** Contestability, attribution, reviewability.
**Must NOT:** Re-argue fairness/dignity.

| Clause | Trigger                                                          | Tier   |
| ------ | ---------------------------------------------------------------- | ------ |
| DP1    | no meaningful appeal, challenge, or review path                  | Tier 3 |
| DP2    | responsibility cannot be assigned to an accountable authority    | Tier 3 |
| DP3    | decision cannot be reconstructed, audited, or explained post-hoc | Tier 2 |

**Interpretation:** If no one can challenge it and no one can own it, the system must not act alone.

---

### 7.6 Precedent & Legitimacy Critic

**Mandate:** Norm creation risk, precedent voids/conflicts, repeatability legitimacy.
**Must NOT:** Score present harm magnitude directly.

| Clause | Trigger                                                                   | Tier   |
| ------ | ------------------------------------------------------------------------- | ------ |
| PR1    | no relevant precedent in high-impact domain                               | Tier 3 |
| PR2    | conflicting precedents require interpretation and explicit acknowledgment | Tier 2 |
| PR3    | decision creates an operational norm without explicit authorization       | Tier 3 |

**Interpretation:** Precedent failures compound. This critic watches **tomorrow**, not just today.

---

### 7.7 Uncertainty Critic

**Mandate:** Epistemic humility, insufficiency detection, competence envelope adherence.
**Must NOT:** Judge morality.

| Clause | Trigger                                                      | Tier   |
| ------ | ------------------------------------------------------------ | ------ |
| U1     | critical context missing for responsible judgment            | Tier 2 |
| U2     | high impact + high uncertainty exceeds safety bounds         | Tier 3 |
| U3     | scenario outside validated competence envelope / assumptions | Tier 3 |

**U3 competence envelope (required triggers include):**

* validated domain assumptions do not hold,
* required evidence sources are unavailable/untrusted,
* confidence bounds are invalidated by missing prerequisites,
* charter-defined scope exclusions apply.

**Interpretation:** Uncertainty is not a flaw. Unacknowledged uncertainty is.

---

## 8. Implementation Discipline

### 8.1 Clause-Aware Escalation Signals

Critics SHALL emit clause-aware escalation signals directly (e.g., `P4`, `DP2`).
Escalation SHALL NOT be inferred at aggregation time.

Clause IDs, tiers, and descriptions should be defined as critic helper methods to prevent drift.

---

### 8.2 Routing and Execution Gating

All post-aggregation decisions SHALL route through human review enforcement before execution.

* Aggregation produces `AggregationResult` (may be gated)
* Human review enforcement produces `ExecutableDecision`
* Execution accepts **only** `ExecutableDecision`

No bypass paths are permitted.

---

### 8.3 CI and Policy Checks

CI SHALL fail if:

* governance tests fail,
* type enforcement fails, or
* execution bypass paths are introduced.

Recommended checks:

* `pytest`
* static typing (e.g., mypy)
* policy scan rejecting direct execution paths outside the approved pipeline

---

### 8.4 Deployment Enforcement (OPA)

OPA policies MAY enforce:

* tier-to-human-action matching,
* linkage to triggering escalation signals,
* denial of execution when escalation requirements are unsatisfied.

OPA acts as an external enforcement layer that does not trust application code.

---

## 9. Amendment Notice (v8.1)

### Added: Privacy Clause P4 (Tier 2)

**P4** formalizes secondary use expansion without renewed consent as a Tier 2 governance trigger.

### Clarified: Uncertainty U3 competence envelope (Tier 3)

U3 triggers based on scope exclusions, missing prerequisites, invalidated confidence bounds, or out-of-domain assumptions.

### Confirmed: Intentional Tier Convergence

Multiple Tier 2 clauses are deliberate and preserve meaningful accountability without artificial precision.

### Implementation alignment note

When validating that a human action references all triggering escalation signals, ensure the validator compares:

* `aggregation_result.escalation_summary.triggering_signals`
  against
* `human_action.linked_escalations`
  using an explicit `aggregation_result` argument (avoid implicit references).

---

## 10. Appendix

### A. Canonical Human Review Statements (Binding)

**Tier 2 — Acknowledgment (HUMAN_ACK)**
“I acknowledge the identified constitutional risks and accept responsibility for proceeding.”

**Tier 3 — Determination (HUMAN_DETERMINATION)**
“I affirmatively determine the appropriate course of action in light of the identified constitutional risks.”

---

### B. Glossary

* **Critic:** Domain-bounded constitutional evaluator that can escalate.
* **EscalationSignal:** Clause-aware governance trigger emitted by a critic.
* **Execution Gate:** Binding decision that automation may not proceed without human action.
* **Dissent Preservation:** Guarantee that minority critic concerns cannot be suppressed.
* **OPA:** External policy engine enforcing governance independently of application code.

---


