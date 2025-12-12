# Governance Layer (Interpretive, Not Enforcement-Based)
Eleanor’s governance layer provides **constitutional assessments and structured oversight metadata**, not behavioral control or enforcement. Governance in Eleanor is an **interpretive layer** designed to support transparency, auditability, traceability, and institutional review — *not permissioning, blocking, or restricting user actions.*

External systems **may** choose to use Eleanor’s assessments as part of their own governance processes, but Eleanor itself **never enforces decisions**.

---

## Purpose of the Governance Layer
The governance layer exists to:

- Provide a **standardized, policy-ready representation** of Eleanor’s deliberation  
- Enable **external oversight systems** (regulators, auditors, institutions) to understand *why* a model output aligns or misaligns with constitutional principles  
- Supply **structured constitutional reasoning**, not commands  
- Support **consultative review**, not control  

This distinction is fundamental to Eleanor’s identity as a **constitutional deliberation engine**, not an alignment enforcer.

---

## OPA Integration (Consultative Policy Assessment)
Eleanor can optionally forward structured output bundles to an external Policy Decision Point (PDP) such as Open Policy Agent (OPA).  
This interaction is **advisory**, not authoritative.

### OPA Client
`engine/governance/opa_client.py` sends the constitutional assessment bundle:

- critic outputs  
- aggregated constitutional interpretation  
- precedent signals  
- uncertainty metadata  
- model metadata  
- input snapshot  

OPA can return its own analysis in response, typically encoded as:

```json
{
  "result": {
    "aligned": true|false,
    "requires_review": true|false,
    "notes": [...]
  }
  }
```



Key Clarification

OPA does not override, replace, or control Eleanor's internal interpretation.
OPA provides external policy context that downstream systems may use for their own compliance or auditing frameworks.

Eleanor does not modify or restrict outputs based on OPA results.

Engine Hook

engine/core/__init__.py allows an optional governance_callback to be injected during engine construction.
Examples:

Default OPA client

A custom institutional reviewer

A logging-based governance observer

A no-op stub for local development

All callbacks operate in read-only advisory mode.

Error Handling

If a governance callback fails (network error, non-200 response, invalid JSON):

The engine records the failure in the evidence package

A requires_human_review flag may be set (if configured)

Engine output is NOT blocked or altered

Failures do not trigger “deny” or “allow” states — the governance system is not an enforcement mechanism.

Configuration

Environment variables configure governance integration:

ELEANOR_GOVERNANCE_URL (e.g., OPA)

ELEANOR_GOVERNANCE_PATH

Authentication headers (optional)

Governance timeout settings

If no governance endpoint is configured, Eleanor defaults to producing constitutional assessments only.

Governance Design Principles
1. Interpretive, Not Coercive

Governance exists to interpret, not to control or enforce.

2. External Policy Independence

OPA or any PDP cannot override Eleanor’s constitutional logic.

3. Auditability Over Control

Evidence bundles exist for transparency and traceability, not for permissioning.

4. Human Oversight First

When ambiguity or high uncertainty arises, Eleanor outputs:

requires_human_review
—not enforcement decisions.

5. Constitutional Supremacy

No external system may alter Eleanor’s lexicographic priority:

rights > autonomy & agency > fairness > truth > risk > pragmatics

6. Optional Governance

Governance integration is fully optional; the engine functions perfectly without OPA or external PDPs.

Future Hardening (All Non-Enforcement)

PDP health checks

Signed evidence payloads

Merkle-stamped evidence for chain-of-custody auditing

Tamper-evident governance logs

Distributed oversight mode

These features increase auditability and trust, not control.

Summary

Eleanor’s governance layer provides:

Constitutional oversight

Advisory assessments

Structured policy signals

Audit-ready evidence

Human-review triggers

It does not provide:

Blocking

Allow/deny authority

Enforcement

Behavioral restrictions

Alignment-like control logic

Eleanor governs interpretation, not users or systems.



