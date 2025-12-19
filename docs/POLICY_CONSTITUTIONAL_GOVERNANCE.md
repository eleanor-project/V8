# Constitutional Governance - CI & Policy Enforcement

This repository enforces constitutional AI governance through code, tests, and policy checks. CI failures are intentional guardrails, not optional warnings.

## Non-Negotiable Invariants
- No execution without human review enforcement: every decision must pass through `enforce_human_review` before `execute_decision` runs.
- Critic-declared escalation: escalation signals are emitted by critics with clause IDs; the aggregator cannot infer, suppress, or downgrade them.
- Tier-based authority: Tier 2 requires explicit human acknowledgment; Tier 3 requires explicit human determination. Execution remains blocked until satisfied.
- Immutable audit trail: escalations and human actions are time-stamped, attributed, and stored as immutable records.

## CI Enforcement
- Pytest proves behavioral invariants across critics, escalation, and execution gating.
- Targeted mypy keeps the escalation schema and enforcement path structurally typed.
- A policy check asserts the executor continues to call `ensure_executable` with an `ExecutableDecision`.

Pull requests that violate these checks fail CI and must not be merged. Governance enforcement is intentional and must not be bypassed for convenience, performance, or expediency.
