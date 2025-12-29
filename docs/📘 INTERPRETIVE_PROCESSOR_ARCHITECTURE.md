📙 INTERPRETIVE_PROCESSOR_CONFIGURATION.md
ELEANOR V8 — Interpretive Processor Configuration Guide

Version: 1.0
Status: Active
Audience: Engineers, platform architects, governance implementers

1. Purpose

This document describes how interpretive processors are configured and assigned within ELEANOR V8.

Interpretive processor configuration governs how critics receive interpretive support, not how constitutional decisions are made. Configuration choices affect interpretive depth, style, and context handling—never constitutional priority, never rights protection, and never decision authority.

This guide replaces legacy “model configuration” assumptions with a governance-aligned, constitutionally safe configuration model.

2. Configuration Principles

All interpretive processor configuration in Eleanor adheres to the following principles:

Constitutional invariants are immutable

Rights and autonomy evaluation cannot be weakened

Lexicographic priority cannot be altered

Processors support critics; they do not decide

Critics interpret

Aggregators decide

Processors assist

Configuration affects interpretation, not obligation

No configuration can reduce dignity, fairness, or safety guarantees

Processor choice is hermeneutic, not economic

Selection is based on interpretive style and suitability

Never cost- or latency-driven for sensitive domains

3. Configuration Hierarchy

Interpretive processor selection follows a strict and explicit priority order:

runtime override
→ explicit critic assignment
→ registry assignment
→ default processor
→ configuration error


This hierarchy ensures deterministic behavior and prevents silent fallback that could obscure interpretive intent.

4. Interpretive Processor Registry

Eleanor uses a centralized Interpretive Processor Registry to manage processor assignments.

4.1 Registry Responsibilities

The registry is responsible for:

mapping critics to interpretive processors

managing interpretive fidelity modes

validating constitutional constraints

enforcing override safety rules

loading configuration from YAML/JSON

5. Interpretive Fidelity Modes

Processors operate under one of three Interpretive Fidelity Modes:

Mode	Description
High Fidelity	Full constitutional depth and contextual expansion
Medium Fidelity	Balanced interpretation with bounded expansion
Light Fidelity	Minimal scaffolding for low-risk contexts
Guarantees

Rights and Autonomy critics always operate at High Fidelity

Fidelity modes never reduce constitutional strength

Fidelity affects how much interpretation occurs, not what must be protected

6. YAML Configuration
6.1 Example: Interpretive Processor Configuration
# config/interpretive_processors.yaml

default_processor: claude-sonnet-4.5

fidelity_modes:
  high: claude-opus-4.5
  medium: claude-sonnet-4.5
  light: claude-haiku-4.0

critics:
  rights:
    fidelity: high

  autonomy:
    fidelity: high

  fairness:
    fidelity: high

  truth:
    fidelity: high

  risk:
    fidelity: medium

  operations:
    fidelity: light

Notes

No critic is assigned a processor directly by default—only a fidelity mode

Fidelity modes map to processors centrally

Changing a processor affects interpretive style, not governance behavior

7. Loading Configuration
from engine.models.registry import InterpretiveProcessorRegistry

registry = InterpretiveProcessorRegistry.from_yaml(
    "config/interpretive_processors.yaml"
)


The registry validates:

presence of required critics

fidelity mode correctness

constitutional constraints

processor availability

Invalid configurations raise explicit errors.

8. Explicit Critic Assignment (Advanced)

Explicit assignment may be used for research, testing, or controlled deployments.

from engine.critics.rights import RightsCriticV8
from engine.models.processors import OpusProcessor

rights = RightsCriticV8(
    processor=OpusProcessor(),
    fidelity="high"
)

Constraints

Explicit assignments cannot downgrade fidelity for protected critics

Rights and Autonomy reject non-high fidelity assignments

Explicit assignment overrides registry only for that critic

9. Runtime Overrides

Runtime overrides allow contextual interpretive adjustment.

await engine.run(
    user_input,
    context={
        "interpretive_override": {
            "operations": {
                "fidelity": "medium"
            }
        }
    }
)

Runtime Rules

Overrides are scoped to a single execution

Rights and Autonomy cannot be overridden

Overrides must pass constitutional validation

Overrides are logged in the evidence record

10. Validation and Safety Checks

Eleanor enforces configuration safety at startup and runtime.

Rejected Configurations Include:

non-high fidelity for Rights or Autonomy

undefined fidelity modes

missing default processor

processor substitution that violates critic constraints

silent downgrade attempts

Validation failures raise explicit, actionable errors.

11. Evidence and Auditability

All interpretive processor decisions are recorded:

processor identifier

critic assignment

fidelity mode

override source (registry / explicit / runtime)

validation outcome

This metadata is included in:

evidence bundles

forensic mode output

governance review payloads

OPA advisory calls (if enabled)

12. OPA and External Governance

Interpretive processor metadata may be forwarded to OPA or other governance systems as advisory context only.

OPA:

does not select processors

does not alter fidelity

does not override critic results

may flag misconfiguration or escalation conditions

13. Migration Guidance
From Legacy Model Configuration

Before

critic.evaluate(model=some_model)


After

registry = InterpretiveProcessorRegistry.from_yaml(...)
critic = RightsCriticV8(registry=registry)

Key Migration Changes

“model” → “interpretive processor”

“tier” → “fidelity mode”

“cost optimization” → “interpretive suitability”

implicit fallback → explicit validation

14. Best Practices

Use YAML for all production deployments

Treat processor changes as governance-impacting changes

Review fidelity mappings during audits

Avoid runtime overrides outside controlled testing

Never tune interpretive processors for cost in sensitive domains

15. Summary

Interpretive processor configuration in Eleanor V8 provides:

flexible interpretive support

strong constitutional guarantees

deterministic governance behavior

audit-ready transparency

safe extensibility across models and environments

This system enables Eleanor to evolve alongside foundation models while preserving dignity, rights, and institutional trust.
