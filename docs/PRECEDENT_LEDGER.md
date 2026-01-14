# Precedent Ledger (Hybrid Core + Overlay)

ELEANOR’s precedent ledger is a **versioned, auditable policy memory**.

## Sanctity rule (separation of powers)
- **Runtime**: Critic Ensemble → Aggregator → Governor (ELEANOR) → Outcome
- **Reviewers**:
  - create/update **future** precedents only
  - do **not** talk to critics
  - do **not** reverse runtime decisions

## Hybrid approach
- **Core rights tags** are the canonical internal vocabulary.
- **Overlays** (UDHR/UNESCO/OrgPolicy/Regulatory) are optional citations to justify and explain.

## Schemas
- Precedent Card: `governance/schemas/precedent.card.schema.json`
- Constraints Bundle: `governance/schemas/constraints.bundle.schema.json`
- Case Packet (for reviewers): `governance/schemas/case.packet.schema.json`

## Versioning
- New guidance is a **new version** of the same `precedent_id`.
- No edits-in-place for active precedents; immutability enables audit.

## Publish control (recommended)
- Optional dual-control for **hard** precedents (two reviewers sign to publish).
