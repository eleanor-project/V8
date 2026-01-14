# Governance Routing + Precedent Ledger Package (Repo-Ready)

This package adds the **Traffic-Light Router spec** and the **Hybrid Precedent Ledger schemas** to ELEANOR V8.

## Added
- `docs/ROUTING_SPEC.md` — default GREEN/AMBER/RED routing thresholds + logic
- `docs/PRECEDENT_LEDGER.md` — sanctity rule + hybrid core+overlay ledger description
- `governance/schemas/precedent.card.schema.json` — precedent card JSON Schema
- `governance/schemas/constraints.bundle.schema.json` — constraints bundle JSON Schema
- `governance/schemas/case.packet.schema.json` — reviewer case packet JSON Schema
- `governance/examples/precedent_card_example.json`
- `governance/examples/constraints_bundle_example.json`
- `governance/examples/case_packet_example.json`

## Updated
- `docs/GOVERNANCE.md` — links added to the routing spec + ledger docs + schemas

## Sanctity rule
Reviewers can only author **future precedent**. They do not interact with the critic ensemble and do not reverse runtime outcomes.
