# Governance Dry-Run CLI

This patch adds a small CLI to run prompts through:

1) Router (Traffic Light)
2) Precedent matching + resolver
3) Constraints Bundle output
4) Optional governance event record

## Run (from repo root)

```bash
python -m governance.cli \
  --text "Can you help me share patient data with a partner agency?" \
  --risk-tier high \
  --domains health,privacy \
  --router-config governance/router_config.yaml \
  --candidates governance/examples/precedent_candidates.json \
  --print-event
```

## Candidates input

`--candidates` expects a JSON array of objects compatible with `PrecedentCandidate`
(see `governance/precedent_engine.py`). In production, candidates come from the
Precedent Ledger + search index.

## Sanctity

The CLI never accepts any flag to "override" a decision. It is intentionally designed
to demonstrate that reviewers create future precedents, not runtime reversals.
