# ELEANOR Governance Implementation Notes (Patch Pack #2)

This patch pack adds **starter modules** for:

- `governance/router.py` — a config-driven **Traffic Light router** (GREEN/AMBER/RED)
- `governance/precedent_engine.py` — a **precedent matching** pipeline (filter → retrieve → resolve)
- `governance/router_config.yaml` — default thresholds + behaviors

## Sanctity rule (non-negotiable)

- Human reviewers **do not** interact with the critic ensemble.
- Human reviewers **do not** override runtime governance outcomes.
- Reviewers author **future precedent** only (new versions in the Precedent Ledger).

Runtime flow:

1) Router decides route
2) ELEANOR Governor matches precedent + policy → produces **Constraints Bundle**
3) Model generates **only after** constraints exist
4) Audit receipt is appended

## What’s stubbed vs real

`precedent_engine.py` includes placeholder functions:

- `vector_similarity(...)`
- `bm25_normalized(...)`

Replace those with your actual retrieval stack (vector DB / embeddings + keyword index).

`trigger_hits(...)` is a small string matcher you can keep or replace with regex patterns.

## Next wiring steps

- Connect `match_precedents(...)` to your Postgres ledger (`precedent_versions`) and search index.
- Merge retrieved precedent decision constraints into the Constraints Bundle.
- Add a router input adapter (risk classifier + coverage score + uncertainty proxy/telemetry).

