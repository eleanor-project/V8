# ELEANOR Orchestrator (v0.2)

This module implements the ELEANOR constitutional decision engine, consisting of:

- Critic callers (via Ollama)
- Multi-critic execution engine
- Lexicographic + uncertainty-based aggregation
- FastAPI service

Run with:

```bash
./run.sh
```

Test with:

```bash
curl -X POST "http://127.0.0.1:8000/evaluate" -H "Content-Type: application/json" -d '{"text":"Example case"}'
```
