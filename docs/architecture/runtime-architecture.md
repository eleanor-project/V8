# Runtime Architecture

Runtime execution is split across focused modules so each stage is isolated and testable.

## Key runtime modules

- `engine/runtime/run.py`: orchestrates the full pipeline for `engine.run()`
- `engine/runtime/streaming.py`: streaming execution for `engine.run_stream()`
- `engine/runtime/critics.py`: critic execution and circuit breaker handling
- `engine/runtime/pipeline.py`: precedent + uncertainty stages
- `engine/runtime/governance.py`: governance gate evaluation

## Runtime lifecycle

1. Validate input and normalize context.
2. Run detectors and enrich context.
3. Route to a model backend.
4. Execute critics (parallel or batched).
5. Aggregate critic results.
6. Perform precedent alignment and uncertainty evaluation (if enabled).
7. Apply governance review gates and escalation.
8. Emit output plus forensic metadata.
