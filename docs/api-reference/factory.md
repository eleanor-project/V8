# DependencyFactory API

`engine/factory.py` constructs engine dependencies with clean defaults and test-friendly
overrides.

## EngineDependencies

Container for router, critics, recorder, and other dependencies.

## DependencyFactory methods

- `create_router()`
- `create_detector_engine()`
- `create_evidence_recorder()`
- `create_precedent_engine()`
- `create_precedent_retriever()`
- `create_uncertainty_engine()`
- `create_aggregator()`
- `create_review_trigger_evaluator()`

See `engine/factory.py` for signatures.
