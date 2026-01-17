# Design Patterns

ELEANOR V8 uses a few core patterns to keep runtime behavior consistent and testable.

## Dependency injection

The engine is assembled via `engine/factory.py` so runtime dependencies can be swapped
for mocks in tests or alternative implementations in production.

## Protocol interfaces

Interfaces live in `engine/protocols.py`. Critics, routers, and recorders are defined
as protocols to keep the runtime loosely coupled.

## Orchestrator + pipeline

Runtime execution is coordinated by `engine/runtime/` modules. The pipeline is designed
to isolate failures and enable degradation strategies.

## Observability hooks

Tracing and metrics are handled by `engine/observability/` and event hooks in runtime.
See `docs/OBSERVABILITY.md`.
