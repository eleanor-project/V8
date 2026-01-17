# Error Codes and Exceptions

ELEANOR V8 surfaces structured exceptions for pipeline failures and governance
signals. Exceptions live in `engine/exceptions.py`.

## Core exceptions

- `EleanorV8Exception` - base class for engine errors.
- `InputValidationError` / `ValidationError` - invalid input or context.
- `CriticEvaluationError` - critic failure during evaluation.
- `RouterSelectionError` - router could not select a model.
- `AggregationError` - critic aggregation failed.
- `PrecedentRetrievalError` - precedent lookup failed.
- `UncertaintyComputationError` - uncertainty computation failed.
- `GovernanceEvaluationError` - governance evaluation failed.
- `GovernanceBlockError` - governance required human review.
- `EvidenceRecordingError` - evidence recorder failed.
- `DetectorExecutionError` - detector pipeline failed.
- `ConfigurationError` - configuration invalid or missing.
- `CircuitBreakerOpenError` - a circuit breaker is open.
- `TimeoutError` - operation exceeded timeout.

## Handling guidance

- Use `details` on the exception to capture structured diagnostics.
- Log `trace_id` when available to correlate across systems.
- Treat `GovernanceBlockError` as a non-fatal governance signal.

For runtime defaults, see `engine/runtime/errors.py`.
