# Schemas and Types

Key schema definitions live in `engine/schemas/` and `engine/types/`.

- `engine/schemas/pipeline_types.py`: CriticResult, AggregationOutput, etc.
- `engine/schemas/critic_schemas.py`: Pydantic critic evaluation models
- `engine/runtime/models.py`: EngineResult, EngineCriticFinding
- `engine/types/definitions.py`: core type aliases

These types define the contract between runtime stages and APIs.
