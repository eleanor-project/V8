# Data Flow

ELEANOR V8 passes a structured context payload through each pipeline stage. Each stage
may add evidence and diagnostics while preserving original input.

```mermaid
sequenceDiagram
    participant Client
    participant Engine
    participant Router
    participant Critics
    participant Aggregator
    participant Governance

    Client->>Engine: Input text + context
    Engine->>Router: Select model
    Router-->>Engine: Model response
    Engine->>Critics: Critic evaluations
    Critics-->>Engine: Critic results
    Engine->>Aggregator: Aggregate signals
    Aggregator-->>Engine: Aggregated output
    Engine->>Governance: Review gate
    Governance-->>Engine: Decision flags
    Engine-->>Client: Output + metadata
```

## Evidence propagation

Evidence is recorded via the recorder interface and attached to the output in detail
levels 2 and 3. See `engine/runtime/run.py` for the evidence lifecycle.
