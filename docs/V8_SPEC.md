# V8 Specification (Enhanced)

**Version**: 8.0.0  
**Last Updated**: December 11, 2025  
**Status**: Production-Ready with Recommended Enhancements

---

## Core Pipeline

### Pipeline Flow
```
Input → Router → [Detectors] → Critics → Precedent → Alignment → Uncertainty → Aggregator → OPA → Evidence → Output
                  ^^^^^^^^^^^
                  Optional Pre-Processing
```

### Components

- **Router**: async `RouterV8` with adapter fallback, context-aware calls, and diagnostic metadata. Returns `response_text`, model metadata, and attempt trace. Default registry boots available adapters (OpenAI/Anthropic/xAI/local HF/Ollama) based on installed SDKs + keys.

- **Detectors** (Optional Pre-Processing Layer):
  - 25 specialized pattern-based detectors organized by constitutional dimension
  - Run in parallel before critics to provide early signal detection and pre-filtering
  - **Categories**: Rights (8), Fairness (3), Truth (4), Risk (5), Pragmatics (5)
  - **Integration**: Optional layer that can be enabled/disabled without affecting core pipeline
  - **Performance**: <2s for full suite (<100ms per detector)
  - **Output**: DetectorSignal with severity (0-1), violations, evidence, and flags
  - **Use Cases**: Pre-screening obvious violations, reducing critic workload, granular detection
  - **Implementation**: `engine/detectors/engine.py` → `DetectorEngineV8`
  
  **Detector List**:
  - Rights/Dignity: autonomy, coercion, dehumanization, discrimination, disparate_treatment, privacy, procedural_fairness, structural_disadvantage
  - Fairness: disparate_impact, embedding_bias, omission
  - Truth: hallucination, factual_accuracy, evidence_grounding, contradiction
  - Risk: physical_safety, psychological_harm, irreversible_harm, cascading_failure, operational_risk
  - Pragmatics: feasibility, resource_burden, time_constraints, environmental_impact, cascading_pragmatic_failure

- **Critics**: parallel execution of Rights, Autonomy, Fairness, Truth, Risk, Pragmatics. Each critic emits severity (0–3), violations, justification, evidence bundle, and flags. Critics may optionally incorporate detector signals from context for enhanced analysis via `context["detector_signals"]`.

- **Precedent**: optional retrieval via `PrecedentRetrievalV8` and alignment via `PrecedentAlignmentEngineV8` (conflict, drift, support strength). Novel cases return neutral alignment. 
  - **Stores**: Weaviate, pgvector, or in-memory with embedding registry for similarity
  - **Conflict Detection**: `PrecedentConflictV8().detect()` returns conflict status and reasons
  - **Drift Detection**: `PrecedentDriftV8().compute_drift()` returns drift score (0-1) and signal (stable/monitor/drift_warning)
  - **Versioning**: Optional precedent database versioning for reproducibility
  - **Caching**: Optional caching layer for frequently accessed precedents (cache_ttl configurable)

- **Uncertainty**: `UncertaintyEngineV8` computes epistemic/aleatoric uncertainty (critic divergence, precedent conflict, model stability) and escalation flags.

- **Aggregator**: lexicographic fusion with priority order `[rights, autonomy, fairness, truth, risk, pragmatics]`, applying precedent and uncertainty weighting. Outputs decision (`allow|constrained_allow|deny|escalate`), scores, and final output text.

- **Evidence**: `EvidenceRecorder` buffers JSONL-ready evidence per critic, including severity label, principle, justification, and detector metadata.
  - **Enhanced Packaging**: `EvidencePackageV8` provides structured bundle builder for oversight
  - **Storage Backends**: JSONL (default), PostgreSQL (`db_sink.py`), MongoDB, Elasticsearch
  - **OPA Integration**: Governance-ready payload format for external oversight tools
  - **Structure**: Includes timestamp, trace_id, input_snapshot, model_used, critic_outputs, uncertainty, precedent, governance_ready_payload

- **Governance**: OPA client wiring available through the engine builder; `opa_callback` can be injected or defaults to `OPAClientV8.evaluate`.

---

## Engine (async)

### Entrypoints
- **Primary**: `engine/engine.py` → `EleanorEngineV8.run` and `run_stream`
- **Streaming**: `run_stream` provides real-time feedback as pipeline stages complete
- **Builder**: `engine/core/__init__.py` → `build_eleanor_engine_v8` for API/websocket bootstraps

### Configuration
- **EngineConfig**: toggles precedent analysis, reflection (uncertainty), and evidence jsonl path
- **Router**: auto-discovery with default echo adapter for local use; supports injected adapters/policy
- **Forensic Mode**: detail_level 3 with timings, router diagnostics, uncertainty graph, and evidence references
- **Detector Integration**: Optional `enable_detectors=True` flag to activate pre-processing layer

### Streaming Support
```python
async for chunk in engine.run_stream(user_text):
    if chunk["stage"] == "router":
        # Router output chunk
    elif chunk["stage"] == "detector":
        # Detector result (name, signal)
    elif chunk["stage"] == "critic":
        # Critic evaluation
    elif chunk["stage"] == "complete":
        # Final decision
```

---

## Critics (V8)

### Core Critics
- **Rights**: discrimination, coercion, dignity attacks, privacy
  - Multi-strategy detection (regex + keywords)
  - Protected characteristic analysis (8 categories)
  - Domain-sensitive severity multipliers
  - UDHR Articles 1, 2, 7 alignment
  
- **Autonomy**: consent bypass, coercion, manipulation, surveillance pressure
  - Coercive language detection
  - Manipulation tactics identification
  - Consent validation
  
- **Fairness**: disparate impact/treatment patterns, protected class cues
  - Group-level outcome analysis
  - Individual fairness assessment
  - Procedural fairness evaluation
  - Domain multipliers (healthcare: 1.4x, lending: 1.4x, criminal_justice: 1.5x, employment: 1.3x)
  
- **Truth**: factual accuracy patterns, evidence grounding
  - Fabrication detection
  - Citation verification
  - Hedging analysis
  - Statistical claim validation
  
- **Risk**: safety domains, irreversibility, vulnerable populations
  - Physical safety assessment
  - Psychological harm detection
  - Irreversibility scoring
  - Vulnerable population multipliers
  
- **Pragmatics**: feasibility, resource burden, operational constraints
  - Timeline realism assessment
  - Resource requirement analysis
  - Technical feasibility evaluation
  - Complexity scoring

### Critic Enhancement Options
- **Domain Multipliers**: Sensitive domains (healthcare, lending, criminal justice) increase severity
- **Protected Group Analysis**: Automatic detection and sensitivity adjustment
- **Confidence Calibration**: Historical accuracy-based confidence scoring (optional)
- **Detector Integration**: Critics can incorporate detector signals for enhanced analysis

---

## Model Registry (Optional Cost Optimization)

### Overview
Centralized model configuration for per-critic model assignment and tier-based routing.

### Features
- **Per-Critic Assignment**: Different models for different critics (e.g., Opus for Rights, Haiku for Pragmatics)
- **Tier-Based Routing**: Premium/Standard/Economy tiers
- **Context-Aware Routing**: Automatic tier selection based on context
- **Cost Tracking**: Built-in cost monitoring and optimization
- **Hot-Reload**: YAML/JSON configuration with dynamic reloading
- **Metrics Callbacks**: Observability integration

### Configuration
```yaml
# model_registry.yaml
critics:
  rights:
    tier: premium
    model: claude-opus-4.5
  fairness:
    tier: standard
    model: claude-sonnet-4.5
  pragmatics:
    tier: economy
    model: claude-haiku-4.0

tiers:
  premium:
    models: [claude-opus-4.5]
    max_tokens: 4096
  standard:
    models: [claude-sonnet-4.5]
    max_tokens: 2048
  economy:
    models: [claude-haiku-4.0, gpt-4o-mini]
    max_tokens: 1024
```

### Usage
```python
from engine.models.registry import ModelRegistry

registry = ModelRegistry()
registry.assign_tier("rights", ModelTier.PREMIUM)
registry.assign_tier("pragmatics", ModelTier.ECONOMY)

# Automatic cost tracking
cost = registry.get_total_cost()
```

---

## Resilience Infrastructure (Optional Production Features)

### Circuit Breaker Pattern
Prevents cascade failures during API outages with automatic recovery.

**Implementation**: `engine/utils/circuit_breaker.py`

**States**: CLOSED (normal) → OPEN (failing, reject calls) → HALF_OPEN (testing recovery)

**Configuration**:
```python
CircuitBreaker(
    failure_threshold=5,      # Failures before opening
    recovery_timeout=30,      # Seconds before testing recovery
    success_threshold=2       # Successes to close circuit
)
```

### Retry with Exponential Backoff
Handles transient failures and rate limits.

**Implementation**: `engine/utils/retry.py`

**Features**:
- Configurable retry logic
- Exponential backoff with jitter
- Retryable/non-retryable exception filtering
- Metrics collection

**Usage**:
```python
@retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=30.0)
async def call_llm_api():
    # API call
```

### Health Checks
Enhanced health check endpoint with dependency status.

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy|degraded",
  "checks": {
    "api": "ok",
    "opa": "ok|unavailable",
    "precedent_store": "ok|unavailable",
    "llm_adapters": "ok|degraded"
  },
  "version": "8.0.0",
  "timestamp": "2025-12-11T..."
}
```

---

## Precedent & Uncertainty

### Precedent System
- **Alignment**: Tolerates missing embeddings and returns novel-case bundle when no precedents exist
- **Conflict Detection**: Identifies contradictions between new case and precedents
- **Drift Detection**: Monitors alignment score changes over time (stable/monitor/drift_warning)
- **Support Strength**: Quantifies how strongly precedents support the current decision
- **Versioning**: Optional database versioning for reproducibility (SHA256 hash of all cases)
- **Caching**: Optional LRU cache with configurable TTL (default: 3600s)

### Uncertainty Quantification
- **Epistemic Uncertainty**: Critic severity variance
- **Aleatoric Uncertainty**: Precedent conflict + model stability heuristics
- **Escalation Triggers**: uncertainty ≥ 0.6 with moderate average severity
- **Uncertainty Graph**: Available in forensic mode (detail_level 3)

---

## Security Hardening (Recommended for Production)

### Priority 1: Critical Vulnerabilities

#### SQL Injection Fix
```python
# engine/precedent/stores.py
from psycopg2 import sql

cur.execute(
    sql.SQL("""
        SELECT text, metadata
        FROM {}
        ORDER BY embedding <-> %s
        LIMIT %s;
    """).format(sql.Identifier(self.table)),
    (embedding, top_k)
)
```

#### CORS Configuration
```python
# api/rest/main.py
ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)
```

#### Authentication
```python
# api/middleware/auth.py
from fastapi import Depends, Security
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def verify_token(credentials = Security(security)):
    token = credentials.credentials
    payload = jwt.decode(token, os.getenv("JWT_SECRET"), algorithms=["HS256"])
    return payload

# Apply to endpoints
@app.post("/deliberate", dependencies=[Depends(verify_token)])
```

### Priority 2: Input Validation

```python
# api/schemas.py
from pydantic import BaseModel, Field, validator

class DeliberationRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=10000)
    context: dict = Field(default_factory=dict)
    
    @validator('input')
    def sanitize_input(cls, v):
        return v.strip()
```

### Priority 3: Rate Limiting

```python
# api/rest/main.py
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/deliberate")
@limiter.limit("10/minute")
async def deliberate_endpoint(request: Request, payload: DeliberationRequest):
    # ...
```

---

## Observability (Recommended for Production)

### Structured Logging
```python
# engine/logging_config.py
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
)

logger = structlog.get_logger()

# Usage
logger.info("deliberation.started", trace_id=trace_id, input_length=len(user_text))
logger.info("deliberation.complete", trace_id=trace_id, decision=decision)
```

### Prometheus Metrics
```python
# api/metrics.py
from prometheus_client import Counter, Histogram, Gauge

DELIBERATION_REQUESTS = Counter(
    'eleanor_deliberation_total',
    'Total deliberation requests',
    ['decision', 'model']
)

DELIBERATION_LATENCY = Histogram(
    'eleanor_deliberation_latency_seconds',
    'Deliberation latency',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

DETECTOR_SEVERITY = Gauge(
    'eleanor_detector_severity',
    'Current detector severity scores',
    ['detector_name']
)
```

### Distributed Tracing
```python
# Integration with OpenTelemetry
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

tracer = trace.get_tracer(__name__)
FastAPIInstrumentor.instrument_app(app)

async def deliberate(self, user_text: str):
    with tracer.start_as_current_span("deliberate") as span:
        span.set_attribute("input.length", len(user_text))
        # ... pipeline execution ...
```

---

## Testing Strategy

### Test Categories

#### Unit Tests
- Individual detector tests (25 detectors)
- Individual critic tests (6 critics)
- Aggregator logic tests
- Precedent alignment tests
- Uncertainty computation tests

#### Integration Tests
```python
@pytest.mark.asyncio
async def test_full_pipeline():
    engine = await build_eleanor_engine_v8()
    result = await engine.deliberate("Should we approve this loan?")
    
    assert "trace_id" in result
    assert "critics" in result
    assert "detector_signals" in result
    assert result["final_decision"] in ["allow", "constrained_allow", "deny", "escalate"]
```

#### Constitutional Invariant Tests
```python
@pytest.mark.asyncio
async def test_dignity_violation_always_blocks():
    """Constitutional invariant: dignity violations cannot be overridden."""
    critics = {
        "rights": {"severity": 3.0, "violations": ["dignity_attack"]},
        "fairness": {"severity": 0.0},
        # ... other critics with low severity
    }
    
    result = aggregator.aggregate(critics, precedent, uncertainty)
    
    # Even with perfect precedent alignment, dignity violation must deny
    assert result["decision"] == "deny"
```

#### Performance Tests
```python
@pytest.mark.asyncio
async def test_detector_suite_performance():
    engine = DetectorEngineV8()
    
    start = time.time()
    signals = await engine.detect_all("Test text", {})
    duration = time.time() - start
    
    assert duration < 2.0  # Full suite must complete in <2s
```

---

## Expected Decisions

### Decision Logic
- **Hard block**: rights/autonomy violations with severity ≥ 2.5
- **Escalate**: uncertainty ≥ 0.6 with moderate average severity (≥ 1.0)
- **Constrained allow**: average severity ≥ 1.0 without hard block
- **Allow**: otherwise, with precedent/uncertainty adjustments applied

### Constitutional Invariants
1. **Dignity Violations**: Always result in "deny", cannot be overridden
2. **Lexicographic Priority**: Rights violations cannot be overridden by lower-priority concerns
3. **High Uncertainty**: Must trigger escalation when combined with moderate severity
4. **Precedent Conflict**: Severe conflicts (score < -0.5) trigger escalation

---

## Deployment Configurations

### Development
```yaml
environment: development
enable_detectors: true
enable_precedent: false  # Use in-memory
enable_opa: false
log_level: DEBUG
```

### Staging
```yaml
environment: staging
enable_detectors: true
enable_precedent: true
precedent_store: pgvector
enable_opa: true
opa_url: http://opa:8181
log_level: INFO
enable_metrics: true
```

### Production
```yaml
environment: production
enable_detectors: true
enable_precedent: true
precedent_store: weaviate
enable_opa: true
opa_url: http://opa:8181
log_level: WARNING
enable_metrics: true
enable_tracing: true
circuit_breaker:
  enabled: true
  failure_threshold: 5
  recovery_timeout: 30
rate_limiting:
  enabled: true
  requests_per_minute: 100
```

---

## API Endpoints

### POST /deliberate
**Request**:
```json
{
  "input": "Should we approve this loan application?",
  "context": {
    "domain": "lending",
    "enable_detectors": true,
    "detail_level": 2
  }
}
```

**Response**:
```json
{
  "trace_id": "uuid-...",
  "timestamp": 1702339200.0,
  "model_used": "claude-sonnet-4.5",
  "detector_signals": {
    "discrimination": {"severity": 0.3, "violations": [...], ...},
    "disparate_impact": {"severity": 0.7, "violations": [...], ...}
  },
  "critics": {
    "rights": {"severity": 1.2, "violations": [...], ...},
    "fairness": {"severity": 2.1, "violations": [...], ...}
  },
  "precedent_alignment": {
    "alignment_score": 0.7,
    "conflict_detected": false,
    "drift_score": 0.1
  },
  "uncertainty": {
    "overall_uncertainty": 0.3,
    "epistemic": 0.2,
    "aleatoric": 0.1
  },
  "aggregator_output": {
    "decision": "constrained_allow",
    "score": {"average_severity": 1.1, "total_severity": 6.6}
  },
  "opa_governance": {
    "allow": true,
    "escalate": false
  },
  "final_decision": "constrained_allow"
}
```

### GET /health
See Resilience Infrastructure section for response format.

### POST /deliberate/stream
Streaming variant that yields JSON chunks as pipeline stages complete.

---

## Future Enhancements (Optional)

### Enhanced Detector Intelligence
- Context-aware domain multipliers
- Temporal impossibility detection
- Citation verification API integration
- Statistical claim validation

### Advanced Precedent Features
- Semantic clustering of precedents
- Automatic precedent generation from decisions
- Precedent explanation generation
- Multi-level precedent hierarchy

### Calibration & Learning
- Confidence calibration based on historical accuracy
- Active learning for detector patterns
- Feedback loop for critic improvement
- A/B testing framework for model selection

### Multi-Model Orchestration
- Ensemble voting across multiple models
- Automatic model selection based on task
- Cost-performance optimization
- Fallback chains for reliability

---

## Version History

### 8.0.0 (Current)
- Production-ready detector system (25 detectors)
- Enhanced evidence packaging
- Model registry for cost optimization
- Resilience infrastructure (circuit breaker, retry)
- Improved precedent system (conflict/drift detection)
- Security hardening recommendations
- Observability framework

### Future Versions
- 8.1.0: Streaming support, full pipeline integration tests
- 8.2.0: Enhanced detector intelligence, confidence calibration
- 8.3.0: Advanced observability, shadow mode deployment
- 9.0.0: Multi-model orchestration, learning framework

---

## References

### Constitutional Frameworks
- Universal Declaration of Human Rights (UDHR)
- UNESCO Recommendation on AI Ethics
- EU AI Act
- NIST AI Risk Management Framework

### Technical Standards
- ISO/IEC 23894:2023 (AI Risk Management)
- ISO/IEC 42001 (AI Management System)
- IEEE 7000-2021 (Systems Design for Ethical Values)

---

**Document Version**: 1.1  
**Specification Status**: Production-Ready with Recommended Enhancements  
**Last Review**: December 11, 2025
