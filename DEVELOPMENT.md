# ELEANOR V8 Development Guide

## Constitutional Principles First

Before contributing code, understand ELEANOR's core mission:

**ELEANOR interprets, evaluates, and protects constitutional governance in AI systems.**

This means:
- **Critics are epistemically isolated** — they evaluate independently without seeing peer outputs
- **Dissent is preserved verbatim** — minority opinions cannot be averaged away
- **Escalation is unilateral** — any critic can gate execution; no veto power
- **Uncertainty is signal, not error** — epistemic limits are governance outputs
- **Evidence is immutable** — audit trails cannot be modified

### Required Reading

1. [`docs/CRITIC_INDEPENDENCE_AND_ESCALATION.md`](docs/CRITIC_INDEPENDENCE_AND_ESCALATION.md)
2. [`docs/ESCALATION_Tiers_Human_Review_Doctrine.md`](docs/ESCALATION_Tiers_Human_Review_Doctrine.md)
3. [`POLICY_CONSTITUTIONAL_GOVERNANCE.md`](POLICY_CONSTITUTIONAL_GOVERNANCE.md)

**Constitutional violations are caught in CI and will block merges.**

---

## Development Environment Setup

### Prerequisites

- Python 3.9+ (3.11+ recommended)
- Git
- Docker (optional, for containerized development)

### Local Setup

```bash
# Clone repository
git clone https://github.com/eleanor-project/V8.git
cd V8

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (including dev tools)
pip install -e ".[dev]"

# Verify installation
python -c "from engine.engine import EleanorEngineV8; print('✓ ELEANOR V8 installed')"
```

### Configuration

Copy environment template:

```bash
cp .env.sample .env
```

Edit `.env` with your configuration:

```env
# LLM Provider Configuration
LLM_PROVIDER=ollama  # or openai, anthropic
OLLAMA_BASE_URL=http://localhost:11434

# Precedent Database (optional)
PRECEDENT_BACKEND=weaviate  # or none
WEAVIATE_URL=http://localhost:8080

# Evidence Recording
EVIDENCE_JSONL_PATH=evidence.jsonl

# Governance
ENFORCE_HUMAN_REVIEW=true  # Never disable in production
```

---

## Architecture Overview

### Core Components

```
engine/
├── engine.py              # Main orchestrator
├── exceptions.py          # Constitutional exception hierarchy
├── types.py               # Type definitions preserving semantics
├── validation.py          # Input validation and sanitization
├── critics/               # Independent constitutional critics
├── aggregator/            # Synthesis preserving dissent
├── escalation/            # Human review gating
├── precedent/             # Case law alignment
├── uncertainty/           # Epistemic quantification
├── execution/             # Execution gating
└── recorder/              # Immutable evidence trail
```

### Data Flow

```
Input (validated)
  ↓
Router (model selection)
  ↓
Critics (parallel, isolated)
  ↓
Aggregator (preserve dissent)
  ↓
Precedent Alignment
  ↓
Uncertainty Quantification
  ↓
Escalation Check (gates execution)
  ↓
Output (with governance metadata)
```

**Key Invariant**: Critics see only model output and input text, never peer evaluations.

---

## Testing

### Running Tests

```bash
# All tests
pytest

# Constitutional guarantees only
pytest tests/test_constitutional_guarantees.py -v

# With coverage
pytest --cov=engine --cov-report=term-missing

# Specific test
pytest tests/test_constitutional_guarantees.py::TestCriticEpistemicIsolation -v
```

### Writing Tests

**Constitutional tests are property tests, not unit tests.**

Good test (verifies invariant):
```python
async def test_critic_does_not_see_peer_outputs():
    """Critics SHALL NOT see peer outputs during evaluation."""
    # Test implementation
    assert no_cross_critic_visibility
```

Bad test (tests implementation detail):
```python
def test_critic_uses_specific_data_structure():
    """Critic stores violations in a list."""
    # This is implementation, not constitutional requirement
```

### Test Requirements

- Constitutional guarantee tests MUST pass (blocking)
- Coverage must be ≥ 80%
- Property-based tests for aggregation logic
- Integration tests for full pipeline

---

## Code Quality Standards

### Type Safety

Use explicit types, avoid `Any`:

```python
# ✗ Bad
def process(data: Any) -> Any:
    return data

# ✓ Good
def process(evaluation: CriticEvaluation) -> AggregatedResult:
    return aggregate(evaluation)
```

### Error Handling

Distinguish signals from errors:

```python
from engine.exceptions import (
    EscalationRequired,  # Signal, not error
    CriticEvaluationError,  # True error
    is_constitutional_signal,
)

try:
    result = await critic.evaluate(...)
except EscalationRequired as escalation:
    # This is GOOD - system working as designed
    handle_escalation(escalation)
except CriticEvaluationError as error:
    # This is FAILURE - critic couldn't function
    log_error_and_retry(error)
```

### Documentation

Document constitutional semantics:

```python
def aggregate_critics(
    evaluations: Dict[str, CriticEvaluation]
) -> AggregatedResult:
    """
    Synthesize critic evaluations preserving dissent.
    
    Constitutional Guarantees:
    - Minority opinions are preserved verbatim
    - No critic output is suppressed to achieve consensus
    - Escalation signals are binding (cannot be vetoed)
    
    Args:
        evaluations: Sealed critic evaluations (immutable)
    
    Returns:
        Aggregated result with dissent records
    """
```

---

## CI/CD Pipeline

### Checks (Automatic)

1. **Constitutional Invariants** (blocking)
   - Epistemic isolation preserved
   - Dissent preservation verified
   - Escalation authority maintained
   - Uncertainty as signal confirmed

2. **Security Scanning** (non-blocking)
   - Bandit (code security)
   - Safety (dependency vulnerabilities)

3. **Code Quality** (non-blocking)
   - Ruff linting
   - MyPy type checking
   - Code formatting

4. **Tests** (blocking)
   - Full test suite
   - Coverage ≥ 80%
   - Multiple Python versions (3.9-3.12)

### Local Pre-Commit

```bash
# Run full CI locally before pushing
./scripts/pre-commit-check.sh
```

Or manually:

```bash
# Format code
ruff format engine/ api/ governance/ tests/

# Lint
ruff check engine/ api/ governance/ tests/

# Type check critical paths
mypy engine/exceptions.py engine/types.py --strict

# Run tests
pytest --cov=engine --cov-fail-under=80

# Security scan
bandit -r engine/ api/ governance/
```

---

## Contributing Workflow

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Follow constitutional principles
- Add tests for new functionality
- Update documentation
- Maintain type safety

### 3. Test Locally

```bash
pytest --cov=engine --cov-fail-under=80
mypy engine/ --ignore-missing-imports
ruff check engine/
```

### 4. Commit with Conventional Commits

```bash
git commit -m "feat: Add uncertainty boundary detection"
git commit -m "fix: Preserve dissent in edge case"
git commit -m "docs: Update escalation tier documentation"
```

Types:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Test updates
- `refactor:` Code refactoring
- `chore:` Maintenance

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Create PR on GitHub with:
- Clear description of changes
- Reference to related issues
- Constitutional justification (if governance-related)
- Test coverage summary

### 6. Address Review Comments

CI will automatically run. Constitutional violations will block merge.

---

## Common Development Tasks

### Adding a New Critic

1. Create critic class in `engine/critics/`:

```python
from engine.critics.base import BaseCritic
from engine.types import CriticEvaluation, EscalationClause

class MyCriticV8(BaseCritic):
    async def evaluate(self, model_adapter, input_text, context):
        # Critic logic (isolated from peers)
        violations = self._detect_violations(input_text)
        
        # Check escalation clauses
        if self._requires_escalation(violations):
            escalation = self._create_escalation_signal(...)
            return {
                "severity": 0.9,
                "violations": violations,
                "escalation": escalation,
            }
        
        return {
            "severity": self._calculate_severity(violations),
            "violations": violations,
        }
```

2. Add tests in `tests/test_critics/`
3. Register in engine configuration
4. Document escalation clauses in governance docs

### Modifying Aggregation Logic

**CRITICAL**: Aggregation changes affect dissent preservation.

1. Verify constitutional guarantee tests still pass
2. Add property-based tests for edge cases
3. Document changes in constitutional terms
4. Get review from governance-focused maintainer

### Adding Evidence Fields

Evidence records are immutable (Pydantic `frozen=True`).

To add fields:
1. Update `EvidenceRecord` in `engine/types.py`
2. Version the change (field should be Optional for backward compat)
3. Update recorder to populate new field
4. Add migration for existing records (if needed)

---

## Debugging

### Enable Detailed Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from engine.engine import create_engine
engine = create_engine()
result = await engine.run(
    "test input",
    detail_level=3,  # Forensic detail
)
```

### Inspect Evidence Trail

```python
# After evaluation
evidence = engine.recorder.buffer
for record in evidence:
    print(f"{record.record_type}: {record.critic} - {record.severity}")
```

### Trace Escalation

```python
if result.human_review_required:
    for escalation in result.escalations:
        print(f"Escalation: {escalation.clause.clause_id}")
        print(f"Critic: {escalation.clause.critic}")
        print(f"Tier: {escalation.clause.tier}")
        print(f"Rationale: {escalation.clause.rationale}")
```

---

## Production Deployment

### Environment Variables (Production)

```env
# CRITICAL: Never disable human review in production
ENFORCE_HUMAN_REVIEW=true

# Evidence recording (required for compliance)
EVIDENCE_STORAGE=postgresql  # or secure alternative
EVIDENCE_DB_URL=postgresql://...

# Monitoring
ENABLE_TELEMETRY=true
OTEL_EXPORTER_ENDPOINT=https://telemetry.example.com

# Security
API_AUTHENTICATION=jwt
JWT_SECRET_KEY=<secure-secret>
```

### Health Checks

```bash
curl http://localhost:8000/health
```

Should return:
```json
{
  "status": "healthy",
  "constitutional_guarantees": "enforced",
  "human_review": "enabled",
  "evidence_recording": "operational"
}
```

---

## Getting Help

- **Constitutional questions**: See governance docs or create discussion
- **Technical issues**: Create issue with reproduction steps
- **Security concerns**: See SECURITY.md for responsible disclosure

---

## License

Apache 2.0 - See LICENSE file
