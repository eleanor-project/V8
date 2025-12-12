# ELEANOR V8 — Installation Guide

## Prerequisites

### System Requirements
- **OS**: macOS 12+ (M1/M2/Intel supported)
- **Python**: 3.9 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: 2GB free space

### Required Software
1. **Python 3.9+**: Check with `python3 --version`
2. **pip**: Included with Python
3. **Git**: For cloning the repository

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/eleanor-project/eleanor-v8.git
cd eleanor-v8
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

### 3. Install Dependencies

```bash
# Install core dependencies
pip install -e .

# Install development dependencies (optional, for testing)
pip install -e ".[dev]"

# Install production dependencies (optional, for monitoring)
pip install -e ".[production]"
```

### 4. Configuration

#### 4.1 Environment Variables

Create `.env` file in project root:

```bash
# API Configuration
ELEANOR_HOST=127.0.0.1
ELEANOR_PORT=8000

# LLM Configuration (choose one or multiple)
OPENAI_API_KEY=your_key_here  # Optional
ANTHROPIC_API_KEY=your_key_here  # Optional

# Database (optional, uses in-memory by default)
# POSTGRES_URL=postgresql://user:pass@localhost:5432/eleanor
# WEAVIATE_URL=http://localhost:8080

# OPA (optional, runs without OPA if not configured)
# OPA_URL=http://localhost:8181

# CORS (for web UI)
CORS_ORIGINS=http://localhost:3000,http://localhost:8000

# Logging
LOG_LEVEL=INFO
```

#### 4.2 Constitutional Configuration

The constitutional configuration is already included in:
- `/governance/constitutional.yaml`

No changes needed unless customizing principles.

### 5. Verify Installation

```bash
# Run tests
pytest

# Start the API server
python -m api.rest.main

# In another terminal, test the health endpoint
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "8.0.0",
  "timestamp": "2025-12-11T..."
}
```

## Quick Start

### Running the API

```bash
# Activate virtual environment
source venv/bin/activate

# Start server
uvicorn api.rest.main:app --reload --host 0.0.0.0 --port 8000
```

Server will be available at: `http://localhost:8000`

### Using the CLI

```bash
# Activate virtual environment
source venv/bin/activate

# Run a deliberation
eleanor deliberate "Should we approve this loan application?"

# Run with streaming
eleanor deliberate --stream "Evaluate this decision"

# Run with detail level
eleanor deliberate --detail 3 "Complex ethical question here"
```

### Python API

```python
import asyncio
from engine.core import build_eleanor_engine_v8

async def main():
    # Build engine
    engine = await build_eleanor_engine_v8()

    # Run deliberation
    result = await engine.deliberate("Should we approve this loan application?")

    print(f"Decision: {result['final_decision']}")
    print(f"Trace ID: {result['trace_id']}")

asyncio.run(main())
```

## Detector System

ELEANOR V8 includes 25 production-ready detectors:

### Rights & Dignity
- **autonomy**: Detects coercion and consent violations
- **coercion**: Detects threatening or manipulative language
- **dehumanization**: Detects dehumanizing language
- **discrimination**: Detects discriminatory patterns

### Fairness
- **disparate_impact**: Detects unequal outcomes
- **disparate_treatment**: Detects differential treatment
- **procedural_fairness**: Detects process fairness issues
- **structural_disadvantage**: Detects systematic barriers
- **embedding_bias**: Detects latent biases

### Truth & Accuracy
- **hallucination**: Detects fabricated citations and false claims
- **factual_accuracy**: Detects factual inaccuracies
- **evidence_grounding**: Detects unsupported claims
- **contradiction**: Detects logical inconsistencies
- **omission**: Detects critical information gaps

### Safety & Risk
- **physical_safety**: Detects physical harm risks
- **psychological_harm**: Detects emotional abuse patterns
- **privacy**: Detects privacy violations
- **irreversible_harm**: Detects permanent consequences
- **cascading_failure**: Detects cascading risk patterns
- **operational_risk**: Detects system failure risks
- **environmental_impact**: Detects environmental concerns

### Pragmatics
- **feasibility**: Detects unrealistic proposals
- **resource_burden**: Detects excessive resource requirements
- **time_constraints**: Detects unrealistic time expectations
- **cascading_pragmatic_failure**: Detects implementation cascades

### Testing Detectors

```python
import asyncio
from engine.detectors.engine import DetectorEngineV8

async def test_detectors():
    engine = DetectorEngineV8()

    # Run all detectors
    text = "You must comply immediately without question."
    signals = await engine.detect_all(text, {})

    # Aggregate results
    aggregated = engine.aggregate_signals(signals)
    print(f"Total detectors: {aggregated['total_detectors']}")
    print(f"Max severity: {aggregated['max_severity']:.2f}")
    print(f"Critical: {aggregated['by_severity']['critical']}")

asyncio.run(test_detectors())
```

## Optional Components

### PostgreSQL (for precedent storage)

```bash
# Install PostgreSQL
brew install postgresql@15
brew services start postgresql@15

# Create database
createdb eleanor

# Install pgvector extension
psql eleanor -c "CREATE EXTENSION vector;"

# Update .env
echo "POSTGRES_URL=postgresql://$(whoami)@localhost:5432/eleanor" >> .env
```

### Weaviate (for vector search)

```bash
# Run Weaviate with Docker
docker run -d \
  -p 8080:8080 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH='/var/lib/weaviate' \
  semitechnologies/weaviate:latest

# Update .env
echo "WEAVIATE_URL=http://localhost:8080" >> .env
```

### OPA (for governance policies)

```bash
# Install OPA
brew install opa

# Run OPA server
opa run --server --addr localhost:8181 governance/

# Update .env
echo "OPA_URL=http://localhost:8181" >> .env
```

## Troubleshooting

### Import Errors

If you see import errors:
```bash
# Ensure you're in the virtual environment
source venv/bin/activate

# Reinstall in editable mode
pip install -e .
```

### Port Already in Use

If port 8000 is already in use:
```bash
# Use a different port
uvicorn api.rest.main:app --port 8001
```

### Database Connection Issues

If PostgreSQL connection fails:
```bash
# Check PostgreSQL is running
brew services list

# Check connection
psql -h localhost -U $(whoami) -d eleanor -c "SELECT 1;"
```

### Detector Loading Issues

If detectors fail to load:
```bash
# Test detector loading
python3 -c "
from engine.detectors.engine import DetectorEngineV8
engine = DetectorEngineV8()
print(f'Loaded {len(engine.detectors)} detectors')
"
```

Expected output: `Loaded 25 detectors`

## Performance Tuning

### Detector Timeout

Adjust detector timeout in code:
```python
from engine.detectors.engine import DetectorEngineV8

# Default is 2.0 seconds
engine = DetectorEngineV8(timeout_seconds=3.0)
```

### Parallel Execution

Detectors run in parallel by default. To adjust:
```python
# Detectors are orchestrated automatically
# Performance target: <2s for all 25 detectors
signals = await engine.detect_all(text, context)
```

## Next Steps

- Read the [Architecture Documentation](docs/ARCHITECTURE.md)
- Explore the [API Documentation](docs/API.md)
- Review [Governance Policies](governance/README.md)
- Run the [Example Notebooks](examples/)

## Support

For issues or questions:
- GitHub Issues: https://github.com/eleanor-project/eleanor-v8/issues
- Documentation: https://github.com/eleanor-project/eleanor-v8#readme
