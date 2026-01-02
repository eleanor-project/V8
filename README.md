# ELEANOR V8 Engine Repository

ELEANOR V8 enforces constitutional governance at runtime: critic-declared escalation gates execution, `enforce_human_review` must approve every decision, and `execute_decision` refuses anything without an audit trail. CI (`.github/workflows/constitutional-ci.yml`) blocks merges if these guardrails drift.

## Production Roadmap

**🚧 Status**: Core engine complete, entering production hardening phase  
**🎯 Target**: Mid-February 2026 (6-8 weeks)

See **[PRODUCTION_ROADMAP.md](./PRODUCTION_ROADMAP.md)** for the complete path to production deployment, including:
- Security hardening (input validation, secrets management)
- Reliability improvements (resource management, configuration)
- Code quality (type safety, comprehensive testing, CI/CD)
- Performance optimization (GPU acceleration - optional)
- Parallel development strategy and timeline
- Risk assessment and acceptance criteria

**Quick Links**:
- [Critical Path to Production](./PRODUCTION_ROADMAP.md#critical-path-to-production) - Must-have items
- [Parallel Development Strategy](./PRODUCTION_ROADMAP.md#parallel-development-strategy) - Team coordination
- [Weekly Progress Tracking](./PRODUCTION_ROADMAP.md#weekly-progress-tracking) - Sprint planning

## Governance and Escalation References

- `docs/CRITIC_INDEPENDENCE_AND_ESCALATION.md` — critic isolation, dissent preservation, critic-initiated escalation.
- `docs/ESCALATION_Tiers_Human_Review_Doctrine.md` — canonical escalation tiers, cross-critic clause matrix, human review duties.
- `POLICY_CONSTITUTIONAL_GOVERNANCE.md` — CI invariants and why governance failures are treated as merge blockers.

## Recent Enhancements

### Performance & Observability (PR #23)
- ✅ Multi-level caching (L1 memory + L2 Redis) with adaptive concurrency
- ✅ Structured logging with OpenTelemetry distributed tracing
- ✅ Circuit breakers and graceful degradation for resilience
- ✅ Component health monitoring and metrics

See: [docs/CACHING.md](./docs/CACHING.md), [docs/OBSERVABILITY.md](./docs/OBSERVABILITY.md), [docs/RESILIENCE.md](./docs/RESILIENCE.md)

## Architecture

ELEANOR V8 implements a constitutional AI governance framework with:

- **Multi-Critic System**: Rights, Risk, Fairness, Truth, Autonomy, Pragmatics critics
- **Precedent Engine**: Case-based reasoning with embedding similarity
- **Uncertainty Quantification**: Epistemic and aleatoric uncertainty modeling
- **Governance Layer**: Human review triggers and escalation protocols
- **Evidence Recording**: Complete audit trails for all decisions
- **Streaming & Batch APIs**: Flexible execution modes

## Project Structure

```
V8/
├── engine/              # Core engine implementation
│   ├── cache/          # Multi-level caching (PR #23)
│   ├── observability/  # Logging and tracing (PR #23)
│   ├── resilience/     # Circuit breakers (PR #23)
│   ├── critics/        # Constitutional critics
│   ├── precedent/      # Precedent retrieval and alignment
│   ├── uncertainty/    # Uncertainty quantification
│   └── engine.py       # Main engine orchestration
├── governance/         # Review triggers and escalation
├── config/             # Configuration files
├── docs/               # Documentation
├── examples/           # Usage examples
├── tests/              # Test suite
└── .github/workflows/  # CI/CD pipelines
```

## Getting Started

### Installation

```bash
# Clone repository
git clone https://github.com/eleanor-project/v8.git
cd v8

# Install dependencies
pip install -r requirements.txt

# Optional: Install observability stack
pip install -r requirements-observability.txt

# Optional: Install GPU support
pip install -r requirements-gpu.txt
```

### Quick Start

```python
from engine import EleanorEngineV8, EngineConfig

# Create engine with default configuration
config = EngineConfig(
    detail_level=2,
    enable_precedent_analysis=True,
    enable_reflection=True
)

engine = EleanorEngineV8(config=config)

# Run evaluation
result = await engine.run(
    text="Should we deploy this AI system in healthcare?",
    context={"domain": "healthcare"}
)

print(result.output_text)
print(result.critic_findings)
print(result.uncertainty)
```

### With Observability (PR #23)

```python
from engine import EleanorEngineV8
from engine.observability import setup_logging, setup_tracing
from engine.cache import CacheManager

# Setup observability
setup_logging(level="INFO", json_output=True)
setup_tracing(service_name="eleanor-v8", jaeger_endpoint="localhost:6831")

# Setup caching
cache_manager = CacheManager(redis_url="redis://localhost:6379")

# Create engine with caching
async with EleanorEngineV8(
    config=config,
    cache_manager=cache_manager
) as engine:
    result = await engine.run(text="...")
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=engine --cov-report=html

# Run specific test suite
pytest tests/unit/test_critics/
```

### Code Quality

```bash
# Linting
ruff check .

# Type checking
mypy engine/ --strict

# Security scanning
bandit -r engine/

# Format code
ruff format .
```

## Contributing

See [PRODUCTION_ROADMAP.md](./PRODUCTION_ROADMAP.md) for current priorities and how to contribute.

**Active Development Areas**:
- 🔴 Security: Input validation, secrets management (Issues #9, #13, #20)
- 🟡 Reliability: Resource management, configuration (Issues #19, #11)
- 🟢 Quality: Type safety, testing, CI/CD (Issues #5, #7, #8)
- 🟢 Performance: GPU acceleration (Issue #25 - optional)

### Development Workflow

1. Check [PRODUCTION_ROADMAP.md](./PRODUCTION_ROADMAP.md) for priorities
2. Pick an issue from the roadmap
3. Create feature branch: `git checkout -b feature/issue-number-description`
4. Implement with tests and documentation
5. Ensure all CI checks pass
6. Submit PR with reference to roadmap

## Documentation

### Core Documentation
- [Production Roadmap](./PRODUCTION_ROADMAP.md) - 🎯 **Start here for production plan**
- [Critic Independence and Escalation](./docs/CRITIC_INDEPENDENCE_AND_ESCALATION.md)
- [Escalation Tiers and Human Review](./docs/ESCALATION_Tiers_Human_Review_Doctrine.md)
- [Constitutional Governance Policy](./POLICY_CONSTITUTIONAL_GOVERNANCE.md)

### Feature Documentation (PR #23)
- [Caching Strategy](./docs/CACHING.md) - Multi-level caching and adaptive concurrency
- [Observability Guide](./docs/OBSERVABILITY.md) - Logging, tracing, and metrics
- [Resilience Patterns](./docs/RESILIENCE.md) - Circuit breakers and degradation

### Architecture
- [Engine Architecture](./docs/ARCHITECTURE.md) - Coming soon
- [API Reference](./docs/API.md) - Coming soon
- [Configuration Guide](./docs/CONFIGURATION.md) - Coming soon

## Monitoring & Operations

### Health Checks

```bash
# Check component health
curl http://localhost:8000/health

# Check detailed status
curl http://localhost:8000/health/detailed
```

### Metrics

With observability enabled (PR #23):
- Structured logs in JSON format
- Distributed traces in Jaeger
- Prometheus metrics endpoint: `/metrics`
- Grafana dashboards: `monitoring/dashboards/`

### Troubleshooting

See [PRODUCTION_ROADMAP.md - Risk Register](./PRODUCTION_ROADMAP.md#risk-register) for common issues and mitigations.

## License

MIT License - See [LICENSE](./LICENSE) for details

## Citation

```bibtex
@software{eleanor_v8,
  title = {ELEANOR V8: Constitutional AI Governance Engine},
  author = {ELEANOR Project},
  year = {2025-2026},
  url = {https://github.com/eleanor-project/v8}
}
```

## Contact

- **Issues**: [GitHub Issues](https://github.com/eleanor-project/v8/issues)
- **Discussions**: [GitHub Discussions](https://github.com/eleanor-project/v8/discussions)
- **Production Roadmap**: [PRODUCTION_ROADMAP.md](./PRODUCTION_ROADMAP.md)

---

**Status**: 🟡 Active Development | **Target**: Production-ready by Mid-February 2026
