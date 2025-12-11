"""
Comprehensive tests for ELEANOR V8 Infrastructure Components.

Tests cover:
- Precedent Store implementations
- Circuit Breaker
- Router with circuit breaker integration
"""

import pytest
import asyncio
import time
from typing import Dict, Any


# ============================================================
# Precedent Store Tests
# ============================================================

class TestInMemoryStore:

    @pytest.fixture
    def store(self):
        from engine.precedent.store import InMemoryStore
        return InMemoryStore()

    @pytest.fixture
    def sample_case(self):
        from engine.precedent.store import PrecedentCase
        return PrecedentCase(
            case_id="test-001",
            query_text="Should we allow this action?",
            decision="allow",
            values=["fairness", "autonomy"],
            aggregate_score=0.85,
            critic_outputs={"rights": {"score": 0.9}},
            rationale="Action aligns with constitutional values.",
        )

    @pytest.fixture
    def sample_embedding(self):
        return [0.1] * 1536  # Standard embedding dimension

    def test_add_and_get(self, store, sample_case, sample_embedding):
        """Test adding and retrieving a case."""
        case_id = store.add(sample_case, sample_embedding)

        assert case_id == "test-001"

        retrieved = store.get(case_id)
        assert retrieved is not None
        assert retrieved["case_id"] == "test-001"
        assert retrieved["decision"] == "allow"

    def test_search_by_embedding(self, store, sample_case, sample_embedding):
        """Test searching by embedding similarity."""
        store.add(sample_case, sample_embedding)

        # Search with similar embedding
        query_embedding = [0.1] * 1536
        results = store.search("test query", top_k=5, embedding=query_embedding)

        assert len(results) >= 1
        assert results[0]["case_id"] == "test-001"
        assert "similarity_score" in results[0]

    def test_delete(self, store, sample_case, sample_embedding):
        """Test deleting a case."""
        case_id = store.add(sample_case, sample_embedding)

        assert store.count() == 1

        deleted = store.delete(case_id)
        assert deleted is True
        assert store.count() == 0
        assert store.get(case_id) is None

    def test_count(self, store, sample_case, sample_embedding):
        """Test counting cases."""
        assert store.count() == 0

        store.add(sample_case, sample_embedding)
        assert store.count() == 1

    def test_search_returns_top_k(self, store, sample_embedding):
        """Test that search respects top_k parameter."""
        from engine.precedent.store import PrecedentCase

        # Add multiple cases
        for i in range(10):
            case = PrecedentCase(
                case_id=f"test-{i:03d}",
                query_text=f"Query {i}",
                decision="allow",
            )
            store.add(case, sample_embedding)

        results = store.search("query", top_k=3, embedding=sample_embedding)
        assert len(results) == 3


class TestJSONFileStore:

    @pytest.fixture
    def store(self, tmp_path):
        from engine.precedent.store import JSONFileStore
        file_path = tmp_path / "test_precedents.json"
        return JSONFileStore(file_path=str(file_path))

    @pytest.fixture
    def sample_case(self):
        from engine.precedent.store import PrecedentCase
        return PrecedentCase(
            case_id="json-test-001",
            query_text="Test query for JSON store",
            decision="deny",
            values=["safety"],
            aggregate_score=0.6,
        )

    @pytest.fixture
    def sample_embedding(self):
        return [0.5] * 1536

    def test_add_and_persist(self, store, sample_case, sample_embedding):
        """Test that cases are persisted to file."""
        case_id = store.add(sample_case, sample_embedding)

        assert store.count() == 1

        retrieved = store.get(case_id)
        assert retrieved is not None

    def test_search_without_embedding(self, store, sample_case, sample_embedding):
        """Test search falls back to recent cases without embedding."""
        store.add(sample_case, sample_embedding)

        results = store.search("query", top_k=5)
        assert len(results) >= 1


class TestPrecedentCaseDataclass:

    def test_to_dict(self):
        """Test case serialization."""
        from engine.precedent.store import PrecedentCase

        case = PrecedentCase(
            case_id="dict-test",
            query_text="Test",
            decision="allow",
            values=["fairness"],
            embedding=[0.1, 0.2, 0.3],
        )

        d = case.to_dict()
        assert "case_id" in d
        assert "embedding" not in d  # Should be excluded

    def test_from_dict(self):
        """Test case deserialization."""
        from engine.precedent.store import PrecedentCase

        data = {
            "case_id": "from-dict-test",
            "query_text": "Test",
            "decision": "deny",
        }

        case = PrecedentCase.from_dict(data)
        assert case.case_id == "from-dict-test"
        assert case.decision == "deny"


# ============================================================
# Circuit Breaker Tests
# ============================================================

class TestCircuitBreaker:

    @pytest.fixture
    def breaker(self):
        from engine.utils.circuit_breaker import CircuitBreaker
        return CircuitBreaker(
            name="test_breaker",
            failure_threshold=3,
            success_threshold=2,
            recovery_timeout=1.0  # Short timeout for testing
        )

    def test_initial_state_closed(self, breaker):
        """Test that circuit starts in closed state."""
        assert breaker.state.value == "closed"

    def test_success_keeps_closed(self, breaker):
        """Test that successful calls keep circuit closed."""
        def success_fn():
            return "success"

        result = breaker.call_sync(success_fn)
        assert result == "success"
        assert breaker.state.value == "closed"

    def test_failures_open_circuit(self, breaker):
        """Test that repeated failures open the circuit."""
        def failure_fn():
            raise Exception("Simulated failure")

        for _ in range(3):
            with pytest.raises(Exception):
                breaker.call_sync(failure_fn)

        assert breaker.state.value == "open"

    def test_open_circuit_rejects_calls(self, breaker):
        """Test that open circuit rejects calls."""
        from engine.utils.circuit_breaker import CircuitBreakerOpen

        def failure_fn():
            raise Exception("Simulated failure")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                breaker.call_sync(failure_fn)

        # Next call should be rejected
        with pytest.raises(CircuitBreakerOpen):
            breaker.call_sync(lambda: "success")

    def test_half_open_after_timeout(self, breaker):
        """Test that circuit transitions to half-open after timeout."""
        def failure_fn():
            raise Exception("Simulated failure")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                breaker.call_sync(failure_fn)

        assert breaker.state.value == "open"

        # Wait for timeout
        time.sleep(1.1)

        # Check state (should transition to half-open on check)
        _ = breaker.state
        assert breaker.state.value == "half_open"

    def test_success_in_half_open_closes(self, breaker):
        """Test that successes in half-open state close the circuit."""
        def failure_fn():
            raise Exception("Simulated failure")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                breaker.call_sync(failure_fn)

        # Wait for timeout
        time.sleep(1.1)

        # Successful calls should close the circuit
        for _ in range(2):  # success_threshold = 2
            result = breaker.call_sync(lambda: "success")
            assert result == "success"

        assert breaker.state.value == "closed"

    def test_reset(self, breaker):
        """Test manual reset."""
        def failure_fn():
            raise Exception("Simulated failure")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(Exception):
                breaker.call_sync(failure_fn)

        assert breaker.state.value == "open"

        # Reset
        breaker.reset()
        assert breaker.state.value == "closed"

    def test_metrics_tracking(self, breaker):
        """Test that metrics are tracked correctly."""
        breaker.call_sync(lambda: "success")
        breaker.call_sync(lambda: "success")

        status = breaker.get_status()
        assert status["metrics"]["total_calls"] == 2
        assert status["metrics"]["successful_calls"] == 2


class TestCircuitBreakerRegistry:

    def test_get_or_create(self):
        """Test registry get_or_create functionality."""
        from engine.utils.circuit_breaker import CircuitBreakerRegistry

        registry = CircuitBreakerRegistry()

        breaker1 = registry.get_or_create("test1")
        breaker2 = registry.get_or_create("test1")  # Same name

        assert breaker1 is breaker2

    def test_get_all_status(self):
        """Test getting status of all breakers."""
        from engine.utils.circuit_breaker import CircuitBreakerRegistry

        registry = CircuitBreakerRegistry()
        registry.get_or_create("breaker1")
        registry.get_or_create("breaker2")

        statuses = registry.get_all_status()
        assert "breaker1" in statuses
        assert "breaker2" in statuses


# ============================================================
# Router with Circuit Breaker Tests
# ============================================================

class TestRouterWithCircuitBreaker:

    @pytest.fixture
    def adapters(self):
        def success_adapter(text):
            return {"response": f"Processed: {text}"}

        def failure_adapter(text):
            raise Exception("Adapter failure")

        return {
            "primary": success_adapter,
            "backup": success_adapter,
            "failing": failure_adapter,
        }

    @pytest.fixture
    def router(self, adapters):
        from engine.router.router import RouterV8

        policy = {
            "primary": "primary",
            "fallback_order": ["backup"],
            "max_retries": 2,
            "circuit_breaker": {
                "enabled": True,
                "failure_threshold": 3,
                "recovery_timeout": 30.0
            }
        }

        return RouterV8(adapters=adapters, routing_policy=policy)

    def test_successful_routing(self, router):
        """Test successful request routing."""
        result = router.route("test input")

        assert result["success"] is True or result.get("model_output") is not None
        assert result["used_adapter"] == "primary"

    def test_circuit_breaker_status(self, router):
        """Test circuit breaker status retrieval."""
        status = router.get_circuit_breaker_status()

        assert isinstance(status, dict)

    def test_fallback_on_failure(self):
        """Test fallback to backup adapter on failure."""
        from engine.router.router import RouterV8

        call_count = {"primary": 0, "backup": 0}

        def failing_primary(text):
            call_count["primary"] += 1
            raise Exception("Primary failed")

        def success_backup(text):
            call_count["backup"] += 1
            return {"response": "backup response"}

        adapters = {
            "primary": failing_primary,
            "backup": success_backup,
        }

        policy = {
            "primary": "primary",
            "fallback_order": ["backup"],
            "max_retries": 1,
            "circuit_breaker": {"enabled": False}  # Disable for simpler test
        }

        router = RouterV8(adapters=adapters, routing_policy=policy)
        result = router.route("test")

        assert result.get("model_output") is not None or result["used_adapter"] == "backup"


# ============================================================
# Precedent Store Factory Tests
# ============================================================

class TestPrecedentStoreFactory:

    def test_create_memory_store(self):
        """Test creating in-memory store."""
        from engine.precedent.store import create_store

        store = create_store("memory")
        assert store is not None
        assert store.count() == 0

    def test_create_json_store(self, tmp_path):
        """Test creating JSON file store."""
        from engine.precedent.store import create_store

        file_path = tmp_path / "test.json"
        store = create_store("json", file_path=str(file_path))
        assert store is not None

    def test_invalid_backend_raises(self):
        """Test that invalid backend raises ValueError."""
        from engine.precedent.store import create_store

        with pytest.raises(ValueError):
            create_store("invalid_backend")
