"""
Comprehensive tests for Precedent modules

Tests coverage for:
- embeddings.py: Embedding adapters and backends
- retrieval.py: Precedent retrieval and alignment scoring
- stores.py: Weaviate and PGVector backends
- alignment.py: Alignment and drift detection
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json


# ============================================================
# Tests for embeddings.py
# ============================================================

class TestBaseEmbeddingAdapter:
    """Test BaseEmbeddingAdapter interface."""

    def test_base_adapter_not_implemented(self):
        """Test that base adapter raises NotImplementedError."""
        from engine.precedent.embeddings import BaseEmbeddingAdapter

        adapter = BaseEmbeddingAdapter()
        with pytest.raises(NotImplementedError):
            adapter.embed("test text")


class TestGPTEmbeddingAdapter:
    """Test GPTEmbeddingAdapter."""

    @patch('engine.precedent.embeddings.OpenAIClient')
    def test_gpt_adapter_initialization(self, mock_client_class):
        """Test GPT adapter initialization."""
        from engine.precedent.embeddings import GPTEmbeddingAdapter

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        adapter = GPTEmbeddingAdapter(model="text-embedding-3-large", api_key="test-key")

        assert adapter.model == "text-embedding-3-large"
        assert adapter.client == mock_client
        mock_client_class.assert_called_once_with(api_key="test-key")

    @patch('engine.precedent.embeddings.OpenAIClient')
    def test_gpt_adapter_embed(self, mock_client_class):
        """Test GPT adapter embed method."""
        from engine.precedent.embeddings import GPTEmbeddingAdapter

        # Mock the client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_data = Mock()
        mock_data.embedding = [0.1, 0.2, 0.3, 0.4]
        mock_response.data = [mock_data]
        mock_client.embeddings.create.return_value = mock_response
        mock_client_class.return_value = mock_client

        adapter = GPTEmbeddingAdapter()
        result = adapter.embed("test text")

        assert result == [0.1, 0.2, 0.3, 0.4]
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-large",
            input="test text"
        )


class TestClaudeEmbeddingAdapter:
    """Test ClaudeEmbeddingAdapter with Voyage AI."""

    @patch.dict('os.environ', {'VOYAGE_API_KEY': 'test-voyage-key'})
    @patch('engine.precedent.embeddings.requests.post')
    def test_claude_adapter_initialization_with_env_key(self, mock_post):
        """Test Claude adapter uses environment variable for API key."""
        from engine.precedent.embeddings import ClaudeEmbeddingAdapter

        adapter = ClaudeEmbeddingAdapter()
        assert adapter.api_key == 'test-voyage-key'
        assert adapter.model == 'voyage-large-2'

    @patch('engine.precedent.embeddings.requests.post')
    def test_claude_adapter_embed_voyage_success(self, mock_post):
        """Test Claude adapter embed with Voyage AI success."""
        from engine.precedent.embeddings import ClaudeEmbeddingAdapter

        # Mock successful Voyage API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"embedding": [0.5, 0.6, 0.7]}]
        }
        mock_post.return_value = mock_response

        adapter = ClaudeEmbeddingAdapter(api_key="test-key")
        result = adapter.embed("test text")

        assert result == [0.5, 0.6, 0.7]

    @patch('engine.precedent.embeddings.requests.post')
    def test_claude_adapter_fallback_to_local(self, mock_post):
        """Test Claude adapter falls back to local model on error."""
        from engine.precedent.embeddings import ClaudeEmbeddingAdapter

        # Mock Voyage API failure
        mock_post.side_effect = Exception("Voyage API unavailable")

        adapter = ClaudeEmbeddingAdapter(api_key="test-key", fallback_to_local=True)

        # This should use the local fallback model
        # The actual implementation may vary, but it should not raise an exception
        try:
            result = adapter.embed("test text")
            # If fallback works, result should be a list of floats
            assert isinstance(result, list)
        except Exception as e:
            # If sentence-transformers not installed, skip
            if "sentence-transformers" in str(e).lower():
                pytest.skip("sentence-transformers not installed")
            else:
                raise


# ============================================================
# Tests for retrieval.py
# ============================================================

class TestPrecedentRetrievalV8:
    """Test PrecedentRetrievalV8."""

    def test_initialization(self):
        """Test retrieval engine initialization."""
        from engine.precedent.retrieval import PrecedentRetrievalV8

        mock_store = Mock()
        retrieval = PrecedentRetrievalV8(mock_store)

        assert retrieval.store == mock_store

    def test_score_alignment_with_value_overlap(self):
        """Test alignment scoring with value overlap."""
        from engine.precedent.retrieval import PrecedentRetrievalV8

        mock_store = Mock()
        retrieval = PrecedentRetrievalV8(mock_store)

        case = {
            "values": ["privacy", "fairness"],
            "aggregate_score": 0.6
        }

        critic_outputs = [
            {"value": "privacy", "score": 0.5},
            {"value": "fairness", "score": 0.7}
        ]

        score = retrieval._score_alignment(case, critic_outputs)

        # Perfect value overlap (2/2) = 1.0
        # Score alignment: 1 - abs(0.6 - 0.6) = 1.0
        # Average: (1.0 + 1.0) / 2 = 1.0
        assert score == 1.0

    def test_score_alignment_with_partial_overlap(self):
        """Test alignment scoring with partial value overlap."""
        from engine.precedent.retrieval import PrecedentRetrievalV8

        mock_store = Mock()
        retrieval = PrecedentRetrievalV8(mock_store)

        case = {
            "values": ["privacy"],
            "aggregate_score": 0.5
        }

        critic_outputs = [
            {"value": "privacy", "score": 0.4},
            {"value": "fairness", "score": 0.6}
        ]

        score = retrieval._score_alignment(case, critic_outputs)

        # Value overlap: 1/2 = 0.5
        # Score alignment: 1 - abs(0.5 - 0.5) = 1.0
        # Average: (0.5 + 1.0) / 2 = 0.75
        assert score == 0.75

    def test_score_alignment_no_current_values(self):
        """Test alignment scoring when no current values."""
        from engine.precedent.retrieval import PrecedentRetrievalV8

        mock_store = Mock()
        retrieval = PrecedentRetrievalV8(mock_store)

        case = {
            "values": ["privacy"],
            "aggregate_score": 0.5
        }

        critic_outputs = [
            {"value": None, "score": 0.5}
        ]

        score = retrieval._score_alignment(case, critic_outputs)

        # No current values, so value_alignment = 0.0
        # Score alignment: 1 - abs(0.5 - 0.5) = 1.0
        # Average: (0.0 + 1.0) / 2 = 0.5
        assert score == 0.5

    def test_retrieve_no_results(self):
        """Test retrieval when no precedents found."""
        from engine.precedent.retrieval import PrecedentRetrievalV8

        mock_store = Mock()
        mock_store.search.return_value = []

        retrieval = PrecedentRetrievalV8(mock_store)

        result = retrieval.retrieve("test query", [], top_k=5)

        assert result["precedent_cases"] == []
        assert result["alignment_score"] == 1.0  # neutral when no precedent
        assert result["top_case"] is None

    def test_retrieve_with_results(self):
        """Test retrieval with precedent results."""
        from engine.precedent.retrieval import PrecedentRetrievalV8

        mock_store = Mock()
        mock_store.search.return_value = [
            {"values": ["privacy"], "aggregate_score": 0.5, "text": "Case 1"},
            {"values": ["fairness"], "aggregate_score": 0.7, "text": "Case 2"}
        ]

        retrieval = PrecedentRetrievalV8(mock_store)

        critic_outputs = [
            {"value": "privacy", "score": 0.5}
        ]

        result = retrieval.retrieve("test query", critic_outputs, top_k=2)

        assert len(result["precedent_cases"]) == 2
        assert result["top_case"] is not None
        assert result["alignment_score"] > 0.0
        # First case has better alignment (privacy value match)
        assert result["top_case"]["text"] == "Case 1"

    def test_retrieve_sorting_by_alignment(self):
        """Test that retrieval sorts cases by alignment score."""
        from engine.precedent.retrieval import PrecedentRetrievalV8

        mock_store = Mock()
        mock_store.search.return_value = [
            {"values": ["other"], "aggregate_score": 0.5, "id": "low"},
            {"values": ["privacy", "fairness"], "aggregate_score": 0.6, "id": "high"}
        ]

        retrieval = PrecedentRetrievalV8(mock_store)

        critic_outputs = [
            {"value": "privacy", "score": 0.6},
            {"value": "fairness", "score": 0.6}
        ]

        result = retrieval.retrieve("test query", critic_outputs, top_k=2)

        # Second case has perfect match, should be top
        assert result["top_case"]["id"] == "high"


# ============================================================
# Tests for stores.py
# ============================================================

class TestEmbedder:
    """Test Embedder utility class."""

    def test_embedder_initialization(self):
        """Test embedder initialization."""
        from engine.precedent.stores import Embedder

        embed_fn = lambda x: [0.1, 0.2, 0.3]
        embedder = Embedder(embed_fn)

        assert embedder.embed_fn == embed_fn

    def test_embedder_embed(self):
        """Test embedder embed method."""
        from engine.precedent.stores import Embedder

        embed_fn = lambda x: [0.1, 0.2, 0.3, len(x)]
        embedder = Embedder(embed_fn)

        result = embedder.embed("test")
        assert result == [0.1, 0.2, 0.3, 4]


class TestWeaviatePrecedentStore:
    """Test WeaviatePrecedentStore."""

    def test_initialization(self):
        """Test Weaviate store initialization."""
        from engine.precedent.stores import WeaviatePrecedentStore

        mock_client = Mock()
        embed_fn = lambda x: [0.1, 0.2]

        store = WeaviatePrecedentStore(mock_client, class_name="TestClass", embed_fn=embed_fn)

        assert store.client == mock_client
        assert store.class_name == "TestClass"
        assert store.embedder.embed_fn == embed_fn

    def test_search_success(self):
        """Test Weaviate search with results."""
        from engine.precedent.stores import WeaviatePrecedentStore

        mock_client = Mock()
        embed_fn = lambda x: [0.1, 0.2, 0.3]

        # Mock Weaviate query chain
        mock_query = Mock()
        mock_get = Mock()
        mock_near = Mock()
        mock_limit = Mock()

        mock_client.query = mock_query
        mock_query.get.return_value = mock_get
        mock_get.with_near_vector.return_value = mock_near
        mock_near.with_limit.return_value = mock_limit
        mock_limit.do.return_value = {
            "data": {
                "Get": {
                    "Precedent": [
                        {"text": "Case 1", "metadata": {"decision": "allow"}},
                        {"text": "Case 2", "metadata": {"decision": "deny"}}
                    ]
                }
            }
        }

        store = WeaviatePrecedentStore(mock_client, embed_fn=embed_fn)
        results = store.search("test query", top_k=2)

        assert len(results) == 2
        assert results[0]["text"] == "Case 1"
        assert results[0]["metadata"] == {"decision": "allow"}
        assert results[1]["text"] == "Case 2"

    def test_search_no_results(self):
        """Test Weaviate search with no results."""
        from engine.precedent.stores import WeaviatePrecedentStore

        mock_client = Mock()
        embed_fn = lambda x: [0.1, 0.2]

        # Mock empty response
        mock_query = Mock()
        mock_get = Mock()
        mock_near = Mock()
        mock_limit = Mock()

        mock_client.query = mock_query
        mock_query.get.return_value = mock_get
        mock_get.with_near_vector.return_value = mock_near
        mock_near.with_limit.return_value = mock_limit
        mock_limit.do.return_value = {}

        store = WeaviatePrecedentStore(mock_client, embed_fn=embed_fn)
        results = store.search("test query", top_k=5)

        assert results == []


class TestPGVectorPrecedentStore:
    """Test PGVectorPrecedentStore."""

    def test_initialization_valid_table_name(self):
        """Test PGVector store initialization with valid table name."""
        from engine.precedent.stores import PGVectorPrecedentStore

        with patch('engine.precedent.stores.psycopg2.connect') as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value = mock_conn
            embed_fn = lambda x: [0.1, 0.2]

            store = PGVectorPrecedentStore("dbname=test", table_name="precedent", embed_fn=embed_fn)

            assert store.table == "precedent"
            assert store.conn == mock_conn
            mock_connect.assert_called_once_with("dbname=test")

    def test_initialization_invalid_table_name(self):
        """Test PGVector store rejects invalid table names."""
        from engine.precedent.stores import PGVectorPrecedentStore

        with patch('engine.precedent.stores.psycopg2.connect'):
            embed_fn = lambda x: [0.1, 0.2]

            # SQL injection attempts
            with pytest.raises(ValueError, match="Invalid table name"):
                PGVectorPrecedentStore("dbname=test", table_name="precedent; DROP TABLE users;", embed_fn=embed_fn)

            with pytest.raises(ValueError, match="Invalid table name"):
                PGVectorPrecedentStore("dbname=test", table_name="123invalid", embed_fn=embed_fn)

            with pytest.raises(ValueError, match="Invalid table name"):
                PGVectorPrecedentStore("dbname=test", table_name="table-name", embed_fn=embed_fn)

    def test_search_success(self):
        """Test PGVector search with results."""
        from engine.precedent.stores import PGVectorPrecedentStore

        with patch('engine.precedent.stores.psycopg2.connect') as mock_connect:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_connect.return_value = mock_conn

            mock_cursor.fetchall.return_value = [
                ("Case 1 text", json.dumps({"decision": "allow"})),
                ("Case 2 text", json.dumps({"decision": "deny"}))
            ]

            embed_fn = lambda x: [0.1, 0.2, 0.3]
            store = PGVectorPrecedentStore("dbname=test", embed_fn=embed_fn)

            results = store.search("test query", top_k=2)

            assert len(results) == 2
            assert results[0]["text"] == "Case 1 text"
            assert results[0]["metadata"]["decision"] == "allow"
            assert results[1]["text"] == "Case 2 text"

    def test_search_empty_results(self):
        """Test PGVector search with no results."""
        from engine.precedent.stores import PGVectorPrecedentStore

        with patch('engine.precedent.stores.psycopg2.connect') as mock_connect:
            mock_conn = Mock()
            mock_cursor = Mock()
            mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
            mock_connect.return_value = mock_conn

            mock_cursor.fetchall.return_value = []

            embed_fn = lambda x: [0.1, 0.2]
            store = PGVectorPrecedentStore("dbname=test", embed_fn=embed_fn)

            results = store.search("test query", top_k=5)

            assert results == []

    def test_context_manager(self):
        """Test PGVector store as context manager."""
        from engine.precedent.stores import PGVectorPrecedentStore

        with patch('engine.precedent.stores.psycopg2.connect') as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value = mock_conn

            embed_fn = lambda x: [0.1]

            with PGVectorPrecedentStore("dbname=test", embed_fn=embed_fn) as store:
                assert store.conn == mock_conn

            # Should close connection on exit
            mock_conn.close.assert_called_once()

    def test_close_method(self):
        """Test PGVector store close method."""
        from engine.precedent.stores import PGVectorPrecedentStore

        with patch('engine.precedent.stores.psycopg2.connect') as mock_connect:
            mock_conn = Mock()
            mock_connect.return_value = mock_conn

            embed_fn = lambda x: [0.1]
            store = PGVectorPrecedentStore("dbname=test", embed_fn=embed_fn)
            store.close()

            mock_conn.close.assert_called_once()


# ============================================================
# Tests for alignment.py
# ============================================================

class TestCosineSimilarity:
    """Test cosine similarity utility function."""

    def test_cosine_identical_vectors(self):
        """Test cosine similarity with identical vectors."""
        from engine.precedent.alignment import cosine

        vec = [1.0, 2.0, 3.0]
        similarity = cosine(vec, vec)

        assert similarity == pytest.approx(1.0)

    def test_cosine_orthogonal_vectors(self):
        """Test cosine similarity with orthogonal vectors."""
        from engine.precedent.alignment import cosine

        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        similarity = cosine(vec1, vec2)

        assert similarity == pytest.approx(0.0)

    def test_cosine_opposite_vectors(self):
        """Test cosine similarity with opposite vectors."""
        from engine.precedent.alignment import cosine

        vec1 = [1.0, 2.0]
        vec2 = [-1.0, -2.0]
        similarity = cosine(vec1, vec2)

        assert similarity == pytest.approx(-1.0)

    def test_cosine_empty_vectors(self):
        """Test cosine similarity with empty vectors."""
        from engine.precedent.alignment import cosine

        assert cosine([], []) == 0.0
        assert cosine([1.0], []) == 0.0
        assert cosine([], [1.0]) == 0.0

    def test_cosine_zero_vectors(self):
        """Test cosine similarity with zero vectors."""
        from engine.precedent.alignment import cosine

        assert cosine([0.0, 0.0], [0.0, 0.0]) == 0.0
        assert cosine([1.0, 2.0], [0.0, 0.0]) == 0.0


class TestPrecedentAlignmentEngineV8:
    """Test PrecedentAlignmentEngineV8."""

    def test_initialization(self):
        """Test alignment engine initialization."""
        from engine.precedent.alignment import PrecedentAlignmentEngineV8

        engine = PrecedentAlignmentEngineV8()
        assert engine is not None

    def test_analyze_no_precedents(self):
        """Test analyze with no precedent cases."""
        from engine.precedent.alignment import PrecedentAlignmentEngineV8

        engine = PrecedentAlignmentEngineV8()

        critics = {
            "fairness": {"severity": 0.5, "violations": []}
        }

        result = engine.analyze(critics, [], [0.1, 0.2, 0.3])

        # Should return high novelty, neutral alignment when no precedents
        assert result["is_novel"] is True
        assert result["precedent_cases"] == []

    def test_analyze_with_precedents(self):
        """Test analyze with precedent cases."""
        from engine.precedent.alignment import PrecedentAlignmentEngineV8

        engine = PrecedentAlignmentEngineV8()

        critics = {
            "fairness": {"severity": 0.6, "violations": ["discrimination"]}
        }

        precedent_cases = [
            {
                "text": "Historical case 1",
                "embedding": [0.5, 0.5, 0.5],
                "metadata": {
                    "decision": "deny",
                    "critic_severity": {"fairness": 0.7}
                }
            },
            {
                "text": "Historical case 2",
                "embedding": [0.3, 0.3, 0.3],
                "metadata": {
                    "decision": "allow",
                    "critic_severity": {"fairness": 0.2}
                }
            }
        ]

        query_embedding = [0.5, 0.5, 0.5]

        result = engine.analyze(critics, precedent_cases, query_embedding)

        assert "alignment_score" in result
        assert "support_strength" in result
        assert "conflict_level" in result
        assert "is_novel" in result
        assert result["is_novel"] is False  # Has precedents
        assert len(result["precedent_cases"]) == 2
