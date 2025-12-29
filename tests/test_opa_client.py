"""
Comprehensive tests for OPAClientV8

Tests all code paths including:
- Initialization
- Health checks
- Policy evaluation (success and error cases)
- Error handling and graceful degradation
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock

import httpx
from engine.governance.opa_client import OPAClientV8


class TestOPAClientInitialization:
    """Test OPAClientV8 initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        client = OPAClientV8()
        assert client.base_url == "http://localhost:8181"
        assert client.policy_path == "v1/data/eleanor/decision"

    def test_custom_base_url(self):
        """Test initialization with custom base URL."""
        client = OPAClientV8(base_url="http://opa.example.com:8181")
        assert client.base_url == "http://opa.example.com:8181"

    def test_base_url_trailing_slash_removed(self):
        """Test that trailing slash is removed from base URL."""
        client = OPAClientV8(base_url="http://localhost:8181/")
        assert client.base_url == "http://localhost:8181"

    def test_custom_policy_path(self):
        """Test initialization with custom policy path."""
        client = OPAClientV8(policy_path="v1/data/custom/policy")
        assert client.policy_path == "v1/data/custom/policy"

    def test_policy_path_leading_slash_removed(self):
        """Test that leading slash is removed from policy path."""
        client = OPAClientV8(policy_path="/v1/data/eleanor/decision")
        assert client.policy_path == "v1/data/eleanor/decision"

    def test_policy_path_trailing_slash_removed(self):
        """Test that trailing slash is removed from policy path."""
        client = OPAClientV8(policy_path="v1/data/eleanor/decision/")
        assert client.policy_path == "v1/data/eleanor/decision"


class TestOPAClientHealth:
    """Test OPAClientV8 health check functionality."""

    @patch('engine.governance.opa_client.httpx.Client')
    def test_health_check_success(self, mock_get):
        """Test successful health check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__.return_value = mock_client
        mock_get.return_value = mock_client

        client = OPAClientV8()
        assert client.health() is True
        mock_client.get.assert_called_once_with("http://localhost:8181/health")

    @patch('engine.governance.opa_client.httpx.Client')
    def test_health_check_failure_non_200(self, mock_get):
        """Test health check with non-200 status code."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__.return_value = mock_client
        mock_get.return_value = mock_client

        client = OPAClientV8()
        assert client.health() is False

    @patch('engine.governance.opa_client.httpx.Client')
    def test_health_check_connection_error(self, mock_get):
        """Test health check with connection error."""
        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.RequestError("Connection refused", request=Mock())
        mock_client.__enter__.return_value = mock_client
        mock_get.return_value = mock_client

        client = OPAClientV8()
        assert client.health() is False

    @patch('engine.governance.opa_client.httpx.Client')
    def test_health_check_timeout(self, mock_get):
        """Test health check with timeout."""
        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.RequestError("Request timeout", request=Mock())
        mock_client.__enter__.return_value = mock_client
        mock_get.return_value = mock_client

        client = OPAClientV8()
        assert client.health() is False


class TestOPAClientEvaluate:
    """Test OPAClientV8 policy evaluation functionality."""

    @patch('engine.governance.opa_client.httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_evaluate_success_allow(self, mock_post):
        """Test successful evaluation with allow=True."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {
                "allow": True,
                "escalate": False,
                "failures": []
            }
        }
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_post.return_value = mock_client

        client = OPAClientV8()
        evidence = {"critics": {"fairness": {"severity": 0.5}}}
        result = await client.evaluate(evidence)

        assert result["allow"] is True
        assert result["escalate"] is False
        assert result["failures"] == []
        mock_client.post.assert_awaited_once_with(
            "http://localhost:8181/v1/data/eleanor/decision",
            json={"input": evidence}
        )

    @patch('engine.governance.opa_client.httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_evaluate_success_deny(self, mock_post):
        """Test successful evaluation with allow=False."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {
                "allow": False,
                "escalate": False,
                "failures": [{"policy": "fairness_threshold", "reason": "Severity too high"}]
            }
        }
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_post.return_value = mock_client

        client = OPAClientV8()
        evidence = {"critics": {"fairness": {"severity": 2.5}}}
        result = await client.evaluate(evidence)

        assert result["allow"] is False
        assert result["escalate"] is False
        assert len(result["failures"]) == 1
        assert result["failures"][0]["policy"] == "fairness_threshold"

    @patch('engine.governance.opa_client.httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_evaluate_success_escalate(self, mock_post):
        """Test successful evaluation with escalate=True."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {
                "allow": False,
                "escalate": True,
                "failures": [{"policy": "constitutional_threshold", "reason": "Tier 3 escalation"}]
            }
        }
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_post.return_value = mock_client

        client = OPAClientV8()
        evidence = {"critics": {"rights": {"severity": 3.0, "escalation": {"tier": "tier_3"}}}}
        result = await client.evaluate(evidence)

        assert result["allow"] is False
        assert result["escalate"] is True
        assert len(result["failures"]) == 1

    @patch('engine.governance.opa_client.httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_evaluate_connection_error(self, mock_post):
        """Test evaluate with connection error."""
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(side_effect=httpx.RequestError("Connection refused", request=Mock()))
        mock_post.return_value = mock_client

        client = OPAClientV8()
        evidence = {"test": "data"}
        result = await client.evaluate(evidence)

        assert result["allow"] is False
        assert result["escalate"] is True
        assert len(result["failures"]) == 1
        assert result["failures"][0]["policy"] == "opa_client_error"
        assert "Connection refused" in result["failures"][0]["reason"]

    @patch('engine.governance.opa_client.httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_evaluate_timeout_error(self, mock_post):
        """Test evaluate with timeout error."""
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(side_effect=httpx.RequestError("Request timeout", request=Mock()))
        mock_post.return_value = mock_client

        client = OPAClientV8()
        evidence = {"test": "data"}
        result = await client.evaluate(evidence)

        assert result["allow"] is False
        assert result["escalate"] is True
        assert len(result["failures"]) == 1
        assert result["failures"][0]["policy"] == "opa_client_error"
        assert "timeout" in result["failures"][0]["reason"].lower()

    @patch('engine.governance.opa_client.httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_evaluate_http_error_500(self, mock_post):
        """Test evaluate with HTTP 500 error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_post.return_value = mock_client

        client = OPAClientV8()
        evidence = {"test": "data"}
        result = await client.evaluate(evidence)

        assert result["allow"] is False
        assert result["escalate"] is True
        assert len(result["failures"]) == 1
        assert result["failures"][0]["policy"] == "opa_http_error"
        assert "HTTP 500" in result["failures"][0]["reason"]

    @patch('engine.governance.opa_client.httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_evaluate_http_error_404(self, mock_post):
        """Test evaluate with HTTP 404 error."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_post.return_value = mock_client

        client = OPAClientV8()
        evidence = {"test": "data"}
        result = await client.evaluate(evidence)

        assert result["allow"] is False
        assert result["escalate"] is True
        assert result["failures"][0]["policy"] == "opa_http_error"
        assert "HTTP 404" in result["failures"][0]["reason"]

    @patch('engine.governance.opa_client.httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_evaluate_invalid_json_response(self, mock_post):
        """Test evaluate with invalid JSON response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.text = "not valid json"
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_post.return_value = mock_client

        client = OPAClientV8()
        evidence = {"test": "data"}
        result = await client.evaluate(evidence)

        assert result["allow"] is False
        assert result["escalate"] is True
        assert len(result["failures"]) == 1
        assert result["failures"][0]["policy"] == "invalid_json"
        assert result["failures"][0]["reason"] == "not valid json"

    @patch('engine.governance.opa_client.httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_evaluate_missing_result_field(self, mock_post):
        """Test evaluate with missing result field in response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}  # No "result" field
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_post.return_value = mock_client

        client = OPAClientV8()
        evidence = {"test": "data"}
        result = await client.evaluate(evidence)

        # Should default to deny/no-escalate with empty failures
        assert result["allow"] is False
        assert result["escalate"] is False
        assert result["failures"] == []

    @patch('engine.governance.opa_client.httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_evaluate_partial_result_fields(self, mock_post):
        """Test evaluate with partial fields in result."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {
                "allow": True
                # Missing "escalate" and "failures"
            }
        }
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_post.return_value = mock_client

        client = OPAClientV8()
        evidence = {"test": "data"}
        result = await client.evaluate(evidence)

        assert result["allow"] is True
        assert result["escalate"] is False  # Default
        assert result["failures"] == []  # Default

    @patch('engine.governance.opa_client.httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_evaluate_with_custom_policy_path(self, mock_post):
        """Test evaluate uses custom policy path correctly."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {"allow": True, "escalate": False, "failures": []}
        }
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_post.return_value = mock_client

        client = OPAClientV8(policy_path="v1/data/custom/policy")
        evidence = {"test": "data"}
        await client.evaluate(evidence)

        mock_client.post.assert_awaited_once_with(
            "http://localhost:8181/v1/data/custom/policy",
            json={"input": evidence}
        )

    @patch('engine.governance.opa_client.httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_evaluate_with_complex_evidence(self, mock_post):
        """Test evaluate with complex evidence payload."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {"allow": False, "escalate": True, "failures": []}
        }
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_post.return_value = mock_client

        client = OPAClientV8()
        evidence = {
            "critics": {
                "fairness": {
                    "severity": 2.5,
                    "violations": ["discrimination"],
                    "escalation": {"tier": "tier_3", "clause_id": "F1"}
                },
                "rights": {
                    "severity": 2.0,
                    "violations": ["dignity violation"]
                }
            },
            "metadata": {
                "timestamp": "2025-01-01T00:00:00Z",
                "model": "claude-sonnet-4.5"
            }
        }
        result = await client.evaluate(evidence)

        assert result["allow"] is False
        assert result["escalate"] is True
        # Verify the evidence was passed correctly
        call_args = mock_client.post.call_args
        assert call_args[1]["json"]["input"] == evidence


class TestOPAClientIntegration:
    """Integration-style tests combining multiple operations."""

    @patch('engine.governance.opa_client.httpx.Client')
    @patch('engine.governance.opa_client.httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_health_then_evaluate(self, mock_async_client, mock_client_cls):
        """Test health check followed by evaluation."""
        # Mock health check
        mock_health_response = Mock()
        mock_health_response.status_code = 200
        mock_client = MagicMock()
        mock_client.get.return_value = mock_health_response
        mock_client.__enter__.return_value = mock_client
        mock_client_cls.return_value = mock_client

        # Mock evaluation
        mock_eval_response = Mock()
        mock_eval_response.status_code = 200
        mock_eval_response.json.return_value = {
            "result": {"allow": True, "escalate": False, "failures": []}
        }
        mock_async = AsyncMock()
        mock_async.__aenter__.return_value = mock_async
        mock_async.post = AsyncMock(return_value=mock_eval_response)
        mock_async_client.return_value = mock_async

        client = OPAClientV8()

        # First check health
        assert client.health() is True

        # Then evaluate
        result = await client.evaluate({"test": "data"})
        assert result["allow"] is True

    @patch('engine.governance.opa_client.httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_multiple_evaluations(self, mock_post):
        """Test multiple consecutive evaluations."""
        # First evaluation: allow
        mock_response1 = Mock()
        mock_response1.status_code = 200
        mock_response1.json.return_value = {
            "result": {"allow": True, "escalate": False, "failures": []}
        }

        # Second evaluation: deny
        mock_response2 = Mock()
        mock_response2.status_code = 200
        mock_response2.json.return_value = {
            "result": {"allow": False, "escalate": False, "failures": [{"policy": "test"}]}
        }

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(side_effect=[mock_response1, mock_response2])
        mock_post.return_value = mock_client

        client = OPAClientV8()

        result1 = await client.evaluate({"severity": 0.5})
        assert result1["allow"] is True

        result2 = await client.evaluate({"severity": 2.5})
        assert result2["allow"] is False

        assert mock_post.call_count == 2
