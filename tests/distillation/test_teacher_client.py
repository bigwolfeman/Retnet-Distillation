"""
Unit tests for TeacherClient with mocked HTTP responses.

Tests:
- Basic HTTP client functionality
- Retry logic with exponential backoff
- Request batching
- Response parsing and validation
- Error handling
"""

import json
import time
from unittest.mock import Mock, patch, MagicMock

import pytest
import requests
from requests.exceptions import ConnectionError, Timeout

from src.distillation.teacher_client import (
    TeacherClient,
    TeacherClientError,
    TeacherServerError,
    TeacherNetworkError,
)
from src.distillation.schemas import TopKResponse


@pytest.fixture
def mock_session():
    """Create a mock requests.Session."""
    with patch("src.distillation.teacher_client.requests.Session") as mock:
        session = MagicMock()
        mock.return_value = session
        yield session


@pytest.fixture
def client():
    """Create a TeacherClient instance for testing."""
    return TeacherClient(
        endpoint_url="http://localhost:8000/v1/topk",
        timeout=10.0,
        verify_ssl=False,
        max_retries=3,
        backoff_base=0.1,  # Fast backoff for testing
    )


@pytest.fixture
def sample_topk_response():
    """Sample TopKResponse for testing."""
    return {
        "indices": [[[100, 200, 300]]],  # batch=1, seq_len=1, k=3
        "values_int8": [[[127, 100, 80]]],
        "scale": [[0.05]],
        "other_mass": [[0.01]],
        "batch_size": 1,
        "num_positions": 1,
        "k": 3,
        "return_dtype": "int8",
    }


class TestTeacherClientInit:
    """Test TeacherClient initialization."""

    def test_init_with_valid_params(self):
        """Test initialization with valid parameters."""
        client = TeacherClient(
            endpoint_url="http://example.com/v1/topk",
            timeout=30.0,
            verify_ssl=True,
            max_retries=5,
        )
        assert client.endpoint_url == "http://example.com/v1/topk"
        assert client.timeout == 30.0
        assert client.verify_ssl is True
        assert client.max_retries == 5

    def test_init_with_empty_endpoint(self):
        """Test initialization fails with empty endpoint URL."""
        with pytest.raises(ValueError, match="endpoint_url cannot be empty"):
            TeacherClient(endpoint_url="")

    def test_init_creates_session(self, mock_session):
        """Test that initialization creates a requests.Session."""
        client = TeacherClient(endpoint_url="http://localhost:8000/v1/topk")
        mock_session.headers.update.assert_called_once_with({"Content-Type": "application/json"})


class TestTeacherClientQueryTopK:
    """Test query_topk method."""

    def test_successful_query(self, client, mock_session, sample_topk_response):
        """Test successful query_topk request."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_topk_response
        mock_session.post = Mock(return_value=mock_response)
        client.session = mock_session

        # Make request
        input_ids = [[1, 2, 3, 4, 5]]
        response = client.query_topk(input_ids=input_ids, topk=3)

        # Verify response
        assert isinstance(response, TopKResponse)
        assert response.batch_size == 1
        assert response.num_positions == 1
        assert response.k == 3
        assert response.indices == [[[100, 200, 300]]]

        # Verify request was made correctly
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert call_args[0][0] == "http://localhost:8000/v1/topk"
        assert call_args[1]["timeout"] == 10.0
        assert call_args[1]["verify"] is False

    def test_request_parameters(self, client, mock_session, sample_topk_response):
        """Test that request parameters are correctly passed."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_topk_response
        mock_session.post = Mock(return_value=mock_response)
        client.session = mock_session

        # Make request with custom parameters
        input_ids = [[1, 2, 3]]
        client.query_topk(
            input_ids=input_ids,
            topk=256,
            return_dtype="float16",
            temperature=2.0,
        )

        # Verify request payload
        call_args = mock_session.post.call_args
        request_data = call_args[1]["json"]
        assert request_data["input_ids"] == [[1, 2, 3]]
        assert request_data["topk"] == 256
        assert request_data["return_dtype"] == "float16"
        assert request_data["temperature"] == 2.0

    @patch("src.distillation.teacher_client.time.sleep")
    def test_server_error_response(self, mock_sleep, client, mock_session):
        """Test handling of 500 server error responses with retry."""
        # Mock error response - 500 is retryable, so it will retry 3 times
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {
            "error": "InternalError",
            "message": "Server crashed",
            "details": {},
        }
        mock_session.post = Mock(return_value=mock_response)
        client.session = mock_session

        # Verify error is raised after retries
        with pytest.raises(TeacherServerError, match="Server error 500 after 3 retries"):
            client.query_topk(input_ids=[[1, 2, 3]], topk=3)

        # Verify retries happened (500 is retryable)
        assert mock_session.post.call_count == 3

    def test_malformed_server_error(self, client, mock_session):
        """Test handling of malformed server error responses."""
        # Mock malformed error response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.text = "Bad request: malformed input"
        mock_session.post = Mock(return_value=mock_response)
        client.session = mock_session

        # Verify error is raised
        with pytest.raises(TeacherServerError, match="Server error.*Bad request"):
            client.query_topk(input_ids=[[1, 2, 3]], topk=3)

    def test_malformed_response_data(self, client, mock_session):
        """Test handling of malformed success response."""
        # Mock response with missing fields
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"indices": [[[]]], "incomplete": True}
        mock_session.post = Mock(return_value=mock_response)
        client.session = mock_session

        # Verify error is raised
        with pytest.raises(TeacherClientError, match="Failed to parse response"):
            client.query_topk(input_ids=[[1, 2, 3]], topk=3)


class TestTeacherClientRetryLogic:
    """Test retry logic with exponential backoff."""

    @patch("src.distillation.teacher_client.time.sleep")
    def test_retry_on_connection_error(self, mock_sleep, client, mock_session, sample_topk_response):
        """Test retry logic on ConnectionError."""
        # Mock: fail twice, then succeed
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_topk_response

        mock_session.post = Mock(
            side_effect=[
                ConnectionError("Connection refused"),
                ConnectionError("Connection refused"),
                mock_response,
            ]
        )
        client.session = mock_session

        # Make request (should succeed after 2 retries)
        response = client.query_topk(input_ids=[[1, 2, 3]], topk=3)

        # Verify retries happened
        assert mock_session.post.call_count == 3
        assert isinstance(response, TopKResponse)

        # Verify exponential backoff: 0.1s * 2^0 = 0.1s, 0.1s * 2^1 = 0.2s
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(0.1)  # First retry backoff
        mock_sleep.assert_any_call(0.2)  # Second retry backoff

    @patch("src.distillation.teacher_client.time.sleep")
    def test_retry_on_timeout(self, mock_sleep, client, mock_session, sample_topk_response):
        """Test retry logic on Timeout."""
        # Mock: timeout once, then succeed
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_topk_response

        mock_session.post = Mock(
            side_effect=[
                Timeout("Request timeout"),
                mock_response,
            ]
        )
        client.session = mock_session

        # Make request (should succeed after 1 retry)
        response = client.query_topk(input_ids=[[1, 2, 3]], topk=3)

        # Verify retry happened
        assert mock_session.post.call_count == 2
        assert isinstance(response, TopKResponse)
        mock_sleep.assert_called_once_with(0.1)

    @patch("src.distillation.teacher_client.time.sleep")
    def test_max_retries_exhausted(self, mock_sleep, client, mock_session):
        """Test that TeacherNetworkError is raised after max retries."""
        # Mock: always fail
        mock_session.post = Mock(side_effect=ConnectionError("Connection refused"))
        client.session = mock_session

        # Verify error after max retries
        with pytest.raises(TeacherNetworkError, match="Failed after 3 retries"):
            client.query_topk(input_ids=[[1, 2, 3]], topk=3)

        # Verify all retries were attempted
        assert mock_session.post.call_count == 3

        # Verify exponential backoff: 0.1s, 0.2s (no sleep after final attempt)
        assert mock_sleep.call_count == 2

    def test_non_retryable_error(self, client, mock_session):
        """Test that non-retryable RequestException is raised immediately."""
        # Mock: non-retryable error
        mock_session.post = Mock(side_effect=requests.exceptions.InvalidURL("Invalid URL"))
        client.session = mock_session

        # Verify error is raised immediately
        with pytest.raises(TeacherNetworkError, match="Request failed"):
            client.query_topk(input_ids=[[1, 2, 3]], topk=3)

        # Verify only one attempt was made
        assert mock_session.post.call_count == 1

    @patch("src.distillation.teacher_client.time.sleep")
    def test_retry_on_503_service_unavailable(self, mock_sleep, client, mock_session, sample_topk_response):
        """Test retry logic on 503 Service Unavailable."""
        # Mock: 503 twice, then succeed
        mock_error_response = Mock()
        mock_error_response.status_code = 503
        mock_error_response.json.return_value = {
            "error": "ServiceUnavailable",
            "message": "Service temporarily unavailable",
            "details": {},
        }

        mock_success_response = Mock()
        mock_success_response.status_code = 200
        mock_success_response.json.return_value = sample_topk_response

        mock_session.post = Mock(
            side_effect=[
                mock_error_response,
                mock_error_response,
                mock_success_response,
            ]
        )
        client.session = mock_session

        # Make request (should succeed after 2 retries)
        response = client.query_topk(input_ids=[[1, 2, 3]], topk=3)

        # Verify retries happened
        assert mock_session.post.call_count == 3
        assert isinstance(response, TopKResponse)

        # Verify exponential backoff: 0.1s * 2^0 = 0.1s, 0.1s * 2^1 = 0.2s
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(0.1)  # First retry backoff
        mock_sleep.assert_any_call(0.2)  # Second retry backoff

    @patch("src.distillation.teacher_client.time.sleep")
    def test_retry_on_502_bad_gateway(self, mock_sleep, client, mock_session, sample_topk_response):
        """Test retry logic on 502 Bad Gateway."""
        # Mock: 502 once, then succeed
        mock_error_response = Mock()
        mock_error_response.status_code = 502
        mock_error_response.json.return_value = {
            "error": "BadGateway",
            "message": "Bad gateway",
            "details": {},
        }

        mock_success_response = Mock()
        mock_success_response.status_code = 200
        mock_success_response.json.return_value = sample_topk_response

        mock_session.post = Mock(
            side_effect=[
                mock_error_response,
                mock_success_response,
            ]
        )
        client.session = mock_session

        # Make request (should succeed after 1 retry)
        response = client.query_topk(input_ids=[[1, 2, 3]], topk=3)

        # Verify retry happened
        assert mock_session.post.call_count == 2
        assert isinstance(response, TopKResponse)
        mock_sleep.assert_called_once_with(0.1)

    @patch("src.distillation.teacher_client.time.sleep")
    def test_retry_on_504_gateway_timeout(self, mock_sleep, client, mock_session, sample_topk_response):
        """Test retry logic on 504 Gateway Timeout."""
        # Mock: 504 once, then succeed
        mock_error_response = Mock()
        mock_error_response.status_code = 504
        mock_error_response.text = "Gateway timeout"

        mock_success_response = Mock()
        mock_success_response.status_code = 200
        mock_success_response.json.return_value = sample_topk_response

        mock_session.post = Mock(
            side_effect=[
                mock_error_response,
                mock_success_response,
            ]
        )
        client.session = mock_session

        # Make request (should succeed after 1 retry)
        response = client.query_topk(input_ids=[[1, 2, 3]], topk=3)

        # Verify retry happened
        assert mock_session.post.call_count == 2
        assert isinstance(response, TopKResponse)
        mock_sleep.assert_called_once_with(0.1)

    @patch("src.distillation.teacher_client.time.sleep")
    def test_5xx_max_retries_exhausted(self, mock_sleep, client, mock_session):
        """Test that TeacherServerError is raised after max retries on 5xx errors."""
        # Mock: always return 503
        mock_error_response = Mock()
        mock_error_response.status_code = 503
        mock_session.post = Mock(return_value=mock_error_response)
        client.session = mock_session

        # Verify error after max retries
        with pytest.raises(TeacherServerError, match="Server error 503 after 3 retries"):
            client.query_topk(input_ids=[[1, 2, 3]], topk=3)

        # Verify all retries were attempted
        assert mock_session.post.call_count == 3

        # Verify exponential backoff: 0.1s, 0.2s (no sleep after final attempt)
        assert mock_sleep.call_count == 2

    def test_4xx_errors_not_retried(self, client, mock_session):
        """Test that 4xx errors are not retried (non-retryable)."""
        # Mock: 400 Bad Request
        mock_error_response = Mock()
        mock_error_response.status_code = 400
        mock_error_response.text = "Bad request body"
        mock_error_response.json.return_value = {
            "error": "BadRequest",
            "message": "Invalid input",
            "details": {},
        }
        mock_session.post = Mock(return_value=mock_error_response)
        client.session = mock_session

        # Verify error is raised immediately without retries
        with pytest.raises(TeacherServerError, match="Server error.*Invalid input"):
            client.query_topk(input_ids=[[1, 2, 3]], topk=3)

        # Verify only one attempt was made (no retries)
        assert mock_session.post.call_count == 1


class TestTeacherClientBatchQuery:
    """Test batch_query method."""

    def test_batch_query_single_batch(self, client, mock_session, sample_topk_response):
        """Test batch_query with single batch (no chunking)."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_topk_response
        mock_session.post = Mock(return_value=mock_response)
        client.session = mock_session

        # Make batched request with 2 sequences (below batch_size=32)
        input_ids_list = [[[1, 2, 3]], [[4, 5, 6]]]
        responses = client.batch_query(input_ids_list, topk=3, batch_size=32)

        # Verify single request was made
        assert len(responses) == 1
        assert mock_session.post.call_count == 1

        # Verify both sequences were included
        call_args = mock_session.post.call_args
        request_data = call_args[1]["json"]
        assert request_data["input_ids"] == [[1, 2, 3], [4, 5, 6]]

    def test_batch_query_multiple_batches(self, client, mock_session, sample_topk_response):
        """Test batch_query with chunking into multiple batches."""
        # Mock successful responses
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_topk_response
        mock_session.post = Mock(return_value=mock_response)
        client.session = mock_session

        # Create 5 sequences with batch_size=2 (should create 3 batches)
        input_ids_list = [[[i]] for i in range(5)]
        responses = client.batch_query(input_ids_list, topk=3, batch_size=2)

        # Verify 3 requests were made
        assert len(responses) == 3
        assert mock_session.post.call_count == 3

    def test_batch_query_empty_list(self, client):
        """Test batch_query with empty input list."""
        responses = client.batch_query([], topk=3, batch_size=32)
        assert responses == []

    def test_batch_query_respects_parameters(self, client, mock_session, sample_topk_response):
        """Test that batch_query passes parameters correctly."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_topk_response
        mock_session.post = Mock(return_value=mock_response)
        client.session = mock_session

        # Make request with custom parameters
        input_ids_list = [[[1, 2, 3]]]
        client.batch_query(
            input_ids_list,
            topk=256,
            return_dtype="float16",
            temperature=2.0,
            batch_size=1,
        )

        # Verify parameters were passed
        call_args = mock_session.post.call_args
        request_data = call_args[1]["json"]
        assert request_data["topk"] == 256
        assert request_data["return_dtype"] == "float16"
        assert request_data["temperature"] == 2.0


class TestTeacherClientHealthCheck:
    """Test health_check method."""

    def test_health_check_success(self, client, mock_session):
        """Test successful health check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_session.get = Mock(return_value=mock_response)
        client.session = mock_session

        # Verify health check succeeds
        assert client.health_check() is True

        # Verify correct endpoint was called
        mock_session.get.assert_called_once()
        call_args = mock_session.get.call_args
        assert "/health" in call_args[0][0]

    def test_health_check_failure(self, client, mock_session):
        """Test failed health check."""
        mock_session.get = Mock(side_effect=ConnectionError("Connection refused"))
        client.session = mock_session

        # Verify health check fails gracefully
        assert client.health_check() is False


class TestTeacherClientContextManager:
    """Test context manager functionality."""

    def test_context_manager(self, mock_session):
        """Test TeacherClient as context manager."""
        with TeacherClient(endpoint_url="http://localhost:8000/v1/topk") as client:
            assert client is not None
            # Session should be created
            assert hasattr(client, "session")

        # Verify session was closed
        # Note: This is a basic check; in real usage, session.close() would be called


class TestTeacherClientEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_large_batch(self, client, mock_session, sample_topk_response):
        """Test batch_query with very large batch."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_topk_response
        mock_session.post = Mock(return_value=mock_response)
        client.session = mock_session

        # 100 sequences with batch_size=10 (should create 10 batches)
        input_ids_list = [[[i]] for i in range(100)]
        responses = client.batch_query(input_ids_list, topk=3, batch_size=10)

        # Verify correct number of batches
        assert len(responses) == 10
        assert mock_session.post.call_count == 10

    def test_single_sequence(self, client, mock_session, sample_topk_response):
        """Test query_topk with single sequence."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_topk_response
        mock_session.post = Mock(return_value=mock_response)
        client.session = mock_session

        # Single sequence
        response = client.query_topk(input_ids=[[1]], topk=3)
        assert isinstance(response, TopKResponse)

    def test_long_sequence(self, client, mock_session):
        """Test query_topk with very long sequence."""
        mock_response = Mock()
        mock_response.status_code = 200

        # Response for long sequence
        long_response = {
            "indices": [[[100, 200, 300]] * 1000],  # 1000 positions
            "values_int8": [[[127, 100, 80]] * 1000],
            "scale": [[0.05] * 1000],
            "other_mass": [[0.01] * 1000],
            "batch_size": 1,
            "num_positions": 1000,
            "k": 3,
            "return_dtype": "int8",
        }
        mock_response.json.return_value = long_response
        mock_session.post = Mock(return_value=mock_response)
        client.session = mock_session

        # Long sequence (4096 tokens)
        long_input = list(range(4096))
        response = client.query_topk(input_ids=[long_input], topk=3)

        assert isinstance(response, TopKResponse)
        assert response.num_positions == 1000


class TestTeacherClientDimensionValidation:
    """Test response dimension validation (C-002)."""

    def test_batch_size_mismatch(self, client, mock_session):
        """Test validation detects batch_size mismatch."""
        # Response with wrong batch_size
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "indices": [[[100, 200, 300]]],
            "values_int8": [[[127, 100, 80]]],
            "scale": [[0.05]],
            "other_mass": [[0.01]],
            "batch_size": 1,  # Server says batch_size=1
            "num_positions": 3,
            "k": 3,
            "return_dtype": "int8",
        }
        mock_session.post = Mock(return_value=mock_response)
        client.session = mock_session

        # Request with batch_size=2
        input_ids = [[1, 2, 3], [4, 5, 6]]

        # Verify error is raised
        with pytest.raises(TeacherClientError, match="Response batch_size mismatch: expected 2, got 1"):
            client.query_topk(input_ids=input_ids, topk=3)

    def test_k_mismatch(self, client, mock_session):
        """Test validation detects k mismatch."""
        # Response with wrong k
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "indices": [[[100, 200, 300]]],
            "values_int8": [[[127, 100, 80]]],
            "scale": [[0.05]],
            "other_mass": [[0.01]],
            "batch_size": 1,
            "num_positions": 3,
            "k": 3,  # Server says k=3
            "return_dtype": "int8",
        }
        mock_session.post = Mock(return_value=mock_response)
        client.session = mock_session

        # Request with topk=128
        input_ids = [[1, 2, 3]]

        # Verify error is raised
        with pytest.raises(TeacherClientError, match="Response k mismatch: expected 128, got 3"):
            client.query_topk(input_ids=input_ids, topk=128)

    def test_num_positions_mismatch(self, client, mock_session):
        """Test validation detects num_positions mismatch for sequences."""
        # Response with wrong num_positions
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "indices": [[[100, 200, 300], [100, 200, 300]]],  # 2 positions
            "values_int8": [[[127, 100, 80], [127, 100, 80]]],
            "scale": [[0.05, 0.05]],
            "other_mass": [[0.01, 0.01]],
            "batch_size": 1,
            "num_positions": 2,
            "k": 3,
            "return_dtype": "int8",
        }
        mock_session.post = Mock(return_value=mock_response)
        client.session = mock_session

        # Request with 5 tokens (expecting 5 positions)
        input_ids = [[1, 2, 3, 4, 5]]

        # Verify error is raised
        with pytest.raises(
            TeacherClientError,
            match="Response num_positions mismatch for sequence 0: expected 5, got 2"
        ):
            client.query_topk(input_ids=input_ids, topk=3)

    def test_indices_length_mismatch(self, client, mock_session):
        """Test validation detects indices length != batch_size."""
        # Response with mismatched array lengths - indices has 2 sequences but batch_size=1
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "indices": [
                [[100, 200, 300]],  # seq 0: 1 position with 3 values
                [[100, 200, 300]]   # seq 1: 1 position with 3 values (EXTRA - shouldn't exist)
            ],  # 2 sequences
            "values_int8": [[[127, 100, 80]]],  # 1 sequence (correct)
            "scale": [[0.05]],
            "other_mass": [[0.01]],
            "batch_size": 1,  # Says batch_size=1
            "num_positions": 1,  # 1 position per sequence
            "k": 3,
            "return_dtype": "int8",
        }
        mock_session.post = Mock(return_value=mock_response)
        client.session = mock_session

        # Request with 1 sequence of 1 token
        input_ids = [[1]]

        # Verify error is raised
        with pytest.raises(TeacherClientError, match="indices length 2 != batch_size 1"):
            client.query_topk(input_ids=input_ids, topk=3)

    def test_values_int8_length_mismatch(self, client, mock_session):
        """Test validation detects values_int8 length != batch_size."""
        # Response with mismatched array lengths - values_int8 has 2 sequences but batch_size=1
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "indices": [[[100, 200, 300]]],  # 1 sequence (correct)
            "values_int8": [
                [[127, 100, 80]],  # seq 0
                [[127, 100, 80]]   # seq 1 (EXTRA - shouldn't exist)
            ],  # 2 sequences
            "scale": [[0.05]],
            "other_mass": [[0.01]],
            "batch_size": 1,
            "num_positions": 1,
            "k": 3,
            "return_dtype": "int8",
        }
        mock_session.post = Mock(return_value=mock_response)
        client.session = mock_session

        # Request with 1 sequence of 1 token
        input_ids = [[1]]

        # Verify error is raised
        with pytest.raises(TeacherClientError, match="values_int8 length 2 != batch_size 1"):
            client.query_topk(input_ids=input_ids, topk=3)

    def test_scale_length_mismatch(self, client, mock_session):
        """Test validation detects scale length != batch_size."""
        # Response with mismatched array lengths - scale has 2 sequences but batch_size=1
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "indices": [[[100, 200, 300]]],  # 1 sequence (correct)
            "values_int8": [[[127, 100, 80]]],  # 1 sequence (correct)
            "scale": [[0.05], [0.05]],  # 2 sequences (EXTRA - shouldn't exist)
            "other_mass": [[0.01]],
            "batch_size": 1,
            "num_positions": 1,
            "k": 3,
            "return_dtype": "int8",
        }
        mock_session.post = Mock(return_value=mock_response)
        client.session = mock_session

        # Request with 1 sequence of 1 token
        input_ids = [[1]]

        # Verify error is raised
        with pytest.raises(TeacherClientError, match="scale length 2 != batch_size 1"):
            client.query_topk(input_ids=input_ids, topk=3)

    def test_other_mass_length_mismatch(self, client, mock_session):
        """Test validation detects other_mass length != batch_size."""
        # Response with mismatched array lengths - other_mass has 2 sequences but batch_size=1
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "indices": [[[100, 200, 300]]],  # 1 sequence (correct)
            "values_int8": [[[127, 100, 80]]],  # 1 sequence (correct)
            "scale": [[0.05]],  # 1 sequence (correct)
            "other_mass": [[0.01], [0.01]],  # 2 sequences (EXTRA - shouldn't exist)
            "batch_size": 1,
            "num_positions": 1,
            "k": 3,
            "return_dtype": "int8",
        }
        mock_session.post = Mock(return_value=mock_response)
        client.session = mock_session

        # Request with 1 sequence of 1 token
        input_ids = [[1]]

        # Verify error is raised
        with pytest.raises(TeacherClientError, match="other_mass length 2 != batch_size 1"):
            client.query_topk(input_ids=input_ids, topk=3)

    def test_valid_dimensions_pass(self, client, mock_session):
        """Test that correctly dimensioned responses pass validation."""
        # Response with correct dimensions
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "indices": [
                [[100, 200, 300], [400, 500, 600], [700, 800, 900]],  # seq 1: 3 positions
                [[101, 201, 301], [401, 501, 601]],  # seq 2: 2 positions
            ],
            "values_int8": [
                [[127, 100, 80], [120, 90, 70], [110, 85, 65]],
                [[125, 95, 75], [115, 88, 68]],
            ],
            "scale": [[0.05, 0.06, 0.07], [0.04, 0.05]],
            "other_mass": [[0.01, 0.02, 0.03], [0.01, 0.02]],
            "batch_size": 2,
            "num_positions": 5,  # total across batch
            "k": 3,
            "return_dtype": "int8",
        }
        mock_session.post = Mock(return_value=mock_response)
        client.session = mock_session

        # Request with matching dimensions
        input_ids = [[1, 2, 3], [4, 5]]

        # Should not raise
        response = client.query_topk(input_ids=input_ids, topk=3)
        assert isinstance(response, TopKResponse)
        assert response.batch_size == 2
        assert response.k == 3

    def test_multi_sequence_position_mismatch(self, client, mock_session):
        """Test validation detects position mismatch in multi-sequence batch."""
        # Response with wrong positions for second sequence
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "indices": [
                [[100, 200, 300], [400, 500, 600], [700, 800, 900]],  # seq 1: 3 positions ✓
                [[101, 201, 301]],  # seq 2: 1 position (but should be 2) ✗
            ],
            "values_int8": [
                [[127, 100, 80], [120, 90, 70], [110, 85, 65]],
                [[125, 95, 75]],
            ],
            "scale": [[0.05, 0.06, 0.07], [0.04]],
            "other_mass": [[0.01, 0.02, 0.03], [0.01]],
            "batch_size": 2,
            "num_positions": 4,
            "k": 3,
            "return_dtype": "int8",
        }
        mock_session.post = Mock(return_value=mock_response)
        client.session = mock_session

        # Request with 3 and 2 token sequences
        input_ids = [[1, 2, 3], [4, 5]]

        # Verify error is raised for sequence 1 (0-indexed)
        with pytest.raises(
            TeacherClientError,
            match="Response num_positions mismatch for sequence 1: expected 2, got 1"
        ):
            client.query_topk(input_ids=input_ids, topk=3)


class TestTeacherClientResourceManagement:
    """Test session resource management (C-003 fix)."""

    def test_close_is_idempotent(self, mock_session):
        """Test that close() can be called multiple times safely."""
        client = TeacherClient(endpoint_url="http://localhost:8000/v1/topk")
        client.session = mock_session

        # Call close() multiple times
        client.close()
        client.close()
        client.close()

        # Should only close session once
        mock_session.close.assert_called_once()

    def test_close_handles_session_close_exception(self, mock_session):
        """Test that close() handles exceptions from session.close()."""
        client = TeacherClient(endpoint_url="http://localhost:8000/v1/topk")
        client.session = mock_session

        # Make session.close() raise an exception
        mock_session.close.side_effect = RuntimeError("Close failed")

        # Should not raise, but log warning
        client.close()

        # Should be marked as closed even after exception
        assert client._closed is True

    def test_close_without_session_attribute(self):
        """Test close() when session attribute doesn't exist."""
        client = TeacherClient(endpoint_url="http://localhost:8000/v1/topk")

        # Delete session attribute to simulate incomplete initialization
        delattr(client, 'session')

        # Should not raise
        client.close()
        assert client._closed is True

    def test_context_manager_closes_on_exit(self, mock_session):
        """Test that context manager closes session on exit."""
        with TeacherClient(endpoint_url="http://localhost:8000/v1/topk") as client:
            client.session = mock_session
            pass

        # Session should be closed after exit
        mock_session.close.assert_called_once()

    def test_context_manager_closes_on_exception(self, mock_session):
        """Test that context manager closes session even when exception occurs."""
        try:
            with TeacherClient(endpoint_url="http://localhost:8000/v1/topk") as client:
                client.session = mock_session
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Session should still be closed
        mock_session.close.assert_called_once()

    def test_context_manager_handles_close_exception(self, mock_session):
        """Test that __exit__ handles exceptions from close()."""
        mock_session.close.side_effect = RuntimeError("Close failed")

        # Should not raise during context manager exit
        with TeacherClient(endpoint_url="http://localhost:8000/v1/topk") as client:
            client.session = mock_session
            pass

        # Should have attempted to close
        mock_session.close.assert_called_once()

    def test_del_finalizer_closes_session(self, mock_session):
        """Test that __del__ finalizer closes session."""
        client = TeacherClient(endpoint_url="http://localhost:8000/v1/topk")
        client.session = mock_session

        # Call __del__ explicitly
        client.__del__()

        # Session should be closed
        mock_session.close.assert_called_once()

    def test_del_suppresses_exceptions(self, mock_session):
        """Test that __del__ suppresses all exceptions."""
        client = TeacherClient(endpoint_url="http://localhost:8000/v1/topk")
        client.session = mock_session

        # Make close() raise exception
        mock_session.close.side_effect = RuntimeError("Close failed")

        # Should not raise during __del__
        client.__del__()

        # Should have attempted to close
        mock_session.close.assert_called_once()

    def test_closed_flag_initial_state(self):
        """Test that _closed flag is initialized to False."""
        client = TeacherClient(endpoint_url="http://localhost:8000/v1/topk")
        assert hasattr(client, '_closed')
        assert client._closed is False

    def test_closed_flag_set_after_close(self, mock_session):
        """Test that _closed flag is set to True after close()."""
        client = TeacherClient(endpoint_url="http://localhost:8000/v1/topk")
        client.session = mock_session

        assert client._closed is False
        client.close()
        assert client._closed is True

    def test_multiple_close_calls_in_context_managers(self, mock_session):
        """Test that multiple context manager uses properly handle close."""
        client = TeacherClient(endpoint_url="http://localhost:8000/v1/topk")
        client.session = mock_session

        # First context manager
        with client:
            assert client._closed is False

        # After first exit, should be closed
        assert client._closed is True

        # Second exit should be no-op
        with client:
            pass

        # Should only have closed once due to idempotent close
        mock_session.close.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
