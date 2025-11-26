"""
Unit tests for teacher server /v1/topk endpoint.

Tests:
- T010: Endpoint unit tests with mock vLLM
- Schema validation
- Top-k correctness
- Int8 quantization accuracy
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from fastapi import HTTPException

from src.distillation.schemas import TopKRequest, TopKResponse, TopKErrorResponse
from src.distillation.teacher_server import (
    TopKLogitProcessor,
    TeacherServer,
    create_app
)


class TestTopKLogitProcessor:
    """Test TopKLogitProcessor functionality."""

    def test_process_logits_shapes(self):
        """Test that process_logits returns correct shapes."""
        processor = TopKLogitProcessor(k=128, temperature=1.0)

        batch_size, seq_len, vocab_size = 2, 10, 32000
        logits = torch.randn(batch_size, seq_len, vocab_size)

        result = processor.process_logits(logits, k=128, temperature=1.0)

        # Check shapes
        assert result["indices"].shape == (batch_size, seq_len, 128)
        assert result["values_int8"].shape == (batch_size, seq_len, 128)
        assert result["scale"].shape == (batch_size, seq_len)
        assert result["other_mass"].shape == (batch_size, seq_len)

        # Check dtypes
        assert result["indices"].dtype == torch.int32
        assert result["values_int8"].dtype == torch.int8
        assert result["scale"].dtype == torch.float32
        assert result["other_mass"].dtype == torch.float32

    def test_topk_correctness(self):
        """Test that top-k selection returns correct indices."""
        processor = TopKLogitProcessor(k=5, temperature=1.0)

        # Create deterministic logits
        batch_size, seq_len, vocab_size = 1, 1, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)

        result = processor.process_logits(logits, k=5, temperature=1.0)

        # Verify top-k indices match torch.topk
        expected_topk_values, expected_topk_indices = torch.topk(
            logits[0, 0],
            k=5,
            largest=True,
            sorted=True
        )

        assert torch.allclose(
            result["indices"][0, 0],
            expected_topk_indices.to(torch.int32)
        )

    def test_int8_quantization_range(self):
        """Test that int8 values are in valid range [-128, 127]."""
        processor = TopKLogitProcessor(k=128, temperature=1.0)

        batch_size, seq_len, vocab_size = 2, 10, 32000
        logits = torch.randn(batch_size, seq_len, vocab_size) * 10  # Large range

        result = processor.process_logits(logits, k=128, temperature=1.0)

        # Check int8 range
        assert result["values_int8"].min() >= -128
        assert result["values_int8"].max() <= 127

    def test_int8_quantization_accuracy(self):
        """Test int8 quantization preserves information within tolerance."""
        processor = TopKLogitProcessor(k=128, temperature=1.0)

        batch_size, seq_len, vocab_size = 1, 5, 32000
        logits = torch.randn(batch_size, seq_len, vocab_size)

        result = processor.process_logits(logits, k=128, temperature=1.0)

        # Dequantize
        dequantized = processor.dequantize(
            result["values_int8"],
            result["scale"]
        )

        # Get original top-k values
        topk_values, _ = torch.topk(logits, k=128, dim=-1, largest=True, sorted=True)

        # Compute relative error
        # Max absolute error should be ~scale (one quantization step)
        max_scale = result["scale"].max().item()
        max_error = (dequantized - topk_values.cpu()).abs().max().item()

        # Error should be within ~1 quantization step
        assert max_error <= max_scale * 1.5, f"Max error {max_error} > {max_scale * 1.5}"

    def test_other_mass_computation(self):
        """Test other_mass computation for sparse-KL."""
        processor = TopKLogitProcessor(k=10, temperature=1.0)

        batch_size, seq_len, vocab_size = 1, 1, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)

        result = processor.process_logits(logits, k=10, temperature=1.0)

        # Verify other_mass properties
        other_mass = result["other_mass"][0, 0].item()

        # Should be in [1e-12, 1-1e-12] (clamped range)
        assert 1e-12 <= other_mass <= 1.0 - 1e-12

        # Compute expected other_mass using CORRECT algorithm:
        # 1. Compute log partition function over FULL vocabulary
        # 2. Compute normalized log-probs for top-k
        # 3. other_mass = 1 - sum(exp(log_probs_topk))
        log_z = torch.logsumexp(logits[0, 0], dim=-1, keepdim=True)
        topk_values, _ = torch.topk(logits[0, 0], k=10, largest=True)
        log_probs_topk = topk_values - log_z
        topk_probs = torch.exp(log_probs_topk)
        expected_other_mass = 1.0 - topk_probs.sum().item()
        expected_other_mass = max(1e-12, min(expected_other_mass, 1.0 - 1e-12))

        # Should match (with some numerical tolerance)
        assert abs(other_mass - expected_other_mass) < 1e-5

    def test_other_mass_verification_cases(self):
        """Test other_mass computation with specific verification cases."""
        processor = TopKLogitProcessor(k=100, temperature=1.0)

        # Test Case 1: Uniform distribution over V=1000 with k=100
        # Expected: other_mass ≈ 0.9 (90% in tail)
        batch_size, seq_len, vocab_size = 1, 1, 1000
        uniform_logits = torch.zeros(batch_size, seq_len, vocab_size)
        result_uniform = processor.process_logits(uniform_logits, k=100, temperature=1.0)
        other_mass_uniform = result_uniform["other_mass"][0, 0].item()

        # For uniform distribution: other_mass = 1 - k/V = 1 - 100/1000 = 0.9
        expected_uniform = 1.0 - (100.0 / 1000.0)
        assert abs(other_mass_uniform - expected_uniform) < 1e-4, \
            f"Uniform dist: expected {expected_uniform}, got {other_mass_uniform}"

        # Test Case 2: Peaked distribution (one token p=0.99)
        # Expected: other_mass ≈ 0.01 if that token is in top-k
        peaked_logits = torch.full((batch_size, seq_len, vocab_size), -10.0)
        peaked_logits[0, 0, 0] = 5.0  # Make first token highly probable
        result_peaked = processor.process_logits(peaked_logits, k=100, temperature=1.0)
        other_mass_peaked = result_peaked["other_mass"][0, 0].item()

        # Compute actual probability mass in top-k
        log_z = torch.logsumexp(peaked_logits[0, 0], dim=-1, keepdim=True)
        topk_values, _ = torch.topk(peaked_logits[0, 0], k=100, largest=True)
        log_probs_topk = topk_values - log_z
        expected_peaked = 1.0 - torch.exp(log_probs_topk).sum().item()

        assert abs(other_mass_peaked - expected_peaked) < 1e-4, \
            f"Peaked dist: expected {expected_peaked}, got {other_mass_peaked}"

        # For peaked distribution, other_mass should be very small
        assert other_mass_peaked < 0.1, \
            f"Peaked dist should have small other_mass, got {other_mass_peaked}"

        # Test Case 3: Verify other_mass is NOT always ≈0 (the bug we're fixing)
        # With random logits and k=10, vocab_size=100, other_mass should be significant
        processor_small_k = TopKLogitProcessor(k=10, temperature=1.0)
        random_logits = torch.randn(batch_size, seq_len, 100)
        result_random = processor_small_k.process_logits(random_logits, k=10, temperature=1.0)
        other_mass_random = result_random["other_mass"][0, 0].item()

        # With random logits, k=10/100, other_mass should be substantial (not ≈0)
        # This would fail with the buggy softmax-on-topk implementation
        assert other_mass_random > 0.5, \
            f"Random dist with k=10/100 should have other_mass > 0.5, got {other_mass_random}"

    def test_temperature_effect(self):
        """Test that temperature affects other_mass correctly."""
        processor = TopKLogitProcessor(k=10)

        batch_size, seq_len, vocab_size = 1, 1, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)

        # Test with different temperatures
        result_t1 = processor.process_logits(logits, k=10, temperature=1.0)
        result_t2 = processor.process_logits(logits, k=10, temperature=2.0)

        # Higher temperature should smooth distribution
        # (result depends on logit distribution, just check valid range)
        assert 0.0 <= result_t1["other_mass"][0, 0].item() <= 1.0
        assert 0.0 <= result_t2["other_mass"][0, 0].item() <= 1.0

    def test_k_exceeds_vocab_size(self):
        """Test handling when k > vocab_size."""
        processor = TopKLogitProcessor(k=128)

        batch_size, seq_len, vocab_size = 1, 1, 50
        logits = torch.randn(batch_size, seq_len, vocab_size)

        # Should clamp k to vocab_size
        result = processor.process_logits(logits, k=128, temperature=1.0)

        # Result should have k=vocab_size
        assert result["indices"].shape[-1] == vocab_size
        assert result["values_int8"].shape[-1] == vocab_size

    def test_dequantize(self):
        """Test dequantization correctness."""
        processor = TopKLogitProcessor()

        # Create test data
        batch_size, seq_len, k = 2, 5, 128
        values_int8 = torch.randint(-128, 128, (batch_size, seq_len, k), dtype=torch.int8)
        scale = torch.rand(batch_size, seq_len) * 0.1

        # Dequantize
        dequantized = processor.dequantize(values_int8, scale)

        # Check shape and dtype
        assert dequantized.shape == (batch_size, seq_len, k)
        assert dequantized.dtype == torch.float32

        # Verify dequantization formula
        for b in range(batch_size):
            for s in range(seq_len):
                expected = values_int8[b, s].float() * scale[b, s]
                assert torch.allclose(dequantized[b, s], expected, atol=1e-6)


class TestTopKSchemas:
    """Test TopKRequest and TopKResponse schemas."""

    def test_topk_request_validation(self):
        """Test TopKRequest validation."""
        # Valid request
        request = TopKRequest(
            input_ids=[[1, 2, 3, 4, 5]],
            topk=128,
            return_dtype="int8",
            temperature=2.0
        )
        assert request.topk == 128
        assert request.temperature == 2.0

    def test_topk_request_defaults(self):
        """Test TopKRequest default values."""
        request = TopKRequest(input_ids=[[1, 2, 3]])
        assert request.topk == 128
        assert request.return_dtype == "int8"
        assert request.temperature == 1.0
        assert request.max_tokens is None

    def test_topk_request_invalid_input_ids(self):
        """Test TopKRequest rejects invalid input_ids."""
        from pydantic import ValidationError

        # Empty sequence
        with pytest.raises(ValueError, match="empty"):
            TopKRequest(input_ids=[[]])

        # Empty batch - pydantic raises ValidationError for min_length constraint
        with pytest.raises(ValidationError, match="at least 1 item"):
            TopKRequest(input_ids=[])

        # Negative token IDs
        with pytest.raises(ValueError, match="negative token ID"):
            TopKRequest(input_ids=[[1, -2, 3]])

    def test_topk_request_invalid_params(self):
        """Test TopKRequest parameter validation."""
        # Import validation constants
        from src.distillation.schemas import MIN_TOPK, MAX_TOPK

        # Invalid topk (too small)
        with pytest.raises(ValueError, match=f"topk must be between {MIN_TOPK}"):
            TopKRequest(input_ids=[[1, 2, 3]], topk=0)

        # Invalid topk (too large)
        with pytest.raises(ValueError, match=f"topk must be between {MIN_TOPK} and {MAX_TOPK}"):
            TopKRequest(input_ids=[[1, 2, 3]], topk=2000)

        # Invalid temperature (too small)
        with pytest.raises(ValueError, match="temperature must be positive"):
            TopKRequest(input_ids=[[1, 2, 3]], temperature=0.0)

        # Invalid temperature (too large)
        with pytest.raises(ValueError, match="temperature must be"):
            TopKRequest(input_ids=[[1, 2, 3]], temperature=11.0)

    def test_topk_request_sequence_length_validation(self):
        """Test F-002: Sequence length validation prevents unbounded memory allocation."""
        from src.distillation.schemas import MAX_SEQUENCE_LENGTH

        # Valid sequence at max length
        request = TopKRequest(input_ids=[[1] * MAX_SEQUENCE_LENGTH])
        assert len(request.input_ids[0]) == MAX_SEQUENCE_LENGTH

        # Invalid: sequence exceeds max length
        with pytest.raises(ValueError, match=f"length .* exceeds maximum {MAX_SEQUENCE_LENGTH}"):
            TopKRequest(input_ids=[[1] * (MAX_SEQUENCE_LENGTH + 1)])

    def test_topk_request_batch_size_validation(self):
        """Test F-004: Batch size validation."""
        from src.distillation.schemas import MAX_BATCH_SIZE

        # Valid batch at max size
        request = TopKRequest(input_ids=[[1, 2, 3]] * MAX_BATCH_SIZE)
        assert len(request.input_ids) == MAX_BATCH_SIZE

        # Invalid: batch exceeds max size
        with pytest.raises(ValueError, match=f"Batch size .* exceeds maximum {MAX_BATCH_SIZE}"):
            TopKRequest(input_ids=[[1, 2, 3]] * (MAX_BATCH_SIZE + 1))

    def test_topk_response_validation(self):
        """Test TopKResponse validation."""
        response = TopKResponse(
            indices=[[[1, 2, 3]]],
            values_int8=[[[100, 50, 0]]],
            scale=[[0.05]],
            other_mass=[[0.01]],
            batch_size=1,
            num_positions=1,
            k=3,
            return_dtype="int8"
        )
        assert response.batch_size == 1
        assert response.k == 3

    def test_topk_response_invalid_int8(self):
        """Test TopKResponse rejects invalid int8 values."""
        # Value out of range (>127)
        with pytest.raises(ValueError, match="out of int8 range"):
            TopKResponse(
                indices=[[[1, 2, 3]]],
                values_int8=[[[128, 50, 0]]],  # 128 > 127
                scale=[[0.05]],
                other_mass=[[0.01]],
                batch_size=1,
                num_positions=1,
                k=3,
                return_dtype="int8"
            )

        # Value out of range (<-128)
        with pytest.raises(ValueError, match="out of int8 range"):
            TopKResponse(
                indices=[[[1, 2, 3]]],
                values_int8=[[[-129, 50, 0]]],  # -129 < -128
                scale=[[0.05]],
                other_mass=[[0.01]],
                batch_size=1,
                num_positions=1,
                k=3,
                return_dtype="int8"
            )


@pytest.fixture
def mock_vllm_engine():
    """Mock vLLM engine for testing."""
    mock_engine = MagicMock()
    mock_engine.model_config.vocab_size = 32000
    return mock_engine


@pytest.fixture
def mock_teacher_server(mock_vllm_engine):
    """Mock TeacherServer for testing."""
    with patch('src.distillation.teacher_server.VLLM_AVAILABLE', True):
        with patch('src.distillation.teacher_server.LLM') as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm.llm_engine = mock_vllm_engine
            mock_llm_class.return_value = mock_llm

            server = TeacherServer(
                model_name="test-model",
                tensor_parallel_size=1,
                max_model_len=4096
            )

            # Override get_logits to return controlled data
            def mock_get_logits(input_ids, max_tokens=None):
                batch_size = len(input_ids)
                max_len = max(len(seq) for seq in input_ids)
                if max_tokens is None:
                    max_tokens = max_len
                return torch.randn(batch_size, max_tokens, 32000)

            server.get_logits = mock_get_logits

            return server


class TestTeacherServer:
    """Test TeacherServer functionality."""

    def test_server_initialization(self, mock_vllm_engine):
        """Test TeacherServer initialization."""
        with patch('src.distillation.teacher_server.VLLM_AVAILABLE', True):
            with patch('src.distillation.teacher_server.LLM') as mock_llm_class:
                mock_llm = MagicMock()
                mock_llm.llm_engine = mock_vllm_engine
                mock_llm_class.return_value = mock_llm

                server = TeacherServer(
                    model_name="test-model",
                    tensor_parallel_size=1,
                    max_model_len=4096
                )

                assert server.model_name == "test-model"
                assert server.tensor_parallel_size == 1
                assert server.max_model_len == 4096
                assert server.processor is not None

    def test_process_topk_request(self, mock_teacher_server):
        """Test processing TopKRequest."""
        request = TopKRequest(
            input_ids=[[1, 2, 3, 4, 5]],
            topk=10,
            temperature=2.0
        )

        response = mock_teacher_server.process_topk_request(request)

        # Check response structure
        assert isinstance(response, TopKResponse)
        assert response.batch_size == 1
        assert response.num_positions == 5
        assert response.k == 10

        # Check array shapes
        assert len(response.indices) == 1
        assert len(response.indices[0]) == 5
        assert len(response.indices[0][0]) == 10

    def test_server_multiple_sequences(self, mock_teacher_server):
        """Test processing multiple sequences in batch."""
        request = TopKRequest(
            input_ids=[
                [1, 2, 3],
                [4, 5, 6, 7]
            ],
            topk=5,
            temperature=1.0
        )

        response = mock_teacher_server.process_topk_request(request)

        assert response.batch_size == 2
        # Max length in batch
        assert response.num_positions == 4
        assert response.k == 5

    def test_vocab_size_validation_token_ids(self, mock_teacher_server):
        """Test F-006: Token ID upper bound validation against vocab size."""
        # Mock vocab_size is 32000
        vocab_size = 32000

        # Valid token IDs (within vocab)
        request = TopKRequest(
            input_ids=[[1, 100, 31999]],
            topk=10
        )
        response = mock_teacher_server.process_topk_request(request)
        assert response is not None

        # Invalid token ID (>= vocab_size)
        request_invalid = TopKRequest(
            input_ids=[[1, 100, 32000]],  # 32000 >= vocab_size
            topk=10
        )
        with pytest.raises(HTTPException) as exc_info:
            mock_teacher_server.process_topk_request(request_invalid)
        assert exc_info.value.status_code == 400
        assert "vocab_size" in str(exc_info.value.detail)

    def test_vocab_size_validation_topk(self, mock_teacher_server):
        """Test F-005: topk validation against vocab size."""
        # Mock vocab_size is 32000
        vocab_size = 32000

        # Valid topk (within limits and vocab)
        request = TopKRequest(
            input_ids=[[1, 2, 3]],
            topk=1000  # Within MAX_TOPK=1024 and vocab_size=32000
        )
        response = mock_teacher_server.process_topk_request(request)
        assert response is not None

        # Note: topk > vocab_size would need to pass schema validation first
        # Schema validation limits topk to MAX_TOPK=1024, which is < 32000
        # But we still validate in server for edge cases

    def test_edge_case_validation(self, mock_teacher_server):
        """Test edge cases for validation."""
        # Single token sequence (edge case but valid)
        request = TopKRequest(
            input_ids=[[1]],
            topk=1
        )
        response = mock_teacher_server.process_topk_request(request)
        assert response.num_positions == 1

        # Large batch at max limit
        from src.distillation.schemas import MAX_BATCH_SIZE
        request_large = TopKRequest(
            input_ids=[[1, 2, 3]] * MAX_BATCH_SIZE,
            topk=10
        )
        response_large = mock_teacher_server.process_topk_request(request_large)
        assert response_large.batch_size == MAX_BATCH_SIZE


class TestFastAPIEndpoint:
    """Test FastAPI endpoint integration."""

    def test_create_app(self, mock_teacher_server):
        """Test FastAPI app creation."""
        app = create_app(mock_teacher_server)

        # Check routes exist
        routes = [route.path for route in app.routes]
        assert "/health" in routes
        assert "/v1/topk" in routes

    def test_health_endpoint(self, mock_teacher_server):
        """Test /health endpoint."""
        from fastapi.testclient import TestClient

        app = create_app(mock_teacher_server)
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert response.json()["model"] == "test-model"

    def test_topk_endpoint_success(self, mock_teacher_server):
        """Test /v1/topk endpoint with valid request."""
        from fastapi.testclient import TestClient

        app = create_app(mock_teacher_server)
        client = TestClient(app)

        request_data = {
            "input_ids": [[1, 2, 3, 4, 5]],
            "topk": 10,
            "temperature": 2.0
        }

        response = client.post("/v1/topk", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["batch_size"] == 1
        assert data["num_positions"] == 5
        assert data["k"] == 10

    def test_topk_endpoint_invalid_request(self, mock_teacher_server):
        """Test /v1/topk endpoint with invalid request."""
        from fastapi.testclient import TestClient

        app = create_app(mock_teacher_server)
        client = TestClient(app)

        # Empty input_ids
        request_data = {
            "input_ids": [],
            "topk": 10
        }

        response = client.post("/v1/topk", json=request_data)
        assert response.status_code == 422  # Validation error


class TestEndToEndFlow:
    """End-to-end tests for complete request/response cycle."""

    def test_complete_topk_flow(self, mock_teacher_server):
        """Test complete flow from request to response."""
        # Create request
        request = TopKRequest(
            input_ids=[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
            topk=128,
            temperature=2.0
        )

        # Process request
        response = mock_teacher_server.process_topk_request(request)

        # Verify response
        assert response.batch_size == 1
        assert response.num_positions == 10
        assert response.k == 128

        # Verify all int8 values are in valid range
        for batch in response.values_int8:
            for pos in batch:
                for val in pos:
                    assert -128 <= val <= 127

        # Verify scales are positive
        for batch in response.scale:
            for scale_val in batch:
                assert scale_val > 0

        # Verify other_mass in [0, 1]
        for batch in response.other_mass:
            for mass in batch:
                assert 0.0 <= mass <= 1.0

    def test_quantization_roundtrip(self):
        """Test that quantization roundtrip preserves information."""
        processor = TopKLogitProcessor(k=128, temperature=1.0)

        # Create test logits
        batch_size, seq_len, vocab_size = 1, 10, 32000
        logits = torch.randn(batch_size, seq_len, vocab_size)

        # Process
        result = processor.process_logits(logits, k=128, temperature=1.0)

        # Dequantize
        dequantized = processor.dequantize(
            result["values_int8"],
            result["scale"]
        )

        # Get original top-k
        topk_values, _ = torch.topk(logits, k=128, dim=-1, largest=True)

        # Compute cross-entropy with both versions
        # This tests the key requirement: CE delta ≤ 1e-3
        ce_original = -torch.log_softmax(topk_values, dim=-1)[:, :, 0].mean()
        ce_quantized = -torch.log_softmax(dequantized, dim=-1)[:, :, 0].mean()

        ce_delta = abs(ce_original.item() - ce_quantized.item())

        # CE delta should be small (exact threshold tested in calibration)
        # Here we just check it's reasonable
        assert ce_delta < 0.1, f"CE delta {ce_delta} too large"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
