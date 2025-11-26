"""
Integration tests for teacher server /v1/topk endpoint.

T011: Integration tests requiring running vLLM server

These tests are marked as slow/integration and require:
- Running vLLM server with actual model
- GPU access
- Network connectivity

Usage:
    # Run integration tests (slow, requires GPU and vLLM)
    pytest tests/distillation/test_teacher_server_integration.py -v -m integration

    # Skip integration tests
    pytest tests/distillation/test_teacher_server_integration.py -v -m "not integration"
"""

import pytest
import requests
import time
import torch
from typing import Optional

from src.distillation.schemas import TopKRequest, TopKResponse


# Integration test configuration
TEST_SERVER_URL = "http://localhost:8000"
TEST_MODEL = "meta-llama/Llama-3.2-1B"
THROUGHPUT_TARGET = 2000  # tok/s (M1 gate requirement)
LATENCY_P95_TARGET = 25  # ms (M1 gate requirement)


@pytest.fixture(scope="module")
def server_available():
    """
    Check if teacher server is running.

    Returns:
        bool: True if server is available, False otherwise
    """
    try:
        response = requests.get(f"{TEST_SERVER_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


@pytest.fixture(scope="module")
def skip_if_no_server(server_available):
    """Skip test if server is not available."""
    if not server_available:
        pytest.skip(
            f"Teacher server not available at {TEST_SERVER_URL}. "
            f"Start server with: python -m src.distillation.teacher_server "
            f"--model {TEST_MODEL} --port 8000"
        )


@pytest.mark.integration
@pytest.mark.slow
class TestServerIntegration:
    """Integration tests for teacher server with real vLLM."""

    def test_server_health(self, skip_if_no_server):
        """Test /health endpoint returns healthy status."""
        response = requests.get(f"{TEST_SERVER_URL}/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "model" in data

    def test_basic_topk_request(self, skip_if_no_server):
        """Test basic /v1/topk request with small sequence."""
        request_data = {
            "input_ids": [[1, 2, 3, 4, 5]],
            "topk": 128,
            "temperature": 1.0
        }

        response = requests.post(
            f"{TEST_SERVER_URL}/v1/topk",
            json=request_data,
            timeout=30
        )

        assert response.status_code == 200

        data = response.json()
        assert data["batch_size"] == 1
        assert data["num_positions"] == 5
        assert data["k"] == 128

        # Validate response structure
        assert len(data["indices"]) == 1
        assert len(data["indices"][0]) == 5
        assert len(data["indices"][0][0]) == 128

    def test_4k_sequence_request(self, skip_if_no_server):
        """Test /v1/topk with 4k token sequence (realistic distillation workload)."""
        # Create 4k token sequence
        seq_len = 4096
        input_ids = list(range(1, seq_len + 1))

        request_data = {
            "input_ids": [input_ids],
            "topk": 256,
            "temperature": 2.0
        }

        start_time = time.time()
        response = requests.post(
            f"{TEST_SERVER_URL}/v1/topk",
            json=request_data,
            timeout=60
        )
        elapsed_time = time.time() - start_time

        assert response.status_code == 200

        data = response.json()
        assert data["batch_size"] == 1
        assert data["num_positions"] == seq_len
        assert data["k"] == 256

        # Check response size (should be ~130KB for 4k sequence)
        response_size_kb = len(response.content) / 1024
        print(f"Response size: {response_size_kb:.2f} KB")
        print(f"Latency: {elapsed_time * 1000:.2f} ms")

        # Response should be compact (target: ~130KB for 4k sequence)
        # Allow some overhead for JSON encoding
        assert response_size_kb < 200, f"Response too large: {response_size_kb:.2f} KB"

    def test_batch_request(self, skip_if_no_server):
        """Test batched sequences in single request."""
        request_data = {
            "input_ids": [
                [1, 2, 3, 4, 5],
                [10, 20, 30, 40, 50, 60],
                [100, 200, 300]
            ],
            "topk": 128,
            "temperature": 1.0
        }

        response = requests.post(
            f"{TEST_SERVER_URL}/v1/topk",
            json=request_data,
            timeout=30
        )

        assert response.status_code == 200

        data = response.json()
        assert data["batch_size"] == 3
        assert data["k"] == 128

    def test_different_topk_values(self, skip_if_no_server):
        """Test different top-k values."""
        input_ids = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]

        for k in [32, 64, 128, 256, 512]:
            request_data = {
                "input_ids": input_ids,
                "topk": k,
                "temperature": 1.0
            }

            response = requests.post(
                f"{TEST_SERVER_URL}/v1/topk",
                json=request_data,
                timeout=30
            )

            assert response.status_code == 200
            data = response.json()
            assert data["k"] == k

    def test_different_temperatures(self, skip_if_no_server):
        """Test different temperature values."""
        input_ids = [[1, 2, 3, 4, 5]]

        for temp in [0.5, 1.0, 2.0, 5.0]:
            request_data = {
                "input_ids": input_ids,
                "topk": 128,
                "temperature": temp
            }

            response = requests.post(
                f"{TEST_SERVER_URL}/v1/topk",
                json=request_data,
                timeout=30
            )

            assert response.status_code == 200

            data = response.json()
            # Other-mass should vary with temperature
            assert 0.0 <= data["other_mass"][0][0] <= 1.0

    def test_invalid_request_handling(self, skip_if_no_server):
        """Test server handles invalid requests correctly."""
        # Empty input_ids
        response = requests.post(
            f"{TEST_SERVER_URL}/v1/topk",
            json={"input_ids": [], "topk": 128},
            timeout=10
        )
        assert response.status_code == 422  # Validation error

        # Negative token IDs
        response = requests.post(
            f"{TEST_SERVER_URL}/v1/topk",
            json={"input_ids": [[1, -2, 3]], "topk": 128},
            timeout=10
        )
        assert response.status_code == 422

        # Invalid topk
        response = requests.post(
            f"{TEST_SERVER_URL}/v1/topk",
            json={"input_ids": [[1, 2, 3]], "topk": 0},
            timeout=10
        )
        assert response.status_code == 422


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.gpu
class TestThroughputAndLatency:
    """
    Throughput and latency tests for M1 gate validation.

    M1 Gate Requirements:
    - Server sustains ≥2k tok/s
    - Client latency <25ms (p95)
    """

    def test_throughput_benchmark(self, skip_if_no_server):
        """
        Test server throughput meets M1 gate (≥2k tok/s).

        This test measures sustained throughput over multiple requests.
        """
        num_requests = 10
        seq_len = 1000
        total_tokens = 0
        total_time = 0

        for i in range(num_requests):
            input_ids = list(range(1, seq_len + 1))
            request_data = {
                "input_ids": [input_ids],
                "topk": 128,
                "temperature": 1.0
            }

            start_time = time.time()
            response = requests.post(
                f"{TEST_SERVER_URL}/v1/topk",
                json=request_data,
                timeout=60
            )
            elapsed_time = time.time() - start_time

            assert response.status_code == 200

            total_tokens += seq_len
            total_time += elapsed_time

        throughput = total_tokens / total_time
        print(f"\nThroughput: {throughput:.2f} tok/s")
        print(f"Total tokens: {total_tokens}")
        print(f"Total time: {total_time:.2f}s")

        # M1 gate: ≥2k tok/s
        assert throughput >= THROUGHPUT_TARGET, \
            f"Throughput {throughput:.2f} tok/s < target {THROUGHPUT_TARGET} tok/s"

    def test_latency_benchmark(self, skip_if_no_server):
        """
        Test client latency meets M1 gate (<25ms p95).

        This test measures latency distribution for many requests.
        """
        num_requests = 100
        seq_len = 100  # Short sequences for latency test
        latencies = []

        for i in range(num_requests):
            input_ids = list(range(1, seq_len + 1))
            request_data = {
                "input_ids": [input_ids],
                "topk": 128,
                "temperature": 1.0
            }

            start_time = time.time()
            response = requests.post(
                f"{TEST_SERVER_URL}/v1/topk",
                json=request_data,
                timeout=30
            )
            elapsed_time = time.time() - start_time

            assert response.status_code == 200
            latencies.append(elapsed_time * 1000)  # Convert to ms

        # Calculate percentiles
        latencies_sorted = sorted(latencies)
        p50 = latencies_sorted[len(latencies) // 2]
        p95 = latencies_sorted[int(len(latencies) * 0.95)]
        p99 = latencies_sorted[int(len(latencies) * 0.99)]

        print(f"\nLatency distribution:")
        print(f"  p50: {p50:.2f} ms")
        print(f"  p95: {p95:.2f} ms")
        print(f"  p99: {p99:.2f} ms")

        # M1 gate: p95 < 25ms
        assert p95 < LATENCY_P95_TARGET, \
            f"Latency p95 {p95:.2f}ms > target {LATENCY_P95_TARGET}ms"

    def test_sustained_load(self, skip_if_no_server):
        """
        Test server handles sustained load without degradation.

        Runs continuous requests for a period and verifies throughput stays stable.
        """
        duration_seconds = 30
        seq_len = 1000
        requests_sent = 0
        total_tokens = 0

        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            input_ids = list(range(1, seq_len + 1))
            request_data = {
                "input_ids": [input_ids],
                "topk": 128,
                "temperature": 1.0
            }

            try:
                response = requests.post(
                    f"{TEST_SERVER_URL}/v1/topk",
                    json=request_data,
                    timeout=30
                )

                if response.status_code == 200:
                    requests_sent += 1
                    total_tokens += seq_len
            except requests.exceptions.RequestException:
                # Allow some failures under sustained load
                pass

        elapsed_time = time.time() - start_time
        throughput = total_tokens / elapsed_time

        print(f"\nSustained load results:")
        print(f"  Duration: {elapsed_time:.2f}s")
        print(f"  Requests: {requests_sent}")
        print(f"  Tokens: {total_tokens}")
        print(f"  Throughput: {throughput:.2f} tok/s")

        # Should sustain reasonable throughput
        assert throughput >= THROUGHPUT_TARGET * 0.8, \
            f"Sustained throughput {throughput:.2f} tok/s too low"


@pytest.mark.integration
@pytest.mark.slow
class TestResponseSizeOptimization:
    """Test response size optimization (15x reduction target)."""

    def test_response_size_4k_sequence(self, skip_if_no_server):
        """
        Test response size for 4k sequence meets optimization target.

        Target: ~130KB for 4k sequence (15x smaller than dense logits)
        Dense logits: 4096 tokens * 128k vocab * 4 bytes (fp32) = ~2GB
        Top-k int8: 4096 tokens * 256 top-k * 1 byte + overhead = ~130KB
        """
        seq_len = 4096
        input_ids = list(range(1, seq_len + 1))

        request_data = {
            "input_ids": [input_ids],
            "topk": 256,
            "temperature": 1.0
        }

        response = requests.post(
            f"{TEST_SERVER_URL}/v1/topk",
            json=request_data,
            timeout=60
        )

        assert response.status_code == 200

        # Measure response size
        response_size_bytes = len(response.content)
        response_size_kb = response_size_bytes / 1024

        # Calculate dense logits size (for comparison)
        vocab_size = 128000  # Approximate for Llama-3.2
        dense_size_mb = (seq_len * vocab_size * 4) / (1024 * 1024)

        compression_ratio = (dense_size_mb * 1024) / response_size_kb

        print(f"\nResponse size analysis:")
        print(f"  Sparse response: {response_size_kb:.2f} KB")
        print(f"  Dense equivalent: {dense_size_mb:.2f} MB")
        print(f"  Compression ratio: {compression_ratio:.1f}x")

        # Target: ~130KB (allow up to 200KB with JSON overhead)
        assert response_size_kb < 200, \
            f"Response size {response_size_kb:.2f}KB exceeds 200KB target"

        # Should achieve >10x compression
        assert compression_ratio > 10, \
            f"Compression ratio {compression_ratio:.1f}x < 10x target"


@pytest.mark.integration
@pytest.mark.slow
class TestQuantizationAccuracy:
    """
    Test int8 quantization accuracy (M2 gate validation).

    M2 Gate: CE delta ≤ 1e-3 vs fp32 logits
    Note: Full calibration done in M2b tests (T023-T026)
    """

    def test_int8_values_in_range(self, skip_if_no_server):
        """Test all int8 values are in valid range."""
        request_data = {
            "input_ids": [[1, 2, 3, 4, 5]],
            "topk": 128,
            "temperature": 1.0
        }

        response = requests.post(
            f"{TEST_SERVER_URL}/v1/topk",
            json=request_data,
            timeout=30
        )

        assert response.status_code == 200
        data = response.json()

        # Check all int8 values are in [-128, 127]
        for batch in data["values_int8"]:
            for pos in batch:
                for val in pos:
                    assert -128 <= val <= 127, f"Int8 value {val} out of range"

    def test_scale_factors_positive(self, skip_if_no_server):
        """Test scale factors are positive."""
        request_data = {
            "input_ids": [[1, 2, 3, 4, 5]],
            "topk": 128,
            "temperature": 1.0
        }

        response = requests.post(
            f"{TEST_SERVER_URL}/v1/topk",
            json=request_data,
            timeout=30
        )

        assert response.status_code == 200
        data = response.json()

        # Check all scale factors are positive
        for batch in data["scale"]:
            for scale in batch:
                assert scale > 0, f"Scale factor {scale} not positive"

    def test_other_mass_valid_range(self, skip_if_no_server):
        """Test other_mass values are in [0, 1]."""
        request_data = {
            "input_ids": [[1, 2, 3, 4, 5]],
            "topk": 128,
            "temperature": 1.0
        }

        response = requests.post(
            f"{TEST_SERVER_URL}/v1/topk",
            json=request_data,
            timeout=30
        )

        assert response.status_code == 200
        data = response.json()

        # Check other_mass in [0, 1]
        for batch in data["other_mass"]:
            for mass in batch:
                assert 0.0 <= mass <= 1.0, f"Other-mass {mass} out of range [0, 1]"


# Manual test instructions
def print_manual_test_instructions():
    """
    Print instructions for running integration tests manually.
    """
    instructions = f"""
    ========================================
    Manual Integration Test Instructions
    ========================================

    These integration tests require a running vLLM teacher server.

    Setup:
    1. Install vLLM: pip install vllm>=0.2.0

    2. Start teacher server:
       python -m src.distillation.teacher_server \\
           --model {TEST_MODEL} \\
           --port 8000 \\
           --host 0.0.0.0

    3. Wait for server to load model (may take 1-2 minutes)

    4. Run integration tests:
       pytest tests/distillation/test_teacher_server_integration.py -v -m integration

    M1 Gate Validation:
    - test_throughput_benchmark: Validates ≥2k tok/s
    - test_latency_benchmark: Validates p95 <25ms
    - test_response_size_4k_sequence: Validates ~130KB response

    Skip integration tests:
       pytest tests/distillation/test_teacher_server_integration.py -v -m "not integration"

    ========================================
    """
    print(instructions)


if __name__ == "__main__":
    print_manual_test_instructions()
    pytest.main([__file__, "-v", "-m", "not integration"])
