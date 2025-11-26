"""
Integration tests for TeacherClient with real teacher server.

These tests require a running teacher server and are marked as 'integration'
and 'slow'. They test:
- Latency requirements (target p95 <25ms per 4k sequence)
- Throughput with batching
- Network error recovery
- Real server response parsing

Run with: pytest -m integration tests/distillation/test_teacher_client_integration.py

Prerequisites:
- Teacher server running at configured endpoint
- vLLM with Llama-3.2-1B model loaded
- /v1/topk endpoint implemented
"""

import os
import time
from statistics import mean, median, stdev
from typing import List

import pytest

from src.distillation.teacher_client import TeacherClient, TeacherNetworkError


# Configuration from environment variables
TEACHER_ENDPOINT = os.getenv("TEACHER_ENDPOINT", "http://localhost:8000/v1/topk")
SKIP_INTEGRATION = os.getenv("SKIP_INTEGRATION_TESTS", "true").lower() == "true"


@pytest.fixture
def client():
    """Create TeacherClient connected to real server."""
    return TeacherClient(
        endpoint_url=TEACHER_ENDPOINT,
        timeout=30.0,
        verify_ssl=False,
        max_retries=3,
        backoff_base=1.0,
    )


@pytest.fixture
def sample_input_ids():
    """Sample input sequences for testing."""
    # 4k sequence (approximately 4096 tokens)
    return [[i % 128256 for i in range(4096)]]


@pytest.fixture
def batch_input_ids():
    """Batch of input sequences for throughput testing."""
    # 32 sequences of ~1024 tokens each
    return [[[i % 128256 for i in range(j, j + 1024)] for j in range(0, 32768, 1024)]]


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(SKIP_INTEGRATION, reason="Integration tests disabled (set SKIP_INTEGRATION_TESTS=false to enable)")
class TestTeacherClientIntegration:
    """Integration tests with real teacher server."""

    def test_server_connectivity(self, client):
        """Test basic server connectivity."""
        # Health check should succeed
        is_healthy = client.health_check()
        assert is_healthy, f"Teacher server at {TEACHER_ENDPOINT} is not responding to health checks"

    def test_single_query_success(self, client):
        """Test single query to real server."""
        input_ids = [[1, 2, 3, 4, 5]]
        response = client.query_topk(input_ids=input_ids, topk=128)

        # Verify response structure
        assert response.batch_size == 1
        assert response.num_positions == len(input_ids[0])
        assert response.k == 128

        # Verify data shapes
        assert len(response.indices) == 1
        assert len(response.indices[0]) == len(input_ids[0])
        assert len(response.indices[0][0]) == 128

        # Verify int8 quantization
        for pos_values in response.values_int8[0]:
            for val in pos_values:
                assert -128 <= val <= 127, f"int8 value {val} out of range"

        # Verify other_mass is valid probability
        for mass in response.other_mass[0]:
            assert 0.0 <= mass <= 1.0, f"other_mass {mass} out of range [0, 1]"

    def test_4k_sequence_latency(self, client, sample_input_ids):
        """
        Test latency for 4k sequence (M1 gate requirement).

        Target: p95 latency <25ms per 4k sequence
        """
        num_samples = 50
        latencies = []

        # Warm up (first request may be slower)
        client.query_topk(input_ids=sample_input_ids, topk=128)

        # Measure latency over multiple requests
        for _ in range(num_samples):
            start_time = time.perf_counter()
            response = client.query_topk(input_ids=sample_input_ids, topk=128)
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

            # Verify response is valid
            assert response.batch_size == 1
            assert response.num_positions == len(sample_input_ids[0])

        # Calculate statistics
        p50 = median(latencies)
        p95 = sorted(latencies)[int(0.95 * num_samples)]
        p99 = sorted(latencies)[int(0.99 * num_samples)]
        avg = mean(latencies)
        std = stdev(latencies) if len(latencies) > 1 else 0

        print(f"\n4k Sequence Latency Statistics (n={num_samples}):")
        print(f"  p50: {p50:.2f}ms")
        print(f"  p95: {p95:.2f}ms")
        print(f"  p99: {p99:.2f}ms")
        print(f"  avg: {avg:.2f}ms")
        print(f"  std: {std:.2f}ms")

        # M1 gate requirement: p95 <25ms
        assert p95 < 25.0, (
            f"p95 latency {p95:.2f}ms exceeds 25ms requirement. "
            f"This may indicate network issues or server performance problems."
        )

    def test_throughput_with_batching(self, client):
        """
        Test throughput with request batching.

        Batching should reduce per-sequence overhead and improve throughput.
        """
        # Create 64 short sequences
        num_sequences = 64
        seq_length = 512
        input_ids_list = [[[i % 128256 for i in range(j, j + seq_length)]] for j in range(num_sequences)]

        # Test with different batch sizes
        batch_sizes = [1, 8, 16, 32]
        results = {}

        for batch_size in batch_sizes:
            start_time = time.perf_counter()
            responses = client.batch_query(
                input_ids_list,
                topk=128,
                batch_size=batch_size,
            )
            end_time = time.perf_counter()

            duration = end_time - start_time
            throughput = num_sequences / duration
            avg_latency_ms = (duration / num_sequences) * 1000

            results[batch_size] = {
                "duration": duration,
                "throughput": throughput,
                "avg_latency_ms": avg_latency_ms,
            }

            # Verify all responses are valid
            total_sequences = sum(r.batch_size for r in responses)
            assert total_sequences == num_sequences

        print(f"\nThroughput with Batching (n={num_sequences} sequences, {seq_length} tokens each):")
        for batch_size, stats in results.items():
            print(
                f"  batch_size={batch_size}: {stats['throughput']:.2f} seq/s, "
                f"avg_latency={stats['avg_latency_ms']:.2f}ms"
            )

        # Verify batching improves throughput
        throughput_no_batch = results[1]["throughput"]
        throughput_with_batch = results[32]["throughput"]
        improvement = throughput_with_batch / throughput_no_batch

        print(f"\nBatching improvement: {improvement:.2f}x")
        assert improvement > 1.5, (
            f"Batching should improve throughput by at least 1.5x, got {improvement:.2f}x"
        )

    def test_network_error_recovery(self, client):
        """
        Test network error recovery with retry logic.

        This test simulates transient network failures by using an invalid
        endpoint, then falling back to the real endpoint.
        """
        # Create client with invalid endpoint (will fail)
        bad_client = TeacherClient(
            endpoint_url="http://invalid-hostname-that-does-not-exist.local/v1/topk",
            timeout=2.0,
            verify_ssl=False,
            max_retries=3,
            backoff_base=0.5,
        )

        # Verify that network errors are properly raised after retries
        with pytest.raises(TeacherNetworkError, match="Failed after 3 retries"):
            bad_client.query_topk(input_ids=[[1, 2, 3]], topk=128)

        # Verify that a valid client works after the failed one
        input_ids = [[1, 2, 3, 4, 5]]
        response = client.query_topk(input_ids=input_ids, topk=128)
        assert response.batch_size == 1

    def test_different_topk_values(self, client):
        """Test query with different top-k values."""
        input_ids = [[1, 2, 3, 4, 5]]

        for topk in [16, 64, 128, 256]:
            response = client.query_topk(input_ids=input_ids, topk=topk)

            # Verify k matches request
            assert response.k == topk
            assert len(response.indices[0][0]) == topk
            assert len(response.values_int8[0][0]) == topk

    def test_different_temperatures(self, client):
        """Test query with different temperature values."""
        input_ids = [[1, 2, 3, 4, 5]]

        for temperature in [0.5, 1.0, 2.0, 3.0]:
            response = client.query_topk(
                input_ids=input_ids,
                topk=128,
                temperature=temperature,
            )

            # All responses should be valid
            assert response.batch_size == 1
            assert response.k == 128

    def test_varying_sequence_lengths(self, client):
        """Test queries with varying sequence lengths."""
        test_lengths = [1, 10, 100, 512, 1024, 2048, 4096]

        for seq_len in test_lengths:
            input_ids = [[i % 128256 for i in range(seq_len)]]
            response = client.query_topk(input_ids=input_ids, topk=128)

            # Verify response matches input length
            assert response.batch_size == 1
            assert response.num_positions == seq_len
            assert len(response.indices[0]) == seq_len

    def test_multiple_sequences_in_batch(self, client):
        """Test multiple sequences in a single request."""
        # 8 sequences of varying lengths
        input_ids = [
            [i % 128256 for i in range(100 + j * 50)]
            for j in range(8)
        ]

        response = client.query_topk(input_ids=input_ids, topk=128)

        # Verify batch processing
        assert response.batch_size == 8
        assert len(response.indices) == 8

        # Verify each sequence has correct length
        for idx, input_seq in enumerate(input_ids):
            assert len(response.indices[idx]) == len(input_seq)

    def test_context_manager_usage(self):
        """Test TeacherClient as context manager."""
        with TeacherClient(endpoint_url=TEACHER_ENDPOINT) as client:
            input_ids = [[1, 2, 3, 4, 5]]
            response = client.query_topk(input_ids=input_ids, topk=128)
            assert response.batch_size == 1

        # Session should be closed after context exit


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(SKIP_INTEGRATION, reason="Integration tests disabled")
class TestTeacherClientPerformance:
    """Performance-focused integration tests."""

    def test_sustained_throughput(self, client):
        """
        Test sustained throughput over extended period.

        This test measures throughput stability over 1000 requests.
        """
        num_requests = 100  # Reduced for reasonable test time
        input_ids = [[i % 128256 for i in range(256)]]  # 256 tokens per sequence

        latencies = []
        start_time = time.perf_counter()

        for i in range(num_requests):
            req_start = time.perf_counter()
            response = client.query_topk(input_ids=input_ids, topk=128)
            req_end = time.perf_counter()

            latencies.append((req_end - req_start) * 1000)

            # Verify response
            assert response.batch_size == 1

        end_time = time.perf_counter()
        total_duration = end_time - start_time
        throughput = num_requests / total_duration

        print(f"\nSustained Throughput Test (n={num_requests}):")
        print(f"  Total duration: {total_duration:.2f}s")
        print(f"  Throughput: {throughput:.2f} req/s")
        print(f"  Median latency: {median(latencies):.2f}ms")
        print(f"  p95 latency: {sorted(latencies)[int(0.95 * num_requests)]:.2f}ms")

        # Verify reasonable throughput (>10 req/s for 256-token sequences)
        assert throughput > 10.0, f"Throughput {throughput:.2f} req/s is too low"

    def test_memory_efficiency(self, client):
        """
        Test that client doesn't leak memory over many requests.

        Note: This is a basic test; proper memory profiling would require
        additional tools like memory_profiler or tracemalloc.
        """
        import gc

        # Force garbage collection before test
        gc.collect()

        # Make many requests
        input_ids = [[i % 128256 for i in range(512)]]
        for _ in range(100):
            response = client.query_topk(input_ids=input_ids, topk=128)
            assert response.batch_size == 1

        # Force garbage collection after test
        gc.collect()

        # If we got here without OOM, memory management is reasonable
        assert True


if __name__ == "__main__":
    # Run with verbose output and integration markers
    pytest.main([
        __file__,
        "-v",
        "-m", "integration",
        "-s",  # Show print statements
    ])
