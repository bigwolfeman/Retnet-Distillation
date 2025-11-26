"""
Unit tests for network throughput test script.

Tests:
- NetworkBenchmark functionality
- Synthetic sequence generation
- Report generation
- Wire size estimation
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch
import time

import numpy as np
import pytest

from src.distillation.test_network import (
    NetworkBenchmark,
    generate_synthetic_sequences,
    generate_report
)
from src.distillation.schemas import TopKResponse


class TestSyntheticSequenceGeneration:
    """Test synthetic sequence generation."""

    def test_basic_generation(self):
        """Test basic synthetic sequence generation."""
        sequences = generate_synthetic_sequences(
            num_sequences=10,
            seq_length=100,
            seed=42
        )

        assert len(sequences) == 10
        assert all(len(seq) == 100 for seq in sequences)
        assert all(isinstance(token_id, int) for seq in sequences for token_id in seq)

    def test_reproducibility(self):
        """Test that generation is reproducible with same seed."""
        seq1 = generate_synthetic_sequences(50, 100, seed=42)
        seq2 = generate_synthetic_sequences(50, 100, seed=42)

        assert seq1 == seq2

    def test_different_seeds(self):
        """Test that different seeds produce different sequences."""
        seq1 = generate_synthetic_sequences(10, 100, seed=42)
        seq2 = generate_synthetic_sequences(10, 100, seed=123)

        assert seq1 != seq2

    def test_vocab_bounds(self):
        """Test that generated tokens are within vocab bounds."""
        vocab_size = 128256
        sequences = generate_synthetic_sequences(10, 100, vocab_size=vocab_size, seed=42)

        for seq in sequences:
            assert all(0 <= token_id < vocab_size for token_id in seq)


class TestNetworkBenchmark:
    """Test NetworkBenchmark class."""

    def test_wire_size_estimation(self):
        """Test wire size estimation."""
        mock_client = Mock()
        benchmarker = NetworkBenchmark(mock_client)

        # Create mock response
        B, L, K = 1, 4096, 128

        response = TopKResponse(
            indices=[[[i for i in range(K)] for _ in range(L)]],
            values_int8=[[[127 - i for i in range(K)] for _ in range(L)]],
            scale=[[0.01 for _ in range(L)]],
            other_mass=[[0.05 for _ in range(L)]],
            batch_size=B,
            num_positions=L,
            k=K
        )

        wire_size = benchmarker._estimate_wire_size(response, K, L)

        # Should be approximately 130KB (130,000 bytes)
        assert isinstance(wire_size, int)
        assert wire_size > 0
        # Allow wide range due to compression estimation
        assert 50_000 < wire_size < 500_000

    def test_benchmark_run_mock(self):
        """Test benchmark run with mocked client."""
        mock_client = Mock()

        # Mock responses with realistic latency
        def mock_query(*args, **kwargs):
            # Simulate network latency
            time.sleep(0.01)  # 10ms

            B, L, K = 1, 100, 128
            return TopKResponse(
                indices=[[[i for i in range(K)] for _ in range(L)]],
                values_int8=[[[127 - i for i in range(K)] for _ in range(L)]],
                scale=[[0.01 for _ in range(L)]],
                other_mass=[[0.05 for _ in range(L)]],
                batch_size=B,
                num_positions=L,
                k=K
            )

        mock_client.query_topk = mock_query

        benchmarker = NetworkBenchmark(mock_client)

        # Create test sequences
        sequences = generate_synthetic_sequences(20, 100, seed=42)

        # Run benchmark (with no warmup for speed)
        result = benchmarker.run_benchmark(
            sequences=sequences,
            topk=128,
            warmup_requests=0
        )

        # Verify report structure
        assert "status" in result
        assert "latency" in result
        assert "throughput" in result
        assert "wire_size" in result
        assert "config" in result
        assert "recommendations" in result

        # Verify latency metrics
        lat = result["latency"]
        assert "p50_ms" in lat
        assert "p95_ms" in lat
        assert "p99_ms" in lat
        assert "mean_ms" in lat
        assert "std_ms" in lat

        # Latency should be roughly 10ms (our mock delay)
        assert 5 < lat["p50_ms"] < 50  # Allow wide range

        # Verify throughput metrics
        thr = result["throughput"]
        assert "sequences_per_sec" in thr
        assert "tokens_per_sec" in thr
        assert thr["sequences_per_sec"] > 0
        assert thr["tokens_per_sec"] > 0

        # Verify wire size
        wire = result["wire_size"]
        assert "mean_bytes" in wire
        assert "median_bytes" in wire
        assert wire["mean_bytes"] > 0

        # Verify config
        cfg = result["config"]
        assert cfg["num_sequences"] == 20
        assert cfg["topk"] == 128

    def test_benchmark_handles_failures(self):
        """Test that benchmark handles request failures gracefully."""
        mock_client = Mock()

        # Mock client that fails sometimes
        call_count = [0]

        def mock_query(*args, **kwargs):
            call_count[0] += 1
            # Fail every 3rd request
            if call_count[0] % 3 == 0:
                raise Exception("Mock network error")

            B, L, K = 1, 100, 128
            return TopKResponse(
                indices=[[[i for i in range(K)] for _ in range(L)]],
                values_int8=[[[127 - i for i in range(K)] for _ in range(L)]],
                scale=[[0.01 for _ in range(L)]],
                other_mass=[[0.05 for _ in range(L)]],
                batch_size=B,
                num_positions=L,
                k=K
            )

        mock_client.query_topk = mock_query

        benchmarker = NetworkBenchmark(mock_client)

        # Create test sequences
        sequences = generate_synthetic_sequences(10, 100, seed=42)

        # Run benchmark - should succeed despite some failures
        result = benchmarker.run_benchmark(
            sequences=sequences,
            topk=128,
            warmup_requests=0
        )

        # Should have some successful requests
        assert result["config"]["num_sequences"] > 0
        assert result["latency"]["p50_ms"] > 0

    def test_latency_percentiles(self):
        """Test that latency percentiles are computed correctly."""
        mock_client = Mock()

        # Mock client with controlled latencies
        latencies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # ms
        latency_idx = [0]

        def mock_query(*args, **kwargs):
            # Use different latency for each call
            idx = latency_idx[0] % len(latencies)
            latency_idx[0] += 1
            time.sleep(latencies[idx] / 1000.0)

            B, L, K = 1, 100, 128
            return TopKResponse(
                indices=[[[i for i in range(K)] for _ in range(L)]],
                values_int8=[[[127 - i for i in range(K)] for _ in range(L)]],
                scale=[[0.01 for _ in range(L)]],
                other_mass=[[0.05 for _ in range(L)]],
                batch_size=B,
                num_positions=L,
                k=K
            )

        mock_client.query_topk = mock_query

        benchmarker = NetworkBenchmark(mock_client)
        sequences = generate_synthetic_sequences(10, 100, seed=42)

        result = benchmarker.run_benchmark(sequences, topk=128, warmup_requests=0)

        # p50 should be around 50-60ms
        # p95 should be around 90-100ms
        # p99 should be around 100ms
        lat = result["latency"]
        assert 40 < lat["p50_ms"] < 70
        assert 85 < lat["p95_ms"] < 110
        assert 95 < lat["p99_ms"] < 110


class TestReportGeneration:
    """Test report generation functions."""

    def test_generate_report_pass(self, tmp_path):
        """Test report generation for passing benchmark."""
        benchmark_result = {
            "status": "PASS",
            "latency": {
                "p50_ms": 15.5,
                "p95_ms": 30.2,
                "p99_ms": 45.8,
                "mean_ms": 18.3,
                "std_ms": 5.2,
                "p50_threshold_ms": 25.0,
                "p95_threshold_ms": 50.0,
                "p99_threshold_ms": 100.0,
                "p50_pass": True,
                "p95_pass": True,
                "p99_pass": True
            },
            "throughput": {
                "sequences_per_sec": 50.5,
                "tokens_per_sec": 206848.0,
                "threshold_tok_per_sec": 2000.0,
                "pass": True
            },
            "wire_size": {
                "mean_bytes": 135000,
                "median_bytes": 134500,
                "mean_kb": 131.8,
                "median_kb": 131.3,
                "target_bytes": 130000,
                "target_kb": 126.9
            },
            "config": {
                "num_sequences": 1000,
                "total_tokens": 4096000,
                "topk": 128,
                "batch_size": 1,
                "warmup_requests": 10,
                "total_time_sec": 19.8
            },
            "recommendations": []
        }

        output_file = tmp_path / "report.json"

        report = generate_report(benchmark_result, output_file)

        # Verify JSON file was created
        assert output_file.exists()

        # Verify JSON content
        with open(output_file) as f:
            saved_data = json.load(f)

        assert saved_data == benchmark_result

        # Verify human-readable report
        assert "PASS" in report
        assert "✓" in report
        assert "15.50ms" in report
        assert "206848" in report

    def test_generate_report_fail(self):
        """Test report generation for failing benchmark."""
        benchmark_result = {
            "status": "FAIL",
            "latency": {
                "p50_ms": 35.5,
                "p95_ms": 75.2,
                "p99_ms": 120.8,
                "mean_ms": 42.3,
                "std_ms": 15.2,
                "p50_threshold_ms": 25.0,
                "p95_threshold_ms": 50.0,
                "p99_threshold_ms": 100.0,
                "p50_pass": False,
                "p95_pass": False,
                "p99_pass": False
            },
            "throughput": {
                "sequences_per_sec": 20.5,
                "tokens_per_sec": 1500.0,
                "threshold_tok_per_sec": 2000.0,
                "pass": False
            },
            "wire_size": {
                "mean_bytes": 135000,
                "median_bytes": 134500,
                "mean_kb": 131.8,
                "median_kb": 131.3,
                "target_bytes": 130000,
                "target_kb": 126.9
            },
            "config": {
                "num_sequences": 1000,
                "total_tokens": 4096000,
                "topk": 128,
                "batch_size": 1,
                "warmup_requests": 10,
                "total_time_sec": 48.8
            },
            "recommendations": [
                "p50 latency (35.5ms) exceeds threshold (25.0ms). Check network bandwidth or server performance.",
                "Throughput (1500 tok/s) below threshold (2000 tok/s). Consider increasing server batch size or GPU count."
            ]
        }

        report = generate_report(benchmark_result)

        assert "FAIL" in report
        assert "✗" in report
        assert "Recommendations:" in report
        assert "network bandwidth" in report


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
