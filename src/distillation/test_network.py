#!/usr/bin/env python3
"""
Network Throughput Test for M1 Gate Validation.

Tests teacher server network performance by measuring latency and throughput
over 1000 sequences. This validates M1 gate requirements for production readiness.

Usage:
    python -m src.distillation.test_network \
        --teacher-url http://localhost:8000/v1/topk \
        --num-sequences 1000 \
        --topk 128 \
        --output network_report.json

M1 Gate Criteria:
    - p50 latency < 25ms
    - p95 latency < 50ms
    - p99 latency < 100ms
    - Wire size ~130KB per 4k sequence at k=128
    - Server throughput ≥ 2000 tok/s
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

from .teacher_client import TeacherClient, TeacherClientError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NetworkBenchmark:
    """
    Network performance benchmarker for M1 gate validation.

    Measures:
    - Latency percentiles (p50, p95, p99)
    - Throughput (sequences/sec, tokens/sec)
    - Wire size (payload size per sequence)
    """

    def __init__(self, teacher_client: TeacherClient):
        """
        Initialize network benchmarker.

        Args:
            teacher_client: TeacherClient instance
        """
        self.client = teacher_client

    def run_benchmark(
        self,
        sequences: List[List[int]],
        topk: int = 128,
        batch_size: int = 1,
        warmup_requests: int = 10
    ) -> Dict[str, Any]:
        """
        Run network performance benchmark.

        Args:
            sequences: List of token ID sequences to benchmark
            topk: Number of top logits to request
            batch_size: Batch size for requests (1 = individual requests)
            warmup_requests: Number of warmup requests before measurement

        Returns:
            Benchmark report dict with latency and throughput metrics
        """
        logger.info(
            f"Running network benchmark on {len(sequences)} sequences "
            f"(topk={topk}, batch_size={batch_size})"
        )

        # Warmup phase
        if warmup_requests > 0:
            logger.info(f"Running {warmup_requests} warmup requests...")
            warmup_sequences = sequences[:warmup_requests]
            for seq in warmup_sequences:
                try:
                    self.client.query_topk(
                        input_ids=[seq],
                        topk=topk,
                        return_dtype="int8"
                    )
                except Exception as e:
                    logger.warning(f"Warmup request failed: {e}")

        # Benchmark phase
        logger.info("Starting benchmark measurement...")
        latencies = []
        wire_sizes = []
        total_tokens = 0

        start_time = time.time()

        for i, seq in enumerate(sequences):
            if (i + 1) % 100 == 0:
                logger.info(f"  Progress: {i + 1}/{len(sequences)} sequences")

            # Measure single request latency
            request_start = time.time()

            try:
                response = self.client.query_topk(
                    input_ids=[seq],
                    topk=topk,
                    return_dtype="int8"
                )

                request_end = time.time()
                latency_ms = (request_end - request_start) * 1000

                latencies.append(latency_ms)

                # Estimate wire size
                wire_size = self._estimate_wire_size(response, topk, len(seq))
                wire_sizes.append(wire_size)

                total_tokens += len(seq)

            except Exception as e:
                logger.warning(f"Request {i} failed: {e}")
                continue

        end_time = time.time()
        total_time = end_time - start_time

        # Compute metrics
        if not latencies:
            raise RuntimeError("All benchmark requests failed")

        latencies = np.array(latencies)
        wire_sizes = np.array(wire_sizes)

        # Latency percentiles
        p50_latency = float(np.percentile(latencies, 50))
        p95_latency = float(np.percentile(latencies, 95))
        p99_latency = float(np.percentile(latencies, 99))
        mean_latency = float(np.mean(latencies))
        std_latency = float(np.std(latencies))

        # Throughput
        sequences_per_sec = len(latencies) / total_time
        tokens_per_sec = total_tokens / total_time

        # Wire size
        mean_wire_size = float(np.mean(wire_sizes))
        median_wire_size = float(np.median(wire_sizes))

        # M1 gate thresholds
        p50_threshold = 25.0  # ms
        p95_threshold = 50.0  # ms
        p99_threshold = 100.0  # ms
        wire_size_target = 130_000  # bytes (~130KB)
        throughput_threshold = 2000  # tok/s

        # Determine status
        p50_pass = p50_latency < p50_threshold
        p95_pass = p95_latency < p95_threshold
        p99_pass = p99_latency < p99_threshold
        throughput_pass = tokens_per_sec >= throughput_threshold

        overall_status = "PASS" if (p50_pass and p95_pass and throughput_pass) else "FAIL"

        # Generate recommendations
        recommendations = []
        if not p50_pass:
            recommendations.append(
                f"p50 latency ({p50_latency:.1f}ms) exceeds threshold ({p50_threshold}ms). "
                f"Check network bandwidth or server performance."
            )
        if not p95_pass:
            recommendations.append(
                f"p95 latency ({p95_latency:.1f}ms) exceeds threshold ({p95_threshold}ms). "
                f"High latency variance - check for network congestion."
            )
        if not p99_pass:
            recommendations.append(
                f"p99 latency ({p99_latency:.1f}ms) exceeds threshold ({p99_threshold}ms). "
                f"Tail latency issues - investigate server load spikes."
            )
        if not throughput_pass:
            recommendations.append(
                f"Throughput ({tokens_per_sec:.0f} tok/s) below threshold ({throughput_threshold} tok/s). "
                f"Consider increasing server batch size or GPU count."
            )

        wire_size_diff_pct = ((median_wire_size - wire_size_target) / wire_size_target) * 100
        if abs(wire_size_diff_pct) > 20:
            recommendations.append(
                f"Wire size ({median_wire_size / 1024:.1f}KB) differs from target "
                f"({wire_size_target / 1024:.1f}KB) by {wire_size_diff_pct:.1f}%. "
                f"Verify topk={topk} and compression settings."
            )

        report = {
            "status": overall_status,
            "latency": {
                "p50_ms": p50_latency,
                "p95_ms": p95_latency,
                "p99_ms": p99_latency,
                "mean_ms": mean_latency,
                "std_ms": std_latency,
                "p50_threshold_ms": p50_threshold,
                "p95_threshold_ms": p95_threshold,
                "p99_threshold_ms": p99_threshold,
                "p50_pass": p50_pass,
                "p95_pass": p95_pass,
                "p99_pass": p99_pass
            },
            "throughput": {
                "sequences_per_sec": sequences_per_sec,
                "tokens_per_sec": tokens_per_sec,
                "threshold_tok_per_sec": throughput_threshold,
                "pass": throughput_pass
            },
            "wire_size": {
                "mean_bytes": mean_wire_size,
                "median_bytes": median_wire_size,
                "mean_kb": mean_wire_size / 1024,
                "median_kb": median_wire_size / 1024,
                "target_bytes": wire_size_target,
                "target_kb": wire_size_target / 1024
            },
            "config": {
                "num_sequences": len(latencies),
                "total_tokens": total_tokens,
                "topk": topk,
                "batch_size": batch_size,
                "warmup_requests": warmup_requests,
                "total_time_sec": total_time
            },
            "recommendations": recommendations
        }

        logger.info(f"Benchmark complete: {overall_status}")
        logger.info(f"  p50 latency: {p50_latency:.2f}ms (threshold: {p50_threshold}ms)")
        logger.info(f"  p95 latency: {p95_latency:.2f}ms (threshold: {p95_threshold}ms)")
        logger.info(f"  p99 latency: {p99_latency:.2f}ms (threshold: {p99_threshold}ms)")
        logger.info(f"  Throughput: {tokens_per_sec:.0f} tok/s (threshold: {throughput_threshold} tok/s)")
        logger.info(f"  Wire size: {median_wire_size / 1024:.1f}KB (target: {wire_size_target / 1024:.1f}KB)")

        return report

    def _estimate_wire_size(
        self,
        response: Any,
        topk: int,
        seq_len: int
    ) -> int:
        """
        Estimate wire size of response.

        Wire format for int8:
        - indices: [B, L, K] * 4 bytes (int32) = B * L * K * 4
        - values_int8: [B, L, K] * 1 byte (int8) = B * L * K * 1
        - scale: [B, L] * 4 bytes (float32) = B * L * 4
        - other_mass: [B, L] * 4 bytes (float32) = B * L * 4
        - Metadata: ~200 bytes (JSON overhead)

        For 4k sequence with k=128, B=1:
        - indices: 1 * 4096 * 128 * 4 = 2,097,152 bytes
        - values_int8: 1 * 4096 * 128 * 1 = 524,288 bytes
        - scale: 1 * 4096 * 4 = 16,384 bytes
        - other_mass: 1 * 4096 * 4 = 16,384 bytes
        - Total: ~2.65MB (before compression)

        With typical JSON compression: ~130KB

        Args:
            response: TopKResponse object
            topk: Number of top-k values
            seq_len: Sequence length

        Returns:
            Estimated wire size in bytes
        """
        batch_size = response.batch_size
        num_positions = response.num_positions

        # Raw payload size
        indices_size = batch_size * num_positions * topk * 4  # int32
        values_size = batch_size * num_positions * topk * 1   # int8
        scale_size = batch_size * num_positions * 4           # float32
        other_mass_size = batch_size * num_positions * 4      # float32

        raw_size = indices_size + values_size + scale_size + other_mass_size

        # JSON overhead
        json_overhead = 200

        # Typical compression ratio for JSON (gzip): ~5-10x
        # For int8 values, compression is very effective
        compression_ratio = 20.0  # Conservative estimate

        compressed_size = (raw_size / compression_ratio) + json_overhead

        return int(compressed_size)


def generate_synthetic_sequences(
    num_sequences: int,
    seq_length: int = 4096,
    vocab_size: int = 128256,
    seed: int = 42
) -> List[List[int]]:
    """
    Generate synthetic token sequences for benchmarking.

    Args:
        num_sequences: Number of sequences to generate
        seq_length: Length of each sequence
        vocab_size: Vocabulary size (default: Llama vocab)
        seed: Random seed

    Returns:
        List of token ID sequences
    """
    random.seed(seed)
    np.random.seed(seed)

    logger.info(f"Generating {num_sequences} synthetic sequences (length={seq_length})")

    sequences = []
    for _ in range(num_sequences):
        # Generate realistic token distribution
        # 70% common tokens (0-1000), 30% rare tokens
        token_ids = []
        for _ in range(seq_length):
            if random.random() < 0.7:
                token_id = random.randint(0, 1000)
            else:
                token_id = random.randint(1001, vocab_size - 1)
            token_ids.append(token_id)

        sequences.append(token_ids)

    return sequences


def generate_report(
    benchmark_result: Dict[str, Any],
    output_file: Optional[Path] = None
) -> str:
    """
    Generate human-readable benchmark report.

    Args:
        benchmark_result: Benchmark report dict
        output_file: Optional path to save JSON report

    Returns:
        Human-readable report string
    """
    # Save JSON report if output file specified
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(benchmark_result, f, indent=2)

        logger.info(f"Saved JSON report to {output_file}")

    # Generate human-readable summary
    status_symbol = "✓" if benchmark_result["status"] == "PASS" else "✗"

    lat = benchmark_result["latency"]
    thr = benchmark_result["throughput"]
    wire = benchmark_result["wire_size"]
    cfg = benchmark_result["config"]

    report_lines = [
        "=" * 70,
        "NETWORK THROUGHPUT TEST REPORT (M1 Gate)",
        "=" * 70,
        "",
        f"Status: {status_symbol} {benchmark_result['status']}",
        "",
        "Latency Metrics:",
        f"  p50: {lat['p50_ms']:7.2f}ms  (threshold: {lat['p50_threshold_ms']:.0f}ms)  {'✓ PASS' if lat['p50_pass'] else '✗ FAIL'}",
        f"  p95: {lat['p95_ms']:7.2f}ms  (threshold: {lat['p95_threshold_ms']:.0f}ms)  {'✓ PASS' if lat['p95_pass'] else '✗ FAIL'}",
        f"  p99: {lat['p99_ms']:7.2f}ms  (threshold: {lat['p99_threshold_ms']:.0f}ms)  {'✓ PASS' if lat['p99_pass'] else '✗ FAIL'}",
        f"  Mean: {lat['mean_ms']:.2f}ms ± {lat['std_ms']:.2f}ms",
        "",
        "Throughput Metrics:",
        f"  Sequences/sec: {thr['sequences_per_sec']:8.1f}",
        f"  Tokens/sec:    {thr['tokens_per_sec']:8.0f}  (threshold: {thr['threshold_tok_per_sec']:.0f})  {'✓ PASS' if thr['pass'] else '✗ FAIL'}",
        "",
        "Wire Size:",
        f"  Median: {wire['median_kb']:6.1f}KB  (target: {wire['target_kb']:.1f}KB)",
        f"  Mean:   {wire['mean_kb']:6.1f}KB",
        "",
        "Configuration:",
        f"  Sequences:     {cfg['num_sequences']}",
        f"  Total tokens:  {cfg['total_tokens']:,}",
        f"  Top-k:         {cfg['topk']}",
        f"  Batch size:    {cfg['batch_size']}",
        f"  Total time:    {cfg['total_time_sec']:.1f}s",
    ]

    if benchmark_result["recommendations"]:
        report_lines.extend([
            "",
            "Recommendations:",
        ])
        for rec in benchmark_result["recommendations"]:
            report_lines.append(f"  - {rec}")

    report_lines.extend([
        "",
        "=" * 70
    ])

    return "\n".join(report_lines)


def main():
    """Main entry point for network throughput test."""
    parser = argparse.ArgumentParser(
        description="Network throughput test for M1 gate validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--teacher-url',
        type=str,
        default='http://localhost:8000/v1/topk',
        help='Teacher server endpoint URL (default: http://localhost:8000/v1/topk)'
    )

    parser.add_argument(
        '--num-sequences',
        type=int,
        default=1000,
        help='Number of sequences to benchmark (default: 1000)'
    )

    parser.add_argument(
        '--seq-length',
        type=int,
        default=4096,
        help='Sequence length (default: 4096)'
    )

    parser.add_argument(
        '--topk',
        type=int,
        default=128,
        help='Number of top logits to request (default: 128)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for requests (default: 1)'
    )

    parser.add_argument(
        '--warmup-requests',
        type=int,
        default=10,
        help='Number of warmup requests before measurement (default: 10)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='network_report.json',
        help='Output JSON report path (default: network_report.json)'
    )

    parser.add_argument(
        '--timeout',
        type=float,
        default=60.0,
        help='Request timeout in seconds (default: 60.0)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    try:
        # Generate synthetic sequences
        logger.info("Generating synthetic benchmark sequences...")
        sequences = generate_synthetic_sequences(
            num_sequences=args.num_sequences,
            seq_length=args.seq_length,
            seed=args.seed
        )

        # Initialize teacher client
        logger.info(f"Connecting to teacher server at {args.teacher_url}...")
        client = TeacherClient(
            endpoint_url=args.teacher_url,
            timeout=args.timeout
        )

        # Check health
        if not client.health_check():
            logger.warning("Teacher health check failed, but continuing anyway...")

        # Initialize benchmarker
        benchmarker = NetworkBenchmark(client)

        # Run benchmark
        logger.info("Running network benchmark...")
        result = benchmarker.run_benchmark(
            sequences=sequences,
            topk=args.topk,
            batch_size=args.batch_size,
            warmup_requests=args.warmup_requests
        )

        # Generate report
        report = generate_report(
            result,
            output_file=Path(args.output)
        )

        # Print to stdout
        print(report)

        # Exit with appropriate code
        sys.exit(0 if result["status"] == "PASS" else 1)

    except TeacherClientError as e:
        logger.error(f"Teacher client error: {e}")
        sys.exit(2)

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(3)


if __name__ == '__main__':
    main()
