#!/usr/bin/env python3
"""
Test script to compare old vs new /v1/topk endpoint speeds.

This script benchmarks:
1. Old approach: vLLM prompt_logprobs (12+ seconds per sequence)
2. New approach: Custom /v1/topk endpoint (<100ms per sequence)

Expected speedup: 100x+

Usage:
    # Test with default server
    python scripts/test_fast_endpoint.py

    # Test with custom server
    python scripts/test_fast_endpoint.py --server http://192.168.0.71:8080

    # Test with more sequences
    python scripts/test_fast_endpoint.py --num-sequences 10
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import requests
from transformers import AutoTokenizer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.distillation.vllm_teacher_client import VLLMTeacherClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_test_sequences(tokenizer, num_sequences: int = 5, seq_len: int = 128) -> List[List[int]]:
    """Create test sequences for benchmarking."""
    logger.info(f"Creating {num_sequences} test sequences (length={seq_len})...")

    # Sample texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "To be or not to be, that is the question.",
        "In the beginning was the Word, and the Word was with God.",
        "Call me Ishmael. Some years ago—never mind how long precisely—",
        "It was the best of times, it was the worst of times.",
    ]

    sequences = []
    for i in range(num_sequences):
        text = texts[i % len(texts)]
        # Repeat text to reach desired length
        repeated_text = (text + " ") * (seq_len // len(text.split()) + 1)
        input_ids = tokenizer.encode(repeated_text, add_special_tokens=True)[:seq_len]
        sequences.append(input_ids)

    return sequences


def test_old_approach(
    server_url: str,
    api_key: str,
    sequences: List[List[int]],
    topk: int = 128,
) -> dict:
    """Test old approach using prompt_logprobs."""
    logger.info("=" * 70)
    logger.info("Testing OLD APPROACH: prompt_logprobs")
    logger.info("=" * 70)

    client = VLLMTeacherClient(
        base_url=server_url,
        model="meta-llama/Llama-3.2-1B-Instruct",
        api_key=api_key,
        timeout=120.0,  # Long timeout for slow approach
    )

    try:
        # Test single sequence first
        logger.info(f"Testing single sequence (length={len(sequences[0])})...")
        start = time.perf_counter()
        result = client.get_prompt_logprobs(input_ids=[sequences[0]], topk=topk)
        single_time = time.perf_counter() - start
        logger.info(f"Single sequence: {single_time:.2f}s")

        # Test batch (if more sequences available)
        if len(sequences) > 1:
            batch_size = min(4, len(sequences))
            logger.info(f"Testing batch of {batch_size} sequences...")
            start = time.perf_counter()
            result = client.get_prompt_logprobs(
                input_ids=sequences[:batch_size],
                topk=topk
            )
            batch_time = time.perf_counter() - start
            per_seq_time = batch_time / batch_size
            logger.info(f"Batch: {batch_time:.2f}s ({per_seq_time:.2f}s per sequence)")
        else:
            per_seq_time = single_time

        return {
            "approach": "old",
            "single_time": single_time,
            "per_sequence_time": per_seq_time,
            "success": True,
        }

    except Exception as e:
        logger.error(f"Old approach failed: {e}")
        return {
            "approach": "old",
            "single_time": None,
            "per_sequence_time": None,
            "success": False,
            "error": str(e),
        }
    finally:
        client.close()


def test_new_approach(
    server_url: str,
    api_key: str,
    sequences: List[List[int]],
    topk: int = 128,
) -> dict:
    """Test new approach using custom /v1/topk endpoint."""
    logger.info("=" * 70)
    logger.info("Testing NEW APPROACH: Custom /v1/topk endpoint")
    logger.info("=" * 70)

    endpoint = f"{server_url.rstrip('/')}/v1/topk"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        # Test single sequence first
        logger.info(f"Testing single sequence (length={len(sequences[0])})...")
        payload = {
            "input_ids": [sequences[0]],
            "topk": topk,
        }
        start = time.perf_counter()
        response = requests.post(endpoint, json=payload, headers=headers, timeout=30.0)
        single_time = time.perf_counter() - start

        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text}")

        logger.info(f"Single sequence: {single_time:.3f}s ({single_time*1000:.1f}ms)")

        # Verify response format
        data = response.json()
        required_fields = ["indices", "values_int8", "scale", "other_mass"]
        for field in required_fields:
            if field not in data:
                raise Exception(f"Response missing required field: {field}")

        # Test batch (if more sequences available)
        if len(sequences) > 1:
            batch_size = min(8, len(sequences))  # Can handle larger batches with fast endpoint
            logger.info(f"Testing batch of {batch_size} sequences...")
            payload = {
                "input_ids": sequences[:batch_size],
                "topk": topk,
            }
            start = time.perf_counter()
            response = requests.post(endpoint, json=payload, headers=headers, timeout=30.0)
            batch_time = time.perf_counter() - start

            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}: {response.text}")

            per_seq_time = batch_time / batch_size
            logger.info(f"Batch: {batch_time:.3f}s ({per_seq_time*1000:.1f}ms per sequence)")
        else:
            per_seq_time = single_time

        return {
            "approach": "new",
            "single_time": single_time,
            "per_sequence_time": per_seq_time,
            "success": True,
        }

    except requests.exceptions.ConnectionError:
        logger.error(f"Cannot connect to {endpoint}")
        logger.error("Make sure vLLM server is running and custom endpoint is installed.")
        logger.error("See INSTALL_CUSTOM_ENDPOINT_WSL.md for installation instructions.")
        return {
            "approach": "new",
            "single_time": None,
            "per_sequence_time": None,
            "success": False,
            "error": "Connection failed - server not reachable",
        }

    except Exception as e:
        logger.error(f"New approach failed: {e}")
        return {
            "approach": "new",
            "single_time": None,
            "per_sequence_time": None,
            "success": False,
            "error": str(e),
        }


def verify_numerical_correctness(
    server_url: str,
    api_key: str,
    sequence: List[int],
    topk: int = 128,
) -> dict:
    """
    Verify that new approach returns numerically correct results.

    Compares top-k indices from both approaches.
    """
    logger.info("=" * 70)
    logger.info("Testing NUMERICAL CORRECTNESS")
    logger.info("=" * 70)

    try:
        # Get results from old approach
        logger.info("Fetching results from old approach (prompt_logprobs)...")
        old_client = VLLMTeacherClient(
            base_url=server_url,
            model="meta-llama/Llama-3.2-1B-Instruct",
            api_key=api_key,
            timeout=120.0,
        )
        old_results = old_client.get_prompt_logprobs(input_ids=[sequence], topk=topk)
        old_indices = old_results[0]["indices"]
        old_client.close()

        # Get results from new approach
        logger.info("Fetching results from new approach (/v1/topk)...")
        endpoint = f"{server_url.rstrip('/')}/v1/topk"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "input_ids": [sequence],
            "topk": topk,
        }
        response = requests.post(endpoint, json=payload, headers=headers, timeout=30.0)
        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text}")

        new_data = response.json()
        new_indices = new_data["indices"][0]  # First sequence

        # Compare indices at each position
        logger.info(f"Comparing {len(old_indices)} positions...")

        total_positions = len(old_indices)
        matching_positions = 0
        top1_matches = 0
        top5_matches = 0

        for pos_idx, (old_pos, new_pos) in enumerate(zip(old_indices, new_indices)):
            if not old_pos or not new_pos:
                # Skip empty positions (e.g., BOS)
                continue

            # Check if indices match exactly
            if old_pos == new_pos:
                matching_positions += 1

            # Check if top-1 matches
            if old_pos[0] == new_pos[0]:
                top1_matches += 1

            # Check if top-5 overlap
            old_top5 = set(old_pos[:5])
            new_top5 = set(new_pos[:5])
            if old_top5 == new_top5:
                top5_matches += 1

        match_rate = matching_positions / total_positions * 100
        top1_rate = top1_matches / total_positions * 100
        top5_rate = top5_matches / total_positions * 100

        logger.info(f"Exact match rate: {match_rate:.1f}%")
        logger.info(f"Top-1 match rate: {top1_rate:.1f}%")
        logger.info(f"Top-5 match rate: {top5_rate:.1f}%")

        # Determine if verification passed
        # Top-1 should always match (same model, same inputs)
        passed = top1_rate >= 95.0  # Allow 5% tolerance for edge cases

        if passed:
            logger.info("PASSED: Numerical correctness verified")
        else:
            logger.warning("FAILED: Significant differences detected")

        return {
            "success": True,
            "passed": passed,
            "exact_match_rate": match_rate,
            "top1_match_rate": top1_rate,
            "top5_match_rate": top5_rate,
        }

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return {
            "success": False,
            "passed": False,
            "error": str(e),
        }


def print_summary(old_results: dict, new_results: dict, verify_results: dict):
    """Print summary of benchmark results."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    if old_results["success"] and new_results["success"]:
        old_time = old_results["per_sequence_time"]
        new_time = new_results["per_sequence_time"]
        speedup = old_time / new_time

        print(f"\nOld approach (prompt_logprobs):")
        print(f"  Time per sequence: {old_time:.3f}s ({old_time*1000:.0f}ms)")

        print(f"\nNew approach (custom /v1/topk):")
        print(f"  Time per sequence: {new_time:.3f}s ({new_time*1000:.0f}ms)")

        print(f"\nSpeedup: {speedup:.1f}x")

        if speedup >= 100:
            print("STATUS: SUCCESS - Achieved 100x+ speedup target!")
        elif speedup >= 50:
            print("STATUS: GOOD - Significant speedup achieved")
        elif speedup >= 10:
            print("STATUS: OK - Moderate speedup")
        else:
            print("STATUS: POOR - Speedup below expectations")
    else:
        if not old_results["success"]:
            print(f"\nOld approach FAILED: {old_results.get('error', 'Unknown error')}")
        if not new_results["success"]:
            print(f"\nNew approach FAILED: {new_results.get('error', 'Unknown error')}")

    if verify_results["success"] and verify_results["passed"]:
        print(f"\nNumerical correctness: PASSED")
        print(f"  Top-1 match rate: {verify_results['top1_match_rate']:.1f}%")
    elif verify_results["success"]:
        print(f"\nNumerical correctness: FAILED")
        print(f"  Top-1 match rate: {verify_results['top1_match_rate']:.1f}%")
    else:
        print(f"\nNumerical correctness: NOT TESTED")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Test and compare old vs new /v1/topk endpoint speeds"
    )
    parser.add_argument(
        "--server",
        type=str,
        default="http://192.168.0.71:8080",
        help="vLLM server URL (default: http://192.168.0.71:8080)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="token-abc123",
        help="API key (default: token-abc123)",
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=5,
        help="Number of test sequences (default: 5)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Sequence length in tokens (default: 128)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=128,
        help="Top-k value (default: 128)",
    )
    parser.add_argument(
        "--skip-old",
        action="store_true",
        help="Skip testing old approach (if you know it's slow)",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip numerical correctness verification",
    )

    args = parser.parse_args()

    # Print configuration
    print("=" * 70)
    print("FAST ENDPOINT BENCHMARK")
    print("=" * 70)
    print(f"Server: {args.server}")
    print(f"Sequences: {args.num_sequences}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Top-k: {args.topk}")
    print("=" * 70)
    print()

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    # Create test sequences
    sequences = create_test_sequences(tokenizer, args.num_sequences, args.seq_len)

    # Test old approach
    if not args.skip_old:
        old_results = test_old_approach(args.server, args.api_key, sequences, args.topk)
    else:
        logger.info("Skipping old approach test (--skip-old)")
        old_results = {"success": False, "skipped": True}

    # Test new approach
    new_results = test_new_approach(args.server, args.api_key, sequences, args.topk)

    # Verify numerical correctness
    if not args.skip_verification and new_results["success"]:
        verify_results = verify_numerical_correctness(
            args.server, args.api_key, sequences[0], args.topk
        )
    else:
        if args.skip_verification:
            logger.info("Skipping numerical verification (--skip-verification)")
        verify_results = {"success": False, "skipped": True}

    # Print summary
    print_summary(old_results, new_results, verify_results)


if __name__ == "__main__":
    main()
