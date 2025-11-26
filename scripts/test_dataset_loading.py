#!/usr/bin/env python3
"""Test dataset loading with existing infrastructure.

Tests that the prepared JSONL datasets can be loaded correctly.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

def test_jsonl_format(file_path: Path) -> Dict:
    """Test JSONL file format and return statistics.

    Args:
        file_path: Path to JSONL file

    Returns:
        Dict with statistics
    """
    print(f"\nTesting: {file_path}")
    print("-" * 60)

    stats = {
        "total_examples": 0,
        "total_tokens": 0,
        "min_tokens": float('inf'),
        "max_tokens": 0,
        "domains": set(),
        "errors": []
    }

    try:
        with open(file_path) as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)

                    # Validate required fields
                    if "text" not in data:
                        stats["errors"].append(f"Line {line_num}: Missing 'text' field")
                        continue

                    if "domain" not in data:
                        stats["errors"].append(f"Line {line_num}: Missing 'domain' field")
                        continue

                    if "num_tokens" not in data:
                        stats["errors"].append(f"Line {line_num}: Missing 'num_tokens' field")
                        continue

                    # Collect statistics
                    stats["total_examples"] += 1
                    stats["total_tokens"] += data["num_tokens"]
                    stats["min_tokens"] = min(stats["min_tokens"], data["num_tokens"])
                    stats["max_tokens"] = max(stats["max_tokens"], data["num_tokens"])
                    stats["domains"].add(data["domain"])

                except json.JSONDecodeError as e:
                    stats["errors"].append(f"Line {line_num}: Invalid JSON - {e}")

    except FileNotFoundError:
        print(f"ERROR: File not found: {file_path}")
        return stats

    # Print statistics
    print(f"Total examples: {stats['total_examples']}")
    print(f"Total tokens: {stats['total_tokens']:,}")

    if stats['total_examples'] > 0:
        avg_tokens = stats['total_tokens'] / stats['total_examples']
        print(f"Average tokens/example: {avg_tokens:.1f}")
        print(f"Token range: {stats['min_tokens']} - {stats['max_tokens']}")

    print(f"Domains: {sorted(stats['domains'])}")

    if stats['errors']:
        print(f"\nErrors found: {len(stats['errors'])}")
        for error in stats['errors'][:5]:  # Show first 5 errors
            print(f"  {error}")
        if len(stats['errors']) > 5:
            print(f"  ... and {len(stats['errors']) - 5} more")
    else:
        print("No errors found")

    return stats


def test_sample_content(file_path: Path, num_samples: int = 3) -> None:
    """Display sample content from dataset.

    Args:
        file_path: Path to JSONL file
        num_samples: Number of samples to display
    """
    print(f"\nSample content from {file_path.name}:")
    print("=" * 60)

    try:
        with open(file_path) as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break

                data = json.loads(line)
                text_preview = data["text"][:200] + "..." if len(data["text"]) > 200 else data["text"]

                print(f"\nExample {i+1}:")
                print(f"  Domain: {data['domain']}")
                print(f"  Tokens: {data['num_tokens']}")
                print(f"  Text preview: {text_preview}")

    except Exception as e:
        print(f"ERROR: {e}")


def test_simple_dataloader(file_path: Path) -> None:
    """Test simple dataloader functionality.

    Args:
        file_path: Path to JSONL file
    """
    print(f"\nTesting dataloader for {file_path.name}:")
    print("-" * 60)

    try:
        # Simple iteration test
        examples = []
        with open(file_path) as f:
            for i, line in enumerate(f):
                if i >= 10:  # Load first 10
                    break
                examples.append(json.loads(line))

        print(f"Successfully loaded {len(examples)} examples")

        # Test shuffling (just verify data is dict)
        if examples:
            print(f"Example structure: {list(examples[0].keys())}")
            print(f"First example has {len(examples[0]['text'])} characters")

        print("Dataloader test: PASSED")

    except Exception as e:
        print(f"Dataloader test: FAILED - {e}")


def main():
    """Run all tests on test datasets."""
    print("="*60)
    print("Dataset Loading Tests")
    print("="*60)

    # Test dataset directory
    test_dir = Path("data/unlabeled/test")

    if not test_dir.exists():
        print(f"\nERROR: Test dataset directory not found: {test_dir}")
        print("Please run: python scripts/prepare_datasets.py --mode test")
        sys.exit(1)

    # Test each domain
    domains = ["text", "code", "math"]
    all_stats = {}

    for domain in domains:
        file_path = test_dir / f"{domain}.jsonl"

        if not file_path.exists():
            print(f"\nWARNING: {domain}.jsonl not found, skipping")
            continue

        # Run tests
        stats = test_jsonl_format(file_path)
        test_sample_content(file_path, num_samples=2)
        test_simple_dataloader(file_path)

        all_stats[domain] = stats

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    total_examples = sum(s["total_examples"] for s in all_stats.values())
    total_tokens = sum(s["total_tokens"] for s in all_stats.values())
    total_errors = sum(len(s["errors"]) for s in all_stats.values())

    print(f"Total examples: {total_examples}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total errors: {total_errors}")

    if total_errors == 0:
        print("\nAll tests PASSED!")
        return 0
    else:
        print(f"\nTests FAILED with {total_errors} errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())
