"""
FORMAT band generator: Single-digit addition with teacher forcing.

This band is used ONLY for quick format alignment (~600 steps).
It uses teacher forcing to bootstrap formatting understanding.
After promotion, it is NEVER revisited (no replay, no review).

Per curriculum.yaml:
- Single-digit addition (0-9 + 0-9, allowing carry to 10-18)
- Answer wrapped as <ANS>result</ANS> during packing
- Teacher forcing enabled (only band with this flag)
- No replay/review after completion
"""

import random
import sys
import uuid
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.schema import DataRecord
from model.dfa import validate_integer


def generate_format_addition(seed: int) -> DataRecord:
    """
    Generate FORMAT single-digit addition.

    Simplest possible arithmetic to learn answer formatting:
    - Q: 3+5
    - A: 8 (wrapped as <ANS>8</ANS> during packing)

    Args:
        seed: Random seed for reproducibility

    Returns:
        DataRecord with single-digit addition
    """
    random.seed(seed)

    # Two single digits (0-9 + 0-9 = 0-18)
    a = random.randint(0, 9)
    b = random.randint(0, 9)
    result = a + b

    # Question: simple format, no number wrapping (single digits)
    question = f"{a}+{b}"

    # Answer: plain number (will be wrapped as <ANS>result</ANS> during packing)
    answer = str(result)
    answer_split = str(result) if result < 10 else f"{result // 10} {result % 10}"

    return DataRecord(
        id=str(uuid.uuid4()),
        band="FORMAT",
        question=question,
        answer=answer,
        answer_split=answer_split,
        hints=None,
        dual=None,
        tool_supervision=None,
        verifier={"ok": validate_integer(answer) and int(answer) == a + b},
    )


def generate_format_batch(
    num_samples: int,
    seed_start: int = 1000,
    split: str = "train",
    band: str = None,
) -> List[DataRecord]:
    """
    Generate a batch of FORMAT examples.

    Args:
        num_samples: Number of samples to generate
        seed_start: Starting seed value
        split: Dataset split (train/val/test)

    Returns:
        List of DataRecords
    """
    records = []

    for i in range(num_samples):
        seed = seed_start + i
        record = generate_format_addition(seed)
        records.append(record)

    return records


# Example usage and testing
if __name__ == "__main__":
    # Fix Unicode encoding on Windows
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=== FORMAT Band Generator Test ===\n")

    # Generate small sample
    print("Generating 20 samples...")
    samples = generate_format_batch(num_samples=20, seed_start=1000)

    print(f"Generated {len(samples)} samples\n")

    # Show first 10
    print("First 10 samples:")
    for i, record in enumerate(samples[:10]):
        print(f"\n{i+1}. Band: {record.band}")
        print(f"   Question: {record.question}")
        print(f"   Answer: {record.answer}")
        print(f"   Answer (split): {record.answer_split}")
        print(f"   Verifier: {record.verifier}")

    # Validate all pass
    all_pass = all(record.verifier["ok"] for record in samples)
    print(f"\n✓ All samples pass verifier: {all_pass}")

    # Show answer distribution
    answer_counts = {}
    for record in samples:
        ans = int(record.answer)
        answer_counts[ans] = answer_counts.get(ans, 0) + 1

    print("\nAnswer distribution (0-18):")
    for ans in sorted(answer_counts.keys()):
        print(f"  {ans}: {answer_counts[ans]} samples")

    print("\n✓ FORMAT band generator working!")
