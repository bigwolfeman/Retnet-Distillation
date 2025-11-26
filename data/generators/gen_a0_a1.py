"""
A0-A1 band generator: Copy, compare, and 1-digit arithmetic.

Band A0: Copy and compare operations
- Copy: Given a number, output the same number
- Compare: Given two numbers, output which is larger

Band A1: Single-digit addition and subtraction
- Addition: 0-9 + 0-9 (no carry)
- Subtraction: 0-9 - 0-9 (result >= 0)

Per tasks.md T016: Seed-based determinism for reproducibility.
"""

import random
import sys
import uuid
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.schema import DataRecord
from model.dfa import validate_integer
from utils.number_wrapping import wrap_numbers, split_num


def generate_a0_copy(seed: int) -> DataRecord:
    """
    Generate A0 copy task: output the same number.

    Args:
        seed: Random seed for reproducibility

    Returns:
        DataRecord with copy task
    """
    random.seed(seed)

    # Single digit number (0-9)
    num = random.randint(0, 9)

    # Question: just the number (no wrapping for single digits)
    question = str(num)

    # Answer: same number
    answer = str(num)

    # Split answer (single digit, no spaces)
    answer_split = str(num)

    return DataRecord(
        id=str(uuid.uuid4()),
        band="A0",
        question=question,
        answer=answer,
        answer_split=answer_split,
        hints=None,  # Deprecated
        dual=None,  # No dual view for copy
        tool_supervision=None,
        verifier={"ok": validate_integer(answer)},
    )


def generate_a0_compare(seed: int) -> DataRecord:
    """
    Generate A0 compare task: output larger of two numbers.

    Args:
        seed: Random seed for reproducibility

    Returns:
        DataRecord with compare task
    """
    random.seed(seed)

    # Two single-digit numbers
    a = random.randint(0, 9)
    b = random.randint(0, 9)

    # Ensure they're different
    while a == b:
        b = random.randint(0, 9)

    # Question: "compare a and b" or just "a vs b"
    question = f"{a} vs {b}"

    # Answer: larger number
    answer = str(max(a, b))
    answer_split = answer

    return DataRecord(
        id=str(uuid.uuid4()),
        band="A0",
        question=question,
        answer=answer,
        answer_split=answer_split,
        hints=None,
        dual=None,
        tool_supervision=None,
        verifier={"ok": validate_integer(answer)},
    )


def generate_a1_addition(seed: int) -> DataRecord:
    """
    Generate A1 single-digit addition (no carry).

    Args:
        seed: Random seed for reproducibility

    Returns:
        DataRecord with addition task
    """
    random.seed(seed)

    # Two digits that don't produce carry
    a = random.randint(0, 9)
    b = random.randint(0, 9 - a)  # Ensure a + b <= 9

    result = a + b

    # Question: wrapped if multi-digit (but result is always single digit here)
    question_text = f"{a}+{b}"
    question = question_text  # Single digits, no wrapping

    # Answer
    answer = str(result)
    answer_split = str(result)  # Single digit

    return DataRecord(
        id=str(uuid.uuid4()),
        band="A1",
        question=question,
        answer=answer,
        answer_split=answer_split,
        hints=None,  # No carry for A1
        dual=None,
        tool_supervision=None,
        verifier={"ok": validate_integer(answer) and int(answer) == a + b},
    )


def generate_a1_subtraction(seed: int) -> DataRecord:
    """
    Generate A1 single-digit subtraction (no borrow).

    Args:
        seed: Random seed for reproducibility

    Returns:
        DataRecord with subtraction task
    """
    random.seed(seed)

    # Ensure a >= b (no borrow)
    a = random.randint(0, 9)
    b = random.randint(0, a)  # b <= a

    result = a - b

    # Question
    question_text = f"{a}-{b}"
    question = question_text

    # Answer
    answer = str(result)
    answer_split = str(result)

    return DataRecord(
        id=str(uuid.uuid4()),
        band="A1",
        question=question,
        answer=answer,
        answer_split=answer_split,
        hints=None,  # No borrow for A1
        dual=None,
        tool_supervision=None,
        verifier={"ok": validate_integer(answer) and int(answer) == a - b},
    )


def generate_a0_a1_batch(
    num_samples: int,
    seed_start: int = 1000,
    split: str = "train",
    band: str = None,
) -> List[DataRecord]:
    """
    Generate a batch of A0-A1 examples.

    Args:
        num_samples: Number of samples to generate
        seed_start: Starting seed value
        split: Dataset split (train/val/test)
        band: Specific band to generate (A0 or A1). If None, generates mixed.

    Returns:
        List of DataRecords
    """
    records = []

    # Determine task distribution based on band parameter
    if band == "A0":
        # A0 only: 80% copy, 20% compare
        task_distribution = [
            ("copy", 0.80),
            ("compare", 0.20),
        ]
    elif band == "A1":
        # A1 only: 50% addition, 50% subtraction
        task_distribution = [
            ("add", 0.50),
            ("sub", 0.50),
        ]
    else:
        # Mixed: A0: 40% copy, 10% compare; A1: 25% addition, 25% subtraction
        task_distribution = [
            ("copy", 0.40),
            ("compare", 0.10),
            ("add", 0.25),
            ("sub", 0.25),
        ]

    for i in range(num_samples):
        seed = seed_start + i

        # Choose task based on distribution
        random.seed(seed)
        rand_val = random.random()

        cumulative = 0.0
        task = task_distribution[0][0]  # Default to first task
        for task_name, prob in task_distribution:
            cumulative += prob
            if rand_val < cumulative:
                task = task_name
                break

        # Generate record
        if task == "copy":
            record = generate_a0_copy(seed)
        elif task == "compare":
            record = generate_a0_compare(seed)
        elif task == "add":
            record = generate_a1_addition(seed)
        elif task == "sub":
            record = generate_a1_subtraction(seed)

        records.append(record)

    return records


# Example usage and testing
if __name__ == "__main__":
    # Fix Unicode encoding on Windows
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=== A0-A1 Generator Test ===\n")

    # Generate small sample
    print("Generating 20 samples...")
    samples = generate_a0_a1_batch(num_samples=20, seed_start=1000)

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

    # Show statistics
    band_counts = {}
    for record in samples:
        band_counts[record.band] = band_counts.get(record.band, 0) + 1

    print("\nBand distribution:")
    for band, count in sorted(band_counts.items()):
        print(f"  {band}: {count} samples ({count/len(samples)*100:.0f}%)")

    print("\n✓ A0-A1 generator working!")
