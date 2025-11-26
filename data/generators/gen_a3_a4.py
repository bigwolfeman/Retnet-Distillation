"""
A3-A4 band generator: Multiplication.

A3: Single-digit × multi-digit (e.g., 7 × 234)
A4: Multi-digit × multi-digit (e.g., 37 × 58)

Per tasks.md T018:
- Multiplication with partial products grid hints (derived from split view)
- 2D carry bitstrings (derived during training)
- Wrapped numbers in questions
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
from utils.number_wrapping import wrap_numbers, split_num


def generate_a3_single_digit_mult(seed: int) -> DataRecord:
    """
    Generate A3: single-digit × multi-digit multiplication.

    Args:
        seed: Random seed

    Returns:
        DataRecord with single-digit multiplication
    """
    random.seed(seed)

    # Single digit (2-9)
    a = random.randint(2, 9)

    # Multi-digit number (2-3 digits)
    num_digits = random.randint(2, 3)
    min_val = 10 ** (num_digits - 1)
    max_val = 10 ** num_digits - 1
    b = random.randint(min_val, max_val)

    result = a * b

    # Build question
    question_text = f"{a}*{b}"
    question = wrap_numbers(question_text)

    # Answers
    answer = str(result)
    answer_split = split_num(answer)

    return DataRecord(
        id=str(uuid.uuid4()),
        band="A3",
        question=question,
        answer=answer,
        answer_split=answer_split,
        hints=None,  # Partial products derived from split view
        dual=None,
        tool_supervision=None,
        verifier={"ok": validate_integer(answer) and int(answer) == a * b},
    )


def generate_a4_multi_digit_mult(seed: int) -> DataRecord:
    """
    Generate A4: multi-digit × multi-digit multiplication.

    Args:
        seed: Random seed

    Returns:
        DataRecord with multi-digit multiplication
    """
    random.seed(seed)

    # Two multi-digit numbers (2-3 digits each)
    num_digits_a = random.randint(2, 3)
    num_digits_b = random.randint(2, 3)

    min_val_a = 10 ** (num_digits_a - 1)
    max_val_a = 10 ** num_digits_a - 1
    a = random.randint(min_val_a, max_val_a)

    min_val_b = 10 ** (num_digits_b - 1)
    max_val_b = 10 ** num_digits_b - 1
    b = random.randint(min_val_b, max_val_b)

    result = a * b

    # Build question
    question_text = f"{a}*{b}"
    question = wrap_numbers(question_text)

    # Answers
    answer = str(result)
    answer_split = split_num(answer)

    return DataRecord(
        id=str(uuid.uuid4()),
        band="A4",
        question=question,
        answer=answer,
        answer_split=answer_split,
        hints=None,  # 2D carry grid derived from split view
        dual=None,
        tool_supervision=None,
        verifier={"ok": validate_integer(answer) and int(answer) == a * b},
    )


def generate_a3_a4_batch(
    num_samples: int,
    seed_start: int = 1000,
    split: str = "train",
    band: str = None,
) -> List[DataRecord]:
    """
    Generate a batch of A3-A4 examples.

    Args:
        num_samples: Number of samples
        seed_start: Starting seed
        split: Dataset split
        band: Specific band to generate (A3 or A4). If None, generates mixed.

    Returns:
        List of DataRecords
    """
    records = []

    for i in range(num_samples):
        seed = seed_start + i
        random.seed(seed)

        # Determine which type to generate based on band parameter
        if band == "A3":
            # A3 only: single-digit multiplication
            record = generate_a3_single_digit_mult(seed)
        elif band == "A4":
            # A4 only: multi-digit multiplication
            record = generate_a4_multi_digit_mult(seed)
        else:
            # Mixed: 30% A3 (single-digit), 70% A4 (multi-digit)
            if random.random() < 0.3:
                record = generate_a3_single_digit_mult(seed)
            else:
                record = generate_a4_multi_digit_mult(seed)

        records.append(record)

    return records


if __name__ == "__main__":
    # Fix Unicode encoding on Windows
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=== A3-A4 Generator Test ===\n")

    samples = generate_a3_a4_batch(num_samples=10, seed_start=3000)
    print(f"Generated {len(samples)} samples\n")

    for i, record in enumerate(samples):
        print(f"{i+1}. [{record.band}] {record.question} = {record.answer}")
        print(f"   Split: {record.answer_split}")
        print(f"   Verifier: {record.verifier}")

    all_pass = all(r.verifier["ok"] for r in samples)
    print(f"\n✓ All pass: {all_pass} ({sum(r.verifier['ok'] for r in samples)}/{len(samples)})")
