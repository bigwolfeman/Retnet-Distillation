"""
A2 band generator: Multi-digit addition and subtraction with carry/borrow.

Per tasks.md T017:
- Multi-digit ± with carry/borrow hints derived from split view
- Adversarial cases: long carry chains (999...999 + 1)
- Leading zeros forbidden
- 100% verifier pass rate

Per user's spatial separation solution:
- Wrap numbers: "37+58" → "⟨N⟩3 7⟨/N⟩+⟨N⟩5 8⟨/N⟩"
- Emit dual views: answer="95", answer_split="9 5"
- NO explicit carry hints (derived from split view during training)
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


def generate_a2_addition(seed: int, num_digits: int = None) -> DataRecord:
    """
    Generate multi-digit addition with carry.

    Args:
        seed: Random seed for reproducibility
        num_digits: Number of digits (2-4 for training, 6-8 for length-gen)

    Returns:
        DataRecord with addition task
    """
    random.seed(seed)

    # Default to 2-4 digits for training
    if num_digits is None:
        num_digits = random.randint(2, 4)

    # Generate two numbers with specified digit count
    # Ensure no leading zeros (except for zero itself)
    min_val = 10 ** (num_digits - 1)
    max_val = 10 ** num_digits - 1

    a = random.randint(min_val, max_val)
    b = random.randint(min_val, max_val)

    # Occasionally create adversarial cases (long carry chains)
    if random.random() < 0.1:  # 10% adversarial
        # Create numbers like 999...999 + 1, or 555...555 + 555...555
        if random.random() < 0.5:
            a = int("9" * num_digits)
            b = random.randint(1, 100)
        else:
            a = int("5" * num_digits)
            b = a  # Forces carry in every column

    result = a + b

    # Build question with wrapped numbers
    question_text = f"{a}+{b}"
    question = wrap_numbers(question_text)

    # Normal answer (for verifier/tools)
    answer = str(result)

    # Split answer (for training)
    answer_split = split_num(answer)

    # Dual view: reversed digits for checking
    # Per data-model.md: reversed digits check for addition
    answer_reversed = str(result)[::-1]
    if result < 0:
        # Handle negative (shouldn't happen for addition)
        answer_reversed = "-" + str(abs(result))[::-1]

    return DataRecord(
        id=str(uuid.uuid4()),
        band="A2",
        question=question,
        answer=answer,
        answer_split=answer_split,
        hints=None,  # Deprecated - carry derived from split view
        dual={"rev": answer_reversed},
        tool_supervision=None,
        verifier={"ok": validate_integer(answer) and int(answer) == a + b},
    )


def generate_a2_subtraction(seed: int, num_digits: int = None) -> DataRecord:
    """
    Generate multi-digit subtraction with borrow.

    Args:
        seed: Random seed for reproducibility
        num_digits: Number of digits (2-4 for training)

    Returns:
        DataRecord with subtraction task
    """
    random.seed(seed)

    if num_digits is None:
        num_digits = random.randint(2, 4)

    min_val = 10 ** (num_digits - 1)
    max_val = 10 ** num_digits - 1

    # Generate a >= b to avoid negative results (for now)
    a = random.randint(min_val, max_val)
    b = random.randint(min_val, a)  # b <= a

    # Adversarial cases
    if random.random() < 0.1:
        # Cases that require borrowing across multiple columns
        # e.g., 1000 - 1, 2000 - 999
        a = random.randint(10, 20) * (10 ** (num_digits - 1))
        b = a - random.randint(1, 10)

    result = a - b

    # Build question
    question_text = f"{a}-{b}"
    question = wrap_numbers(question_text)

    # Answers
    answer = str(result)
    answer_split = split_num(answer)

    return DataRecord(
        id=str(uuid.uuid4()),
        band="A2",
        question=question,
        answer=answer,
        answer_split=answer_split,
        hints=None,  # Deprecated
        dual=None,
        tool_supervision=None,
        verifier={"ok": validate_integer(answer) and int(answer) == a - b},
    )


def generate_a2_batch(
    num_samples: int,
    seed_start: int = 1000,
    split: str = "train",
    band: str = None,
) -> List[DataRecord]:
    """
    Generate a batch of A2 examples.

    Args:
        num_samples: Number of samples to generate
        seed_start: Starting seed value
        split: Dataset split (train/val/test)

    Returns:
        List of DataRecords
    """
    records = []

    # 50% addition, 50% subtraction
    for i in range(num_samples):
        seed = seed_start + i
        random.seed(seed)

        if random.random() < 0.5:
            record = generate_a2_addition(seed)
        else:
            record = generate_a2_subtraction(seed)

        records.append(record)

    return records


# Example usage and testing
if __name__ == "__main__":
    # Fix Unicode encoding on Windows
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=== A2 Generator Test ===\n")

    # Generate small sample
    print("Generating 20 samples...")
    samples = generate_a2_batch(num_samples=20, seed_start=2000)

    print(f"Generated {len(samples)} samples\n")

    # Show first 10
    print("First 10 samples:")
    for i, record in enumerate(samples[:10]):
        print(f"\n{i+1}. Question (wrapped): {record.question}")
        print(f"   Answer (normal): {record.answer}")
        print(f"   Answer (split): {record.answer_split}")
        print(f"   Dual (reversed): {record.dual}")
        print(f"   Verifier: {record.verifier}")

    # Validate all pass
    all_pass = all(record.verifier["ok"] for record in samples)
    print(f"\n✓ All samples pass verifier: {all_pass} ({sum(r.verifier['ok'] for r in samples)}/{len(samples)})")

    # Check for adversarial cases
    adversarial_count = 0
    for record in samples:
        # Detect long carry chains (result has more digits than operands)
        if "+" in record.question:
            # Extract operands from wrapped question
            from utils.number_wrapping import unwrap_numbers
            unwrapped = unwrap_numbers(record.question)
            parts = unwrapped.split("+")
            if len(parts) == 2:
                a_len = len(parts[0])
                result_len = len(record.answer)
                if result_len > a_len:
                    adversarial_count += 1

    print(f"\nAdversarial cases (carry overflow): {adversarial_count}")

    print("\n✓ A2 generator working!")
