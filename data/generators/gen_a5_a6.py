"""
A5-A6 band generator: Division with remainder.

A5: Multi-digit ÷ single-digit
A6: Multi-digit ÷ multi-digit

Per tasks.md T019: Division with long division trace
Per data-model.md verifier: dividend == divisor*quotient + remainder AND remainder < divisor
"""

import random
import sys
import uuid
from pathlib import Path
from typing import List

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.schema import DataRecord
from model.dfa import validate_division, validate_integer
from utils.number_wrapping import wrap_numbers, split_num


def generate_a5_division(seed: int) -> DataRecord:
    """A5: Multi-digit ÷ single-digit."""
    random.seed(seed)

    divisor = random.randint(2, 9)
    num_digits = random.randint(2, 3)
    min_val = 10 ** (num_digits - 1)
    max_val = 10 ** num_digits - 1
    dividend = random.randint(min_val, max_val)

    quotient = dividend // divisor
    remainder = dividend % divisor

    question_text = f"{dividend}/{divisor}"
    question = wrap_numbers(question_text)

    # Answer format: "quotient R remainder"
    answer = f"{quotient} R {remainder}"
    answer_split = f"{split_num(str(quotient))} R {remainder}"

    return DataRecord(
        id=str(uuid.uuid4()),
        band="A5",
        question=question,
        answer=answer,
        answer_split=answer_split,
        hints=None,
        dual=None,
        tool_supervision=None,
        verifier={"ok": validate_division(dividend, divisor, quotient, remainder)},
    )


def generate_a6_division(seed: int) -> DataRecord:
    """A6: Multi-digit ÷ multi-digit."""
    random.seed(seed)

    num_digits_divisor = random.randint(2, 2)
    min_div = 10 ** (num_digits_divisor - 1)
    max_div = 10 ** num_digits_divisor - 1
    divisor = random.randint(min_div, max_div)

    num_digits_dividend = random.randint(3, 4)
    min_dividend = 10 ** (num_digits_dividend - 1)
    max_dividend = 10 ** num_digits_dividend - 1
    dividend = random.randint(min_dividend, max_dividend)

    quotient = dividend // divisor
    remainder = dividend % divisor

    question_text = f"{dividend}/{divisor}"
    question = wrap_numbers(question_text)

    answer = f"{quotient} R {remainder}"
    answer_split = f"{split_num(str(quotient))} R {remainder}"

    return DataRecord(
        id=str(uuid.uuid4()),
        band="A6",
        question=question,
        answer=answer,
        answer_split=answer_split,
        hints=None,
        dual=None,
        tool_supervision=None,
        verifier={"ok": validate_division(dividend, divisor, quotient, remainder)},
    )


def generate_a5_a6_batch(num_samples: int, seed_start: int = 1000, split: str = "train", band: str = None) -> List[DataRecord]:
    records = []
    for i in range(num_samples):
        seed = seed_start + i
        random.seed(seed)
        if random.random() < 0.4:
            record = generate_a5_division(seed)
        else:
            record = generate_a6_division(seed)
        records.append(record)
    return records


if __name__ == "__main__":
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=== A5-A6 Generator Test ===\n")
    samples = generate_a5_a6_batch(10, 5000)
    for i, r in enumerate(samples):
        print(f"{i+1}. [{r.band}] {r.question} = {r.answer}")
    print(f"\n✓ All pass: {all(r.verifier['ok'] for r in samples)}")
