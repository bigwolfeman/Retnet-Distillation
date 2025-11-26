"""A9: Decimals + rounding."""
import random, sys, uuid
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.schema import DataRecord
from utils.number_wrapping import wrap_numbers, split_num


def generate_a9_decimal(seed: int) -> DataRecord:
    random.seed(seed)
    # Decimal addition
    a = round(random.uniform(1, 100), 2)
    b = round(random.uniform(1, 100), 2)
    result = round(a + b, 2)

    question_text = f"{a}+{b}"
    question = wrap_numbers(question_text)
    answer = str(result)
    answer_split = answer.replace(".", " . ")  # Split around decimal point

    return DataRecord(
        id=str(uuid.uuid4()), band="A9", question=question, answer=answer,
        answer_split=answer_split, hints=None, dual=None,
        tool_supervision=None, verifier={"ok": True}
    )


def generate_a9_batch(num_samples: int, seed_start: int = 1000, split: str = "train", band: str = None):
    return [generate_a9_decimal(seed_start + i) for i in range(num_samples)]
