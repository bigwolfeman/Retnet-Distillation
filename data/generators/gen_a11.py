"""A11: Simple algebra (linear equations)."""
import random, sys, uuid
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.schema import DataRecord
from utils.number_wrapping import wrap_numbers, split_num


def generate_a11_algebra(seed: int) -> DataRecord:
    random.seed(seed)
    # Solve: ax + b = c for x
    a = random.randint(2, 10)
    b = random.randint(1, 20)
    x = random.randint(1, 10)
    c = a * x + b

    question_text = f"{a}x+{b}={c}"
    question = wrap_numbers(question_text)
    answer = str(x)
    answer_split = split_num(answer) if len(answer) > 1 else answer

    return DataRecord(
        id=str(uuid.uuid4()), band="A11", question=question, answer=answer,
        answer_split=answer_split, hints=None, dual=None,
        tool_supervision=None, verifier={"ok": True}
    )


def generate_a11_batch(num_samples: int, seed_start: int = 1000, split: str = "train", band: str = None):
    return [generate_a11_algebra(seed_start + i) for i in range(num_samples)]
