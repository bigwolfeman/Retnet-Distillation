"""A10: Modular arithmetic."""
import random, sys, uuid
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.schema import DataRecord
from utils.number_wrapping import wrap_numbers, split_num


def generate_a10_modular(seed: int) -> DataRecord:
    random.seed(seed)
    a = random.randint(10, 999)
    m = random.randint(2, 20)
    result = a % m

    question_text = f"{a} mod {m}"
    question = wrap_numbers(question_text)
    answer = str(result)
    answer_split = split_num(answer) if len(answer) > 1 else answer

    return DataRecord(
        id=str(uuid.uuid4()), band="A10", question=question, answer=answer,
        answer_split=answer_split, hints=None, dual=None,
        tool_supervision=None, verifier={"ok": True}
    )


def generate_a10_batch(num_samples: int, seed_start: int = 1000, split: str = "train", band: str = None):
    return [generate_a10_modular(seed_start + i) for i in range(num_samples)]
