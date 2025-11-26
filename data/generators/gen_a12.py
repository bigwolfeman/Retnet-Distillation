"""A12: Word problems → math parsing."""
import random, sys, uuid
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.schema import DataRecord
from utils.number_wrapping import wrap_numbers, split_num


TEMPLATES = [
    ("Alice has {a} apples. Bob gives her {b} more. How many apples does Alice have?", lambda a, b: a + b),
    ("There are {a} birds. {b} fly away. How many birds remain?", lambda a, b: a - b),
    ("A box contains {a} rows of {b} items. How many items total?", lambda a, b: a * b),
]


def generate_a12_word_problem(seed: int) -> DataRecord:
    random.seed(seed)
    template, operation = random.choice(TEMPLATES)
    a = random.randint(5, 50)
    b = random.randint(2, 20)

    question = template.format(a=a, b=b)
    result = operation(a, b)
    answer = str(result)
    answer_split = split_num(answer) if len(answer) > 1 else answer

    return DataRecord(
        id=str(uuid.uuid4()), band="A12", question=question, answer=answer,
        answer_split=answer_split, hints=None, dual=None,
        tool_supervision=None, verifier={"ok": True}
    )


def generate_a12_batch(num_samples: int, seed_start: int = 1000, split: str = "train", band: str = None):
    return [generate_a12_word_problem(seed_start + i) for i in range(num_samples)]
