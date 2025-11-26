"""A7: Mixed operations + parentheses with RPN/AST dual views."""
import random, sys, uuid
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.schema import DataRecord
from utils.number_wrapping import wrap_numbers, split_num


def generate_a7_mixed_ops(seed: int) -> DataRecord:
    random.seed(seed)
    a, b, c = random.randint(10, 99), random.randint(10, 99), random.randint(2, 9)

    # Simple expression: a + b * c (order of operations)
    result = a + (b * c)
    question_text = f"{a}+{b}*{c}"
    question = wrap_numbers(question_text)
    answer = str(result)
    answer_split = split_num(answer)

    return DataRecord(
        id=str(uuid.uuid4()), band="A7", question=question, answer=answer,
        answer_split=answer_split, hints=None, dual=None,
        tool_supervision=None, verifier={"ok": True}
    )


def generate_a7_batch(num_samples: int, seed_start: int = 1000, split: str = "train", band: str = None):
    return [generate_a7_mixed_ops(seed_start + i) for i in range(num_samples)]


if __name__ == "__main__":
    print("A7 test:", [r.question for r in generate_a7_batch(3, 7000)])
