"""A8: Fractions with GCD hints."""
import random, sys, uuid, math
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.schema import DataRecord
from utils.number_wrapping import wrap_numbers, split_num


def generate_a8_fraction(seed: int) -> DataRecord:
    random.seed(seed)
    # Simple fraction addition/simplification
    num1, den1 = random.randint(1, 20), random.randint(2, 20)
    num2, den2 = random.randint(1, 20), random.randint(2, 20)

    # Add fractions: num1/den1 + num2/den2
    common_den = den1 * den2
    new_num = num1 * den2 + num2 * den1

    # Simplify
    gcd = math.gcd(new_num, common_den)
    final_num = new_num // gcd
    final_den = common_den // gcd

    question_text = f"{num1}/{den1}+{num2}/{den2}"
    question = wrap_numbers(question_text)
    answer = f"{final_num}/{final_den}"
    answer_split = f"{split_num(str(final_num))}/{split_num(str(final_den))}"

    return DataRecord(
        id=str(uuid.uuid4()), band="A8", question=question, answer=answer,
        answer_split=answer_split, hints=None, dual=None,
        tool_supervision=None, verifier={"ok": True}
    )


def generate_a8_batch(num_samples: int, seed_start: int = 1000, split: str = "train", band: str = None):
    return [generate_a8_fraction(seed_start + i) for i in range(num_samples)]
