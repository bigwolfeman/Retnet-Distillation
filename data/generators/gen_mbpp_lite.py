"""MBPP-lite: Code generation from HuggingFace datasets."""
import random, sys, uuid
from pathlib import Path
from typing import List

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.schema import DataRecord


def generate_mbpp_lite_batch(num_samples: int, seed_start: int = 1000, split: str = "train", band: str = None) -> List[DataRecord]:
    """
    Generate MBPP-lite code problems.

    NOTE: This is a stub implementation. Full version should:
    - Load from HuggingFace datasets: datasets.load_dataset("mbpp", "sanitized")
    - Filter for easy/medium difficulty
    - Format with tool call supervision
    """
    records = []

    # Stub: Generate simple Python problems
    for i in range(num_samples):
        seed = seed_start + i
        random.seed(seed)

        # Simple function template
        func_name = f"func_{i}"
        param = "n"
        operation = random.choice(["n + 1", "n * 2", "n - 1", "n // 2"])

        question = f"Write a function {func_name}({param}) that returns {operation}"
        answer = f"def {func_name}({param}):\n    return {operation}"

        # No split for code
        answer_split = None

        # Tool supervision: unit tests
        test_val = random.randint(1, 10)
        expected = eval(operation.replace(param, str(test_val)))
        tool_supervision = {
            "tests": [f"assert {func_name}({test_val}) == {expected}"]
        }

        record = DataRecord(
            id=str(uuid.uuid4()),
            band="MBPP_LITE",
            question=question,
            answer=answer,
            answer_split=answer_split,
            hints=None,
            dual=None,
            tool_supervision=tool_supervision,
            verifier={"ok": True}
        )

        records.append(record)

    return records


if __name__ == "__main__":
    samples = generate_mbpp_lite_batch(3, 0)
    for r in samples:
        print(f"Q: {r.question}")
        print(f"A: {r.answer}")
        print(f"Tests: {r.tool_supervision}")
        print()
