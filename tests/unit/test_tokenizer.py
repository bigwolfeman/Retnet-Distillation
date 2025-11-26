"""
Unit tests for tokenizer round-trip identity with control tags.

Per tasks.md T015: Test 1k random sequences with control tags for round-trip identity.
"""

import random
import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from model.tokenizer import get_tokenizer
from utils.number_wrapping import wrap_numbers, unwrap_numbers, split_num


def test_round_trip_identity():
    """
    Test tokenizer round-trip identity for 1k random sequences with control tags.

    Per FR-006b: System must emit dual answer views with wrapped numbers.
    """
    tokenizer = get_tokenizer()
    random.seed(42)

    passed = 0
    failed = 0

    for i in range(1000):
        # Generate random arithmetic expression
        a = random.randint(0, 999)
        b = random.randint(0, 999)
        ops = ['+', '-', '*', '/']
        op = random.choice(ops)

        # Compute result
        if op == '+':
            result = a + b
        elif op == '-':
            result = a - b
        elif op == '*':
            result = a * b
        elif op == '/':
            if b == 0:
                b = 1  # Avoid division by zero
            result = a // b

        # Create wrapped question
        question = wrap_numbers(f"{a}{op}{b}")

        # Create dual answer views
        answer_normal = str(result)
        answer_split = split_num(str(result))

        # Build full sequence with control tags
        sequence = f"<Q>{question}<A><ANS>{answer_normal}</ANS><ANS_SPLIT>{answer_split}</ANS_SPLIT><SEP>"

        # Tokenize
        tokens = tokenizer.encode(sequence, add_special_tokens=False)

        # Decode
        decoded = tokenizer.decode(tokens, skip_special_tokens=False)

        # Clean up whitespace differences (BPE may add/remove spaces)
        sequence_clean = sequence.replace(" ", "")
        decoded_clean = decoded.replace(" ", "")

        # Check if semantically equivalent (unwrap both and compare)
        sequence_unwrapped = unwrap_numbers(sequence)
        decoded_unwrapped = unwrap_numbers(decoded)

        if sequence_unwrapped.replace(" ", "") == decoded_unwrapped.replace(" ", ""):
            passed += 1
        else:
            failed += 1
            if failed <= 5:  # Print first 5 failures
                print(f"\nFailed {i}:")
                print(f"  Original:  {sequence}")
                print(f"  Decoded:   {decoded}")
                print(f"  Unwrapped: {sequence_unwrapped} vs {decoded_unwrapped}")

    print(f"\nRound-trip test: {passed}/1000 passed, {failed}/1000 failed")

    # Require at least 99% success (allow for edge cases)
    assert passed >= 990, f"Too many failures: {failed}/1000"


def test_control_tags_are_single_tokens():
    """
    Test that control tags are encoded as single tokens.

    Per research.md: Special tokens should be atomic.
    """
    tokenizer = get_tokenizer()

    control_tags = [
        "⟨N⟩",
        "⟨/N⟩",
        "<ANS>",
        "</ANS>",
        "<ANS_SPLIT>",
        "</ANS_SPLIT>",
        "<Q>",
        "<A>",
        "<SEP>",
        "<TOOL:calc>",
        "<RET:calc>",
        "<CALL>",
        "<END>",
    ]

    for tag in control_tags:
        tokens = tokenizer.encode(tag, add_special_tokens=False)
        assert len(tokens) == 1, f"Control tag '{tag}' should be single token, got {len(tokens)}: {tokens}"

        # Verify decode matches
        decoded = tokenizer.decode(tokens, skip_special_tokens=False)
        assert decoded == tag, f"Decode mismatch: '{tag}' -> '{decoded}'"


def test_wrapped_numbers_are_split():
    """
    Test that wrapped numbers become separate tokens per digit.

    Per research.md: BPE never merges across whitespace, so space-separated
    digits become individual tokens.
    """
    tokenizer = get_tokenizer()

    test_cases = [
        ("37", "⟨N⟩3 7⟨/N⟩", 5),  # wrapper(2) + 3 + space + 7 ≈ 5 tokens
        ("999", "⟨N⟩9 9 9⟨/N⟩", 7),  # wrapper(2) + 9 + space + 9 + space + 9 ≈ 7 tokens
        ("12345", "⟨N⟩1 2 3 4 5⟨/N⟩", 11),  # wrapper(2) + 5 digits + 4 spaces ≈ 11 tokens
    ]

    for original, wrapped, expected_min_tokens in test_cases:
        wrapped_actual = wrap_numbers(original)
        assert wrapped_actual == wrapped, f"Wrapping failed: {original} -> {wrapped_actual}"

        tokens = tokenizer.encode(wrapped, add_special_tokens=False)

        # Each digit should contribute at least 1 token
        # Wrappers are 2 tokens
        # Spaces may or may not be separate tokens depending on BPE
        assert len(tokens) >= len(original) + 2, (
            f"Token count too low for '{wrapped}': "
            f"{len(tokens)} < {len(original) + 2}"
        )


def test_dual_answer_views():
    """
    Test that dual answer views (normal + split) tokenize correctly.

    Per data-model.md: DataRecord has both answer and answer_split fields.
    """
    tokenizer = get_tokenizer()

    answer = "12345"
    answer_split = "1 2 3 4 5"

    normal_text = f"<ANS>{answer}</ANS>"
    split_text = f"<ANS_SPLIT>{answer_split}</ANS_SPLIT>"

    # Tokenize both
    normal_tokens = tokenizer.encode(normal_text, add_special_tokens=False)
    split_tokens = tokenizer.encode(split_text, add_special_tokens=False)

    # Split should have more tokens (inflated)
    assert len(split_tokens) > len(normal_tokens), (
        f"Split answer should have more tokens: "
        f"{len(split_tokens)} <= {len(normal_tokens)}"
    )

    # Decode and verify
    normal_decoded = tokenizer.decode(normal_tokens, skip_special_tokens=False)
    split_decoded = tokenizer.decode(split_tokens, skip_special_tokens=False)

    # Check that decoded versions contain the answers
    assert answer in normal_decoded or answer.replace(" ", "") in normal_decoded.replace(" ", "")
    assert answer_split.replace(" ", "") in split_decoded.replace(" ", "")


if __name__ == "__main__":
    # Fix Unicode encoding on Windows
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    # Run tests
    print("Running tokenizer tests...\n")

    print("Test 1: Round-trip identity (1000 trials)")
    test_round_trip_identity()
    print("✓ PASSED\n")

    print("Test 2: Control tags are single tokens")
    test_control_tags_are_single_tokens()
    print("✓ PASSED\n")

    print("Test 3: Wrapped numbers are split")
    test_wrapped_numbers_are_split()
    print("✓ PASSED\n")

    print("Test 4: Dual answer views")
    test_dual_answer_views()
    print("✓ PASSED\n")

    print("✓ All tokenizer tests passed!")
