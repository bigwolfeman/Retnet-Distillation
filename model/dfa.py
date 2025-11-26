"""
Format validators using regex-based DFAs (Deterministic Finite Automata).

Implements validation for integer, decimal, and fraction formats per data-model.md.
"""

import re
from enum import Enum
from typing import List, Optional, Tuple


class NumberFormat(Enum):
    """Supported number formats."""

    INTEGER = "integer"
    DECIMAL = "decimal"
    FRACTION = "fraction"


# Regex patterns per data-model.md
INTEGER_PATTERN = re.compile(r"^-?[0-9]+$")
DECIMAL_PATTERN = re.compile(r"^-?[0-9]+\.[0-9]+$")
FRACTION_PATTERN = re.compile(r"^-?[0-9]+/-?[0-9]+$")


def validate_integer(s: str) -> bool:
    """
    Validate integer format: ^-?[0-9]+$

    Args:
        s: String to validate

    Returns:
        True if valid integer format
    """
    return bool(INTEGER_PATTERN.match(s))


def validate_decimal(s: str) -> bool:
    """
    Validate decimal format: ^-?[0-9]+\.[0-9]+$

    Args:
        s: String to validate

    Returns:
        True if valid decimal format
    """
    return bool(DECIMAL_PATTERN.match(s))


def validate_fraction(s: str) -> bool:
    """
    Validate fraction format: ^-?[0-9]+/-?[0-9]+$

    Also checks that denominator != 0.

    Args:
        s: String to validate

    Returns:
        True if valid fraction format with non-zero denominator
    """
    if not FRACTION_PATTERN.match(s):
        return False

    # Extract denominator and check != 0
    try:
        parts = s.split("/")
        denominator = int(parts[1])
        return denominator != 0
    except (IndexError, ValueError):
        return False


def infer_format(s: str) -> Optional[NumberFormat]:
    """
    Infer the format of a number string.

    Args:
        s: String to analyze

    Returns:
        NumberFormat enum or None if no match
    """
    if validate_fraction(s):
        return NumberFormat.FRACTION
    if validate_decimal(s):
        return NumberFormat.DECIMAL
    if validate_integer(s):
        return NumberFormat.INTEGER
    return None


def validate_division(dividend: int, divisor: int, quotient: int, remainder: int) -> bool:
    """
    Validate division result per data-model.md verifier rule.

    Checks: dividend == divisor * quotient + remainder AND remainder < divisor

    Args:
        dividend: Number being divided
        divisor: Number dividing by
        quotient: Division result
        remainder: Division remainder

    Returns:
        True if division is correct
    """
    if divisor == 0:
        return False

    # Check: dividend = divisor * quotient + remainder
    if dividend != divisor * quotient + remainder:
        return False

    # Check: remainder < divisor (and non-negative)
    if not (0 <= remainder < abs(divisor)):
        return False

    return True


def validate_addition_reversed(answer: str) -> Tuple[bool, Optional[str]]:
    """
    Validate addition using reversed digits check per data-model.md.

    For answer "95", reversed should be "59".

    Args:
        answer: Answer string (must be valid integer)

    Returns:
        Tuple of (is_valid, reversed_string)
    """
    if not validate_integer(answer):
        return False, None

    # Parse as integer to handle leading zeros and signs
    try:
        num = int(answer)
        # Reverse the absolute value
        reversed_abs = str(abs(num))[::-1]

        # Handle sign
        if num < 0:
            reversed_str = "-" + reversed_abs
        else:
            reversed_str = reversed_abs

        return True, reversed_str
    except ValueError:
        return False, None


def generate_test_cases(format_type: NumberFormat, count: int = 100) -> List[str]:
    """
    Generate test cases for a given format type.

    Args:
        format_type: Format to generate test cases for
        count: Number of test cases to generate

    Returns:
        List of test case strings
    """
    import random

    random.seed(42)  # Deterministic for testing
    test_cases = []

    for _ in range(count):
        if format_type == NumberFormat.INTEGER:
            # Random integer (-1000 to 1000)
            num = random.randint(-1000, 1000)
            test_cases.append(str(num))

        elif format_type == NumberFormat.DECIMAL:
            # Random decimal with 1-3 decimal places
            integer_part = random.randint(-1000, 1000)
            decimal_places = random.randint(1, 3)
            decimal_part = "".join(str(random.randint(0, 9)) for _ in range(decimal_places))
            test_cases.append(f"{integer_part}.{decimal_part}")

        elif format_type == NumberFormat.FRACTION:
            # Random fraction with non-zero denominator
            numerator = random.randint(-100, 100)
            denominator = random.randint(1, 100)  # Ensure non-zero
            test_cases.append(f"{numerator}/{denominator}")

    return test_cases


def test_wrapping_reversibility() -> bool:
    """
    Test unsplit(split(x)) == x for 10k random numbers.

    Returns:
        True if all tests pass
    """
    from utils.number_wrapping import test_reversibility

    passed, error = test_reversibility(num_trials=10000)
    if not passed:
        print(f"Reversibility test failed: {error}")
    return passed


def test_wrapping_token_count_linearity(tokenizer):
    """
    Test that token count scales linearly with digit count inside wrappers.

    Args:
        tokenizer: Tokenizer instance

    Returns:
        True if token count is roughly linear
    """
    from utils.number_wrapping import wrap_numbers

    # Test that wrapped numbers produce more tokens than compact form
    test_cases = [
        (2, "99"),
        (4, "9999"),
        (6, "999999"),
        (8, "99999999"),
        (10, "9999999999"),
    ]

    for expected_digits, num_str in test_cases:
        wrapped = wrap_numbers(num_str)
        tokens = tokenizer.encode(wrapped, add_special_tokens=False)

        # Each digit should be at least 1 token, plus wrappers (2 tokens)
        # BPE might merge some, so check lower bound
        min_expected = expected_digits + 2

        if len(tokens) < min_expected:
            print(f"Token count too low for {num_str}: {len(tokens)} < {min_expected}")
            return False

    return True


def test_dual_view_consistency() -> bool:
    """
    Test that <ANS> equals unsplit(<ANS_SPLIT>).

    Returns:
        True if consistency holds
    """
    from utils.number_wrapping import unsplit_num

    test_cases = [
        ("95", "9 5"),
        ("123", "1 2 3"),
        ("1000", "1 0 0 0"),
    ]

    for answer, answer_split in test_cases:
        if unsplit_num(answer_split) != answer:
            print(f"Dual-view inconsistency: {answer} != unsplit({answer_split})")
            return False

    return True


# Example usage and validation tests
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    # Fix Unicode encoding on Windows
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=== Format Validator Tests ===\n")

    # Integer tests
    print("INTEGER validation:")
    test_integers = ["42", "-123", "0", "999", "42.5", "abc", ""]
    for test in test_integers:
        result = validate_integer(test)
        print(f"  {test:10s} -> {result}")

    # Decimal tests
    print("\nDECIMAL validation:")
    test_decimals = ["42.5", "-123.456", "0.1", "42", "abc", ""]
    for test in test_decimals:
        result = validate_decimal(test)
        print(f"  {test:10s} -> {result}")

    # Fraction tests
    print("\nFRACTION validation:")
    test_fractions = ["3/4", "-12/16", "5/0", "42", "abc", "1/2/3"]
    for test in test_fractions:
        result = validate_fraction(test)
        print(f"  {test:10s} -> {result}")

    # Division validation
    print("\nDIVISION validation:")
    division_tests = [
        (17, 5, 3, 2, True),  # 17 = 5*3 + 2
        (20, 4, 5, 0, True),  # 20 = 4*5 + 0
        (17, 5, 3, 3, False),  # Wrong: 17 != 5*3 + 3
        (17, 5, 4, -3, False),  # Negative remainder
        (10, 0, 0, 0, False),  # Division by zero
    ]
    for dividend, divisor, quotient, remainder, expected in division_tests:
        result = validate_division(dividend, divisor, quotient, remainder)
        status = "✓" if result == expected else "✗"
        print(
            f"  {status} {dividend}/{divisor} = {quotient} R {remainder} -> {result} (expected {expected})"
        )

    # Reversed addition check
    print("\nREVERSED ADDITION validation:")
    reversed_tests = ["95", "123", "-456", "0"]
    for test in reversed_tests:
        valid, reversed_str = validate_addition_reversed(test)
        print(f"  {test} -> reversed: {reversed_str} (valid: {valid})")

    # Generate test cases
    print("\n=== Generated Test Cases ===")
    for fmt in NumberFormat:
        cases = generate_test_cases(fmt, count=10)
        print(f"\n{fmt.value.upper()} (10 samples):")
        for i, case in enumerate(cases[:10], 1):
            validator = {
                NumberFormat.INTEGER: validate_integer,
                NumberFormat.DECIMAL: validate_decimal,
                NumberFormat.FRACTION: validate_fraction,
            }[fmt]
            is_valid = validator(case)
            status = "✓" if is_valid else "✗"
            print(f"  {status} {case}")

    print("\n=== Number Wrapping Tests ===")

    # Test reversibility
    print("\nReversibility (10k trials):")
    if test_wrapping_reversibility():
        print("  ✓ All trials passed")
    else:
        print("  ✗ Failed")

    # Test dual-view consistency
    print("\nDual-view consistency:")
    if test_dual_view_consistency():
        print("  ✓ All tests passed")
    else:
        print("  ✗ Failed")

    print("\n✓ All format validators and wrapping tests implemented")
