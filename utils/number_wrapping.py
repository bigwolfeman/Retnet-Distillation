"""
Number wrapping utilities for spatial digit separation with StarCoder2 tokenizer.

Exploits BPE property: never merges across whitespace, so space-separated digits
become individual tokens without custom vocabulary.
"""

import json
import re
from typing import Tuple


# Unicode-based wrappers to avoid conflicts with existing syntax
NUM_WRAP_OPEN = "⟨N⟩"
NUM_WRAP_CLOSE = "⟨/N⟩"


def split_num(s: str) -> str:
    """
    Split digits with spaces: '37' → '3 7'

    Args:
        s: String of consecutive digits

    Returns:
        Space-separated digits
    """
    return " ".join(list(s))


def unsplit_num(s: str) -> str:
    """
    Unsplit digits: '9 5' → '95'

    Args:
        s: Space-separated digits

    Returns:
        Concatenated digits (spaces removed)
    """
    return s.replace(" ", "")


def wrap_numbers(text: str, exclude_json: bool = True) -> str:
    """
    Wrap multi-digit numbers in text with spatial separation.

    Examples:
        "37+58" → "⟨N⟩3 7⟨/N⟩+⟨N⟩5 8⟨/N⟩"
        "-123" → "-⟨N⟩1 2 3⟨/N⟩"
        "12.34" → "⟨N⟩1 2⟨/N⟩.⟨N⟩3 4⟨/N⟩"

    Does NOT wrap:
        - Single digits: "7" stays "7"
        - JSON blocks: {"expr":"999"} stays unchanged (if exclude_json=True)
        - Code literals (if inside JSON or between braces)

    Args:
        text: Input text with numbers
        exclude_json: If True, skip wrapping if text looks like JSON

    Returns:
        Text with wrapped numbers
    """
    # Skip if text looks like JSON/code
    if exclude_json:
        # Detect JSON objects
        if text.strip().startswith('{') or text.strip().startswith('['):
            return text
        # Detect quoted strings (likely code/JSON values)
        if '"' in text and ':' in text:
            return text

    def replace_num(match):
        """Replace a matched number with wrapped version."""
        num_str = match.group(0)

        # Check for minus sign
        if num_str.startswith('-'):
            minus = '-'
            digits = num_str[1:]
        else:
            minus = ''
            digits = num_str

        # Only wrap if 2+ digits
        if len(digits) < 2:
            return num_str  # Keep single digits unwrapped

        # Split and wrap
        split_digits = split_num(digits)
        return f"{minus}{NUM_WRAP_OPEN}{split_digits}{NUM_WRAP_CLOSE}"

    # Match: optional minus + 2 or more consecutive digits
    # This pattern handles:
    # - Integers: 37, -123, 9999
    # - Parts of decimals: 12.34 → wraps "12" and "34" separately
    # - Parts of fractions: 12/34 → wraps "12" and "34" separately
    pattern = r'-?\d{2,}'

    return re.sub(pattern, replace_num, text)


def unwrap_numbers(text: str) -> str:
    """
    Remove wrappers and unsplit digits.

    Example:
        "⟨N⟩9 5⟨/N⟩" → "95"
        "⟨N⟩1 2⟨/N⟩.⟨N⟩3 4⟨/N⟩" → "12.34"

    Args:
        text: Text with wrapped numbers

    Returns:
        Text with unwrapped numbers
    """
    # Regex: find wrapped numbers
    # Pattern matches: ⟨N⟩ followed by digits/spaces, ending with ⟨/N⟩
    pattern = rf'{re.escape(NUM_WRAP_OPEN)}([0-9 ]+){re.escape(NUM_WRAP_CLOSE)}'

    def replace_wrapped(match):
        split_digits = match.group(1)
        return unsplit_num(split_digits)

    return re.sub(pattern, replace_wrapped, text)


def validate_wrapping(text: str) -> bool:
    """
    Validate wrapped number format.

    Valid pattern: ⟨N⟩([0-9]( [0-9])*)⟨/N⟩
    - Must have at least one digit
    - Digits must be space-separated (if multiple)
    - No other characters inside wrappers

    Args:
        text: Text to validate

    Returns:
        True if all wrapped regions are valid
    """
    # Find all wrapped regions
    wrapped_regions = re.findall(
        rf'{re.escape(NUM_WRAP_OPEN)}(.*?){re.escape(NUM_WRAP_CLOSE)}',
        text
    )

    if not wrapped_regions:
        return True  # No wrapped regions is valid

    # Valid pattern: digit, optionally followed by (space digit)*
    valid_pattern = r'^[0-9]( [0-9])*$'

    for region in wrapped_regions:
        if not re.match(valid_pattern, region):
            return False

    return True


def test_reversibility(num_trials: int = 10000) -> Tuple[bool, str]:
    """
    Test unsplit(split(x)) == x for random numbers.

    Args:
        num_trials: Number of random test cases

    Returns:
        Tuple of (all_passed, error_message)
    """
    import random

    random.seed(42)  # Deterministic

    for _ in range(num_trials):
        # Test various number types
        num = random.randint(-999999, 999999)
        num_str = str(num)

        # Test split/unsplit
        split = split_num(num_str.lstrip('-'))  # Split absolute value
        if num < 0:
            split = split  # Negative handled by wrapping logic
        unsplit = unsplit_num(split)

        if unsplit != num_str.lstrip('-'):
            return False, f"Failed on {num_str}: split={split}, unsplit={unsplit}"

        # Test wrap/unwrap
        wrapped = wrap_numbers(num_str)
        unwrapped = unwrap_numbers(wrapped)

        # Handle single digits (not wrapped)
        if -9 <= num <= 9:
            expected = num_str
        else:
            expected = num_str

        if unwrapped != expected:
            return False, f"Failed wrap/unwrap on {num_str}: wrapped={wrapped}, unwrapped={unwrapped}"

    return True, ""


def test_dual_view_consistency() -> Tuple[bool, str]:
    """
    Test that <ANS> equals unsplit(<ANS_SPLIT>).

    Returns:
        Tuple of (all_passed, error_message)
    """
    test_cases = [
        ("95", "9 5"),
        ("123", "1 2 3"),
        ("9", "9"),  # Single digit
        ("1000", "1 0 0 0"),
        ("42", "4 2"),
    ]

    for answer, answer_split in test_cases:
        unsplit = unsplit_num(answer_split)
        if unsplit != answer:
            return False, f"Failed on answer={answer}, split={answer_split}, unsplit={unsplit}"

    return True, ""


# Run tests on import (for development validation)
if __name__ == "__main__":
    import sys

    # Fix Unicode encoding on Windows
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=== Number Wrapping Utilities ===\n")

    # Test wrapping
    test_cases = [
        ("37+58", "⟨N⟩3 7⟨/N⟩+⟨N⟩5 8⟨/N⟩"),
        ("-123", "-⟨N⟩1 2 3⟨/N⟩"),
        ("12.34", "⟨N⟩1 2⟨/N⟩.⟨N⟩3 4⟨/N⟩"),
        ("7", "7"),  # Single digit, no wrap
        ("12/34", "⟨N⟩1 2⟨/N⟩/⟨N⟩3 4⟨/N⟩"),
    ]

    print("Wrapping tests:")
    for input_text, expected in test_cases:
        result = wrap_numbers(input_text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{input_text}' → '{result}'")
        if result != expected:
            print(f"     Expected: '{expected}'")

    # Test unwrapping
    print("\nUnwrapping tests:")
    for expected, input_text in test_cases:
        if input_text == expected:  # Skip single digit
            continue
        result = unwrap_numbers(input_text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{input_text}' → '{result}'")

    # Test JSON exclusion
    print("\nJSON exclusion tests:")
    json_cases = [
        '{"expr":"999*999"}',
        '{"result":"998001"}',
        '[1, 2, 3]',
    ]
    for json_str in json_cases:
        result = wrap_numbers(json_str, exclude_json=True)
        status = "✓" if result == json_str else "✗"
        print(f"  {status} JSON unchanged: {result == json_str}")

    # Test validation
    print("\nValidation tests:")
    valid_cases = [
        "⟨N⟩3 7⟨/N⟩+⟨N⟩5 8⟨/N⟩",  # Valid
        "⟨N⟩1 2 3⟨/N⟩",  # Valid
        "⟨N⟩9⟨/N⟩",  # Valid (single digit)
    ]
    invalid_cases = [
        "⟨N⟩37⟨/N⟩",  # Invalid (no spaces)
        "⟨N⟩3  7⟨/N⟩",  # Invalid (double space)
        "⟨N⟩3-7⟨/N⟩",  # Invalid (minus inside)
    ]

    for case in valid_cases:
        result = validate_wrapping(case)
        status = "✓" if result else "✗"
        print(f"  {status} Valid: '{case}' → {result}")

    for case in invalid_cases:
        result = validate_wrapping(case)
        status = "✓" if not result else "✗"
        print(f"  {status} Invalid: '{case}' → {result}")

    # Test reversibility
    print("\nReversibility test:")
    passed, error = test_reversibility(num_trials=10000)
    if passed:
        print("  ✓ All 10,000 trials passed")
    else:
        print(f"  ✗ Failed: {error}")

    # Test dual-view consistency
    print("\nDual-view consistency test:")
    passed, error = test_dual_view_consistency()
    if passed:
        print("  ✓ All dual-view tests passed")
    else:
        print(f"  ✗ Failed: {error}")

    print("\n✓ Number wrapping utilities complete")
