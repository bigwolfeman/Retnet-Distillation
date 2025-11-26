"""
StarCoder2 tokenizer integration with spatial digit separation.

Uses bigcode/starcoder2-15b base tokenizer with added special tokens for math curriculum.
"""

from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer


# Special tokens for math curriculum
CURRICULUM_SPECIAL_TOKENS = [
    # Number wrappers
    "⟨N⟩",
    "⟨/N⟩",
    # Answer views
    "<ANS>",
    "</ANS>",
    "<ANS_SPLIT>",
    "</ANS_SPLIT>",
    # Structure
    "<Q>",
    "<A>",
    "<SEP>",
    # Tool calls
    "<TOOL:calc>",
    "<TOOL:cas>",
    "<TOOL:test>",
    "<RET:calc>",
    "<RET:cas>",
    "<RET:test>",
    "<CALL>",
    "<END>",
]


def create_curriculum_tokenizer(
    base_model: str = "bigcode/starcoder2-15b",
    cache_dir: Optional[Path] = None,
) -> AutoTokenizer:
    """
    Create StarCoder2 tokenizer with curriculum-specific special tokens.

    Args:
        base_model: HuggingFace model identifier for StarCoder2
        cache_dir: Optional directory to cache tokenizer

    Returns:
        Configured AutoTokenizer instance
    """
    # Load StarCoder2 base tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        cache_dir=str(cache_dir) if cache_dir else None,
        trust_remote_code=True,  # StarCoder2 may need this
    )

    # Add special tokens for math curriculum
    special_tokens_dict = {
        "additional_special_tokens": CURRICULUM_SPECIAL_TOKENS
    }

    num_added = tokenizer.add_special_tokens(special_tokens_dict)

    print(f"Added {num_added} special tokens to StarCoder2 tokenizer")
    print(f"Vocabulary size: {len(tokenizer)}")

    return tokenizer


def get_vocab_size(tokenizer: AutoTokenizer) -> int:
    """
    Get total vocabulary size including special tokens.

    Args:
        tokenizer: Tokenizer instance

    Returns:
        Total number of tokens
    """
    return len(tokenizer)


# Tokenizer singleton cache
_TOKENIZER_CACHE: Optional[AutoTokenizer] = None


def get_tokenizer(cache: bool = True, base_model: str = "bigcode/starcoder2-15b") -> AutoTokenizer:
    """
    Get or create curriculum tokenizer instance.

    Args:
        cache: If True, cache tokenizer for reuse
        base_model: StarCoder2 model identifier

    Returns:
        Tokenizer instance
    """
    global _TOKENIZER_CACHE

    if cache and _TOKENIZER_CACHE is not None:
        return _TOKENIZER_CACHE

    tokenizer = create_curriculum_tokenizer(base_model=base_model)

    if cache:
        _TOKENIZER_CACHE = tokenizer

    return tokenizer


# Example usage and validation
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

    from utils.number_wrapping import wrap_numbers, unwrap_numbers

    print("=== StarCoder2 Tokenizer Integration ===\n")

    # Create tokenizer
    tokenizer = create_curriculum_tokenizer()

    print(f"Base model: bigcode/starcoder2-15b")
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")

    # Test special tokens
    print("\n=== Special Token IDs ===")
    for token in CURRICULUM_SPECIAL_TOKENS[:5]:  # Show first 5
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"  {token}: {token_id}")

    # Test encoding/decoding with wrapped numbers
    test_cases = [
        ("37+58", "⟨N⟩3 7⟨/N⟩+⟨N⟩5 8⟨/N⟩"),
        ("999*999", "⟨N⟩9 9 9⟨/N⟩*⟨N⟩9 9 9⟨/N⟩"),
        ("-123", "-⟨N⟩1 2 3⟨/N⟩"),
    ]

    print("\n=== Tokenization Tests ===")
    for original, wrapped in test_cases:
        # Wrap the number
        wrapped_text = wrap_numbers(original)
        assert wrapped_text == wrapped, f"Wrapping failed: {wrapped_text} != {wrapped}"

        # Tokenize
        tokens = tokenizer.encode(wrapped_text, add_special_tokens=False)

        # Decode
        decoded = tokenizer.decode(tokens, skip_special_tokens=False)

        # Unwrap
        unwrapped = unwrap_numbers(decoded)

        # Check round-trip
        match = "✓" if unwrapped.replace(" ", "") == original.replace(" ", "") else "✗"

        print(f"\n{match} Input:     {original}")
        print(f"  Wrapped:   {wrapped_text}")
        print(f"  Tokens:    {tokens} (count: {len(tokens)})")
        print(f"  Decoded:   {decoded}")
        print(f"  Unwrapped: {unwrapped}")

    # Test token count linearity
    print("\n=== Token Count Linearity Test ===")
    for num_digits in [2, 4, 6, 8, 10]:
        num_str = "9" * num_digits
        wrapped = wrap_numbers(num_str)

        tokens = tokenizer.encode(wrapped, add_special_tokens=False)

        # Expected: wrappers (2) + num_digits + spaces (handled by BPE, minimal)
        # BPE might merge some patterns, but each digit should be at least 1 token
        print(f"  {num_digits} digits: {len(tokens)} tokens ('{wrapped}')")

    # Test dual answer views
    print("\n=== Dual Answer View Test ===")
    answer = "95"
    answer_split = "9 5"

    normal_text = f"<ANS>{answer}</ANS>"
    split_text = f"<ANS_SPLIT>{answer_split}</ANS_SPLIT>"

    normal_tokens = tokenizer.encode(normal_text, add_special_tokens=False)
    split_tokens = tokenizer.encode(split_text, add_special_tokens=False)

    print(f"  Normal:    '{normal_text}' → {len(normal_tokens)} tokens")
    print(f"  Split:     '{split_text}' → {len(split_tokens)} tokens")
    print(f"  Inflation: {len(split_tokens) / len(normal_tokens):.2f}x")

    # Test JSON preservation
    print("\n=== JSON Preservation Test ===")
    json_str = '{"expr":"999*999"}'
    # Should NOT wrap numbers inside JSON
    wrapped_json = wrap_numbers(json_str, exclude_json=True)
    status = "✓" if wrapped_json == json_str else "✗"
    print(f"  {status} JSON unchanged: '{json_str}' → '{wrapped_json}'")

    print("\n✓ StarCoder2 tokenizer integration complete")
