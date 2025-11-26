"""
DFA Canary Test: Verify model never generates closing tags before opening tags.

This test ensures that after training with FSM penalty and constrained decoding,
the model respects structural constraints and never emits closing tags (</ANS>,
</ANS_SPLIT>) before their corresponding opening tags.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from train.curriculum import CurriculumState
from train.curriculum_dataset_causal import CurriculumDatasetCausal
from model.tokenizer import get_tokenizer
from utils.structure_fsm import StructureFSM
from src.models.core import RetNetHRMModel, ModelConfig


def validate_tag_order(sequence: str, tokenizer) -> tuple[bool, str]:
    """
    Validate that closing tags appear after opening tags.

    Returns:
        (is_valid, error_message)
    """
    # Check <ANS>...</ANS> structure
    ans_open_pos = sequence.find('<ANS>')
    ans_close_pos = sequence.find('</ANS>')

    if ans_close_pos != -1 and ans_open_pos == -1:
        return False, "</ANS> appears without preceding <ANS>"

    if ans_open_pos != -1 and ans_close_pos != -1:
        if ans_close_pos < ans_open_pos:
            return False, f"</ANS> at position {ans_close_pos} appears before <ANS> at position {ans_open_pos}"

    # Check <ANS_SPLIT>...</ANS_SPLIT> structure
    split_open_pos = sequence.find('<ANS_SPLIT>')
    split_close_pos = sequence.find('</ANS_SPLIT>')

    if split_close_pos != -1 and split_open_pos == -1:
        return False, "</ANS_SPLIT> appears without preceding <ANS_SPLIT>"

    if split_open_pos != -1 and split_close_pos != -1:
        if split_close_pos < split_open_pos:
            return False, f"</ANS_SPLIT> at position {split_close_pos} appears before <ANS_SPLIT> at position {split_open_pos}"

    # Check for multiple consecutive closing tags (like </ANS></ANS></ANS>)
    if '</ANS></ANS>' in sequence or '</ANS_SPLIT></ANS_SPLIT>' in sequence:
        return False, "Multiple consecutive closing tags detected"

    return True, ""


def test_no_close_before_open():
    """
    DFA Canary Test: Verify FSM constrained decoding prevents invalid tag sequences.

    This lightweight test:
    1. Creates mock sequences with various structures
    2. Applies FSM constrained decoding at each step
    3. Validates that constrained decoding prevents:
       - </ANS> appearing before <ANS>
       - </ANS_SPLIT> appearing before <ANS_SPLIT>
       - Multiple consecutive closing tags

    Note: This tests the FSM logic directly without training. For end-to-end
    validation, run a full training script with the FSM enabled.
    """
    # Initialize tokenizer and FSM
    tokenizer = get_tokenizer()
    fsm = StructureFSM(tokenizer)
    device = 'cpu'  # Use CPU for fast unit test

    print("Testing FSM constrained decoding...")

    # Use vocab_size for all tests
    vocab_size = tokenizer.vocab_size
    print(f"  tokenizer.vocab_size = {vocab_size}")
    print(f"  len(tokenizer) = {len(tokenizer)}")
    close_ans_id = tokenizer.convert_tokens_to_ids('</ANS>')
    print(f"  close_ans_id = {close_ans_id}")

    # Use len(tokenizer) which includes special tokens
    vocab_size = len(tokenizer)

    # Test 1: Verify FSM prevents </ANS> before <ANS>
    print("\nTest 1: FSM should prevent </ANS> before <ANS>")
    input_ids = torch.tensor([[
        tokenizer.convert_tokens_to_ids('<Q>'),
        tokenizer.convert_tokens_to_ids('<A>'),
    ]], device=device)

    # Create mock logits that heavily favor </ANS> (invalid token)
    logits = torch.ones(1, 2, vocab_size, device=device) * -100.0  # Match seq_len from input_ids
    logits[0, 1, close_ans_id] = 100.0  # High score for invalid token at position 1

    # Apply FSM constraints
    states = fsm.compute_states_from_tokens(input_ids)
    constrained_logits = fsm.constrain_logits(logits, states)

    # Check that </ANS> is masked (should be -inf) at the last position
    assert constrained_logits[0, 1, close_ans_id] == float('-inf'), \
        "FSM failed to mask </ANS> before <ANS>"
    print("  ✓ FSM correctly masks </ANS> in OUT state")

    # Test 2: Verify FSM allows </ANS> after <ANS>
    print("\nTest 2: FSM should allow </ANS> after <ANS>")
    input_ids = torch.tensor([[
        tokenizer.convert_tokens_to_ids('<Q>'),
        tokenizer.convert_tokens_to_ids('<A>'),
        tokenizer.convert_tokens_to_ids('<ANS>'),
        tokenizer.convert_tokens_to_ids('8'),  # Some digit
    ]], device=device)

    logits = torch.ones(1, 4, vocab_size, device=device) * -100.0  # Match seq_len
    logits[0, 3, close_ans_id] = 100.0  # Check position 3 (after digit '8')

    states = fsm.compute_states_from_tokens(input_ids)
    constrained_logits = fsm.constrain_logits(logits, states)

    # Check that </ANS> is NOT masked (should be > -inf) at position 3 (in IN_ANS state)
    assert constrained_logits[0, 3, close_ans_id] > float('-inf'), \
        "FSM incorrectly masked </ANS> in IN_ANS state"
    print("  ✓ FSM correctly allows </ANS> in IN_ANS state")

    # Test 3: Generate a few sequences with constrained decoding
    print("\nTest 3: Generate sequences with FSM constrained decoding")
    num_sequences = 10
    invalid_sequences = []

    for i in range(num_sequences):
        # Start with <Q> token
        input_ids = torch.tensor([[tokenizer.convert_tokens_to_ids('<Q>')]], device=device)
        max_len = 50

        for step in range(max_len):
            # Create random logits for full sequence (use full vocab size for generation)
            seq_len = input_ids.shape[1]
            logits = torch.randn(1, seq_len, vocab_size, device=device)

            # Apply FSM constraints
            states = fsm.compute_states_from_tokens(input_ids)
            constrained_logits = fsm.constrain_logits(logits, states)

            # Greedy decode the last position
            next_token = torch.argmax(constrained_logits[0, -1]).unsqueeze(0).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop if we hit <SEP>
            if next_token.item() == tokenizer.convert_tokens_to_ids('<SEP>'):
                break

        # Decode and validate
        sequence = tokenizer.decode(input_ids[0], skip_special_tokens=False)
        is_valid, error_msg = validate_tag_order(sequence, tokenizer)

        if not is_valid:
            invalid_sequences.append((i, sequence, error_msg))

    print(f"  Generated {num_sequences} sequences")
    print(f"  Valid sequences: {num_sequences - len(invalid_sequences)}/{num_sequences}")

    # Assert all sequences are valid
    if len(invalid_sequences) > 0:
        error_report = "\n".join([
            f"    Sequence {i}: {msg}\n      Output: {seq[:100]}..."
            for i, seq, msg in invalid_sequences[:3]
        ])
        pytest.fail(
            f"\n❌ FSM failed to prevent invalid sequences:\n{error_report}\n"
        )

    print("  ✓ All generated sequences have valid tag order")
    print("\n✓ All FSM constrained decoding tests passed!")


def test_validate_tag_order_function():
    """Test the validate_tag_order helper function."""
    tokenizer = get_tokenizer()

    # Valid sequences
    valid_sequences = [
        "<Q>9-1<A><ANS>8</ANS><SEP>",
        "<Q>37+58<A><ANS_SPLIT>9 5</ANS_SPLIT><SEP>",
        "<Q>test<A><ANS>42</ANS><ANS_SPLIT>4 2</ANS_SPLIT><SEP>",
        "<Q>test<A>",  # Incomplete but not invalid
    ]

    for seq in valid_sequences:
        is_valid, msg = validate_tag_order(seq, tokenizer)
        assert is_valid, f"Expected valid but got: {msg} for sequence: {seq}"

    # Invalid sequences
    invalid_sequences = [
        "</ANS><ANS>8</ANS>",  # Close before open
        "<Q>test</ANS>",  # Close without open
        "<ANS>8</ANS></ANS>",  # Multiple closes
        "<ANS_SPLIT>9</ANS_SPLIT></ANS_SPLIT>",  # Multiple closes
    ]

    for seq in invalid_sequences:
        is_valid, msg = validate_tag_order(seq, tokenizer)
        assert not is_valid, f"Expected invalid but got valid for sequence: {seq}"

    print("✓ validate_tag_order function tests passed")


if __name__ == "__main__":
    # Fix Unicode encoding on Windows
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=== DFA Canary Test Suite ===\n")

    # Test validation function first
    print("Testing validate_tag_order function...")
    test_validate_tag_order_function()
    print()

    # Run main canary test
    print("Running main canary test (no close before open)...")
    test_no_close_before_open()

    print("\n✓ All canary tests passed!")
