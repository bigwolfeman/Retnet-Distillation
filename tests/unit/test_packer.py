"""
Unit tests for sequence packing module.

Per tasks.md T029-T030: Test loss mask coverage and length weighting.
"""

import sys
from pathlib import Path

import pytest
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.generators.gen_a0_a1 import generate_a0_a1_batch
from data.generators.gen_a2 import generate_a2_batch
from train.packer import pack_examples, PackedSequence


def test_loss_mask_coverage():
    """
    T029: Test that loss mask covers only answer tokens.

    Requirement: Only tokens inside <ANS>...</ANS> and <ANS_SPLIT>...</ANS_SPLIT>
    should have mask=1 (True). Question tokens and delimiters should have mask=0 (False).
    """
    # Generate test records
    records = generate_a0_a1_batch(num_samples=10, seed_start=1000, split="test")

    # Pack sequences
    packed = pack_examples(records, max_tokens=512)

    # Verify loss mask is boolean
    assert packed.loss_mask.dtype == torch.bool, "loss_mask must be boolean tensor"

    # Verify loss mask has correct shape
    assert packed.loss_mask.shape == packed.token_ids.shape, \
        "loss_mask must have same shape as token_ids"

    # Verify at least some tokens have loss=True
    assert packed.loss_mask.sum() > 0, "loss_mask must have at least one True value"

    # Verify not all tokens have loss=True (questions should be False)
    assert packed.loss_mask.sum() < packed.total_length, \
        "loss_mask should not be True for all tokens (questions should be False)"

    # For this test, we expect loss mask to cover roughly 10-30% of tokens
    # (answer tokens are shorter than questions for simple A0-A1)
    coverage_ratio = packed.loss_mask.sum().item() / packed.total_length
    assert 0.05 <= coverage_ratio <= 0.5, \
        f"Loss mask coverage {coverage_ratio:.2%} should be between 5% and 50%"

    print(f"✓ Loss mask coverage test passed: {coverage_ratio:.1%} of tokens have loss=True")


def test_length_weighting():
    """
    T030: Test that length weighting formula is correct.

    Requirement: Apply weight = clip(8 / |answer_tokens|, 1.0, 4.0) per example.
    - Short answers (1-2 tokens): weight = 4.0
    - Medium answers (3-8 tokens): weight = 1.0-2.67
    - Long answers (>8 tokens): weight = 1.0
    """
    # Generate records with varying answer lengths
    records_a0_a1 = generate_a0_a1_batch(num_samples=5, seed_start=1000, split="test")  # Short answers
    records_a2 = generate_a2_batch(num_samples=5, seed_start=2000, split="test")  # Longer answers

    # Pack each set
    packed_short = pack_examples(records_a0_a1, max_tokens=256)
    packed_long = pack_examples(records_a2, max_tokens=512)

    # Verify weights are non-negative
    assert (packed_short.example_weights >= 0).all(), "Weights must be non-negative"
    assert (packed_long.example_weights >= 0).all(), "Weights must be non-negative"

    # Verify weights are in valid range [1.0, 4.0]
    assert (packed_short.example_weights >= 1.0).all(), "Weights must be >= 1.0"
    assert (packed_short.example_weights <= 4.0).all(), "Weights must be <= 4.0"
    assert (packed_long.example_weights >= 1.0).all(), "Weights must be >= 1.0"
    assert (packed_long.example_weights <= 4.0).all(), "Weights must be <= 4.0"

    # Verify short answers have higher weights than long answers
    avg_weight_short = packed_short.example_weights.mean().item()
    avg_weight_long = packed_long.example_weights.mean().item()

    print(f"  Average weight (short answers): {avg_weight_short:.2f}")
    print(f"  Average weight (long answers): {avg_weight_long:.2f}")

    assert avg_weight_short >= avg_weight_long, \
        "Short answers should have higher average weight than long answers"

    print("✓ Length weighting test passed")


def test_packed_sequence_validation():
    """Test that PackedSequence validates its constraints."""
    # Valid packed sequence
    valid = PackedSequence(
        token_ids=torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
        loss_mask=torch.tensor([False, False, True, True, False], dtype=torch.bool),
        example_weights=torch.tensor([1.0, 1.0, 2.0, 2.0, 1.0], dtype=torch.float),
        total_length=5
    )
    # Should not raise

    # Test total_length mismatch
    with pytest.raises(AssertionError, match="total_length"):
        PackedSequence(
            token_ids=torch.tensor([1, 2, 3], dtype=torch.long),
            loss_mask=torch.tensor([True, True, True], dtype=torch.bool),
            example_weights=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float),
            total_length=5  # Wrong!
        )

    # Test loss_mask shape mismatch
    with pytest.raises(AssertionError, match="loss_mask shape"):
        PackedSequence(
            token_ids=torch.tensor([1, 2, 3], dtype=torch.long),
            loss_mask=torch.tensor([True, True], dtype=torch.bool),  # Wrong shape!
            example_weights=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float),
            total_length=3
        )

    # Test no answer tokens (all False)
    with pytest.raises(AssertionError, match="at least one True"):
        PackedSequence(
            token_ids=torch.tensor([1, 2, 3], dtype=torch.long),
            loss_mask=torch.tensor([False, False, False], dtype=torch.bool),  # No answer tokens!
            example_weights=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float),
            total_length=3
        )

    # Test negative weights
    with pytest.raises(AssertionError, match="non-negative"):
        PackedSequence(
            token_ids=torch.tensor([1, 2, 3], dtype=torch.long),
            loss_mask=torch.tensor([False, True, True], dtype=torch.bool),
            example_weights=torch.tensor([1.0, -1.0, 1.0], dtype=torch.float),  # Negative!
            total_length=3
        )

    print("✓ PackedSequence validation test passed")


def test_packing_max_tokens():
    """Test that packing respects max_tokens limit."""
    # Generate many records
    records = generate_a2_batch(num_samples=100, seed_start=3000, split="test")

    # Pack with various max_tokens
    packed_128 = pack_examples(records, max_tokens=128)
    packed_512 = pack_examples(records, max_tokens=512)
    packed_2048 = pack_examples(records, max_tokens=2048)

    # Verify lengths respect max_tokens
    assert packed_128.total_length <= 128, "Packed length must not exceed max_tokens"
    assert packed_512.total_length <= 512, "Packed length must not exceed max_tokens"
    assert packed_2048.total_length <= 2048, "Packed length must not exceed max_tokens"

    # Verify larger max_tokens produces longer sequences
    assert packed_512.total_length >= packed_128.total_length, \
        "Larger max_tokens should produce longer sequences"
    assert packed_2048.total_length >= packed_512.total_length, \
        "Larger max_tokens should produce longer sequences"

    print(f"  Packed lengths: {packed_128.total_length}, {packed_512.total_length}, {packed_2048.total_length}")
    print("✓ Max tokens test passed")


if __name__ == "__main__":
    # Fix Unicode encoding on Windows
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=== Running Packer Unit Tests ===\n")

    print("Test 1: Loss mask coverage")
    test_loss_mask_coverage()

    print("\nTest 2: Length weighting")
    test_length_weighting()

    print("\nTest 3: PackedSequence validation")
    test_packed_sequence_validation()

    print("\nTest 4: Max tokens limit")
    test_packing_max_tokens()

    print("\n✓ All packer tests passed!")
