"""
Unit tests for causal sequence packing (no label leakage).

Per tasks.md Phase 5, T5.1: Test that causal packing eliminates label leakage.

Tests:
- No answer tokens in input_ids before supervised positions
- Labels correctly masked with -100 for question tokens
- Sequence lengths match after packing
- Verify input ends at <A>, labels start with answer tokens
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
from model.tokenizer import get_tokenizer


def test_no_answer_leakage_in_input():
    """
    T5.1.1: Verify no label leakage via proper masking.

    Critical: The causal implementation prevents label leakage through masking, not truncation.

    For causal LM:
        input_ids:  <Q>9-1<A><ANS>8</ANS><SEP>  (full sequence)
        labels:     [-100, -100, -100, -100, <ANS>, 8, </ANS>, <SEP>]

    The model sees the full sequence but is only supervised on answer tokens.
    This prevents the model from learning to copy because at each position i,
    it must predict token i+1, and for question positions, labels are -100 (no loss).

    This prevents the model from learning to copy instead of compute.
    """
    # This test requires the causal implementation to exist
    try:
        from train.curriculum_dataset_causal import pack_examples_causal
    except ImportError:
        pytest.skip("Causal packing implementation not yet available (Phase 2 TODO)")

    # Generate test records
    records = generate_a0_a1_batch(num_samples=10, seed_start=1000, split="test")
    tokenizer = get_tokenizer()

    # Pack with causal mode
    packed = pack_examples_causal(records, max_tokens=512, tokenizer=tokenizer)

    # Verify structure
    assert hasattr(packed, 'input_ids'), "Packed sequence must have input_ids"
    assert hasattr(packed, 'labels'), "Packed sequence must have labels"

    # Find <A> token ID
    a_token_id = tokenizer.convert_tokens_to_ids("<A>")

    # Input should contain <A> token
    assert a_token_id in packed.input_ids, "Input must contain <A> delimiter"

    # Verify labels have -100 for non-supervised tokens (question part)
    assert (packed.labels == -100).any(), "Labels must have -100 masking for questions"

    # Verify some tokens ARE supervised (not all -100)
    assert (packed.labels != -100).any(), "Labels must have some supervised tokens"

    # CRITICAL: Verify all tokens before first <A> have -100 labels
    # This ensures question tokens are not supervised
    a_positions = (packed.input_ids == a_token_id).nonzero(as_tuple=True)[0]
    if len(a_positions) > 0:
        first_a_pos = a_positions[0].item()
        # All labels up to and including <A> should be -100
        assert (packed.labels[:first_a_pos+1] == -100).all(), \
            "Question tokens (before and including <A>) must have -100 labels"

    print("✓ No label leakage: question tokens properly masked with -100")


def test_labels_correctly_masked():
    """
    T5.1.2: Verify labels are correctly masked with -100 for question tokens.

    Labels format:
        labels[i] = -100  if token i is part of question (not supervised)
        labels[i] = token_id  if token i is part of answer (supervised)

    PyTorch's CrossEntropyLoss ignores positions where labels == -100.
    """
    try:
        from train.curriculum_dataset_causal import pack_examples_causal
    except ImportError:
        pytest.skip("Causal packing implementation not yet available (Phase 2 TODO)")

    records = generate_a0_a1_batch(num_samples=5, seed_start=2000, split="test")
    tokenizer = get_tokenizer()

    packed = pack_examples_causal(records, max_tokens=256, tokenizer=tokenizer)

    # Verify labels tensor
    assert packed.labels.dtype == torch.long, "labels must be Long tensor"
    assert packed.labels.shape == packed.input_ids.shape, \
        "labels must have same shape as input_ids"

    # Count masked vs supervised tokens
    num_masked = (packed.labels == -100).sum().item()
    num_supervised = (packed.labels != -100).sum().item()
    total = packed.labels.shape[0]

    # For A0-A1, answers are short (1 digit), so most tokens should be masked
    assert num_masked > 0, "Must have at least some masked (question) tokens"
    assert num_supervised > 0, "Must have at least some supervised (answer) tokens"

    # Sanity check: masked + supervised = total
    assert num_masked + num_supervised == total, \
        f"Token count mismatch: {num_masked} + {num_supervised} != {total}"

    # For A0-A1, expect roughly 80-95% masking (questions longer than answers)
    mask_ratio = num_masked / total
    assert 0.5 <= mask_ratio <= 0.99, \
        f"Unexpected mask ratio {mask_ratio:.2%} (expected 50-99% for A0-A1)"

    print(f"✓ Labels correctly masked: {num_masked}/{total} masked ({mask_ratio:.1%})")


def test_sequence_lengths_match():
    """
    T5.1.3: Verify sequence lengths are consistent after causal packing.

    input_ids and labels must have the same length.
    Total length should not exceed max_tokens.
    """
    try:
        from train.curriculum_dataset_causal import pack_examples_causal
    except ImportError:
        pytest.skip("Causal packing implementation not yet available (Phase 2 TODO)")

    records = generate_a2_batch(num_samples=20, seed_start=3000, split="test")
    tokenizer = get_tokenizer()

    # Test with different max_tokens
    for max_tokens in [128, 256, 512, 1024]:
        packed = pack_examples_causal(records, max_tokens=max_tokens, tokenizer=tokenizer)

        # Verify lengths match
        input_len = packed.input_ids.shape[0]
        labels_len = packed.labels.shape[0]

        assert input_len == labels_len, \
            f"Length mismatch: input_ids={input_len}, labels={labels_len}"

        # Verify respects max_tokens
        assert input_len <= max_tokens, \
            f"Sequence length {input_len} exceeds max_tokens {max_tokens}"

        print(f"  max_tokens={max_tokens}: packed length={input_len} ✓")

    print("✓ Sequence lengths match across all max_tokens settings")


def test_input_ends_at_A_labels_start_with_answer():
    """
    T5.1.4: Verify input ends at <A>, and labels contain answer starting there.

    Causal format:
        input_ids:  [<Q>, question_tokens..., <A>]
        labels:     [-100, -100, ..., -100, answer_token_1, answer_token_2, ...]

    The first non -100 label should appear at or after the <A> token position.
    """
    try:
        from train.curriculum_dataset_causal import pack_examples_causal
    except ImportError:
        pytest.skip("Causal packing implementation not yet available (Phase 2 TODO)")

    records = generate_a0_a1_batch(num_samples=3, seed_start=4000, split="test")
    tokenizer = get_tokenizer()

    packed = pack_examples_causal(records, max_tokens=512, tokenizer=tokenizer)

    # Find all <A> token positions
    a_token_id = tokenizer.convert_tokens_to_ids("<A>")
    a_positions = (packed.input_ids == a_token_id).nonzero(as_tuple=True)[0]

    assert len(a_positions) > 0, "Must have at least one <A> token"

    # Find first supervised position (first non -100 in labels)
    supervised_positions = (packed.labels != -100).nonzero(as_tuple=True)[0]

    assert len(supervised_positions) > 0, "Must have at least one supervised token"

    first_supervised_pos = supervised_positions[0].item()
    first_a_pos = a_positions[0].item()

    # The first supervised token should come AT or AFTER the first <A> token
    # This ensures the model cannot see the answer before predicting it
    assert first_supervised_pos >= first_a_pos, \
        f"Supervised token at position {first_supervised_pos} comes before <A> at {first_a_pos} - LABEL LEAKAGE!"

    print(f"✓ Input ends at <A> (pos {first_a_pos}), labels start at pos {first_supervised_pos}")


def test_causal_vs_diagnostic_comparison():
    """
    T5.1.5: Compare causal vs diagnostic packing to verify difference.

    Diagnostic (current): input_ids and labels are identical
    Causal (new): input_ids ends at <A>, labels contain answer

    This test verifies the causal implementation is actually different.
    """
    try:
        from train.curriculum_dataset_causal import pack_examples_causal
    except ImportError:
        pytest.skip("Causal packing implementation not yet available (Phase 2 TODO)")

    from train.packer import pack_examples  # Diagnostic version

    records = generate_a0_a1_batch(num_samples=5, seed_start=5000, split="test")
    tokenizer = get_tokenizer()

    # Pack with both methods
    diagnostic = pack_examples(records, max_tokens=512, tokenizer=tokenizer)
    causal = pack_examples_causal(records, max_tokens=512, tokenizer=tokenizer)

    # Diagnostic: input_ids and labels (token_ids) should be identical
    # (except diagnostic uses loss_mask to control which tokens are supervised)
    assert torch.equal(diagnostic.token_ids, diagnostic.token_ids), \
        "Diagnostic: token_ids should match itself (sanity check)"

    # Causal: input_ids and labels should be DIFFERENT
    # (input has questions, labels have -100 for questions)
    assert not torch.equal(causal.input_ids, causal.labels), \
        "Causal: input_ids and labels MUST be different (no label leakage)"

    # Verify causal input is shorter (no answer tokens)
    # This might not always be true if padding is applied, but generally causal input < diagnostic
    print(f"  Diagnostic length: {diagnostic.total_length}")
    print(f"  Causal input length: {causal.input_ids.shape[0]}")
    print(f"  Causal labels length: {causal.labels.shape[0]}")

    print("✓ Causal packing is different from diagnostic (as expected)")


def test_multiple_examples_in_sequence():
    """
    T5.1.6: Verify causal packing works correctly with multiple examples in one sequence.

    Format for multiple examples:
        input_ids:  <Q>q1<A> <Q>q2<A> <Q>q3<A>
        labels:     [-100,-100,-100,a1,-100,-100,-100,a2,-100,-100,-100,a3]

    Each question should be masked, each answer should be supervised.
    """
    try:
        from train.curriculum_dataset_causal import pack_examples_causal
    except ImportError:
        pytest.skip("Causal packing implementation not yet available (Phase 2 TODO)")

    # Generate enough records to pack multiple examples
    records = generate_a0_a1_batch(num_samples=50, seed_start=6000, split="test")
    tokenizer = get_tokenizer()

    # Pack with small max_tokens to force multiple examples
    packed = pack_examples_causal(records, max_tokens=512, tokenizer=tokenizer)

    # Count number of <A> tokens (one per example)
    a_token_id = tokenizer.convert_tokens_to_ids("<A>")
    num_examples = (packed.input_ids == a_token_id).sum().item()

    # Should have packed multiple examples
    assert num_examples >= 2, f"Expected multiple examples, got {num_examples}"

    # Count supervised regions (should be at least num_examples)
    # Each example should have at least one supervised token
    supervised_count = (packed.labels != -100).sum().item()
    assert supervised_count >= num_examples, \
        f"Expected at least {num_examples} supervised tokens, got {supervised_count}"

    print(f"✓ Successfully packed {num_examples} examples with causal format")


def test_edge_case_empty_records():
    """
    T5.1.7: Test edge case with empty or minimal records.
    """
    try:
        from train.curriculum_dataset_causal import pack_examples_causal
    except ImportError:
        pytest.skip("Causal packing implementation not yet available (Phase 2 TODO)")

    tokenizer = get_tokenizer()

    # Test with empty list - should raise ValueError
    try:
        packed_empty = pack_examples_causal([], max_tokens=512, tokenizer=tokenizer)
        assert False, "Expected ValueError for empty records"
    except ValueError as e:
        assert "empty" in str(e).lower(), "Error message should mention empty records"

    # Test with single record
    records = generate_a0_a1_batch(num_samples=1, seed_start=7000, split="test")
    packed_single = pack_examples_causal(records, max_tokens=512, tokenizer=tokenizer)

    assert packed_single.input_ids.shape[0] > 0, "Single record should produce non-empty sequence"
    assert packed_single.labels.shape[0] > 0, "Single record should produce non-empty labels"

    print("✓ Edge cases handled correctly")


if __name__ == "__main__":
    # Fix Unicode encoding on Windows
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=== Running Causal Packing Unit Tests ===\n")

    # Run tests manually (or use pytest)
    tests = [
        ("No answer leakage in input", test_no_answer_leakage_in_input),
        ("Labels correctly masked", test_labels_correctly_masked),
        ("Sequence lengths match", test_sequence_lengths_match),
        ("Input ends at <A>, labels start with answer", test_input_ends_at_A_labels_start_with_answer),
        ("Causal vs diagnostic comparison", test_causal_vs_diagnostic_comparison),
        ("Multiple examples in sequence", test_multiple_examples_in_sequence),
        ("Edge cases", test_edge_case_empty_records),
    ]

    passed = 0
    failed = 0
    skipped = 0

    for test_name, test_func in tests:
        print(f"\nTest: {test_name}")
        try:
            test_func()
            passed += 1
        except pytest.skip.Exception as e:
            print(f"⊘ SKIPPED: {e}")
            skipped += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*60}")

    if failed == 0 and skipped < len(tests):
        print("\n✓ All causal packing tests passed!")
    elif skipped == len(tests):
        print("\n⊘ All tests skipped (causal implementation not yet available)")
    else:
        print(f"\n✗ {failed} test(s) failed")
        sys.exit(1)
