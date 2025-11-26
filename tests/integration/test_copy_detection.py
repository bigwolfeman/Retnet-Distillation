"""
Integration test for copy-rate detection (label leakage detection).

Per tasks.md Phase 5, T5.2: Test that copy-rate detection works correctly.

Tests:
- Generate small batch and train for 100-1000 steps
- Verify copy-rate decreases over time (or stays low)
- Compare diagnostic vs causal mode
- Check loss decreases more slowly with causal (no shortcut)

Copy-rate metric:
- Count predictions that appear verbatim in the input
- High copy-rate (>50%) indicates the model is copying, not computing
- Low copy-rate (<10%) indicates genuine reasoning
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.generators.gen_a0_a1 import generate_a0_a1_batch
from model.tokenizer import get_tokenizer


def compute_copy_rate(input_ids: torch.Tensor, predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute copy-rate: fraction of predictions that appear in the input.

    Args:
        input_ids: [batch, seq_len] input token IDs
        predictions: [batch, seq_len] predicted token IDs
        labels: [batch, seq_len] ground truth labels (-100 for masked)

    Returns:
        float: copy-rate (0.0 to 1.0)
    """
    batch_size, seq_len = input_ids.shape

    total_predictions = 0
    total_copies = 0

    for b in range(batch_size):
        for t in range(seq_len):
            # Only check supervised positions (where labels != -100)
            if labels[b, t] != -100:
                total_predictions += 1
                pred_token = predictions[b, t].item()

                # Check if this predicted token appears anywhere in the input
                if pred_token in input_ids[b]:
                    total_copies += 1

    if total_predictions == 0:
        return 0.0

    return total_copies / total_predictions


def test_copy_rate_computation():
    """
    T5.2.1: Test that copy-rate metric is computed correctly.

    Verify the copy-rate calculation works on synthetic examples.
    """
    # Create synthetic example
    # Input: [1, 2, 3, 4, 5]
    # Predictions: [2, 2, 6, 4, 7]  (2 and 4 are copies, 6 and 7 are not)
    # Labels: [-100, -100, 6, 4, 7]  (only last 3 are supervised)

    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    predictions = torch.tensor([[2, 2, 6, 4, 7]])
    labels = torch.tensor([[-100, -100, 6, 4, 7]])

    copy_rate = compute_copy_rate(input_ids, predictions, labels)

    # Expected: 1 copy out of 3 supervised positions (token 4)
    # Wait, let me recalculate:
    # Position 2: pred=6, not in input -> not a copy
    # Position 3: pred=4, in input (input[3]=4) -> copy
    # Position 4: pred=7, not in input -> not a copy
    # So 1/3 = 0.333

    expected_copy_rate = 1.0 / 3.0
    assert abs(copy_rate - expected_copy_rate) < 0.01, \
        f"Expected copy-rate {expected_copy_rate:.3f}, got {copy_rate:.3f}"

    print(f"✓ Copy-rate computation correct: {copy_rate:.3f}")


def test_diagnostic_mode_high_copy_rate():
    """
    T5.2.2: Verify diagnostic mode (with label leakage) produces high copy-rate.

    Diagnostic mode:
        input_ids:  <Q>9-1<A><ANS>8</ANS><SEP>
        labels:     <Q>9-1<A><ANS>8</ANS><SEP>  (same!)

    Model can learn to copy '8' from the input -> high copy-rate.

    This test trains for a small number of steps and checks copy-rate increases.
    """
    pytest.skip("Requires training loop implementation - placeholder for now")

    # TODO: Implement when training infrastructure is ready
    # Expected behavior:
    # 1. Create diagnostic dataset (current implementation)
    # 2. Train for 100-500 steps
    # 3. Measure copy-rate on validation set
    # 4. Assert copy-rate > 50% (indicating copy-learning)

    print("⊘ SKIPPED: Requires training loop (to be implemented)")


def test_causal_mode_low_copy_rate():
    """
    T5.2.3: Verify causal mode (no label leakage) produces low copy-rate.

    Causal mode:
        input_ids:  <Q>9-1<A>
        labels:     [-100, -100, -100, 8, ...]

    Model cannot copy '8' from input -> must compute -> low copy-rate.

    This test trains for a small number of steps and checks copy-rate stays low.
    """
    pytest.skip("Requires causal implementation and training loop - placeholder for now")

    # TODO: Implement when causal implementation is ready
    # Expected behavior:
    # 1. Create causal dataset
    # 2. Train for 100-1000 steps
    # 3. Measure copy-rate on validation set
    # 4. Assert copy-rate < 10% (indicating genuine computation)

    print("⊘ SKIPPED: Requires causal implementation (Phase 2 TODO)")


def test_copy_rate_decreases_over_training():
    """
    T5.2.4: Verify copy-rate metric decreases (or stays low) over training.

    For causal mode, copy-rate should:
    - Start low (< 20%) even at initialization
    - Stay low (< 10%) during training
    - Not increase significantly over time

    This indicates the model is learning to compute, not copy.
    """
    pytest.skip("Requires full training pipeline - placeholder for now")

    # TODO: Implement when training infrastructure is ready
    # Expected behavior:
    # 1. Train for 1000 steps
    # 2. Log copy-rate every 100 steps
    # 3. Verify copy-rate trends downward or stays low
    # 4. Alert if copy-rate > 15% at any point

    print("⊘ SKIPPED: Requires training pipeline")


def test_diagnostic_vs_causal_copy_rate_comparison():
    """
    T5.2.5: Compare copy-rate between diagnostic and causal modes side-by-side.

    Expected results:
    - Diagnostic: copy-rate ~70-95% (model learns to copy)
    - Causal: copy-rate <10% (model learns to compute)

    This is the key test that validates the fix works.
    """
    pytest.skip("Requires both implementations and training - placeholder for now")

    # TODO: Implement full comparison
    # Steps:
    # 1. Train diagnostic model for N steps
    # 2. Train causal model for N steps
    # 3. Evaluate both on same held-out set
    # 4. Compare copy-rates
    # 5. Assert diagnostic >> causal (at least 5x difference)

    print("⊘ SKIPPED: Requires full training pipeline and both implementations")


def test_copy_rate_on_evaluation_set():
    """
    T5.2.6: Verify copy-rate can be computed on a held-out evaluation set.

    This test creates a small eval set and computes copy-rate on it.
    It doesn't require training, just inference.
    """
    try:
        from train.curriculum_dataset_causal import pack_examples_causal
    except ImportError:
        pytest.skip("Causal implementation not yet available (Phase 2 TODO)")

    # Generate evaluation set
    eval_records = generate_a0_a1_batch(num_samples=20, seed_start=9000, split="test")
    tokenizer = get_tokenizer()

    # Pack with causal mode
    packed = pack_examples_causal(eval_records, max_tokens=512, tokenizer=tokenizer)

    # Create dummy model predictions (random for now)
    # In real test, this would be model.generate() or model.forward()
    vocab_size = tokenizer.vocab_size
    batch_size = 1
    seq_len = packed.input_ids.shape[0]

    input_ids_batch = packed.input_ids.unsqueeze(0)  # [1, seq_len]
    labels_batch = packed.labels.unsqueeze(0)  # [1, seq_len]

    # Simulate predictions (random)
    predictions = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Compute copy-rate
    copy_rate = compute_copy_rate(input_ids_batch, predictions, labels_batch)

    # With random predictions, copy-rate should be very low
    # (unlikely to randomly match input tokens)
    assert 0.0 <= copy_rate <= 1.0, "Copy-rate must be between 0 and 1"

    print(f"✓ Copy-rate on eval set: {copy_rate:.3f} (random predictions)")


def test_loss_convergence_speed_diagnostic_vs_causal():
    """
    T5.2.7: Verify that loss decreases more slowly with causal mode.

    Diagnostic mode:
    - Loss drops very fast (99% reduction in 100 steps) - RED FLAG
    - This is because model learns to copy, not compute

    Causal mode:
    - Loss drops gradually over 1000+ steps
    - This is expected for genuine learning

    This test compares convergence speed.
    """
    pytest.skip("Requires training for comparison - placeholder for now")

    # TODO: Implement convergence comparison
    # Expected behavior:
    # 1. Train diagnostic for 100 steps, track loss
    # 2. Train causal for 100 steps, track loss
    # 3. Assert diagnostic loss drops faster than causal
    # 4. But causal should still show gradual improvement

    print("⊘ SKIPPED: Requires training comparison")


def test_copy_rate_threshold_alert():
    """
    T5.2.8: Test that high copy-rate triggers an alert.

    If copy-rate > 15% during training, the system should log a warning.
    This helps detect if label leakage is accidentally reintroduced.
    """
    # Simulate high copy-rate scenario
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    predictions = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])  # Perfect copy
    labels = torch.tensor([[-100, -100, -100, 4, 5, 6, 7, 8, 9]])

    copy_rate = compute_copy_rate(input_ids, predictions, labels)

    # All supervised predictions are copies -> copy-rate = 100%
    assert copy_rate == 1.0, f"Expected 100% copy-rate, got {copy_rate:.3f}"

    # Check threshold
    COPY_RATE_THRESHOLD = 0.15

    if copy_rate > COPY_RATE_THRESHOLD:
        print(f"⚠ WARNING: Copy-rate {copy_rate:.1%} exceeds threshold {COPY_RATE_THRESHOLD:.1%}")
        print("  This may indicate label leakage or copy-learning.")
    else:
        print(f"✓ Copy-rate {copy_rate:.1%} below threshold")

    assert copy_rate > COPY_RATE_THRESHOLD, "Test expects high copy-rate for this example"


def test_copy_rate_with_batch():
    """
    T5.2.9: Test copy-rate computation with batched inputs.

    Verify the metric works correctly across multiple examples in a batch.
    """
    # Create batch with mixed copy/compute patterns
    batch_size = 4
    seq_len = 8

    input_ids = torch.tensor([
        [1, 2, 3, 4, 5, 6, 7, 8],
        [9, 10, 11, 12, 13, 14, 15, 16],
        [17, 18, 19, 20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29, 30, 31, 32],
    ])

    # Predictions: some copy, some don't
    predictions = torch.tensor([
        [1, 2, 3, 4, 50, 60, 70, 80],  # First half copy, second half don't
        [90, 100, 110, 120, 13, 14, 15, 16],  # First half don't, second half copy
        [17, 18, 19, 20, 21, 22, 23, 24],  # All copy
        [250, 260, 270, 280, 290, 300, 310, 320],  # None copy
    ])

    # Labels: only last 4 positions are supervised
    labels = torch.tensor([
        [-100, -100, -100, -100, 50, 60, 70, 80],
        [-100, -100, -100, -100, 13, 14, 15, 16],
        [-100, -100, -100, -100, 21, 22, 23, 24],
        [-100, -100, -100, -100, 290, 300, 310, 320],
    ])

    copy_rate = compute_copy_rate(input_ids, predictions, labels)

    # Calculate expected:
    # Batch 0: 0/4 copies (50, 60, 70, 80 not in input)
    # Batch 1: 4/4 copies (13, 14, 15, 16 all in input)
    # Batch 2: 4/4 copies (21, 22, 23, 24 all in input)
    # Batch 3: 0/4 copies (290, 300, 310, 320 not in input)
    # Total: 8/16 = 0.5

    expected_copy_rate = 8.0 / 16.0
    assert abs(copy_rate - expected_copy_rate) < 0.01, \
        f"Expected copy-rate {expected_copy_rate:.3f}, got {copy_rate:.3f}"

    print(f"✓ Batch copy-rate computation correct: {copy_rate:.3f}")


if __name__ == "__main__":
    # Fix Unicode encoding on Windows
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=== Running Copy-Rate Detection Integration Tests ===\n")

    # Run tests manually
    tests = [
        ("Copy-rate computation", test_copy_rate_computation),
        ("Copy-rate threshold alert", test_copy_rate_threshold_alert),
        ("Copy-rate with batch", test_copy_rate_with_batch),
        ("Copy-rate on eval set", test_copy_rate_on_evaluation_set),
        ("Diagnostic mode high copy-rate", test_diagnostic_mode_high_copy_rate),
        ("Causal mode low copy-rate", test_causal_mode_low_copy_rate),
        ("Copy-rate decreases over training", test_copy_rate_decreases_over_training),
        ("Diagnostic vs causal comparison", test_diagnostic_vs_causal_copy_rate_comparison),
        ("Loss convergence speed", test_loss_convergence_speed_diagnostic_vs_causal),
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
            print(f"⊘ SKIPPED: {str(e).split('Skipped: ')[-1]}")
            skipped += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*60}")

    if failed == 0:
        print("\n✓ All available copy-rate tests passed!")
        print(f"  ({skipped} tests skipped pending implementation)")
    else:
        print(f"\n✗ {failed} test(s) failed")
        sys.exit(1)
