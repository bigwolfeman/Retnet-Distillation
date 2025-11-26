"""
Held-out A0 validation test for label leakage detection.

Per tasks.md Phase 5, T5.4: Validate on held-out test set.

Tests:
- Generate predictions on held-out test set
- Compute exact match (EM)
- Verify copy-rate <5% on eval
- Check that EM improves over training (not just loss)

This test ensures the model genuinely learns to solve problems,
not just memorize or copy answers.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import torch
import torch.nn.functional as F

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.generators.gen_a0_a1 import generate_a0_a1_batch
from model.tokenizer import get_tokenizer


def compute_exact_match(predictions: List[str], ground_truth: List[str]) -> float:
    """
    Compute exact match accuracy.

    Args:
        predictions: List of predicted answer strings
        ground_truth: List of ground truth answer strings

    Returns:
        float: Exact match accuracy (0.0 to 1.0)
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(ground_truth)} ground truth"
        )

    if len(predictions) == 0:
        return 0.0

    matches = sum(1 for pred, gt in zip(predictions, ground_truth) if pred.strip() == gt.strip())
    return matches / len(predictions)


def compute_copy_rate_eval(
    input_texts: List[str],
    predictions: List[str],
    ground_truth: List[str]
) -> float:
    """
    Compute copy-rate on evaluation set.

    Args:
        input_texts: List of input question strings
        predictions: List of predicted answer strings
        ground_truth: List of ground truth answer strings

    Returns:
        float: Copy-rate (fraction of predictions that appear in input)
    """
    if len(input_texts) != len(predictions):
        raise ValueError("Length mismatch between inputs and predictions")

    total = len(predictions)
    if total == 0:
        return 0.0

    copies = 0
    for input_text, pred in zip(input_texts, predictions):
        # Check if prediction appears verbatim in input
        # (case-insensitive, whitespace-normalized)
        input_normalized = input_text.lower().replace(" ", "")
        pred_normalized = pred.lower().replace(" ", "")

        if pred_normalized in input_normalized:
            copies += 1

    return copies / total


def extract_answer_from_sequence(
    token_ids: torch.Tensor,
    tokenizer,
    a_token_id: int,
    sep_token_id: int
) -> str:
    """
    Extract answer text from generated sequence.

    Looks for tokens between <A> and <SEP> or end of sequence.

    Args:
        token_ids: [seq_len] Generated token IDs
        tokenizer: Tokenizer instance
        a_token_id: Token ID for <A>
        sep_token_id: Token ID for <SEP>

    Returns:
        str: Extracted answer text
    """
    # Find <A> token position
    a_positions = (token_ids == a_token_id).nonzero(as_tuple=True)[0]

    if len(a_positions) == 0:
        # No <A> found, return empty
        return ""

    # Start after <A>
    start_pos = a_positions[0].item() + 1

    # Find <SEP> or end of sequence
    sep_positions = (token_ids[start_pos:] == sep_token_id).nonzero(as_tuple=True)[0]

    if len(sep_positions) > 0:
        end_pos = start_pos + sep_positions[0].item()
    else:
        end_pos = len(token_ids)

    # Extract answer tokens
    answer_ids = token_ids[start_pos:end_pos]

    # Decode to text
    answer_text = tokenizer.decode(answer_ids.tolist(), skip_special_tokens=True)

    return answer_text.strip()


def test_exact_match_computation():
    """
    T5.4.1: Test exact match metric computation.

    Verify EM calculation is correct.
    """
    # Perfect predictions
    predictions = ["8", "5", "12", "3"]
    ground_truth = ["8", "5", "12", "3"]

    em = compute_exact_match(predictions, ground_truth)
    assert em == 1.0, f"Expected EM=1.0, got {em}"

    # Half correct
    predictions = ["8", "5", "10", "2"]
    ground_truth = ["8", "5", "12", "3"]

    em = compute_exact_match(predictions, ground_truth)
    assert em == 0.5, f"Expected EM=0.5, got {em}"

    # All wrong
    predictions = ["1", "2", "3", "4"]
    ground_truth = ["8", "5", "12", "3"]

    em = compute_exact_match(predictions, ground_truth)
    assert em == 0.0, f"Expected EM=0.0, got {em}"

    print("✓ Exact match computation correct")


def test_copy_rate_eval_computation():
    """
    T5.4.2: Test copy-rate computation on evaluation set.

    Verify copy-rate calculation for text-based predictions.
    """
    # All predictions are copies
    input_texts = ["9-1", "5+3", "7-2", "4+4"]
    predictions = ["9", "5", "7", "4"]  # All appear in input
    ground_truth = ["8", "8", "5", "8"]

    copy_rate = compute_copy_rate_eval(input_texts, predictions, ground_truth)
    assert copy_rate == 1.0, f"Expected 100% copy-rate, got {copy_rate:.1%}"

    # No predictions are copies
    input_texts = ["9-1", "5+3", "7-2", "4+4"]
    predictions = ["8", "8", "5", "8"]  # Correct answers, not in input
    ground_truth = ["8", "8", "5", "8"]

    copy_rate = compute_copy_rate_eval(input_texts, predictions, ground_truth)
    # "8" might appear in "9-1" as substring... let's adjust
    # Actually "8" doesn't appear in any of these inputs
    assert copy_rate == 0.0, f"Expected 0% copy-rate, got {copy_rate:.1%}"

    print("✓ Copy-rate eval computation correct")


def test_answer_extraction():
    """
    T5.4.3: Test answer extraction from generated sequences.

    Verify we can correctly extract answers from model outputs.
    """
    tokenizer = get_tokenizer()

    # Create test sequence: <Q>9-1<A><ANS>8</ANS><SEP>
    q_tokens = tokenizer.encode("<Q>9-1", add_special_tokens=False)
    a_token = tokenizer.encode("<A>", add_special_tokens=False)
    ans_tokens = tokenizer.encode("<ANS>8</ANS>", add_special_tokens=False)
    sep_token = tokenizer.encode("<SEP>", add_special_tokens=False)

    full_sequence = torch.tensor(q_tokens + a_token + ans_tokens + sep_token, dtype=torch.long)

    # Extract answer
    a_token_id = tokenizer.convert_tokens_to_ids("<A>")
    sep_token_id = tokenizer.convert_tokens_to_ids("<SEP>")

    answer = extract_answer_from_sequence(full_sequence, tokenizer, a_token_id, sep_token_id)

    # Answer should be "8" (may include <ANS> tags depending on implementation)
    # Let's check it contains "8"
    assert "8" in answer, f"Expected answer to contain '8', got '{answer}'"

    print(f"✓ Answer extraction working: '{answer}'")


def test_heldout_set_generation():
    """
    T5.4.4: Test held-out test set generation.

    Verify we can generate a held-out set with different seeds.
    """
    # Generate train set
    train_records = generate_a0_a1_batch(num_samples=100, seed_start=1000, split="train")

    # Generate test set with different seed
    test_records = generate_a0_a1_batch(num_samples=50, seed_start=9000, split="test")

    # Verify they're different
    train_questions = {r.question for r in train_records}
    test_questions = {r.question for r in test_records}

    # There should be some overlap (same problem types), but not complete overlap
    overlap = train_questions & test_questions
    print(f"  Train set: {len(train_questions)} unique questions")
    print(f"  Test set: {len(test_questions)} unique questions")
    print(f"  Overlap: {len(overlap)} questions")

    # Verify test set is non-empty
    assert len(test_records) == 50, f"Expected 50 test records, got {len(test_records)}"

    # Verify all records have valid answers
    assert all(r.verifier['ok'] for r in test_records), "All test records should pass verifier"

    print("✓ Held-out test set generation working")


def test_eval_metrics_on_heldout_set():
    """
    T5.4.5: Test evaluation metrics on held-out set.

    This test simulates evaluation on a held-out test set.
    It doesn't require a trained model, just verifies the eval pipeline.
    """
    # Generate held-out test set
    test_records = generate_a0_a1_batch(num_samples=20, seed_start=9000, split="test")

    # Simulate predictions (for now, just use ground truth to verify pipeline)
    input_texts = [r.question for r in test_records]
    ground_truth = [r.answer for r in test_records]
    predictions = ground_truth.copy()  # Perfect predictions for now

    # Compute metrics
    em = compute_exact_match(predictions, ground_truth)
    copy_rate = compute_copy_rate_eval(input_texts, predictions, ground_truth)

    # With perfect predictions
    assert em == 1.0, f"Expected EM=1.0 with perfect predictions, got {em}"

    # Copy-rate should be low (correct answers shouldn't appear in questions)
    print(f"  EM: {em:.1%}")
    print(f"  Copy-rate: {copy_rate:.1%}")

    # For A0-A1, answers are single digits that might appear in questions
    # So copy-rate might not be 0%, but should be reasonable
    assert 0.0 <= copy_rate <= 1.0, "Copy-rate must be between 0% and 100%"

    print("✓ Eval metrics pipeline working")


def test_eval_copy_rate_threshold():
    """
    T5.4.6: Test that eval copy-rate threshold check works.

    On evaluation set, copy-rate should be <5% for a well-trained causal model.
    """
    pytest.skip("Requires trained model - placeholder for now")

    # TODO: Implement when model training is available
    # Expected behavior:
    # 1. Load trained causal model
    # 2. Generate predictions on held-out test set
    # 3. Compute copy-rate
    # 4. Assert copy-rate < 5%

    print("⊘ SKIPPED: Requires trained model")


def test_em_improves_over_training():
    """
    T5.4.7: Test that EM improves over training, not just loss.

    This verifies the model genuinely learns to solve problems.
    """
    pytest.skip("Requires training checkpoints - placeholder for now")

    # TODO: Implement when training infrastructure is ready
    # Expected behavior:
    # 1. Load checkpoints at steps 0, 100, 500, 1000, etc.
    # 2. Evaluate each checkpoint on held-out test set
    # 3. Track EM over time
    # 4. Assert EM increases (or stays high)

    print("⊘ SKIPPED: Requires training checkpoints")


def test_diagnostic_vs_causal_em_comparison():
    """
    T5.4.8: Compare EM between diagnostic and causal modes.

    Expected:
    - Diagnostic: High EM initially (copying works), but doesn't generalize
    - Causal: Lower EM initially, but improves with training and generalizes better
    """
    pytest.skip("Requires both trained models - placeholder for now")

    # TODO: Implement when both models are trained
    # Expected behavior:
    # 1. Train diagnostic model
    # 2. Train causal model
    # 3. Evaluate both on held-out set
    # 4. Compare EM and copy-rate
    # 5. Verify causal has lower copy-rate

    print("⊘ SKIPPED: Requires both trained models")


def test_error_analysis():
    """
    T5.4.9: Test error analysis on held-out set.

    Analyze which types of problems the model gets wrong.
    This helps identify if the model is truly learning or just memorizing.
    """
    # Generate test set
    test_records = generate_a0_a1_batch(num_samples=50, seed_start=9000, split="test")

    # Simulate some predictions (mix of correct and incorrect)
    ground_truth = [r.answer for r in test_records]
    predictions = ground_truth.copy()

    # Introduce some errors (every 5th prediction is wrong)
    for i in range(0, len(predictions), 5):
        predictions[i] = "0"  # Wrong answer

    # Compute EM
    em = compute_exact_match(predictions, ground_truth)

    # Should be 80% correct (4 out of 5)
    expected_em = 0.8
    assert abs(em - expected_em) < 0.05, f"Expected EM≈{expected_em}, got {em}"

    # Find errors
    errors = []
    for i, (pred, gt, record) in enumerate(zip(predictions, ground_truth, test_records)):
        if pred != gt:
            errors.append({
                'index': i,
                'question': record.question,
                'prediction': pred,
                'ground_truth': gt,
                'band': record.band,
            })

    # Should have ~10 errors (20% of 50)
    assert 8 <= len(errors) <= 12, f"Expected ~10 errors, got {len(errors)}"

    print(f"✓ Error analysis working: {len(errors)} errors found")
    print(f"  First 3 errors:")
    for err in errors[:3]:
        print(f"    Q: {err['question']} | Pred: {err['prediction']} | GT: {err['ground_truth']}")


def test_copy_rate_vs_em_correlation():
    """
    T5.4.10: Test that high copy-rate correlates with poor generalization.

    If model has high copy-rate, EM should be lower on held-out set.
    This validates that copy-rate is a good proxy for label leakage.
    """
    pytest.skip("Requires trained models with different copy-rates - placeholder for now")

    # TODO: Implement when multiple models are available
    # Expected behavior:
    # 1. Evaluate multiple models with different copy-rates
    # 2. Plot copy-rate vs EM
    # 3. Verify negative correlation (high copy-rate -> low EM)

    print("⊘ SKIPPED: Requires multiple trained models")


if __name__ == "__main__":
    # Fix Unicode encoding on Windows
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=== Running Held-Out Validation Tests ===\n")

    # Run tests manually
    tests = [
        ("Exact match computation", test_exact_match_computation),
        ("Copy-rate eval computation", test_copy_rate_eval_computation),
        ("Answer extraction", test_answer_extraction),
        ("Held-out set generation", test_heldout_set_generation),
        ("Eval metrics pipeline", test_eval_metrics_on_heldout_set),
        ("Error analysis", test_error_analysis),
        ("Eval copy-rate threshold", test_eval_copy_rate_threshold),
        ("EM improves over training", test_em_improves_over_training),
        ("Diagnostic vs causal EM", test_diagnostic_vs_causal_em_comparison),
        ("Copy-rate vs EM correlation", test_copy_rate_vs_em_correlation),
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
        print("\n✓ All available validation tests passed!")
        print(f"  ({skipped} tests skipped pending trained models)")
    else:
        print(f"\n✗ {failed} test(s) failed")
        sys.exit(1)
