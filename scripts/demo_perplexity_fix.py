#!/usr/bin/env python3
"""
Demonstration of the perplexity double-shift bug fix.

This script shows the difference between:
1. BUGGY: Shifting labels twice (dataset + evaluator)
2. FIXED: Shifting labels once (dataset only)

Run with: python scripts/demo_perplexity_fix.py
"""

import torch
import torch.nn.functional as F


def demonstrate_double_shift_bug():
    """Show how double-shifting labels causes astronomical perplexity."""

    print("="*80)
    print("DEMONSTRATION: Double-Shift Bug in Perplexity Evaluation")
    print("="*80)
    print()

    # Create example sequence: [A, B, C, D, E]
    # Using token IDs: A=100, B=101, C=102, D=103, E=104
    input_ids = torch.tensor([[100, 101, 102, 103, 104]])
    vocab_size = 128256  # Llama vocab size

    print("Example Sequence:")
    print(f"  input_ids = [A=100, B=101, C=102, D=103, E=104]")
    print()

    # Simulate dataset: labels are already shifted
    # labels[i] = input_ids[i+1]
    labels = torch.tensor([[101, 102, 103, 104, -100]])  # [B, C, D, E, -100]
    print("Dataset Output (labels already shifted):")
    print(f"  labels = [B=101, C=102, D=103, E=104, -100]")
    print(f"  ✓ labels[0] = 101 = input_ids[1] (correct shift)")
    print()

    # Create perfect model predictions
    # For each position i, model should predict input_ids[i+1]
    logits = torch.zeros(1, 5, vocab_size)
    for i in range(4):
        logits[0, i, input_ids[0, i+1]] = 10.0  # High score for correct next token
    logits[0, 4, :] = -10.0  # Random for last position (no valid target)

    print("Model Predictions:")
    print(f"  logits[0] has high score for token B=101 (predicts input_ids[1])")
    print(f"  logits[1] has high score for token C=102 (predicts input_ids[2])")
    print(f"  logits[2] has high score for token D=103 (predicts input_ids[3])")
    print(f"  logits[3] has high score for token E=104 (predicts input_ids[4])")
    print()

    # METHOD 1: CORRECT - No additional shift (fixed code)
    print("-"*80)
    print("METHOD 1: FIXED (no additional shift in evaluator)")
    print("-"*80)

    flat_logits = logits.view(-1, vocab_size)
    flat_labels = labels.view(-1)

    loss_correct = F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=-100,
        reduction='mean'
    )
    ppl_correct = torch.exp(loss_correct)

    print(f"Computation:")
    print(f"  logits[0] vs labels[0]=101 → logits[0] predicts 101 ✓ MATCH")
    print(f"  logits[1] vs labels[1]=102 → logits[1] predicts 102 ✓ MATCH")
    print(f"  logits[2] vs labels[2]=103 → logits[2] predicts 103 ✓ MATCH")
    print(f"  logits[3] vs labels[3]=104 → logits[3] predicts 104 ✓ MATCH")
    print()
    print(f"Results:")
    print(f"  Loss: {loss_correct.item():.6f} nats")
    print(f"  Perplexity: {ppl_correct.item():.2f}")
    print(f"  ✓ Near-perfect predictions → near-zero loss")
    print()

    # METHOD 2: BUGGY - Double shift (old code)
    print("-"*80)
    print("METHOD 2: BUGGY (shift again in evaluator - OLD CODE)")
    print("-"*80)

    shift_logits = logits[:, :-1, :]  # Drop last position [0:4]
    shift_labels = labels[:, 1:]      # Drop first position [1:5]

    flat_shift_logits = shift_logits.reshape(-1, vocab_size)
    flat_shift_labels = shift_labels.reshape(-1)

    loss_buggy = F.cross_entropy(
        flat_shift_logits,
        flat_shift_labels,
        ignore_index=-100,
        reduction='mean'
    )
    ppl_buggy = torch.exp(loss_buggy)

    print(f"Computation:")
    print(f"  logits[0] vs shift_labels[0]=labels[1]=102 → logits[0] predicts 101 ✗ MISMATCH")
    print(f"  logits[1] vs shift_labels[1]=labels[2]=103 → logits[1] predicts 102 ✗ MISMATCH")
    print(f"  logits[2] vs shift_labels[2]=labels[3]=104 → logits[2] predicts 103 ✗ MISMATCH")
    print(f"  logits[3] vs shift_labels[3]=labels[4]=-100 → IGNORED")
    print()
    print(f"Results:")
    print(f"  Loss: {loss_buggy.item():.6f} nats")
    print(f"  Perplexity: {ppl_buggy.item():.2f}")
    print(f"  ✗ Predictions misaligned → astronomical loss!")
    print()

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Fixed method:  Loss = {loss_correct.item():.6f}, PPL = {ppl_correct.item():.2f}")
    print(f"Buggy method:  Loss = {loss_buggy.item():.6f}, PPL = {ppl_buggy.item():.2f}")
    print(f"Loss increase: {loss_buggy.item() / max(loss_correct.item(), 1e-6):.1f}x")
    print()
    print("Explanation:")
    print("  • Dataset shifts labels: labels[i] = input_ids[i+1]")
    print("  • Model predicts: logits[i] → input_ids[i+1]")
    print("  • Correct: Compare logits[i] with labels[i] (which is input_ids[i+1])")
    print("  • Buggy: Shift labels again → compare logits[i] with input_ids[i+2]!")
    print()
    print("The fix: Remove the additional shift in perplexity.py lines 134-138")
    print("="*80)


if __name__ == "__main__":
    demonstrate_double_shift_bug()
