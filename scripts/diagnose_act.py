#!/usr/bin/env python3
"""
Diagnostic script to verify ACT is actually working.

Tests:
1. Pondering steps vary per token (not constant)
2. Halting probabilities change meaningfully
3. Ponder penalty is non-zero and affects loss
4. Outputs differ based on max_steps
5. Gradients flow through halting units
6. Halting units learn (params change after backward)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from transformers import AutoTokenizer
from src.models.retnet.act_wrapper import create_act_retnet, ACTLoss


def test_pondering_variance():
    """Test 1: Do different tokens get different pondering steps?"""
    print("\n" + "="*80)
    print("TEST 1: Pondering Variance")
    print("="*80)

    model = create_act_retnet(
        vocab_size=128256,
        d_model=512,
        n_layers=4,
        n_heads=8,
        act_max_steps=10,
        act_epsilon=0.01,
    ).cuda()

    # Create a real text sequence
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    text = "The quick brown fox jumps over the lazy dog. Supercalifragilisticexpialidocious!"
    input_ids = tokenizer.encode(text, return_tensors="pt").cuda()

    print(f"Input text: {text}")
    print(f"Sequence length: {input_ids.shape[1]}")

    with torch.no_grad():
        output, act_info = model(input_ids, return_act_info=True)

    ponder_steps = act_info['pondering_cost'][0].cpu().numpy()  # [seq_len]

    print(f"\nPondering steps per token:")
    for i, (token_id, steps) in enumerate(zip(input_ids[0].cpu().tolist(), ponder_steps)):
        token = tokenizer.decode([token_id])
        print(f"  [{i:2d}] '{token:20s}' -> {steps:.2f} steps")

    print(f"\nStatistics:")
    print(f"  Mean:   {ponder_steps.mean():.3f}")
    print(f"  Std:    {ponder_steps.std():.3f}")
    print(f"  Min:    {ponder_steps.min():.3f}")
    print(f"  Max:    {ponder_steps.max():.3f}")
    print(f"  Range:  {ponder_steps.max() - ponder_steps.min():.3f}")

    # Verdict
    if ponder_steps.std() < 0.01:
        print("\n❌ FAIL: All tokens get same steps (std < 0.01)")
        return False
    elif ponder_steps.std() < 0.1:
        print("\n⚠️  WARN: Very low variance (std < 0.1) - might not be learning")
        return True
    else:
        print("\n✅ PASS: Tokens get different pondering steps")
        return True


def test_halting_evolution():
    """Test 2: Do halting probabilities change across pondering steps?"""
    print("\n" + "="*80)
    print("TEST 2: Halting Probability Evolution")
    print("="*80)

    model = create_act_retnet(
        vocab_size=128256,
        d_model=512,
        n_layers=4,
        n_heads=8,
        act_max_steps=10,
        act_epsilon=0.01,
    ).cuda()

    input_ids = torch.randint(0, 128256, (1, 16)).cuda()

    with torch.no_grad():
        output, act_info = model(input_ids, return_act_info=True)

    # Get halting probs: [batch, seq_len, max_steps]
    halting_probs = act_info['halting_probs'][0]  # [seq_len, max_steps]

    print(f"\nHalting probabilities for first 5 tokens:")
    for pos in range(min(5, halting_probs.shape[0])):
        probs = halting_probs[pos].cpu().numpy()
        print(f"\n  Token {pos}:")
        for step, prob in enumerate(probs[:5]):  # Show first 5 steps
            print(f"    Step {step}: {prob:.4f}")

    # Check if probabilities vary across steps
    step_variances = []
    for pos in range(halting_probs.shape[0]):
        probs = halting_probs[pos].cpu().numpy()
        step_variances.append(probs.std())

    mean_variance = np.mean(step_variances)
    print(f"\nMean variance across steps: {mean_variance:.4f}")

    # Verdict
    if mean_variance < 0.001:
        print("\n❌ FAIL: Halting probs constant across steps (var < 0.001)")
        return False
    elif mean_variance < 0.01:
        print("\n⚠️  WARN: Low variance (< 0.01) - halting might not be adaptive")
        return True
    else:
        print("\n✅ PASS: Halting probabilities evolve across steps")
        return True


def test_loss_components():
    """Test 3: Is ponder penalty non-zero and affecting total loss?"""
    print("\n" + "="*80)
    print("TEST 3: Loss Components")
    print("="*80)

    model = create_act_retnet(
        vocab_size=128256,
        d_model=512,
        n_layers=4,
        n_heads=8,
        act_max_steps=10,
        act_ponder_penalty=0.01,
    ).cuda()

    criterion = ACTLoss(act_model=model)

    input_ids = torch.randint(0, 128256, (1, 32)).cuda()
    labels = torch.randint(0, 128256, (1, 32)).cuda()

    # Forward pass
    output, act_info = model(input_ids, return_act_info=True)

    # Compute task loss (fake CE loss for demo)
    task_loss = torch.nn.functional.cross_entropy(
        output.view(-1, output.size(-1)),
        labels.view(-1),
    )

    # Compute total loss
    total_loss, loss_dict = criterion(task_loss, act_info)

    print(f"\nLoss breakdown:")
    print(f"  Task loss:    {task_loss.item():.4f}")
    print(f"  Ponder loss:  {loss_dict['ponder_loss']:.4f}")
    print(f"  Total loss:   {total_loss.item():.4f}")
    print(f"  Difference:   {(total_loss - task_loss).item():.4f}")

    print(f"\nPondering statistics:")
    print(f"  Mean steps:   {act_info['pondering_cost'].mean().item():.3f}")
    print(f"  Max steps:    {act_info['pondering_cost'].max().item():.3f}")

    # Verdict
    if loss_dict['ponder_loss'] < 1e-6:
        print("\n❌ FAIL: Ponder loss is zero - ACT not active")
        return False
    elif abs(total_loss.item() - task_loss.item()) < 1e-6:
        print("\n❌ FAIL: Total loss equals task loss - ponder penalty not applied")
        return False
    else:
        print("\n✅ PASS: Ponder penalty is non-zero and affects total loss")
        return True


def test_max_steps_effect():
    """Test 4: Does changing max_steps change outputs?"""
    print("\n" + "="*80)
    print("TEST 4: Max Steps Effect")
    print("="*80)

    input_ids = torch.randint(0, 128256, (1, 16)).cuda()

    results = {}
    for max_steps in [3, 5, 10]:
        model = create_act_retnet(
            vocab_size=128256,
            d_model=512,
            n_layers=4,
            n_heads=8,
            act_max_steps=max_steps,
        ).cuda()

        with torch.no_grad():
            output, act_info = model(input_ids, return_act_info=True)

        results[max_steps] = {
            'output': output,
            'mean_steps': act_info['pondering_cost'].mean().item(),
            'max_actual': act_info['pondering_cost'].max().item(),
        }

        print(f"\nmax_steps={max_steps}:")
        print(f"  Mean actual steps: {results[max_steps]['mean_steps']:.3f}")
        print(f"  Max actual steps:  {results[max_steps]['max_actual']:.3f}")

    # Check if outputs differ
    out_3 = results[3]['output']
    out_5 = results[5]['output']
    out_10 = results[10]['output']

    diff_3_5 = (out_3 - out_5).abs().mean().item()
    diff_5_10 = (out_5 - out_10).abs().mean().item()

    print(f"\nOutput differences:")
    print(f"  max_steps 3 vs 5:   {diff_3_5:.6f}")
    print(f"  max_steps 5 vs 10:  {diff_5_10:.6f}")

    # Verdict
    if diff_3_5 < 1e-6 and diff_5_10 < 1e-6:
        print("\n❌ FAIL: Outputs identical regardless of max_steps")
        return False
    elif diff_3_5 < 0.001 or diff_5_10 < 0.001:
        print("\n⚠️  WARN: Outputs barely change with max_steps")
        return True
    else:
        print("\n✅ PASS: Outputs differ meaningfully with max_steps")
        return True


def test_gradient_flow():
    """Test 5: Do gradients flow through halting units?"""
    print("\n" + "="*80)
    print("TEST 5: Gradient Flow")
    print("="*80)

    model = create_act_retnet(
        vocab_size=128256,
        d_model=512,
        n_layers=4,
        n_heads=8,
        act_max_steps=5,
    ).cuda()

    criterion = ACTLoss(act_model=model)

    input_ids = torch.randint(0, 128256, (1, 16)).cuda()
    labels = torch.randint(0, 128256, (1, 16)).cuda()

    # Forward pass
    output, act_info = model(input_ids, return_act_info=True)

    task_loss = torch.nn.functional.cross_entropy(
        output.view(-1, output.size(-1)),
        labels.view(-1),
    )

    total_loss, loss_dict = criterion(task_loss, act_info)

    # Zero grads and backward
    model.zero_grad()
    total_loss.backward()

    # Check halting unit gradients
    halting_grads = []
    for name, param in model.named_parameters():
        if 'halting_unit' in name and param.grad is not None:
            grad_norm = param.grad.norm().item()
            halting_grads.append(grad_norm)
            print(f"\n  {name}:")
            print(f"    Grad norm: {grad_norm:.6f}")

    if not halting_grads:
        print("\n❌ FAIL: No gradients found for halting units")
        return False

    mean_grad = np.mean(halting_grads)
    print(f"\nMean halting unit grad norm: {mean_grad:.6f}")

    # Verdict
    if mean_grad < 1e-8:
        print("\n❌ FAIL: Halting unit gradients are zero")
        return False
    elif mean_grad < 1e-5:
        print("\n⚠️  WARN: Very small gradients (< 1e-5)")
        return True
    else:
        print("\n✅ PASS: Gradients flow through halting units")
        return True


def test_learning():
    """Test 6: Do halting units learn (params change)?"""
    print("\n" + "="*80)
    print("TEST 6: Learning (Parameter Updates)")
    print("="*80)

    model = create_act_retnet(
        vocab_size=128256,
        d_model=512,
        n_layers=4,
        n_heads=8,
        act_max_steps=5,
    ).cuda()

    criterion = ACTLoss(act_model=model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Save initial params
    initial_params = {}
    for name, param in model.named_parameters():
        if 'halting_unit' in name:
            initial_params[name] = param.data.clone()

    print(f"\nInitial halting unit params:")
    for name, param in initial_params.items():
        print(f"  {name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}")

    # Run 10 training steps
    print(f"\nRunning 10 training steps...")
    for step in range(10):
        input_ids = torch.randint(0, 128256, (2, 32)).cuda()
        labels = torch.randint(0, 128256, (2, 32)).cuda()

        output, act_info = model(input_ids, return_act_info=True)
        task_loss = torch.nn.functional.cross_entropy(
            output.view(-1, output.size(-1)),
            labels.view(-1),
        )

        total_loss, loss_dict = criterion(task_loss, act_info)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 3 == 0:
            print(f"  Step {step}: loss={total_loss.item():.4f}, ponder={loss_dict['ponder_loss']:.4f}")

    # Check param changes
    print(f"\nParameter changes:")
    changes = []
    for name, initial in initial_params.items():
        current = dict(model.named_parameters())[name].data
        change = (current - initial).abs().mean().item()
        changes.append(change)
        print(f"  {name}:")
        print(f"    Mean absolute change: {change:.6f}")

    mean_change = np.mean(changes)
    print(f"\nMean parameter change: {mean_change:.6f}")

    # Verdict
    if mean_change < 1e-6:
        print("\n❌ FAIL: Parameters didn't change - not learning")
        return False
    elif mean_change < 1e-4:
        print("\n⚠️  WARN: Very small parameter changes (< 1e-4)")
        return True
    else:
        print("\n✅ PASS: Halting units are learning")
        return True


def main():
    print("\n" + "="*80)
    print("ACT DIAGNOSTIC SUITE")
    print("="*80)
    print("\nThis will verify that ACT is actually doing something, not just passing tests.")
    print("Expected time: ~30 seconds")

    tests = [
        ("Pondering Variance", test_pondering_variance),
        ("Halting Evolution", test_halting_evolution),
        ("Loss Components", test_loss_components),
        ("Max Steps Effect", test_max_steps_effect),
        ("Gradient Flow", test_gradient_flow),
        ("Learning", test_learning),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ ERROR in {name}: {e}")
            results.append((name, False))
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    num_passed = sum(passed for _, passed in results)
    num_total = len(results)

    print(f"\nOverall: {num_passed}/{num_total} tests passed")

    if num_passed == num_total:
        print("\n🎉 ALL TESTS PASSED - ACT is working as expected!")
    elif num_passed >= num_total * 0.8:
        print("\n⚠️  MOSTLY WORKING - Some concerns but ACT appears functional")
    else:
        print("\n❌ MULTIPLE FAILURES - ACT may not be working correctly")


if __name__ == "__main__":
    main()
