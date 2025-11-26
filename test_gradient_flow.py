#!/usr/bin/env python3
"""
Minimal test script to diagnose zero gradient issue.

Tests multiple scenarios to identify where gradient flow breaks:
- Scenario A: FP32 model (baseline)
- Scenario B: BF16 model without autocast
- Scenario C: BF16 model with autocast (current setup)
- Scenario D: FP32 model with autocast BF16 (recommended)
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.retnet.backbone import RetNetBackbone, RetNetOutputHead
from distillation.student_config import create_student_config

def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def check_model_params(model, name="Model"):
    """Check if model parameters have requires_grad and gradients."""
    print(f"\n{name} Parameter Status:")
    total_params = 0
    params_with_grad = 0
    params_with_grad_values = 0

    for n, p in model.named_parameters():
        total_params += 1
        if p.requires_grad:
            params_with_grad += 1
        if p.grad is not None:
            params_with_grad_values += 1
            grad_norm = p.grad.norm().item()
            if grad_norm > 0:
                print(f"  ✓ {n}: requires_grad={p.requires_grad}, grad_norm={grad_norm:.6f}")
            else:
                print(f"  ✗ {n}: requires_grad={p.requires_grad}, grad_norm=0.0000")

    print(f"\nSummary:")
    print(f"  Total params: {total_params}")
    print(f"  Params with requires_grad=True: {params_with_grad}")
    print(f"  Params with actual gradients: {params_with_grad_values}")

    return params_with_grad_values > 0

def test_scenario(scenario_name, model, use_autocast, autocast_dtype=None):
    """Test a single scenario."""
    print_section(f"SCENARIO: {scenario_name}")

    # Check model dtype
    first_param = next(model.parameters())
    print(f"Model dtype: {first_param.dtype}")
    print(f"Model device: {first_param.device}")
    print(f"Use autocast: {use_autocast}")
    if autocast_dtype:
        print(f"Autocast dtype: {autocast_dtype}")

    # Create dummy batch
    batch_size = 2
    seq_len = 128
    vocab_size = 100352

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')

    # Mock teacher logits (sparse top-k format)
    K = 128
    teacher_topk_indices = torch.randint(0, vocab_size, (batch_size, seq_len, K), device='cuda')
    teacher_topk_values = torch.randn(batch_size, seq_len, K, device='cuda')
    teacher_other_mass = torch.randn(batch_size, seq_len, 1, device='cuda')

    print(f"\nInput shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  labels: {labels.shape}")
    print(f"  teacher_topk_indices: {teacher_topk_indices.shape}")

    # Forward pass
    print(f"\n[1] Running forward pass...")
    try:
        if use_autocast and autocast_dtype:
            from torch.amp import autocast
            with autocast('cuda', dtype=autocast_dtype, enabled=True):
                hidden_states = model.forward_train(input_ids)
                student_logits = model.lm_head(hidden_states)
        else:
            hidden_states = model.forward_train(input_ids)
            student_logits = model.lm_head(hidden_states)

        print(f"  ✓ Forward pass successful")
        print(f"  hidden_states: {hidden_states.shape}, dtype={hidden_states.dtype}")
        print(f"  student_logits: {student_logits.shape}, dtype={student_logits.dtype}")
        print(f"  student_logits.requires_grad: {student_logits.requires_grad}")
        print(f"  student_logits.grad_fn: {student_logits.grad_fn}")
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        return False

    # Compute loss
    print(f"\n[2] Computing loss...")
    try:
        from distillation.losses import SparseKLLoss
        loss_fn = SparseKLLoss(temperature=2.0, alpha=0.2)

        if use_autocast and autocast_dtype:
            from torch.amp import autocast
            with autocast('cuda', dtype=autocast_dtype, enabled=True):
                loss = loss_fn(
                    student_logits=student_logits,
                    teacher_topk_indices=teacher_topk_indices,
                    teacher_topk_values=teacher_topk_values,
                    teacher_other_mass=teacher_other_mass,
                    hard_targets=labels,
                )
        else:
            loss = loss_fn(
                student_logits=student_logits,
                teacher_topk_indices=teacher_topk_indices,
                teacher_topk_values=teacher_topk_values,
                teacher_other_mass=teacher_other_mass,
                hard_targets=labels,
            )

        print(f"  ✓ Loss computation successful")
        print(f"  loss: {loss.item():.4f}")
        print(f"  loss.requires_grad: {loss.requires_grad}")
        print(f"  loss.grad_fn: {loss.grad_fn}")
        print(f"  loss.dtype: {loss.dtype}")
    except Exception as e:
        print(f"  ✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Backward pass
    print(f"\n[3] Running backward pass...")
    try:
        # Scale loss by gradient accumulation (like in training)
        loss_scaled = loss / 256
        print(f"  Scaled loss: {loss_scaled.item():.6f}")

        loss_scaled.backward()
        print(f"  ✓ Backward pass successful")
    except Exception as e:
        print(f"  ✗ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Check gradients
    print(f"\n[4] Checking gradients...")
    has_gradients = check_model_params(model, scenario_name)

    # Compute gradient norm
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
    print(f"\nGradient norm: {grad_norm.item():.6f}")

    # Clear gradients for next scenario
    for p in model.parameters():
        p.grad = None

    return has_gradients

def main():
    """Main test runner."""
    print_section("ZERO GRADIENT DIAGNOSTIC TEST")
    print("Testing multiple scenarios to identify where gradient flow breaks")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This test requires GPU.")
        return

    # Load student config
    print("\nLoading student configuration...")
    student_config = create_student_config("350M")
    print(f"  Variant: {student_config.variant}")
    print(f"  d_model: {student_config.d_model}")
    print(f"  n_layers: {student_config.n_layers}")

    # Create model
    print("\nCreating model...")
    model_kwargs = student_config.to_retnet_backbone_args()

    results = {}

    # =========================================================================
    # SCENARIO A: FP32 model (baseline)
    # =========================================================================
    print_section("SCENARIO A: FP32 Model (Baseline)")
    model_fp32 = RetNetBackbone(**model_kwargs)
    model_fp32.lm_head = RetNetOutputHead(
        d_model=student_config.d_model,
        vocab_size=student_config.vocab_size,
        tie_weights=True,
        embedding_layer=model_fp32.embed,
    )
    model_fp32 = model_fp32.to(device)
    # Keep in FP32

    results['A_FP32'] = test_scenario(
        scenario_name="A: FP32 Model",
        model=model_fp32,
        use_autocast=False,
    )

    del model_fp32
    torch.cuda.empty_cache()

    # =========================================================================
    # SCENARIO B: BF16 model without autocast
    # =========================================================================
    print_section("SCENARIO B: BF16 Model WITHOUT Autocast")
    model_bf16_no_autocast = RetNetBackbone(**model_kwargs)
    model_bf16_no_autocast.lm_head = RetNetOutputHead(
        d_model=student_config.d_model,
        vocab_size=student_config.vocab_size,
        tie_weights=True,
        embedding_layer=model_bf16_no_autocast.embed,
    )
    model_bf16_no_autocast = model_bf16_no_autocast.to(device)
    model_bf16_no_autocast = model_bf16_no_autocast.to(dtype=torch.bfloat16)

    # Check if requires_grad is preserved after .to(dtype=bfloat16)
    print("\nChecking requires_grad after .to(dtype=bfloat16):")
    sample_params = list(model_bf16_no_autocast.named_parameters())[:5]
    for n, p in sample_params:
        print(f"  {n}: requires_grad={p.requires_grad}, dtype={p.dtype}")

    results['B_BF16_no_autocast'] = test_scenario(
        scenario_name="B: BF16 Model (no autocast)",
        model=model_bf16_no_autocast,
        use_autocast=False,
    )

    del model_bf16_no_autocast
    torch.cuda.empty_cache()

    # =========================================================================
    # SCENARIO C: BF16 model WITH autocast (current setup)
    # =========================================================================
    print_section("SCENARIO C: BF16 Model WITH Autocast (Current Setup)")
    model_bf16_autocast = RetNetBackbone(**model_kwargs)
    model_bf16_autocast.lm_head = RetNetOutputHead(
        d_model=student_config.d_model,
        vocab_size=student_config.vocab_size,
        tie_weights=True,
        embedding_layer=model_bf16_autocast.embed,
    )
    model_bf16_autocast = model_bf16_autocast.to(device)
    model_bf16_autocast = model_bf16_autocast.to(dtype=torch.bfloat16)

    results['C_BF16_with_autocast'] = test_scenario(
        scenario_name="C: BF16 Model + Autocast",
        model=model_bf16_autocast,
        use_autocast=True,
        autocast_dtype=torch.bfloat16,
    )

    del model_bf16_autocast
    torch.cuda.empty_cache()

    # =========================================================================
    # SCENARIO D: FP32 model WITH autocast (recommended setup)
    # =========================================================================
    print_section("SCENARIO D: FP32 Model WITH Autocast BF16 (Recommended)")
    model_fp32_autocast = RetNetBackbone(**model_kwargs)
    model_fp32_autocast.lm_head = RetNetOutputHead(
        d_model=student_config.d_model,
        vocab_size=student_config.vocab_size,
        tie_weights=True,
        embedding_layer=model_fp32_autocast.embed,
    )
    model_fp32_autocast = model_fp32_autocast.to(device)
    # Keep in FP32, let autocast handle conversion

    results['D_FP32_with_autocast'] = test_scenario(
        scenario_name="D: FP32 Model + Autocast BF16",
        model=model_fp32_autocast,
        use_autocast=True,
        autocast_dtype=torch.bfloat16,
    )

    del model_fp32_autocast
    torch.cuda.empty_cache()

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print_section("FINAL SUMMARY")
    print("\nGradient Flow Results:")
    for scenario, has_grads in results.items():
        status = "✓ PASS" if has_grads else "✗ FAIL"
        print(f"  {scenario}: {status}")

    print("\nDiagnosis:")
    if results['A_FP32'] and not results['B_BF16_no_autocast']:
        print("  → BF16 conversion (.to(dtype=bfloat16)) BREAKS gradient flow")
        print("  → This is the root cause!")
    elif results['A_FP32'] and not results['C_BF16_with_autocast']:
        print("  → BF16 model + autocast BREAKS gradient flow")
        print("  → This is the root cause!")
    elif results['D_FP32_with_autocast']:
        print("  → FP32 model with autocast BF16 WORKS")
        print("  → Recommended fix: Keep model in FP32, use autocast for BF16 computation")
    else:
        print("  → Multiple scenarios failing, needs deeper investigation")

    print("\nRecommended fix:")
    print("  1. Remove: model = model.to(dtype=torch.bfloat16)")
    print("  2. Keep: model = model.to(device)")
    print("  3. Use autocast context for BF16 computation (already in place)")
    print("  4. This allows gradients to accumulate in FP32 while forward pass uses BF16")

if __name__ == "__main__":
    main()
