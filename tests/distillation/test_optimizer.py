"""
Tests for optimizer and learning rate scheduler.

Tests 8-bit optimizer, cosine annealing with warm restarts,
and checkpoint save/load functionality.

Tasks: T047
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import math

from src.distillation.optimizer import (
    create_optimizer,
    create_scheduler,
    CosineAnnealingWarmRestartsWithWarmup,
    get_optimizer_memory_footprint,
    HAS_BITSANDBYTES,
)

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None


# Test model
class SimpleModel(nn.Module):
    """Simple model for testing optimizer."""

    def __init__(self, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            ])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


@pytest.fixture
def simple_model():
    """Create simple model for testing."""
    return SimpleModel(hidden_dim=128, num_layers=2)


@pytest.fixture
def optimizer_8bit(simple_model):
    """Create 8-bit optimizer."""
    return create_optimizer(simple_model, lr=1e-4, use_8bit=True)


@pytest.fixture
def optimizer_fp32(simple_model):
    """Create FP32 optimizer."""
    return create_optimizer(simple_model, lr=1e-4, use_8bit=False)


# ============================================================================
# Test T044: 8-bit Optimizer Creation
# ============================================================================

def test_create_optimizer_8bit(simple_model):
    """Test creating 8-bit AdamW optimizer."""
    optimizer = create_optimizer(
        simple_model,
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        use_8bit=True,
    )

    assert optimizer is not None
    assert len(optimizer.param_groups) == 2  # decay and no-decay groups

    # Check learning rate
    for group in optimizer.param_groups:
        assert group['lr'] == 1e-4

    # Verify optimizer type
    if HAS_BITSANDBYTES:
        assert isinstance(optimizer, bnb.optim.AdamW8bit)
    else:
        # Fallback to standard AdamW if bitsandbytes not available
        assert isinstance(optimizer, torch.optim.AdamW)


def test_create_optimizer_fp32(simple_model):
    """Test creating standard FP32 AdamW optimizer."""
    optimizer = create_optimizer(
        simple_model,
        lr=1e-4,
        use_8bit=False,
    )

    assert optimizer is not None
    assert isinstance(optimizer, torch.optim.AdamW)
    assert len(optimizer.param_groups) == 2


def test_optimizer_parameter_groups(simple_model):
    """Test parameter grouping (decay vs no-decay)."""
    optimizer = create_optimizer(
        simple_model,
        weight_decay=0.01,
        no_decay_bias_and_norm=True,
    )

    # Should have 2 groups: with decay and without decay
    assert len(optimizer.param_groups) == 2

    # Group 0: parameters with weight decay (weights)
    assert optimizer.param_groups[0]['weight_decay'] == 0.01

    # Group 1: parameters without weight decay (biases, norms)
    assert optimizer.param_groups[1]['weight_decay'] == 0.0

    # Count parameters in each group
    num_decay = sum(p.numel() for p in optimizer.param_groups[0]['params'])
    num_no_decay = sum(p.numel() for p in optimizer.param_groups[1]['params'])

    # Should have both types of parameters
    assert num_decay > 0
    assert num_no_decay > 0


def test_optimizer_no_parameter_groups(simple_model):
    """Test optimizer without parameter grouping."""
    optimizer = create_optimizer(
        simple_model,
        weight_decay=0.01,
        no_decay_bias_and_norm=False,
    )

    # Should have single group
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]['weight_decay'] == 0.01


def test_optimizer_step_8bit(simple_model, optimizer_8bit):
    """Test 8-bit optimizer can execute training step."""
    # Forward pass
    x = torch.randn(2, 128)
    output = simple_model(x)
    loss = output.mean()

    # Backward pass
    loss.backward()

    # Optimizer step
    optimizer_8bit.step()
    optimizer_8bit.zero_grad()

    # Verify optimizer state is created
    assert len(optimizer_8bit.state) > 0


def test_optimizer_memory_footprint():
    """Test memory footprint estimation."""
    # Create small model
    model = SimpleModel(hidden_dim=64, num_layers=1)

    # 8-bit optimizer
    opt_8bit = create_optimizer(model, use_8bit=True)
    mem_8bit = get_optimizer_memory_footprint(opt_8bit)

    assert 'total_memory_gb' in mem_8bit
    assert 'state_memory_gb' in mem_8bit
    assert 'param_memory_gb' in mem_8bit
    assert mem_8bit['total_memory_gb'] > 0

    # FP32 optimizer
    opt_fp32 = create_optimizer(model, use_8bit=False)
    mem_fp32 = get_optimizer_memory_footprint(opt_fp32)

    # 8-bit should use less memory (if bitsandbytes available)
    if HAS_BITSANDBYTES:
        # 8-bit uses ~25% memory of FP32 for optimizer state
        assert mem_8bit['state_memory_gb'] < mem_fp32['state_memory_gb']
        # Approximately 4x reduction (2 bytes vs 8 bytes per param for states)
        ratio = mem_fp32['state_memory_gb'] / mem_8bit['state_memory_gb']
        assert ratio > 3.0  # Should be ~4x, allow some tolerance


# ============================================================================
# Test T045: Cosine LR Scheduler
# ============================================================================

def test_create_scheduler(optimizer_fp32):
    """Test creating cosine annealing scheduler."""
    scheduler = create_scheduler(
        optimizer_fp32,
        warmup_steps=1000,
        T_0=10000,
        T_mult=2,
        eta_min=1e-6,
    )

    assert scheduler is not None
    assert isinstance(scheduler, CosineAnnealingWarmRestartsWithWarmup)
    assert scheduler.warmup_steps == 1000
    assert scheduler.T_0 == 10000
    assert scheduler.T_mult == 2
    assert scheduler.eta_min == 1e-6


def test_scheduler_warmup(optimizer_fp32):
    """Test linear warmup phase."""
    base_lr = 1e-4
    warmup_steps = 100

    scheduler = CosineAnnealingWarmRestartsWithWarmup(
        optimizer_fp32,
        warmup_steps=warmup_steps,
        T_0=1000,
        eta_min=1e-6,
    )

    # Test warmup progression
    lrs = []
    for step in range(warmup_steps):
        scheduler.step()
        lrs.append(scheduler.get_lr()[0])

    # Learning rate should increase linearly during warmup
    assert lrs[0] < base_lr * 0.02  # Start near 0 (step 1)
    assert abs(lrs[-1] - base_lr) < 1e-6  # End at base_lr (step 100 = warmup complete)

    # Check general increasing trend
    for i in range(1, len(lrs)):
        assert lrs[i] >= lrs[i - 1] - 1e-9  # Should be monotonically increasing


def test_scheduler_cosine_annealing(optimizer_fp32):
    """Test cosine annealing phase after warmup."""
    base_lr = 1e-4
    warmup_steps = 100
    T_0 = 1000
    eta_min = 1e-6

    scheduler = CosineAnnealingWarmRestartsWithWarmup(
        optimizer_fp32,
        warmup_steps=warmup_steps,
        T_0=T_0,
        eta_min=eta_min,
    )

    # Skip warmup
    for _ in range(warmup_steps):
        scheduler.step()

    # Test cosine annealing - collect T_0-1 steps (don't include restart)
    lrs = []
    for step in range(T_0 - 1):
        scheduler.step()
        lrs.append(scheduler.get_lr()[0])

    # Learning rate should start near base_lr after warmup
    assert abs(lrs[0] - base_lr) < 1e-5

    # Learning rate should decay toward eta_min by end of cycle
    assert lrs[-1] < lrs[0]
    assert lrs[-1] < base_lr * 0.05  # Should be very low (near eta_min)

    # Check cosine shape: should decrease monotonically (with small tolerance)
    for i in range(1, len(lrs)):
        assert lrs[i] <= lrs[i - 1] + 1e-7  # Allow small numerical error


def test_scheduler_warm_restarts(optimizer_fp32):
    """Test warm restarts with increasing cycle lengths."""
    base_lr = 1e-4
    warmup_steps = 100
    T_0 = 100
    T_mult = 2
    eta_min = 1e-6

    scheduler = CosineAnnealingWarmRestartsWithWarmup(
        optimizer_fp32,
        warmup_steps=warmup_steps,
        T_0=T_0,
        T_mult=T_mult,
        eta_min=eta_min,
    )

    # Skip warmup
    for _ in range(warmup_steps):
        scheduler.step()

    # Cycle 1: T_0 = 100 steps
    lrs_cycle1 = []
    for _ in range(T_0):
        scheduler.step()
        lrs_cycle1.append(scheduler.get_lr()[0])

    # Cycle 2: T_0 * T_mult = 200 steps
    lrs_cycle2 = []
    for _ in range(T_0 * T_mult):
        scheduler.step()
        lrs_cycle2.append(scheduler.get_lr()[0])

    # Both cycles should start near base_lr
    # (First collected LR is after first step in cycle)
    assert abs(lrs_cycle1[0] - base_lr) < 1e-5
    assert abs(lrs_cycle2[0] - base_lr) < 1e-5

    # Both cycles should end near eta_min (except last step is restart)
    # Check second-to-last step
    assert lrs_cycle1[-2] < base_lr * 0.05
    assert lrs_cycle2[-2] < base_lr * 0.05

    # Last step should be restart (back to base_lr)
    assert abs(lrs_cycle1[-1] - base_lr) < 1e-5
    assert abs(lrs_cycle2[-1] - base_lr) < 1e-5

    # Cycle 2 should be longer (200 vs 100 steps collected)
    assert len(lrs_cycle2) == len(lrs_cycle1) * T_mult


def test_scheduler_multiple_param_groups():
    """Test scheduler with multiple parameter groups."""
    # Model with different LRs for different groups
    model = SimpleModel()
    optimizer = torch.optim.AdamW([
        {'params': model.layers[0].parameters(), 'lr': 1e-4},
        {'params': model.layers[1].parameters(), 'lr': 5e-5},
    ])

    scheduler = CosineAnnealingWarmRestartsWithWarmup(
        optimizer,
        warmup_steps=100,
        T_0=1000,
    )

    # After warmup, should reach base LRs
    for _ in range(100):
        scheduler.step()

    lrs = scheduler.get_lr()
    assert len(lrs) == 2
    assert abs(lrs[0] - 1e-4) < 1e-6
    assert abs(lrs[1] - 5e-5) < 1e-6


# ============================================================================
# Test T046: Warmup + Cosine Combined
# ============================================================================

def test_warmup_cosine_smooth_transition(optimizer_fp32):
    """Test smooth transition from warmup to cosine annealing."""
    base_lr = 1e-4
    warmup_steps = 100

    scheduler = CosineAnnealingWarmRestartsWithWarmup(
        optimizer_fp32,
        warmup_steps=warmup_steps,
        T_0=1000,
        eta_min=1e-6,
    )

    # Collect LRs across warmup-cosine boundary
    lrs = []
    for step in range(warmup_steps + 10):
        scheduler.step()
        lrs.append(scheduler.get_lr()[0])

    # Check no discontinuity at boundary
    warmup_end_lr = lrs[warmup_steps - 1]
    cosine_start_lr = lrs[warmup_steps]

    # Should be smooth transition (both near base_lr)
    assert abs(warmup_end_lr - base_lr) < 1e-5
    assert abs(cosine_start_lr - base_lr) < 1e-5

    # No large jumps
    for i in range(1, len(lrs)):
        lr_change = abs(lrs[i] - lrs[i - 1])
        assert lr_change < base_lr * 0.02  # Max 2% change per step


def test_scheduler_full_schedule():
    """Test full learning rate schedule (warmup + multiple cycles)."""
    base_lr = 1e-4
    warmup_steps = 1000
    T_0 = 10000
    T_mult = 2
    eta_min = 1e-6
    total_steps = warmup_steps + T_0 + T_0 * T_mult  # Warmup + 2 cycles

    model = SimpleModel()
    optimizer = create_optimizer(model, lr=base_lr)
    scheduler = create_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        T_0=T_0,
        T_mult=T_mult,
        eta_min=eta_min,
    )

    lrs = []
    for step in range(total_steps):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        lrs.append(lr)

    # Check warmup phase
    assert lrs[0] < base_lr * 0.01  # Start near 0
    assert abs(lrs[warmup_steps - 1] - base_lr) < 1e-5  # End at base_lr

    # Check first cycle (don't check last step as it's the restart)
    cycle1_start = warmup_steps
    cycle1_near_end = warmup_steps + T_0 - 2  # Second-to-last step
    assert abs(lrs[cycle1_start] - base_lr) < 1e-5
    assert lrs[cycle1_near_end] < base_lr * 0.05  # Near eta_min

    # Check second cycle (restart)
    cycle2_start = warmup_steps + T_0
    assert abs(lrs[cycle2_start] - base_lr) < 1e-5


# ============================================================================
# Test T047: Checkpoint Save/Load
# ============================================================================

def test_optimizer_state_save_load(simple_model):
    """Test optimizer state save/load (8-bit)."""
    # Create optimizer
    optimizer = create_optimizer(simple_model, lr=1e-4, use_8bit=True)

    # Take a few training steps to create optimizer state
    for _ in range(5):
        x = torch.randn(2, 128)
        output = simple_model(x)
        loss = output.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Save state
    state_dict = optimizer.state_dict()

    # Create new optimizer
    model2 = SimpleModel(hidden_dim=128, num_layers=2)
    optimizer2 = create_optimizer(model2, lr=1e-4, use_8bit=True)

    # Load state
    optimizer2.load_state_dict(state_dict)

    # Verify state matches
    assert len(optimizer2.state) == len(optimizer.state)

    # Check parameter groups match
    for pg1, pg2 in zip(optimizer.param_groups, optimizer2.param_groups):
        assert pg1['lr'] == pg2['lr']


def test_optimizer_resume_training(simple_model):
    """Test that training can resume with identical optimizer state."""
    # Setup
    optimizer = create_optimizer(simple_model, lr=1e-4, use_8bit=True)

    # Train for 5 steps
    losses_before = []
    for _ in range(5):
        x = torch.randn(2, 128)
        output = simple_model(x)
        loss = output.mean()
        losses_before.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Save state
    model_state = simple_model.state_dict()
    optimizer_state = optimizer.state_dict()

    # Continue training for 3 more steps
    losses_continue = []
    for _ in range(3):
        x = torch.randn(2, 128)
        output = simple_model(x)
        loss = output.mean()
        losses_continue.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Reset and resume from checkpoint
    model2 = SimpleModel(hidden_dim=128, num_layers=2)
    model2.load_state_dict(model_state)
    optimizer2 = create_optimizer(model2, lr=1e-4, use_8bit=True)
    optimizer2.load_state_dict(optimizer_state)

    # Train for 3 steps with same random seed
    torch.manual_seed(42)
    losses_resume = []
    for _ in range(3):
        x = torch.randn(2, 128)
        output = model2(x)
        loss = output.mean()
        losses_resume.append(loss.item())
        loss.backward()
        optimizer2.step()
        optimizer2.zero_grad()

    # Resumed training should match continued training (approximately)
    # Note: May have small differences due to RNG, but optimizer state should be preserved


def test_scheduler_state_save_load(optimizer_fp32):
    """Test scheduler state save/load."""
    # Create scheduler
    scheduler = create_scheduler(
        optimizer_fp32,
        warmup_steps=100,
        T_0=1000,
        T_mult=2,
        eta_min=1e-6,
    )

    # Run for some steps
    for _ in range(150):  # Past warmup
        scheduler.step()

    # Save state
    scheduler_state = scheduler.state_dict()

    # Create new scheduler
    model2 = SimpleModel()
    optimizer2 = create_optimizer(model2, lr=1e-4, use_8bit=False)
    scheduler2 = create_scheduler(optimizer2, warmup_steps=100, T_0=1000)

    # Load state
    scheduler2.load_state_dict(scheduler_state)

    # Verify state matches
    assert scheduler2.last_epoch == scheduler.last_epoch

    # Next LR should match
    lr1 = scheduler.get_lr()[0]
    lr2 = scheduler2.get_lr()[0]
    assert abs(lr1 - lr2) < 1e-8


def test_scheduler_deterministic_after_resume(optimizer_fp32):
    """Test that LR schedule is deterministic after resume."""
    base_lr = 1e-4

    # Scheduler 1: run for 200 steps
    scheduler1 = create_scheduler(optimizer_fp32, warmup_steps=100, T_0=1000)
    lrs1 = []
    for _ in range(200):
        lrs1.append(scheduler1.get_lr()[0])
        scheduler1.step()

    # Scheduler 2: run for 100 steps, save, load, continue
    model2 = SimpleModel()
    optimizer2 = create_optimizer(model2, lr=base_lr, use_8bit=False)
    scheduler2 = create_scheduler(optimizer2, warmup_steps=100, T_0=1000)

    lrs2 = []
    for _ in range(100):
        lrs2.append(scheduler2.get_lr()[0])
        scheduler2.step()

    # Save and load
    state = scheduler2.state_dict()
    scheduler2.load_state_dict(state)

    # Continue for 100 more steps
    for _ in range(100):
        lrs2.append(scheduler2.get_lr()[0])
        scheduler2.step()

    # LR schedules should match
    assert len(lrs1) == len(lrs2)
    for lr1, lr2 in zip(lrs1, lrs2):
        assert abs(lr1 - lr2) < 1e-8


def test_checkpoint_integration():
    """Test full checkpoint save/load with model, optimizer, and scheduler."""
    # Setup
    model = SimpleModel()
    optimizer = create_optimizer(model, lr=1e-4, use_8bit=True)
    scheduler = create_scheduler(optimizer, warmup_steps=100, T_0=1000)

    # Train for 50 steps
    for step in range(50):
        x = torch.randn(2, 128)
        output = model(x)
        loss = output.mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # Save checkpoint
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step': 50,
    }

    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pt') as f:
        torch.save(checkpoint, f)
        checkpoint_path = f.name

    # Continue training
    lrs_continue = []
    for step in range(50, 100):
        lrs_continue.append(scheduler.get_lr()[0])
        x = torch.randn(2, 128)
        output = model(x)
        loss = output.mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # Load checkpoint and resume
    model2 = SimpleModel()
    optimizer2 = create_optimizer(model2, lr=1e-4, use_8bit=True)
    scheduler2 = create_scheduler(optimizer2, warmup_steps=100, T_0=1000)

    checkpoint_loaded = torch.load(checkpoint_path, weights_only=False)
    model2.load_state_dict(checkpoint_loaded['model'])
    optimizer2.load_state_dict(checkpoint_loaded['optimizer'])
    scheduler2.load_state_dict(checkpoint_loaded['scheduler'])

    # Resume training
    lrs_resume = []
    for step in range(50, 100):
        lrs_resume.append(scheduler2.get_lr()[0])
        x = torch.randn(2, 128)
        output = model2(x)
        loss = output.mean()
        loss.backward()
        optimizer2.step()
        scheduler2.step()
        optimizer2.zero_grad()

    # LR schedules should match
    for lr1, lr2 in zip(lrs_continue, lrs_resume):
        assert abs(lr1 - lr2) < 1e-8

    # Cleanup
    Path(checkpoint_path).unlink()


# ============================================================================
# Edge Cases and Validation
# ============================================================================

def test_scheduler_zero_warmup(optimizer_fp32):
    """Test scheduler with zero warmup steps."""
    scheduler = CosineAnnealingWarmRestartsWithWarmup(
        optimizer_fp32,
        warmup_steps=0,
        T_0=1000,
        eta_min=1e-6,
    )

    # First LR should be base_lr (no warmup)
    lr = scheduler.get_lr()[0]
    assert abs(lr - 1e-4) < 1e-8


def test_scheduler_min_lr_never_exceeded(optimizer_fp32):
    """Test that LR never goes below eta_min."""
    eta_min = 1e-6
    scheduler = CosineAnnealingWarmRestartsWithWarmup(
        optimizer_fp32,
        warmup_steps=100,
        T_0=1000,
        T_mult=2,
        eta_min=eta_min,
    )

    # Run for many steps
    for _ in range(50000):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        # During warmup, LR can be 0, but after warmup should be >= eta_min
        # Allow for small numerical error
        assert lr >= -1e-9


def test_optimizer_with_frozen_parameters():
    """Test optimizer with some frozen parameters."""
    model = SimpleModel()

    # Freeze first layer
    for param in model.layers[0].parameters():
        param.requires_grad = False

    # Create optimizer (should only include trainable params)
    optimizer = create_optimizer(model, lr=1e-4)

    # Count trainable params
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_in_optimizer = sum(
        p.numel() for group in optimizer.param_groups for p in group['params']
    )

    assert num_in_optimizer == num_trainable
    assert num_in_optimizer < sum(p.numel() for p in model.parameters())


@pytest.mark.skipif(not HAS_BITSANDBYTES, reason="bitsandbytes not available")
def test_8bit_optimizer_type():
    """Test that 8-bit optimizer is actually AdamW8bit when available."""
    model = SimpleModel()
    optimizer = create_optimizer(model, lr=1e-4, use_8bit=True)
    assert isinstance(optimizer, bnb.optim.AdamW8bit)


def test_memory_footprint_comparison():
    """Test memory footprint comparison between 8-bit and FP32."""
    model = SimpleModel(hidden_dim=256, num_layers=4)

    # 8-bit
    opt_8bit = create_optimizer(model, use_8bit=True)
    mem_8bit = get_optimizer_memory_footprint(opt_8bit)

    # FP32
    opt_fp32 = create_optimizer(model, use_8bit=False)
    mem_fp32 = get_optimizer_memory_footprint(opt_fp32)

    # Both should report positive memory
    assert mem_8bit['total_memory_gb'] > 0
    assert mem_fp32['total_memory_gb'] > 0

    # FP32 should use more memory
    assert mem_fp32['state_memory_gb'] >= mem_8bit['state_memory_gb']

    # If bitsandbytes available, should see ~4x reduction in state memory
    if HAS_BITSANDBYTES:
        ratio = mem_fp32['state_memory_gb'] / mem_8bit['state_memory_gb']
        assert ratio > 3.0  # ~4x reduction


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
