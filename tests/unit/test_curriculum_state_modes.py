"""
Unit tests for CurriculumState mode tracking.

Per teacher-forcing-fix.md Task 1.4: Test state serialization, mode reset, and field initialization.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from train.curriculum import CurriculumState


def test_default_mode_is_teacher_forcing():
    """Test that default training mode is teacher_forcing."""
    state = CurriculumState()
    assert state.training_mode == "teacher_forcing"
    assert state.mode_transition_step == 0


def test_mode_serialization_roundtrip():
    """Test serialization and deserialization of all three modes."""
    # Test all three modes
    for mode in ["teacher_forcing", "scheduled_sampling", "gradient_accumulation"]:
        state = CurriculumState()
        state.training_mode = mode
        state.mode_transition_step = 5000
        state.ce_loss_history = [0.8, 0.7, 0.6]

        data = state.to_dict()
        restored = CurriculumState.from_dict(data)

        assert restored.training_mode == mode
        assert restored.mode_transition_step == 5000
        assert restored.ce_loss_history == [0.8, 0.7, 0.6]


def test_backward_compatibility():
    """Test that old state files without mode fields load correctly."""
    # Old state without mode fields
    old_data = {
        "current_band": "A1",
        "step_count": 10000,
        "promotion_history": [],
        "band_metrics": {},
        "last_eval_step": 0
    }
    state = CurriculumState.from_dict(old_data)
    assert state.training_mode == "teacher_forcing"
    assert state.ce_loss_history == []


def test_mode_reset():
    """Test that mode reset clears history and resets to teacher_forcing."""
    state = CurriculumState()
    state.training_mode = "gradient_accumulation"
    state.mode_transition_step = 5000
    state.ce_loss_history = [0.1, 0.2, 0.3]
    state.fsm_loss_history = [0.01, 0.02]

    state.reset_training_mode(step=10000)

    assert state.training_mode == "teacher_forcing"
    assert state.mode_transition_step == 10000
    assert state.ce_loss_history == []
    assert state.fsm_loss_history == []


if __name__ == "__main__":
    # Fix Unicode encoding on Windows
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    # Run tests
    print("Running CurriculumState mode tracking tests...\n")

    print("Test 1: Default mode is teacher_forcing")
    test_default_mode_is_teacher_forcing()
    print("✓ PASSED\n")

    print("Test 2: Mode serialization roundtrip")
    test_mode_serialization_roundtrip()
    print("✓ PASSED\n")

    print("Test 3: Backward compatibility")
    test_backward_compatibility()
    print("✓ PASSED\n")

    print("Test 4: Mode reset")
    test_mode_reset()
    print("✓ PASSED\n")

    print("✓ All CurriculumState mode tracking tests passed!")
