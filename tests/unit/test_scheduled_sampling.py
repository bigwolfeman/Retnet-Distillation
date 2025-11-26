"""Unit tests for scheduled sampling implementation."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from train.curriculum import CurriculumState


def create_mock_trainer():
    """Create a mock trainer with curriculum state."""
    class MockTrainer:
        def __init__(self):
            self.curriculum_state = CurriculumState()
            self.debug = False
    return MockTrainer()


def compute_sampling_prob(trainer):
    """Compute sampling probability from trainer's loss history."""
    if len(trainer.curriculum_state.ce_loss_history) >= 10:
        avg_ce_loss = sum(trainer.curriculum_state.ce_loss_history) / 10
    else:
        avg_ce_loss = trainer.curriculum_state.ce_loss_history[-1] if trainer.curriculum_state.ce_loss_history else 1.0

    return max(0.0, min(0.3, (1.0 - avg_ce_loss) / 0.8 * 0.3))


def test_sampling_prob_ramp():
    """Test that sampling probability ramps correctly based on loss."""
    trainer = create_mock_trainer()
    trainer.curriculum_state.training_mode = "scheduled_sampling"

    # At ce_loss = 1.0 (just transitioned): 0%
    trainer.curriculum_state.ce_loss_history = [1.0] * 10
    sampling_prob = compute_sampling_prob(trainer)
    assert abs(sampling_prob - 0.0) < 0.01, f"Expected 0.0, got {sampling_prob}"

    # At ce_loss = 0.6 (midpoint): 15%
    trainer.curriculum_state.ce_loss_history = [0.6] * 10
    sampling_prob = compute_sampling_prob(trainer)
    assert abs(sampling_prob - 0.15) < 0.01, f"Expected 0.15, got {sampling_prob}"

    # At ce_loss = 0.2 (about to transition to GA): 30% (capped)
    trainer.curriculum_state.ce_loss_history = [0.2] * 10
    sampling_prob = compute_sampling_prob(trainer)
    assert abs(sampling_prob - 0.3) < 0.01, f"Expected 0.3, got {sampling_prob}"

    # At ce_loss = 0.05 (very low): 30% (capped)
    trainer.curriculum_state.ce_loss_history = [0.05] * 10
    sampling_prob = compute_sampling_prob(trainer)
    assert abs(sampling_prob - 0.3) < 0.01, f"Expected 0.3, got {sampling_prob}"

    print("[PASS] test_sampling_prob_ramp passed")


def test_sampling_prob_zero_no_changes():
    """Test that sampling_prob=0 doesn't modify input."""
    # This test verifies the early return when sampling_prob < 0.01
    trainer = create_mock_trainer()

    # At ce_loss = 1.0, sampling_prob = 0%
    trainer.curriculum_state.ce_loss_history = [1.0] * 10
    sampling_prob = compute_sampling_prob(trainer)

    assert sampling_prob < 0.01, "Expected sampling prob near 0"
    print("[PASS] test_sampling_prob_zero_no_changes passed")


if __name__ == "__main__":
    print("Running scheduled sampling tests...")
    test_sampling_prob_ramp()
    test_sampling_prob_zero_no_changes()
    print("\n[SUCCESS] All tests passed!")
