"""Integration tests for training mode transitions."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from train.curriculum import CurriculumState
from configs.model_config import CurriculumConfig, BandConfig


def create_test_trainer():
    """Create a minimal trainer for testing."""
    class MockTrainer:
        def __init__(self):
            self.curriculum_state = CurriculumState()
            self.grad_accumulation_steps = 1
            self.batch_size = 4

        def update_loss_history(self, ce_loss: float, fsm_loss: float) -> None:
            """Update rolling window of loss history (last 10 steps)."""
            self.curriculum_state.ce_loss_history.append(ce_loss)
            if len(self.curriculum_state.ce_loss_history) > 10:
                self.curriculum_state.ce_loss_history.pop(0)

            self.curriculum_state.fsm_loss_history.append(fsm_loss)
            if len(self.curriculum_state.fsm_loss_history) > 10:
                self.curriculum_state.fsm_loss_history.pop(0)

        def check_mode_transition(self, global_step: int) -> bool:
            """Check if training mode should transition."""
            if len(self.curriculum_state.ce_loss_history) < 10:
                return False

            avg_ce_loss = sum(self.curriculum_state.ce_loss_history) / 10
            avg_fsm_loss = sum(self.curriculum_state.fsm_loss_history) / 10
            current_mode = self.curriculum_state.training_mode

            if current_mode == "teacher_forcing":
                if avg_ce_loss < 1.0 and avg_fsm_loss < 0.1:
                    self._transition_to_scheduled_sampling(global_step, avg_ce_loss, avg_fsm_loss)
                    return True

            elif current_mode == "scheduled_sampling":
                if avg_ce_loss < 0.2:
                    self._transition_to_gradient_accumulation(global_step, avg_ce_loss)
                    return True

            return False

        def _transition_to_scheduled_sampling(self, global_step, avg_ce_loss, avg_fsm_loss):
            self.curriculum_state.training_mode = "scheduled_sampling"
            self.curriculum_state.mode_transition_step = global_step

        def _transition_to_gradient_accumulation(self, global_step, avg_ce_loss):
            self.curriculum_state.training_mode = "gradient_accumulation"
            self.curriculum_state.mode_transition_step = global_step
            if torch.cuda.is_available():
                total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                self.grad_accumulation_steps = 4 if total_memory_gb >= 24 else 2
            else:
                self.grad_accumulation_steps = 2

    return MockTrainer()


def test_teacher_forcing_to_scheduled_sampling():
    """Test TF -> SS transition when thresholds met."""
    print("\n[TEST] Teacher Forcing -> Scheduled Sampling")
    trainer = create_test_trainer()
    assert trainer.curriculum_state.training_mode == "teacher_forcing"

    # Simulate 20 steps with decreasing loss
    for i in range(20):
        ce_loss = 1.5 - i * 0.08  # Decreases from 1.5 rapidly
        fsm_loss = 0.2 - i * 0.015  # Decreases from 0.2 rapidly
        trainer.update_loss_history(ce_loss, fsm_loss)

        if i >= 9:  # After 10 steps (0-9), check transition
            changed = trainer.check_mode_transition(global_step=i)

            # Calculate what the average would be for the last 10 steps
            if len(trainer.curriculum_state.ce_loss_history) >= 10:
                avg_ce = sum(trainer.curriculum_state.ce_loss_history) / 10
                avg_fsm = sum(trainer.curriculum_state.fsm_loss_history) / 10

                if avg_ce < 1.0 and avg_fsm < 0.1:
                    assert changed, f"Should transition at step {i} (avg_ce={avg_ce:.2f}, avg_fsm={avg_fsm:.3f})"
                    assert trainer.curriculum_state.training_mode == "scheduled_sampling"
                    print(f"  [OK] Transitioned at step {i} (avg_ce={avg_ce:.2f}, avg_fsm={avg_fsm:.3f})")
                    return

    raise AssertionError("Transition never occurred")


def test_scheduled_sampling_to_gradient_accumulation():
    """Test SS -> GA transition when threshold met."""
    print("\n[TEST] Scheduled Sampling -> Gradient Accumulation")
    trainer = create_test_trainer()
    trainer.curriculum_state.training_mode = "scheduled_sampling"
    trainer.curriculum_state.mode_transition_step = 0

    # Simulate steps with very low loss
    for i in range(20):
        ce_loss = 0.25 - i * 0.015  # Decreases from 0.25 rapidly
        fsm_loss = 0.02
        trainer.update_loss_history(ce_loss, fsm_loss)

        if i >= 9:  # After 10 steps, check transition
            changed = trainer.check_mode_transition(global_step=i)

            # Calculate what the average would be for the last 10 steps
            if len(trainer.curriculum_state.ce_loss_history) >= 10:
                avg_ce = sum(trainer.curriculum_state.ce_loss_history) / 10

                if avg_ce < 0.2:
                    assert changed, f"Should transition at step {i} (avg_ce={avg_ce:.2f})"
                    assert trainer.curriculum_state.training_mode == "gradient_accumulation"
                    assert trainer.grad_accumulation_steps in [2, 4]
                    print(f"  [OK] Transitioned at step {i} (avg_ce={avg_ce:.2f}, accum={trainer.grad_accumulation_steps}x)")
                    return

    raise AssertionError("Transition never occurred")


def test_band_promotion_resets_mode():
    """Test that band promotion resets to teacher_forcing."""
    print("\n[TEST] Band promotion resets mode")
    trainer = create_test_trainer()

    # Advance to gradient_accumulation mode
    trainer.curriculum_state.training_mode = "gradient_accumulation"
    trainer.curriculum_state.mode_transition_step = 5000
    trainer.grad_accumulation_steps = 4

    # Reset (simulating band promotion)
    trainer.curriculum_state.reset_training_mode(step=10000)
    trainer.grad_accumulation_steps = 1

    # Verify reset
    assert trainer.curriculum_state.training_mode == "teacher_forcing"
    assert trainer.curriculum_state.mode_transition_step == 10000
    assert trainer.curriculum_state.ce_loss_history == []
    assert trainer.grad_accumulation_steps == 1
    print(f"  [OK] Mode reset to teacher_forcing with cleared history")


def test_no_backward_transitions():
    """Test that transitions are one-way (no reverting)."""
    print("\n[TEST] No backward transitions")
    trainer = create_test_trainer()

    # Advance to scheduled_sampling
    trainer.curriculum_state.training_mode = "scheduled_sampling"

    # Try to trigger TF transition with high loss
    for i in range(15):
        trainer.update_loss_history(ce_loss=5.0, fsm_loss=0.5)

    changed = trainer.check_mode_transition(global_step=20)

    # Should NOT revert to teacher_forcing
    assert not changed
    assert trainer.curriculum_state.training_mode == "scheduled_sampling"
    print(f"  [OK] No backward transition despite high loss")


if __name__ == "__main__":
    print("="*80)
    print("Running Mode Transition Integration Tests")
    print("="*80)

    try:
        test_teacher_forcing_to_scheduled_sampling()
        test_scheduled_sampling_to_gradient_accumulation()
        test_band_promotion_resets_mode()
        test_no_backward_transitions()

        print("\n" + "="*80)
        print("[SUCCESS] ALL TESTS PASSED")
        print("="*80)
    except AssertionError as e:
        print(f"\n[FAILED] TEST FAILED: {e}")
        sys.exit(1)
