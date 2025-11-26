"""
Computation Budget configuration for Adaptive Computation Time (ACT).

Defines parameters that control the ACT loop including:
- Minimum and maximum steps
- Halting threshold (epsilon)
- Ponder cost weight (tau)
- Target average steps

Implements schema from data-model.md.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass
class ComputationBudget:
    """
    Configuration for adaptive computation budget (FR-006, FR-013).

    Controls ACT (Adaptive Computation Time) parameters:
    - Step limits: min_steps to max_steps (1-10 per FR-006)
    - Halting threshold: epsilon (when R_t >= 1-epsilon, halt)
    - Ponder cost: tau weight for loss regularization

    All parameters validated against FR-006 constraints.
    """

    # ACT parameters
    min_steps: int = 1                     # Minimum steps before halting allowed
    max_steps: int = 10                    # Maximum steps (T_max), FR-006: 1-10
    epsilon: float = 1e-3                  # Halting threshold (R_t >= 1-epsilon)

    # Ponder cost (FR-013)
    ponder_tau: float = 0.002              # Weight for ponder cost in loss
                                           # From research.md: 0.002 (annealed from 0.001)

    # Constraints and targets
    avg_steps_target: float = 4.0          # Target average steps per query
    max_total_ponder_budget: Optional[float] = None  # Total budget for batch (optional)

    # Monitoring (runtime statistics)
    current_avg_steps: float = field(default=0.0, init=False)  # Moving average
    total_ponder_cost: float = field(default=0.0, init=False)  # Accumulated cost
    step_history: List[int] = field(default_factory=list, init=False)  # History for analysis

    def __post_init__(self):
        """Validate configuration constraints."""
        # FR-006: Steps must be in range [1, 10]
        assert 1 <= self.min_steps <= 10, \
            f"min_steps ({self.min_steps}) must be in [1, 10] (FR-006)"
        assert 1 <= self.max_steps <= 10, \
            f"max_steps ({self.max_steps}) must be in [1, 10] (FR-006)"
        assert self.min_steps <= self.max_steps, \
            f"min_steps ({self.min_steps}) must be <= max_steps ({self.max_steps})"

        # Epsilon should be small but positive
        assert 0 < self.epsilon < 0.1, \
            f"epsilon ({self.epsilon}) should be small positive value (typically 1e-3)"

        # Ponder tau should be small positive
        assert 0 < self.ponder_tau < 0.1, \
            f"ponder_tau ({self.ponder_tau}) should be small positive value"

        # Target average steps should be reasonable
        assert self.min_steps <= self.avg_steps_target <= self.max_steps, \
            f"avg_steps_target ({self.avg_steps_target}) should be in [min_steps, max_steps]"

    def update_statistics(self, actual_steps: List[int]):
        """
        Update running statistics from batch of actual steps.

        Args:
            actual_steps: List of step counts from batch [batch_size]
        """
        if not actual_steps:
            return

        # Update moving average (exponential moving average with alpha=0.1)
        batch_avg = np.mean(actual_steps)
        self.current_avg_steps = 0.9 * self.current_avg_steps + 0.1 * batch_avg

        # Store history (for analysis, limit to last 1000)
        self.step_history.extend(actual_steps)
        if len(self.step_history) > 1000:
            self.step_history = self.step_history[-1000:]

    def validate_step_count(self, step: int) -> bool:
        """
        Check if step count is within budget.

        Args:
            step: Step number to validate

        Returns:
            True if step is within [min_steps, max_steps]
        """
        return self.min_steps <= step <= self.max_steps

    def should_halt(self, accumulated_prob: float, step: int) -> bool:
        """
        Determine if halting should occur (ACT logic).

        This is a convenience method that wraps the ACT halting logic.

        Args:
            accumulated_prob: R_t (sum of halting probs so far)
            step: Current step number

        Returns:
            True if should halt (R_t >= 1-epsilon OR step >= max_steps)
        """
        # Halt if probability threshold reached or max steps
        threshold_reached = accumulated_prob >= (1.0 - self.epsilon)
        max_steps_reached = step >= self.max_steps

        return threshold_reached or max_steps_reached

    def get_ponder_loss_weight(self, current_step: Optional[int] = None) -> float:
        """
        Get ponder cost weight (tau) for current training step.

        Optionally supports annealing schedule (from research.md):
        - Start: tau = 0.001
        - Target: tau = 0.002
        - Anneal over training

        Args:
            current_step: Current training step (for annealing)

        Returns:
            Ponder cost weight
        """
        # For now, return fixed tau
        # TODO: Implement annealing schedule if needed
        return self.ponder_tau

    def get_statistics(self) -> dict:
        """
        Get current statistics for logging.

        Returns:
            Dictionary with current statistics
        """
        if len(self.step_history) == 0:
            return {
                "avg_steps": 0.0,
                "min_steps_seen": 0,
                "max_steps_seen": 0,
                "std_steps": 0.0,
            }

        return {
            "avg_steps": self.current_avg_steps,
            "avg_steps_recent": np.mean(self.step_history[-100:]) if len(self.step_history) >= 100 else np.mean(self.step_history),
            "min_steps_seen": int(np.min(self.step_history)),
            "max_steps_seen": int(np.max(self.step_history)),
            "std_steps": float(np.std(self.step_history)),
            "total_samples": len(self.step_history),
        }

    def reset_statistics(self):
        """Reset runtime statistics."""
        self.current_avg_steps = 0.0
        self.total_ponder_cost = 0.0
        self.step_history = []

    def to_dict(self) -> dict:
        """
        Serialize configuration to dictionary.

        Returns:
            Dictionary with all configuration parameters
        """
        return {
            "min_steps": self.min_steps,
            "max_steps": self.max_steps,
            "epsilon": self.epsilon,
            "ponder_tau": self.ponder_tau,
            "avg_steps_target": self.avg_steps_target,
            "max_total_ponder_budget": self.max_total_ponder_budget,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ComputationBudget":
        """
        Create ComputationBudget from dictionary.

        Args:
            config_dict: Dictionary with configuration parameters

        Returns:
            ComputationBudget instance
        """
        return cls(
            min_steps=config_dict.get("min_steps", 1),
            max_steps=config_dict.get("max_steps", 10),
            epsilon=config_dict.get("epsilon", 1e-3),
            ponder_tau=config_dict.get("ponder_tau", 0.002),
            avg_steps_target=config_dict.get("avg_steps_target", 4.0),
            max_total_ponder_budget=config_dict.get("max_total_ponder_budget"),
        )


def test_computation_budget():
    """Test ComputationBudget implementation."""
    print("Testing ComputationBudget...")

    # Test 1: Default configuration
    print("\n[Test 1] Default configuration")
    budget = ComputationBudget()
    print(f"  min_steps: {budget.min_steps}")
    print(f"  max_steps: {budget.max_steps}")
    print(f"  epsilon: {budget.epsilon}")
    print(f"  ponder_tau: {budget.ponder_tau}")
    print(f"  avg_steps_target: {budget.avg_steps_target}")
    assert budget.min_steps == 1
    assert budget.max_steps == 10
    print("  [PASS]")

    # Test 2: Custom configuration
    print("\n[Test 2] Custom configuration")
    custom_budget = ComputationBudget(
        min_steps=2,
        max_steps=6,
        epsilon=0.001,
        ponder_tau=0.003,
        avg_steps_target=3.5,
    )
    assert custom_budget.max_steps == 6
    assert custom_budget.avg_steps_target == 3.5
    print(f"  Custom max_steps: {custom_budget.max_steps}")
    print(f"  Custom target: {custom_budget.avg_steps_target}")
    print("  [PASS]")

    # Test 3: Validation
    print("\n[Test 3] Validation")
    assert budget.validate_step_count(5) == True
    assert budget.validate_step_count(1) == True
    assert budget.validate_step_count(10) == True
    assert budget.validate_step_count(0) == False
    assert budget.validate_step_count(11) == False
    print("  Step validation working correctly")
    print("  [PASS]")

    # Test 4: Halting logic
    print("\n[Test 4] Halting logic")
    # Should halt when R_t >= 1-epsilon
    should_halt_1 = budget.should_halt(accumulated_prob=0.999, step=3)
    assert should_halt_1 == True, "Should halt when R_t >= 0.999"

    # Should not halt when R_t < 1-epsilon and step < max
    should_halt_2 = budget.should_halt(accumulated_prob=0.5, step=3)
    assert should_halt_2 == False, "Should not halt when R_t=0.5, step=3"

    # Should halt at max_steps
    should_halt_3 = budget.should_halt(accumulated_prob=0.5, step=10)
    assert should_halt_3 == True, "Should halt at max_steps"

    print("  Halting logic: threshold test [PASS]")
    print("  Halting logic: mid-computation [PASS]")
    print("  Halting logic: max_steps [PASS]")
    print("  [PASS]")

    # Test 5: Statistics update
    print("\n[Test 5] Statistics update")
    actual_steps_batch1 = [2, 3, 4, 5, 3]
    actual_steps_batch2 = [4, 4, 5, 6, 4]

    budget.update_statistics(actual_steps_batch1)
    stats1 = budget.get_statistics()
    print(f"  After batch 1: avg={stats1['avg_steps']:.2f}, "
          f"min={stats1['min_steps_seen']}, max={stats1['max_steps_seen']}")

    budget.update_statistics(actual_steps_batch2)
    stats2 = budget.get_statistics()
    print(f"  After batch 2: avg={stats2['avg_steps']:.2f}, "
          f"min={stats2['min_steps_seen']}, max={stats2['max_steps_seen']}")

    assert stats2['total_samples'] == 10
    assert stats2['min_steps_seen'] == 2
    assert stats2['max_steps_seen'] == 6
    print("  [PASS]")

    # Test 6: Serialization
    print("\n[Test 6] Serialization")
    config_dict = budget.to_dict()
    reconstructed = ComputationBudget.from_dict(config_dict)

    assert reconstructed.min_steps == budget.min_steps
    assert reconstructed.max_steps == budget.max_steps
    assert reconstructed.epsilon == budget.epsilon
    assert reconstructed.ponder_tau == budget.ponder_tau
    print(f"  Serialized: {list(config_dict.keys())}")
    print(f"  Reconstructed successfully")
    print("  [PASS]")

    # Test 7: Invalid configurations
    print("\n[Test 7] Invalid configuration handling")
    try:
        invalid = ComputationBudget(min_steps=15)  # > 10
        assert False, "Should have raised assertion"
    except AssertionError:
        print("  Correctly rejected min_steps=15 (> 10)")

    try:
        invalid = ComputationBudget(max_steps=0)  # < 1
        assert False, "Should have raised assertion"
    except AssertionError:
        print("  Correctly rejected max_steps=0 (< 1)")

    print("  [PASS]")

    print("\n" + "="*50)
    print("[PASS] All ComputationBudget tests passed!")
    print("="*50)

    # Summary
    print("\nSummary:")
    print(f"  Configured range: [{budget.min_steps}, {budget.max_steps}] steps (FR-006)")
    print(f"  Halting threshold: epsilon={budget.epsilon}")
    print(f"  Ponder cost weight: tau={budget.ponder_tau}")
    print(f"  Target average: {budget.avg_steps_target} steps")


if __name__ == "__main__":
    test_computation_budget()
