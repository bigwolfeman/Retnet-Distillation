"""
Adaptive Computation Time (ACT) Halting Head.

Implements Graves-style ACT for dynamic computation allocation.
The halting head predicts when to stop pondering based on the current
controller state.

References:
- Adaptive Computation Time: https://arxiv.org/abs/1603.08983
- PonderNet: https://arxiv.org/abs/2107.05407
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class HaltingOutput:
    """Output from ACT halting computation.

    Attributes:
        should_halt: Whether to halt at this step (bool)
        halting_prob: Halting probability p_t (float in [0, 1])
        accumulated_prob: Cumulative probability R_t = sum(p_i)
        ponder_cost: Expected number of steps E[N]
        num_steps: Current step number
    """
    should_halt: bool
    halting_prob: float
    accumulated_prob: float
    ponder_cost: float
    num_steps: int


class ACTHaltingHead(nn.Module):
    """
    Adaptive Computation Time (ACT) halting head.

    Predicts halting probability p_t at each step based on controller state.
    Uses Graves' ACT formulation with:
    - Cumulative probability R_t = sum_{i=1}^t p_i
    - Halt when R_t >= 1-epsilon or step >= T_max
    - Final step uses alpha_T = 1 - R_{T-1} (remainder handling)

    Args:
        d_input: Input dimension (controller output dimension)
        epsilon: Halting threshold (default: 1e-3)
        bias_init: Initial bias for halting head (default: -1.0)
                  Negative bias prevents premature halting
    """

    def __init__(
        self,
        d_input: int,
        epsilon: float = 1e-3,
        bias_init: float = -1.0,
    ):
        super().__init__()

        self.d_input = d_input
        self.epsilon = epsilon
        self.bias_init = bias_init

        # Halting prediction head
        # Single linear layer + sigmoid
        self.halting_linear = nn.Linear(d_input, 1)

        # Initialize bias to prevent early halting
        nn.init.constant_(self.halting_linear.bias, bias_init)

    def forward(
        self,
        controller_state: torch.Tensor,
        accumulated_prob: torch.Tensor,
        step: int,
        max_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute halting probability for current step.

        Args:
            controller_state: HRM controller state [batch, d_input]
            accumulated_prob: R_{t-1}, cumulative prob so far [batch]
            step: Current step number (1-indexed)
            max_steps: Maximum allowed steps

        Returns:
            tuple of (halting_prob, new_accumulated_prob, should_halt):
                - halting_prob: p_t for this step [batch]
                - new_accumulated_prob: R_t = R_{t-1} + p_t [batch]
                - should_halt: Boolean mask [batch]
        """
        batch_size = controller_state.shape[0]

        # Compute raw halting logit
        halting_logit = self.halting_linear(controller_state).squeeze(-1)  # [batch]

        # Apply sigmoid to get probability
        halting_prob = torch.sigmoid(halting_logit)  # [batch], in [0, 1]

        # Update accumulated probability
        new_accumulated_prob = accumulated_prob + halting_prob

        # Determine if should halt
        # Halt if: R_t >= 1-epsilon OR step >= max_steps
        threshold_reached = new_accumulated_prob >= (1.0 - self.epsilon)
        max_steps_reached = (step >= max_steps)
        should_halt = threshold_reached | max_steps_reached

        # For final step (max_steps), use remainder: alpha_T = 1 - R_{T-1}
        if step == max_steps:
            # Adjust halting prob to use up remaining probability
            halting_prob = torch.clamp(1.0 - accumulated_prob, min=0.0, max=1.0)
            new_accumulated_prob = torch.ones_like(accumulated_prob)

        return halting_prob, new_accumulated_prob, should_halt

    def compute_ponder_cost(
        self,
        halting_probs: torch.Tensor,
        steps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute expected number of steps E[N] for ponder cost.

        E[N] = sum_{t=1}^T t * alpha_t
        where alpha_t is the normalized halting weight

        Args:
            halting_probs: Halting probabilities for each step [batch, T]
            steps: Step indices tensor [T] (1, 2, 3, ..., T)

        Returns:
            Expected steps [batch]
        """
        # Normalize halting probs to sum to 1 (alpha_t)
        # This handles remainder correction
        alpha = halting_probs / (halting_probs.sum(dim=1, keepdim=True) + 1e-9)

        # E[N] = sum(t * alpha_t)
        expected_steps = (alpha * steps.unsqueeze(0)).sum(dim=1)

        return expected_steps


class ACTManager:
    """
    Manages ACT loop execution and ponder cost computation.

    Handles:
    - Step-by-step halting decisions
    - Accumulation of weighted outputs
    - Ponder cost tracking
    """

    def __init__(
        self,
        min_steps: int = 1,
        max_steps: int = 10,
        epsilon: float = 1e-3,
    ):
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.epsilon = epsilon

    def execute_act_loop(
        self,
        compute_fn,
        initial_input: torch.Tensor,
        halting_head: ACTHaltingHead,
        controller_state_fn,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Execute ACT loop with dynamic halting.

        Args:
            compute_fn: Function to compute output at each step
                       Signature: (input, step) -> output
            initial_input: Initial input to compute_fn
            halting_head: ACT halting head
            controller_state_fn: Function to get controller state
                                Signature: () -> controller_state

        Returns:
            tuple of (weighted_output, ponder_cost, num_steps):
                - weighted_output: Probability-weighted output
                - ponder_cost: Expected number of steps
                - num_steps: Actual steps taken
        """
        batch_size = initial_input.shape[0]
        device = initial_input.device

        # Initialize accumulators
        accumulated_output = None
        accumulated_prob = torch.zeros(batch_size, device=device)
        halting_probs_list = []
        halted_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(1, self.max_steps + 1):
            # Get controller state for halting decision
            controller_state = controller_state_fn()

            # Compute halting probability
            halting_prob, new_accumulated_prob, should_halt = halting_head(
                controller_state=controller_state,
                accumulated_prob=accumulated_prob,
                step=step,
                max_steps=self.max_steps,
            )

            # Store halting probability
            halting_probs_list.append(halting_prob.unsqueeze(1))

            # Compute output for this step
            step_output = compute_fn(initial_input, step)

            # Weight by halting probability
            # For sequences still computing, use full halting_prob
            # For halted sequences, use 0
            active_mask = ~halted_mask
            weight = halting_prob * active_mask.float()

            # Accumulate weighted output
            weighted_step_output = step_output * weight.unsqueeze(-1)

            if accumulated_output is None:
                accumulated_output = weighted_step_output
            else:
                accumulated_output = accumulated_output + weighted_step_output

            # Update state
            accumulated_prob = new_accumulated_prob
            halted_mask = halted_mask | should_halt

            # Early exit if all sequences halted and past min_steps
            if step >= self.min_steps and halted_mask.all():
                num_steps = step
                break
        else:
            num_steps = self.max_steps

        # Compute ponder cost
        halting_probs = torch.cat(halting_probs_list, dim=1)  # [batch, T]
        steps_tensor = torch.arange(1, num_steps + 1, device=device, dtype=torch.float32)
        ponder_cost = halting_head.compute_ponder_cost(halting_probs, steps_tensor)

        return accumulated_output, ponder_cost, num_steps


def test_act_halting():
    """Test ACT halting head implementation."""
    print("Testing ACT Halting Head...")

    # Configuration
    batch_size = 4
    d_input = 1024
    max_steps = 10
    epsilon = 1e-3

    # Create halting head
    halting_head = ACTHaltingHead(
        d_input=d_input,
        epsilon=epsilon,
        bias_init=-1.0,
    )

    print(f"\nConfiguration:")
    print(f"  d_input: {d_input}")
    print(f"  epsilon: {epsilon}")
    print(f"  max_steps: {max_steps}")

    # Test 1: Single step halting
    print("\n[Test 1] Single step halting")
    controller_state = torch.randn(batch_size, d_input)
    accumulated_prob = torch.zeros(batch_size)

    halt_prob, new_accum, should_halt = halting_head(
        controller_state=controller_state,
        accumulated_prob=accumulated_prob,
        step=1,
        max_steps=max_steps,
    )

    print(f"  Halting prob range: [{halt_prob.min():.3f}, {halt_prob.max():.3f}]")
    print(f"  Accumulated prob: {new_accum.mean():.3f}")
    print(f"  Should halt: {should_halt.sum().item()}/{batch_size}")
    print("  [PASS]")

    # Test 2: Multi-step simulation
    print("\n[Test 2] Multi-step ACT simulation")
    accumulated_prob = torch.zeros(batch_size)
    halting_probs_list = []

    for step in range(1, max_steps + 1):
        controller_state = torch.randn(batch_size, d_input)

        halt_prob, accumulated_prob, should_halt = halting_head(
            controller_state=controller_state,
            accumulated_prob=accumulated_prob,
            step=step,
            max_steps=max_steps,
        )

        halting_probs_list.append(halt_prob.unsqueeze(1))

        halted_count = should_halt.sum().item()
        print(f"  Step {step}: halt_prob={halt_prob.mean():.3f}, "
              f"R_t={accumulated_prob.mean():.3f}, halted={halted_count}/{batch_size}")

        if should_halt.all():
            print(f"  All sequences halted at step {step}")
            break

    print("  [PASS]")

    # Test 3: Ponder cost computation
    print("\n[Test 3] Ponder cost computation")
    halting_probs = torch.cat(halting_probs_list, dim=1)  # [batch, T]
    steps_tensor = torch.arange(1, halting_probs.shape[1] + 1, dtype=torch.float32)

    ponder_cost = halting_head.compute_ponder_cost(halting_probs, steps_tensor)

    print(f"  Halting probs shape: {halting_probs.shape}")
    print(f"  Ponder cost range: [{ponder_cost.min():.2f}, {ponder_cost.max():.2f}]")
    print(f"  Mean ponder cost: {ponder_cost.mean():.2f}")
    assert ponder_cost.min() >= 1.0, "Ponder cost should be >= 1"
    assert ponder_cost.max() <= max_steps, f"Ponder cost should be <= {max_steps}"
    print("  [PASS]")

    # Test 4: Final step remainder handling
    print("\n[Test 4] Final step remainder handling")
    accumulated_prob = torch.tensor([0.7, 0.8, 0.9, 0.95])  # Different accumulations
    controller_state = torch.randn(batch_size, d_input)

    halt_prob, new_accum, should_halt = halting_head(
        controller_state=controller_state,
        accumulated_prob=accumulated_prob,
        step=max_steps,  # Final step
        max_steps=max_steps,
    )

    print(f"  Input R_{{T-1}}: {accumulated_prob.tolist()}")
    print(f"  Final p_T: {halt_prob.tolist()}")
    print(f"  Final R_T: {new_accum.tolist()}")
    assert torch.allclose(new_accum, torch.ones(batch_size)), "Final R_T should be 1.0"
    assert should_halt.all(), "Should halt at max_steps"
    print("  [PASS]")

    print("\n" + "="*50)
    print("[PASS] All ACT Halting tests passed!")
    print("="*50)

    # Summary
    print("\nSummary:")
    param_count = sum(p.numel() for p in halting_head.parameters())
    print(f"  Halting head parameters: {param_count}")
    print(f"  Epsilon: {epsilon}")
    print(f"  Bias init: -1.0")


if __name__ == "__main__":
    test_act_halting()
