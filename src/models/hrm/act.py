"""Adaptive Computation Time (ACT) implementation for RetNet-HRM.

Implements Graves-style ACT halting with correct probability weighting.
Expert implementation from docs/snippets.

This module will be integrated in Phase 5 (US3: Adaptive Computation).
For MVP (US1), this is not used yet - included for completeness.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple


class ACT(nn.Module):
    """Adaptive Computation Time halting mechanism.

    Implements:
    - Per-step halting probability prediction
    - Remainder calculation for final step
    - Ponder cost computation (τ·E[steps])
    - Correct weighted output aggregation

    Based on Graves (2016): https://arxiv.org/abs/1603.08983
    """

    def __init__(
        self,
        d_model: int,
        tau: float = 2e-3,      # Ponder cost weight (from research.md)
        eps: float = 1e-3,      # Halting threshold
        t_min: int = 1,         # Minimum steps before halting allowed
        init_bias: float = -1.0, # Initial halting bias (prevents early halting)
    ):
        """Initialize ACT module.

        Args:
            d_model: Hidden dimension
            tau: Ponder cost weight (FR-013)
            eps: Halting threshold
            t_min: Minimum steps before halting
            init_bias: Initial bias for halting head (negative prevents early halting)
        """
        super().__init__()

        # Halting probability predictor
        self.halt = nn.Linear(d_model, 1)
        nn.init.constant_(self.halt.bias, init_bias)

        self.tau = tau
        self.eps = eps
        self.t_min = t_min

    def forward(
        self,
        h_seq: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, any]]:
        """Compute ACT-weighted output.

        Args:
            h_seq: List of hidden states from outer steps
                   Each tensor: (batch_size, d_model)

        Returns:
            Tuple of:
                - h_hat: Weighted output, shape (batch_size, d_model)
                - stats: Dict with p_t, alphas, steps, ponder cost
        """
        ps, Rs, alphas = [], [], []

        # Initialize remainder accumulator
        R = h_seq[0].new_zeros((h_seq[0].size(0), 1))

        # Compute halting probabilities
        for t, h in enumerate(h_seq, start=1):
            p = torch.sigmoid(self.halt(h))  # (B, 1)

            # Force p=0 for first t_min steps (minimum computation)
            if t <= self.t_min:
                p = p * 0.0

            R = R + p
            ps.append(p)
            Rs.append(R.clamp(max=1.0))

        # Compute alphas (weights for each step)
        alphas = []
        R_prev = h_seq[0].new_zeros((h_seq[0].size(0), 1))

        for p, R in zip(ps, Rs):
            # Alpha is p if we haven't halted, else remainder
            a = torch.where(R < 1 - self.eps, p, (1 - self.eps) - R_prev)
            a = a.clamp(min=0.0)
            alphas.append(a)
            R_prev = R

        # Weighted sum of hidden states
        H = torch.stack(h_seq, dim=0)        # (T, B, d)
        A = torch.stack(alphas, dim=0)       # (T, B, 1)
        h_hat = (A * H).sum(dim=0)           # (B, d)

        # Compute stats
        steps = (torch.cat(Rs, dim=-1) < 1 - self.eps).sum(dim=-1) + 1  # (B,)
        ponder = self.tau * steps.float().mean()  # Scalar ponder cost

        stats = {
            "p_t": ps,                # List of halting probabilities
            "alphas": alphas,         # List of weights
            "steps": steps,           # Tensor of steps per example
            "ponder": ponder,         # Scalar ponder cost (for loss)
        }

        return h_hat, stats
