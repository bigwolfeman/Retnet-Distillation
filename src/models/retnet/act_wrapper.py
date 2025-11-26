"""Adaptive Computation Time (ACT) wrapper for RetNet backbone.

Implements ACT mechanism from Graves (2016): https://arxiv.org/abs/1603.08983
Allows the model to adaptively determine computation depth per input.

Key Features:
- Halting probability mechanism (learns when to stop pondering)
- Weighted output aggregation across pondering steps
- Pondering cost penalty (encourages efficiency)
- Compatible with both training and recurrent inference modes

Design Decisions:
- ACT wraps the entire RetNet backbone (not individual layers)
- Each "pondering step" = full forward pass through all RetNet layers
- Halting is computed per-token (each position can ponder independently)
- Epsilon threshold = 0.01 (allows single-step computation if sufficient)

References:
- Graves (2016): Adaptive Computation Time for Recurrent Neural Networks
- PonderNet (2021): Learning to Ponder (DeepMind improvement)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math

from .backbone import RetNetBackbone


class ACTRetNetBackbone(nn.Module):
    """RetNet backbone with Adaptive Computation Time.

    Wraps RetNetBackbone to add adaptive pondering capability.
    Each input can take 1 to max_steps forward passes through the network,
    with the model learning to halt early for simple inputs.

    Args:
        backbone: RetNetBackbone instance to wrap
        max_steps: Maximum number of pondering steps (default: 10)
        epsilon: Halting threshold (default: 0.01)
        ponder_penalty: Weight for pondering cost loss (default: 0.01)
        use_geometric_prior: Use geometric distribution prior for regularization (default: False)
        prior_lambda: Lambda for geometric prior (default: 0.5)
    """

    def __init__(
        self,
        backbone: RetNetBackbone,
        max_steps: int = 10,
        epsilon: float = 0.01,
        ponder_penalty: float = 0.01,
        use_geometric_prior: bool = False,
        prior_lambda: float = 0.5,
    ):
        super().__init__()

        self.backbone = backbone
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.ponder_penalty = ponder_penalty
        self.use_geometric_prior = use_geometric_prior
        self.prior_lambda = prior_lambda

        # Halting unit: projects hidden states to halting probability
        # One per position (per-token halting)
        self.halting_unit = nn.Sequential(
            nn.Linear(backbone.d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output in (0, 1)
        )

        # Initialize halting unit to start with moderate halting probabilities
        # This encourages exploration early in training
        nn.init.xavier_uniform_(self.halting_unit[0].weight)
        nn.init.constant_(self.halting_unit[0].bias, 0.0)
        nn.init.xavier_uniform_(self.halting_unit[2].weight)
        nn.init.constant_(self.halting_unit[2].bias, 1.0)  # Bias toward ~0.7 initial halting prob

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: Optional[torch.Tensor] = None,
        return_act_info: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Forward pass with adaptive computation time.

        Args:
            input_ids: Token IDs, shape (batch_size, seq_len)
            segment_ids: Optional segment IDs for packed sequences
            return_act_info: If True, return ACT diagnostic info

        Returns:
            Tuple of:
                - Final hidden states, shape (batch_size, seq_len, d_model)
                - ACT info dict (if return_act_info=True), contains:
                    - 'pondering_cost': Average number of steps taken (batch_size, seq_len)
                    - 'halting_probs': Halting probabilities per step (batch_size, seq_len, max_steps)
                    - 'remainders': Final remainder probabilities (batch_size, seq_len)
                    - 'n_steps': Actual number of steps per position (batch_size, seq_len)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Initialize accumulators
        accumulated_output = torch.zeros(
            batch_size, seq_len, self.backbone.d_model,
            device=device, dtype=torch.float32
        )
        accumulated_halting_prob = torch.zeros(
            batch_size, seq_len, device=device, dtype=torch.float32
        )

        # Track which positions are still pondering
        still_running = torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)

        # Store diagnostics (if requested)
        if return_act_info:
            all_halting_probs = torch.zeros(
                batch_size, seq_len, self.max_steps,
                device=device, dtype=torch.float32
            )
            n_steps = torch.zeros(batch_size, seq_len, device=device, dtype=torch.long)

        # Pondering loop
        for step in range(self.max_steps):
            # Forward pass through backbone
            hidden_states = self.backbone.forward_train(input_ids, segment_ids)

            # Compute halting probability for this step
            # Shape: (batch_size, seq_len, 1) -> (batch_size, seq_len)
            halting_logit = self.halting_unit(hidden_states).squeeze(-1)

            # Determine which positions halt at this step
            # For still-running positions, check if cumulative prob + new halt >= (1 - epsilon)
            can_halt = still_running & (accumulated_halting_prob + halting_logit >= 1.0 - self.epsilon)

            # For halting positions, use remainder to make cumulative prob = 1.0
            # For continuing positions, use the full halting probability
            halting_prob = torch.where(
                can_halt,
                1.0 - accumulated_halting_prob,  # Remainder
                halting_logit * still_running.float()  # Full prob (0 if stopped)
            )

            # Accumulate weighted output
            accumulated_output += hidden_states * halting_prob.unsqueeze(-1)
            accumulated_halting_prob += halting_prob

            # Update running mask
            still_running = still_running & ~can_halt

            # Store diagnostics
            if return_act_info:
                all_halting_probs[:, :, step] = halting_prob
                n_steps += still_running.long()

            # Early exit if all positions have halted
            if not still_running.any():
                break

        # Handle any positions that didn't halt (force halt at max_steps)
        if still_running.any():
            # Assign remaining probability mass to final step
            remainder = 1.0 - accumulated_halting_prob
            accumulated_output += hidden_states * remainder.unsqueeze(-1)
            accumulated_halting_prob += remainder

            if return_act_info:
                all_halting_probs[:, :, self.max_steps - 1] += remainder

        # Prepare return values
        act_info = None
        if return_act_info:
            # Compute pondering cost (average steps taken per position)
            # Sum of halting probs weighted by step number: Σ(n * p_n)
            step_weights = torch.arange(1, self.max_steps + 1, device=device, dtype=torch.float32)
            pondering_cost = (all_halting_probs * step_weights.view(1, 1, -1)).sum(dim=-1)

            act_info = {
                'pondering_cost': pondering_cost,  # (batch_size, seq_len)
                'halting_probs': all_halting_probs,  # (batch_size, seq_len, max_steps)
                'remainders': 1.0 - accumulated_halting_prob,  # (batch_size, seq_len)
                'n_steps': n_steps + 1,  # +1 because we count from 1, not 0
            }

        return accumulated_output, act_info

    def compute_act_loss(
        self,
        act_info: Dict[str, torch.Tensor],
        reduction: str = 'mean',
    ) -> torch.Tensor:
        """Compute ACT pondering loss.

        This is the regularization term that penalizes excessive computation.
        Two variants:
        - Simple: L_ponder = ponder_penalty * mean(pondering_cost)
        - Geometric prior: L_ponder = KL(halting_dist || geometric_prior)

        Args:
            act_info: ACT info dict from forward pass
            reduction: 'mean', 'sum', or 'none'

        Returns:
            Pondering loss (scalar if reduction != 'none')
        """
        if self.use_geometric_prior:
            # Geometric prior regularization (PonderNet style)
            # KL(p || p_G) where p_G is geometric distribution
            return self._compute_kl_geometric_loss(act_info, reduction)
        else:
            # Simple pondering cost penalty (ACT style)
            pondering_cost = act_info['pondering_cost']

            if reduction == 'mean':
                return self.ponder_penalty * pondering_cost.mean()
            elif reduction == 'sum':
                return self.ponder_penalty * pondering_cost.sum()
            else:  # 'none'
                return self.ponder_penalty * pondering_cost

    def _compute_kl_geometric_loss(
        self,
        act_info: Dict[str, torch.Tensor],
        reduction: str = 'mean',
    ) -> torch.Tensor:
        """Compute KL divergence against geometric prior.

        KL(p || p_G) = Σ p_n * log(p_n / p_G_n)
        where p_G_n = (1-λ)^(n-1) * λ is geometric distribution

        Args:
            act_info: ACT info dict
            reduction: 'mean', 'sum', or 'none'

        Returns:
            KL divergence loss
        """
        halting_probs = act_info['halting_probs']  # (batch_size, seq_len, max_steps)

        # Compute geometric prior probabilities
        # p_G_n = (1-λ)^(n-1) * λ for n = 1, 2, ..., N
        steps = torch.arange(self.max_steps, device=halting_probs.device, dtype=torch.float32)
        geometric_probs = (1 - self.prior_lambda) ** steps * self.prior_lambda
        geometric_probs = geometric_probs / geometric_probs.sum()  # Normalize
        geometric_probs = geometric_probs.view(1, 1, -1)  # Broadcast shape

        # KL divergence: KL(p || q) = Σ p * log(p/q)
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        kl_div = halting_probs * (
            torch.log(halting_probs + eps) - torch.log(geometric_probs + eps)
        )
        kl_div = kl_div.sum(dim=-1)  # Sum over steps

        if reduction == 'mean':
            return self.ponder_penalty * kl_div.mean()
        elif reduction == 'sum':
            return self.ponder_penalty * kl_div.sum()
        else:  # 'none'
            return self.ponder_penalty * kl_div

    @torch.no_grad()
    def forward_recurrent(
        self,
        input_ids: torch.Tensor,
        state: Optional[tuple] = None,
        max_ponder_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, tuple, Dict[str, torch.Tensor]]:
        """Forward pass in recurrent mode with ACT.

        NOTE: ACT in recurrent mode is tricky because we need to maintain
        both the recurrent state AND the pondering state. This implementation
        uses a simplified approach where each chunk ponders independently.

        Args:
            input_ids: Token IDs, shape (batch_size, chunk_size)
            state: Tuple of (retnet_state, ponder_state)
            max_ponder_steps: Override max_steps for inference

        Returns:
            Tuple of:
                - Final hidden states
                - Updated state tuple
                - ACT diagnostics
        """
        if max_ponder_steps is None:
            max_ponder_steps = self.max_steps

        batch_size, chunk_size = input_ids.shape
        device = input_ids.device

        # Unpack state
        if state is None:
            retnet_state = None
        else:
            retnet_state, _ = state

        # Initialize accumulators
        accumulated_output = torch.zeros(
            batch_size, chunk_size, self.backbone.d_model,
            device=device, dtype=torch.float32
        )
        accumulated_halting_prob = torch.zeros(
            batch_size, chunk_size, device=device, dtype=torch.float32
        )
        still_running = torch.ones(batch_size, chunk_size, device=device, dtype=torch.bool)

        # Diagnostics
        all_halting_probs = []

        # Pondering loop
        for step in range(max_ponder_steps):
            # Recurrent forward pass
            hidden_states, retnet_state = self.backbone.forward_recurrent(
                input_ids, retnet_state
            )

            # Compute halting probability
            halting_logit = self.halting_unit(hidden_states).squeeze(-1)

            # Determine halting
            can_halt = still_running & (accumulated_halting_prob + halting_logit >= 1.0 - self.epsilon)
            halting_prob = torch.where(
                can_halt,
                1.0 - accumulated_halting_prob,
                halting_logit * still_running.float()
            )

            # Accumulate
            accumulated_output += hidden_states * halting_prob.unsqueeze(-1)
            accumulated_halting_prob += halting_prob
            still_running = still_running & ~can_halt

            all_halting_probs.append(halting_prob)

            if not still_running.any():
                break

        # Force halt if needed
        if still_running.any():
            remainder = 1.0 - accumulated_halting_prob
            accumulated_output += hidden_states * remainder.unsqueeze(-1)

        # Pack state
        new_state = (retnet_state, None)  # No persistent ponder state for now

        # Diagnostics
        diagnostics = {
            'halting_probs': torch.stack(all_halting_probs, dim=-1),
            'mean_steps': accumulated_halting_prob.sum(dim=-1).mean().item(),
        }

        return accumulated_output, new_state, diagnostics


class ACTLoss(nn.Module):
    """Combined loss for training ACT models.

    Combines:
    - Task loss (e.g., cross-entropy for language modeling)
    - Pondering loss (regularization on computation cost)

    Usage:
        criterion = ACTLoss(task_loss_fn=nn.CrossEntropyLoss())
        loss, loss_dict = criterion(
            logits=model_output,
            targets=targets,
            act_info=act_info
        )
    """

    def __init__(
        self,
        act_model: ACTRetNetBackbone,
        task_loss_weight: float = 1.0,
    ):
        """Initialize ACT loss.

        Args:
            act_model: ACT-wrapped model (needed to compute pondering loss)
            task_loss_weight: Weight for task loss (default: 1.0)
        """
        super().__init__()
        self.act_model = act_model
        self.task_loss_weight = task_loss_weight

    def forward(
        self,
        task_loss: torch.Tensor,
        act_info: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute combined loss.

        Args:
            task_loss: Task-specific loss (e.g., CE loss)
            act_info: ACT diagnostics from forward pass

        Returns:
            Tuple of:
                - Total loss (scalar)
                - Loss breakdown dict
        """
        # Task loss (already computed)
        weighted_task_loss = self.task_loss_weight * task_loss

        # Pondering loss
        ponder_loss = self.act_model.compute_act_loss(act_info, reduction='mean')

        # Total loss
        total_loss = weighted_task_loss + ponder_loss

        # Breakdown for logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'task_loss': task_loss.item(),
            'ponder_loss': ponder_loss.item(),
            'mean_ponder_steps': act_info['pondering_cost'].mean().item(),
        }

        return total_loss, loss_dict


def create_act_retnet(
    vocab_size: int = 100352,
    d_model: int = 2816,
    n_layers: int = 28,
    n_heads: int = 12,
    dropout: float = 0.0,
    max_seq_len: int = 65536,
    # ACT-specific args
    act_max_steps: int = 10,
    act_epsilon: float = 0.01,
    act_ponder_penalty: float = 0.01,
    act_use_geometric_prior: bool = False,
    act_prior_lambda: float = 0.5,
    debug: bool = False,
) -> ACTRetNetBackbone:
    """Factory function to create ACT-wrapped RetNet.

    Args:
        vocab_size: Vocabulary size
        d_model: Hidden dimension
        n_layers: Number of RetNet layers
        n_heads: Number of retention heads
        dropout: Dropout rate
        max_seq_len: Maximum sequence length
        act_max_steps: Maximum pondering steps
        act_epsilon: Halting threshold
        act_ponder_penalty: Weight for pondering cost
        act_use_geometric_prior: Use geometric prior regularization
        act_prior_lambda: Lambda for geometric prior
        debug: Enable debug output

    Returns:
        ACT-wrapped RetNet model
    """
    # Create base RetNet backbone
    backbone = RetNetBackbone(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        max_seq_len=max_seq_len,
        debug=debug,
    )

    # Wrap with ACT
    act_model = ACTRetNetBackbone(
        backbone=backbone,
        max_steps=act_max_steps,
        epsilon=act_epsilon,
        ponder_penalty=act_ponder_penalty,
        use_geometric_prior=act_use_geometric_prior,
        prior_lambda=act_prior_lambda,
    )

    return act_model
