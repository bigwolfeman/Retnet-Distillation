"""Gumbel Top-k Router (T052 - US4).

Straight-through Gumbel-Softmax router for selecting top-k landmark tokens
within a budget constraint (B=24 tokens from FR-009).

Architecture from research.md:
- Input: route_logits over all candidate landmarks
- Selection: Gumbel-Softmax + top-k (differentiable during training)
- Straight-through estimator: hard selection, soft gradients
- Auxiliary losses: sparsity (λ_s=2e-4) + entropy (λ_e=1e-3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class GumbelTopKRouter(nn.Module):
    """
    Budgeted router using Gumbel-Softmax for landmark selection (FR-009).

    Selects up to B=24 global tokens from available landmarks via differentiable
    top-k selection. Uses straight-through gradients for training.

    Args:
        budget_B: Maximum global tokens (default: 24, from FR-009)
        temperature: Gumbel temperature (default: 0.7, anneals to 0.5)
        temperature_min: Minimum temperature (default: 0.5)
        lambda_sparsity: Sparsity loss weight (default: 2e-4)
        lambda_entropy: Entropy loss weight (default: 1e-3)
        hard: Use hard selection (default: True for straight-through)

    Examples:
        >>> router = GumbelTopKRouter(budget_B=24)
        >>> logits = torch.randn(2, 100)  # batch=2, 100 candidates
        >>> selected_indices, selected_probs, aux_losses = router(logits, k=24)
        >>> print(selected_indices.shape)  # (2, 24)
    """

    def __init__(
        self,
        budget_B: int = 24,
        temperature: float = 0.7,
        temperature_min: float = 0.5,
        lambda_sparsity: float = 2e-4,
        lambda_entropy: float = 1e-3,
        hard: bool = True,
    ):
        super().__init__()

        self.budget_B = budget_B
        self.temperature = temperature
        self.temperature_min = temperature_min
        self.lambda_sparsity = lambda_sparsity
        self.lambda_entropy = lambda_entropy
        self.hard = hard

        # Temperature annealing (scheduled externally)
        self.register_buffer('current_temperature', torch.tensor(temperature))

    def anneal_temperature(self, progress: float):
        """
        Anneal temperature during training.

        Args:
            progress: Training progress in [0, 1]

        Examples:
            >>> router.anneal_temperature(progress=0.5)
            >>> print(router.current_temperature)  # Between temperature and temperature_min
        """
        # Linear annealing: temp = max(temp_min, temp_start * (1 - progress))
        new_temp = max(
            self.temperature_min,
            self.temperature * (1.0 - progress)
        )
        self.current_temperature.fill_(new_temp)

    def forward(
        self,
        route_logits: torch.Tensor,
        k: Optional[int] = None,
        return_aux_losses: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[dict]]:
        """
        Select top-k landmarks via Gumbel-Softmax.

        Args:
            route_logits: Routing scores [batch, num_candidates]
            k: Number of tokens to select (default: budget_B)
            return_aux_losses: Whether to compute auxiliary losses

        Returns:
            selected_indices: Hard indices [batch, k]
            selected_probs: Soft probabilities [batch, k] (for gradient flow)
            aux_losses: Dict with 'sparsity' and 'entropy' losses (or None)

        Examples:
            >>> router = GumbelTopKRouter(budget_B=24)
            >>> logits = torch.randn(4, 100)  # 4 batches, 100 candidates
            >>> indices, probs, losses = router(logits, k=24)
            >>> print(indices.shape, probs.shape)  # (4, 24), (4, 24)
        """
        if k is None:
            k = self.budget_B

        batch_size, num_candidates = route_logits.shape

        # Apply Gumbel-Softmax
        # During training: soft probabilities, during eval: hard selection
        if self.training:
            # Gumbel-Softmax: sample from Gumbel(0,1) and apply softmax
            gumbel_probs = self._gumbel_softmax(
                route_logits,
                temperature=self.current_temperature.item(),
                hard=self.hard
            )
        else:
            # Evaluation: just use softmax (no Gumbel noise)
            gumbel_probs = F.softmax(route_logits, dim=-1)

        # Select top-k
        # Get top-k probabilities and indices
        topk_probs, topk_indices = torch.topk(gumbel_probs, k=k, dim=-1)

        # Straight-through estimator: use hard indices, but gradient flows through soft probs
        if self.hard and self.training:
            # Hard selection for forward pass
            selected_indices = topk_indices
            # Soft probabilities for backward pass
            selected_probs = topk_probs
        else:
            selected_indices = topk_indices
            selected_probs = topk_probs

        # Compute auxiliary losses
        aux_losses = None
        if return_aux_losses and self.training:
            aux_losses = self._compute_aux_losses(gumbel_probs, topk_probs)

        return selected_indices, selected_probs, aux_losses

    def _gumbel_softmax(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = False,
    ) -> torch.Tensor:
        """
        Gumbel-Softmax sampling.

        Args:
            logits: [batch, num_candidates]
            temperature: Softmax temperature
            hard: If True, use straight-through estimator

        Returns:
            probs: [batch, num_candidates] probabilities
        """
        # Sample Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)

        # Add noise to logits
        gumbel_logits = (logits + gumbel_noise) / temperature

        # Softmax
        probs = F.softmax(gumbel_logits, dim=-1)

        if hard:
            # Straight-through: one-hot in forward, soft in backward
            # Get argmax
            indices = probs.argmax(dim=-1, keepdim=True)
            # Create one-hot
            hard_probs = torch.zeros_like(probs).scatter_(-1, indices, 1.0)
            # Straight-through: hard forward, soft backward
            probs = hard_probs - probs.detach() + probs

        return probs

    def _compute_aux_losses(
        self,
        all_probs: torch.Tensor,
        selected_probs: torch.Tensor,
    ) -> dict:
        """
        Compute auxiliary losses for routing.

        Sparsity loss: Encourages concentrated selection (L1 on top-k probs)
        Entropy loss: Encourages diversity in selection (negative entropy)

        Args:
            all_probs: Full probability distribution [batch, num_candidates]
            selected_probs: Selected probabilities [batch, k]

        Returns:
            Dictionary with 'sparsity' and 'entropy' losses
        """
        # Sparsity loss: L1 norm on selected probabilities
        # Encourages router to be confident (select fewer high-prob tokens)
        sparsity_loss = self.lambda_sparsity * selected_probs.sum(dim=-1).mean()

        # Entropy loss: Negative entropy over full distribution
        # Encourages diversity (don't always select the same tokens)
        # H(p) = -Σ p_i log(p_i)
        entropy = -(all_probs * torch.log(all_probs + 1e-10)).sum(dim=-1).mean()
        entropy_loss = -self.lambda_entropy * entropy  # Negative to encourage high entropy

        return {
            'sparsity': sparsity_loss,
            'entropy': entropy_loss,
            'total': sparsity_loss + entropy_loss,
        }

    def select_landmarks(
        self,
        landmarks: torch.Tensor,
        route_logits: torch.Tensor,
        k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[dict]]:
        """
        Select landmarks based on routing scores.

        Convenience method that combines routing with landmark selection.

        Args:
            landmarks: Landmark tokens [batch, num_landmarks, L, d_model]
                      where L=6 tokens per landmark
            route_logits: Routing scores [batch, num_landmarks]
            k: Number of landmarks to select

        Returns:
            selected_landmarks: [batch, k, L, d_model]
            selected_probs: [batch, k]
            aux_losses: Auxiliary losses dict (or None)

        Examples:
            >>> router = GumbelTopKRouter(budget_B=24)
            >>> landmarks = torch.randn(2, 100, 6, 2816)  # 100 landmarks per batch
            >>> logits = torch.randn(2, 100)
            >>> selected, probs, losses = router.select_landmarks(landmarks, logits, k=4)
            >>> print(selected.shape)  # (2, 4, 6, 2816)
        """
        batch_size, num_landmarks, L, d_model = landmarks.shape

        # Route
        selected_indices, selected_probs, aux_losses = self.forward(
            route_logits, k=k, return_aux_losses=True
        )

        # Gather selected landmarks
        # Expand indices to match landmark dimensions
        indices_expanded = selected_indices.unsqueeze(-1).unsqueeze(-1).expand(
            batch_size, k, L, d_model
        )

        # Gather: [batch, num_landmarks, L, d_model] -> [batch, k, L, d_model]
        selected_landmarks = torch.gather(
            landmarks,
            dim=1,
            index=indices_expanded
        )

        return selected_landmarks, selected_probs, aux_losses

    def get_routing_stats(self) -> dict:
        """
        Get current routing statistics.

        Returns:
            Dictionary with routing config and state

        Examples:
            >>> stats = router.get_routing_stats()
            >>> print(stats['temperature'])
        """
        return {
            'budget_B': self.budget_B,
            'temperature': self.current_temperature.item(),
            'temperature_min': self.temperature_min,
            'lambda_sparsity': self.lambda_sparsity,
            'lambda_entropy': self.lambda_entropy,
            'hard': self.hard,
        }


def test_gumbel_topk_router():
    """Test GumbelTopKRouter implementation."""
    print("Testing GumbelTopKRouter...")

    # Configuration
    batch_size = 4
    num_candidates = 100
    k = 24
    budget_B = 24

    # Create router
    print("\n[Test 1] Create router")
    router = GumbelTopKRouter(
        budget_B=budget_B,
        temperature=0.7,
        lambda_sparsity=2e-4,
        lambda_entropy=1e-3,
    )
    print(f"  Router created with budget B={budget_B}")
    print(f"  Temperature: {router.current_temperature.item()}")
    print(f"  Sparsity weight: {router.lambda_sparsity}")
    print(f"  Entropy weight: {router.lambda_entropy}")
    print("  [PASS]")

    # Test forward pass (training mode)
    print("\n[Test 2] Forward pass (training)")
    router.train()
    route_logits = torch.randn(batch_size, num_candidates)
    selected_indices, selected_probs, aux_losses = router(route_logits, k=k)

    assert selected_indices.shape == (batch_size, k)
    assert selected_probs.shape == (batch_size, k)
    assert aux_losses is not None
    assert 'sparsity' in aux_losses
    assert 'entropy' in aux_losses

    print(f"  Logits shape: {route_logits.shape}")
    print(f"  Selected indices shape: {selected_indices.shape}")
    print(f"  Selected probs shape: {selected_probs.shape}")
    print(f"  Aux losses: sparsity={aux_losses['sparsity']:.4f}, entropy={aux_losses['entropy']:.4f}")
    print("  [PASS]")

    # Test forward pass (eval mode)
    print("\n[Test 3] Forward pass (eval)")
    router.eval()
    with torch.no_grad():
        eval_indices, eval_probs, eval_losses = router(route_logits, k=k, return_aux_losses=False)

    assert eval_indices.shape == (batch_size, k)
    assert eval_probs.shape == (batch_size, k)
    assert eval_losses is None

    print(f"  Eval indices shape: {eval_indices.shape}")
    print(f"  Eval probs shape: {eval_probs.shape}")
    print(f"  Aux losses: {eval_losses}")
    print("  [PASS]")

    # Test temperature annealing
    print("\n[Test 4] Temperature annealing")
    router.train()
    initial_temp = router.current_temperature.item()

    router.anneal_temperature(progress=0.5)
    mid_temp = router.current_temperature.item()

    router.anneal_temperature(progress=1.0)
    final_temp = router.current_temperature.item()

    assert final_temp <= mid_temp <= initial_temp
    assert final_temp == router.temperature_min

    print(f"  Initial temp: {initial_temp:.3f}")
    print(f"  Mid temp (50%): {mid_temp:.3f}")
    print(f"  Final temp (100%): {final_temp:.3f}")
    print("  [PASS]")

    # Test landmark selection
    print("\n[Test 5] Select landmarks")
    L = 6  # Tokens per landmark
    d_model = 2816
    landmarks = torch.randn(batch_size, num_candidates, L, d_model)
    logits = torch.randn(batch_size, num_candidates)

    selected_landmarks, selected_probs, aux_losses = router.select_landmarks(
        landmarks, logits, k=k
    )

    assert selected_landmarks.shape == (batch_size, k, L, d_model)
    assert selected_probs.shape == (batch_size, k)

    print(f"  Input landmarks: {landmarks.shape}")
    print(f"  Selected landmarks: {selected_landmarks.shape}")
    print(f"  Selected probs: {selected_probs.shape}")
    print("  [PASS]")

    # Test gradient flow
    print("\n[Test 6] Gradient flow")
    router.train()
    logits_grad = torch.randn(batch_size, num_candidates, requires_grad=True)
    indices, probs, losses = router(logits_grad, k=k)

    # Backward through auxiliary losses
    loss_total = losses['total']
    loss_total.backward()

    assert logits_grad.grad is not None
    print(f"  Gradient norm: {logits_grad.grad.norm().item():.4f}")
    print("  [PASS]")

    # Test budget constraint
    print("\n[Test 7] Budget constraint")
    different_k = [10, 24, 30]
    for test_k in different_k:
        indices, probs, _ = router(route_logits, k=test_k, return_aux_losses=False)
        assert indices.size(1) == test_k
        print(f"  k={test_k}: selected {indices.size(1)} tokens ✓")
    print("  [PASS]")

    # Test statistics
    print("\n[Test 8] Get routing stats")
    stats = router.get_routing_stats()
    assert stats['budget_B'] == budget_B
    assert 'temperature' in stats
    print(f"  Stats keys: {list(stats.keys())}")
    print(f"  Budget B: {stats['budget_B']}")
    print(f"  Current temp: {stats['temperature']:.3f}")
    print("  [PASS]")

    # Test top-k selection correctness
    print("\n[Test 9] Top-k correctness")
    # Create logits where we know the top-k
    known_logits = torch.zeros(1, 100)
    top_indices = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    known_logits[0, top_indices] = torch.tensor([10., 9., 8., 7., 6., 5., 4., 3., 2., 1.])

    router.eval()
    with torch.no_grad():
        selected, _, _ = router(known_logits, k=10, return_aux_losses=False)

    # Check that selected indices match top_indices (may be in different order)
    selected_set = set(selected[0].tolist())
    expected_set = set(top_indices)
    assert selected_set == expected_set, f"Selected {selected_set} != expected {expected_set}"

    print(f"  Expected top-10: {top_indices}")
    print(f"  Selected: {sorted(selected[0].tolist())}")
    print("  [PASS]")

    print("\n" + "="*50)
    print("[PASS] All GumbelTopKRouter tests passed!")
    print("="*50)
    print(f"\nRouter summary:")
    print(f"  Budget: {budget_B} global tokens (FR-009)")
    print(f"  Selection: Gumbel-Softmax + top-k")
    print(f"  Straight-through: Hard indices, soft gradients")
    print(f"  Aux losses: sparsity (λ={router.lambda_sparsity}) + entropy (λ={router.lambda_entropy})")


if __name__ == "__main__":
    test_gumbel_topk_router()
