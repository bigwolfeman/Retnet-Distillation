"""
Loss functions for sparse knowledge distillation.

Implements sparse-KL divergence with renormalization, temperature scaling,
and numerically stable computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SparseKLLoss(nn.Module):
    """
    Sparse KL divergence loss for knowledge distillation.

    Computes KL divergence over k+1 items (top-k + "other" bucket) instead of
    full vocabulary. This implementation uses NO DENSIFICATION - all operations
    are performed on sparse tensors of shape [B, L, K] or [B, L, K+1].

    Mathematical formulation:
    Let K = top-k indices, V = full vocab, C = V \\ K (complement)
    Let P = student distribution, Q = teacher distribution

    KL(P||Q) = Σ(i∈K) P_i(log P_i - log Q_i) + P_other(log P_other - log Q_other)

    where:
    - P_other = Σ(j∈C) P_j (sum over non-top-k, computed via log-subtract trick)
    - Q_other = Σ(j∈C) Q_j (passed from server as teacher_other_mass)

    Implementation steps:
    1. Gather student logits at top-k indices (NO full vocab tensor created)
    2. Compute P_other using: log(sum_all - sum_topk) with log1p for stability
    3. Build sparse distributions [top-k logits, other logit] with shape [B, L, K+1]
    4. Apply softmax and compute KL divergence
    5. Mix with hard CE loss if targets provided

    Memory efficiency:
    - NO tensors of shape [B, L, V] created (except input student_logits)
    - All intermediate tensors are [B, L, K] or [B, L, K+1] or [B, L, 1]
    - For V=128k, K=128: ~1000x memory reduction for intermediate computations

    Args:
        temperature: Temperature for softening logits (default: 2.0)
        alpha: Mixing coefficient for hard vs soft targets (default: 0.2)
               alpha=0.2 means 20% hard CE + 80% soft KL
        epsilon: Small constant for numerical stability (default: 1e-8)
                 Applied INSIDE log to prevent probability sum violation

    Example:
        >>> loss_fn = SparseKLLoss(temperature=2.0, alpha=0.2)
        >>> student_logits = torch.randn(2, 4096, 128256)  # [B, L, V]
        >>> teacher_topk_indices = torch.randint(0, 128256, (2, 4096, 128))  # [B, L, K]
        >>> teacher_topk_logprobs = torch.randn(2, 4096, 128)  # [B, L, K] (log-probs, fp32 or dequantized)
        >>> teacher_other_logprob = torch.randn(2, 4096, 1)  # [B, L, 1] (log-probability, not linear prob!)
        >>> loss = loss_fn(student_logits, teacher_topk_indices, teacher_topk_logprobs, teacher_other_logprob)
    """

    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.2,
        epsilon: float = 1e-8,
        reverse_kl: bool = False
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.epsilon = epsilon
        self.reverse_kl = reverse_kl

        # Validate parameters
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        if not 0 <= alpha <= 1:
            raise ValueError(f"Alpha must be in [0, 1], got {alpha}")
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_topk_indices: torch.Tensor,
        teacher_topk_logprobs: torch.Tensor,
        teacher_other_logprob: torch.Tensor,
        hard_targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute sparse-KL distillation loss.

        Args:
            student_logits: Student model logits [B, L, V] where V is vocab size
            teacher_topk_indices: Top-k token indices from teacher [B, L, K]
            teacher_topk_logprobs: Top-k log-probabilities from teacher [B, L, K]
            teacher_other_logprob: Log-probability for tail (logsumexp of non-top-k) [B, L, 1]
            hard_targets: Optional ground truth labels [B, L] for CE loss mixing
            attention_mask: Optional mask for padding positions [B, L], 1 = valid, 0 = padding

        Returns:
            Scalar loss value
        """
        # Validate input shapes
        B, L, V = student_logits.shape
        if teacher_topk_indices.shape[:2] != (B, L):
            raise ValueError(f"teacher_topk_indices shape {teacher_topk_indices.shape} doesn't match student_logits batch/seq {(B, L)}")
        if teacher_topk_logprobs.shape != teacher_topk_indices.shape:
            raise ValueError(f"teacher_topk_logprobs shape {teacher_topk_logprobs.shape} != teacher_topk_indices shape {teacher_topk_indices.shape}")
        if teacher_other_logprob.shape != (B, L, 1):
            raise ValueError(f"teacher_other_logprob shape {teacher_other_logprob.shape} != expected {(B, L, 1)}")

        K = teacher_topk_indices.shape[2]

        # Step 1: Build teacher sparse log-probabilities [B, L, K+1]
        # Teacher already sent us log-probabilities (not logits), so we just concatenate
        # Note: Temperature scaling is NOT applied here because inputs are already log-probs
        # Temperature scaling should only be applied to student logits

        # Concatenate top-k log-probs with other_logprob: [B, L, K] + [B, L, 1] -> [B, L, K+1]
        teacher_logprobs_sparse = torch.cat([teacher_topk_logprobs, teacher_other_logprob], dim=-1)

        # Step 2: Gather student logits at top-k indices (NO DENSIFICATION)
        # Apply temperature scaling to student logits BEFORE gathering
        student_logits_scaled = student_logits / self.temperature

        # Gather student logits at top-k indices
        # teacher_topk_indices: [B, L, K], student_logits_scaled: [B, L, V]
        student_topk_logits = torch.gather(
            student_logits_scaled,
            dim=-1,
            index=teacher_topk_indices
        )  # [B, L, K]

        # Step 3: Compute student P_other using log-subtract trick (NO DENSIFICATION)
        # P_other = sum of all probs NOT in top-k
        # Use: log(sum_all - sum_topk) = log(exp(lse_all) - exp(lse_topk))
        #    = lse_all + log(1 - exp(lse_topk - lse_all))

        total_lse = torch.logsumexp(student_logits_scaled, dim=-1, keepdim=True)  # [B, L, 1]
        topk_lse = torch.logsumexp(student_topk_logits, dim=-1, keepdim=True)  # [B, L, 1]

        # Numerically stable log-subtract
        # Handle edge case where K=V (all vocab in top-k): topk_lse ≈ total_lse
        # In this case, exp(topk_lse - total_lse) ≈ 1, so log1p(-1) -> -inf
        # We clamp to prevent this edge case
        ratio = torch.clamp(topk_lse - total_lse, max=-1e-7)  # Ensure exp(ratio) < 1
        student_other_logit = total_lse + torch.log1p(-torch.exp(ratio))  # [B, L, 1]

        # Step 4: Build student sparse logits [B, L, K+1]
        student_logits_sparse = torch.cat([student_topk_logits, student_other_logit], dim=-1)

        # Step 5: Compute probabilities over sparse space (K+1 items)
        # Teacher inputs are already log-probabilities, so convert to probabilities
        p_teacher = torch.exp(teacher_logprobs_sparse)  # [B, L, K+1]
        p_student = F.softmax(student_logits_sparse, dim=-1)  # [B, L, K+1]

        # Step 6: Compute KL divergence with epsilon INSIDE log (not on probabilities)
        # This prevents probability sum violation from clamping
        #
        # Forward KL (default): KL(teacher || student) = sum(teacher * log(teacher / student))
        #   - Mode-seeking: student focuses on high-probability regions of teacher
        #   - Avoids placing mass where teacher has none (safe, conservative)
        #
        # Reverse KL: KL(student || teacher) = sum(student * log(student / teacher))
        #   - Mean-seeking: student tries to cover all modes of teacher
        #   - Can place mass where teacher is weak (risky, exploratory)
        if self.reverse_kl:
            # Reverse KL: KL(student || teacher)
            kl_loss = torch.sum(
                p_student * (torch.log(p_student + self.epsilon) - torch.log(p_teacher + self.epsilon)),
                dim=-1
            )  # [B, L]
        else:
            # Forward KL: KL(teacher || student) - default behavior
            kl_loss = torch.sum(
                p_teacher * (torch.log(p_teacher + self.epsilon) - torch.log(p_student + self.epsilon)),
                dim=-1
            )  # [B, L]

        # Average over batch and sequence (with optional masking)
        # FIX: Mask padding positions to avoid gradient noise
        if attention_mask is not None:
            # attention_mask: [B, L], 1 = valid, 0 = padding
            # Zero out loss at padding positions
            kl_loss = kl_loss * attention_mask
            # Compute mean over non-padding positions
            kl_loss = kl_loss.sum() / attention_mask.sum().clamp(min=1.0)
        else:
            kl_loss = kl_loss.mean()

        # Step 7: Mix with hard CE loss if targets provided
        if hard_targets is not None and self.alpha > 0:
            # Compute standard cross-entropy on hard targets
            # Use original (unscaled) student logits for CE
            if attention_mask is not None:
                # Compute per-position CE loss
                ce_loss_per_pos = F.cross_entropy(
                    student_logits.reshape(-1, V),
                    hard_targets.reshape(-1),
                    reduction='none'
                ).reshape(B, L)  # [B, L]
                # Apply mask and compute mean over non-padding positions
                ce_loss = (ce_loss_per_pos * attention_mask).sum() / attention_mask.sum().clamp(min=1.0)
            else:
                ce_loss = F.cross_entropy(
                    student_logits.reshape(-1, V),
                    hard_targets.reshape(-1),
                    reduction='mean'
                )

            # Combine: alpha * CE + (1-alpha) * KL
            # Note: Scale KL by temperature^2 to match gradient magnitude
            # (standard practice in knowledge distillation)
            total_loss = self.alpha * ce_loss + (1 - self.alpha) * kl_loss * (self.temperature ** 2)
        else:
            # Pure KL loss (scale by temperature^2)
            total_loss = kl_loss * (self.temperature ** 2)

        return total_loss

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f'temperature={self.temperature}, alpha={self.alpha}, epsilon={self.epsilon}, reverse_kl={self.reverse_kl}'


def sparse_kl_loss(
    student_logits: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    teacher_other_logprob: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.2,
    epsilon: float = 1e-8,
    hard_targets: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Functional interface for sparse-KL loss.

    See SparseKLLoss class for detailed documentation.

    Args:
        student_logits: Student model logits [B, L, V]
        teacher_topk_indices: Top-k token indices from teacher [B, L, K]
        teacher_topk_logprobs: Top-k log-probabilities from teacher [B, L, K]
        teacher_other_logprob: Log-probability for tail (logsumexp of non-top-k) [B, L, 1]
        temperature: Temperature for softening logits (default: 2.0)
        alpha: Mixing coefficient for hard vs soft targets (default: 0.2)
        epsilon: Small constant for numerical stability (default: 1e-8)
        hard_targets: Optional ground truth labels [B, L] for CE loss mixing

    Returns:
        Scalar loss value
    """
    loss_fn = SparseKLLoss(temperature=temperature, alpha=alpha, epsilon=epsilon)
    return loss_fn(student_logits, teacher_topk_indices, teacher_topk_logprobs, teacher_other_logprob, hard_targets)
