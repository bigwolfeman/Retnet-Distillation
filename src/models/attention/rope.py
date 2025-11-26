"""
Rotary Position Embedding (RoPE) for the attention band.

RoPE is used ONLY in the thin attention band, not in RetNet layers
(RetNet uses retention-based positional encoding).

References:
- RoFormer: https://arxiv.org/abs/2104.09864
- Implementation adapted from Hugging Face transformers
"""

import torch
import torch.nn as nn
from typing import Tuple


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation.

    Precomputes sin/cos tables for efficiency and applies rotary embeddings
    to query and key tensors.

    Args:
        dim: Dimension per attention head
        max_seq_len: Maximum sequence length (default: 128000 for stretch goal)
        base: Base for frequency computation (default: 10000)
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 128000,
        base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequency table
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute sin/cos tables for max sequence length
        self._precompute_freqs_cis(max_seq_len)

    def _precompute_freqs_cis(self, seq_len: int):
        """
        Precompute complex exponentials (cos + i*sin) for all positions.

        Args:
            seq_len: Sequence length to precompute
        """
        # Create position indices
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)

        # Compute frequencies: [seq_len, dim/2]
        freqs = torch.outer(t, self.inv_freq)

        # Compute cos and sin
        # Shape: [seq_len, dim/2]
        freqs_cos = torch.cos(freqs)
        freqs_sin = torch.sin(freqs)

        # Store as buffers (non-trainable)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate half the hidden dims of the input.

        This is the core RoPE rotation operation:
        [x1, x2, x3, x4] -> [-x2, x1, -x4, x3]

        Args:
            x: Input tensor of shape [..., dim]

        Returns:
            Rotated tensor of same shape
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embeddings to queries and keys.

        Args:
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            k: Key tensor [batch, num_heads, seq_len, head_dim]
            position_ids: Optional position indices [batch, seq_len]
                         If None, assumes positions 0, 1, 2, ..., seq_len-1

        Returns:
            (q_rotated, k_rotated): Rotated query and key tensors
        """
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Get position IDs
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=q.device).unsqueeze(0)

        # Ensure we have precomputed frequencies for this sequence length
        if seq_len > self.freqs_cos.shape[0]:
            self._precompute_freqs_cis(seq_len)

        # Get cos and sin for these positions
        # Shape: [batch, seq_len, dim/2]
        cos = self.freqs_cos[position_ids]
        sin = self.freqs_sin[position_ids]

        # Expand for num_heads dimension
        # Shape: [batch, 1, seq_len, head_dim/2]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        # Repeat for full head_dim (since we split in half)
        # Shape: [batch, 1, seq_len, head_dim]
        cos = cos.repeat(1, 1, 1, 2)
        sin = sin.repeat(1, 1, 1, 2)

        # Apply rotation to queries and keys
        # RoPE formula: x_rotated = x * cos + rotate_half(x) * sin
        q_rotated = q * cos + self._rotate_half(q) * sin
        k_rotated = k * cos + self._rotate_half(k) * sin

        return q_rotated, k_rotated

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass - applies rotary embeddings to Q and K.

        Args:
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            k: Key tensor [batch, num_heads, seq_len, head_dim]
            position_ids: Optional position indices [batch, seq_len]

        Returns:
            (q_rotated, k_rotated): Rotated query and key tensors
        """
        return self.apply_rotary_emb(q, k, position_ids)


def test_rope():
    """Test RoPE implementation."""
    print("Testing RoPE...")

    # Configuration
    batch_size = 2
    num_heads = 12
    seq_len = 1024
    head_dim = 64

    # Create RoPE module
    rope = RotaryPositionEmbedding(dim=head_dim, max_seq_len=2048)

    # Create dummy Q and K
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Apply RoPE
    q_rot, k_rot = rope(q, k)

    # Check shapes
    assert q_rot.shape == q.shape, f"Q shape mismatch: {q_rot.shape} vs {q.shape}"
    assert k_rot.shape == k.shape, f"K shape mismatch: {k_rot.shape} vs {k.shape}"

    # Check that rotation actually changed the tensors
    assert not torch.allclose(q, q_rot), "Q should be modified"
    assert not torch.allclose(k, k_rot), "K should be modified"

    # Check with custom position IDs
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    q_rot2, k_rot2 = rope(q, k, position_ids)

    # Should be same as default (sequential positions)
    assert torch.allclose(q_rot, q_rot2), "Custom position IDs should match default"

    print("[PASS] RoPE tests passed!")
    print(f"  Input shape: {q.shape}")
    print(f"  Output shape: {q_rot.shape}")
    print(f"  Max sequence length: {rope.max_seq_len}")
    print(f"  Dimension: {rope.dim}")


if __name__ == "__main__":
    test_rope()
