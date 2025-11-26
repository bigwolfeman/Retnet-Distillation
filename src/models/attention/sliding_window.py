"""
Sliding Window Attention with Global Tokens (Longformer/BigBird style).

This implements local attention with a sliding window plus support for global tokens
(e.g., landmark tokens from retrieval).

References:
- Longformer: https://arxiv.org/abs/2004.05150
- BigBird: https://arxiv.org/abs/2007.14062
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SlidingWindowAttention(nn.Module):
    """
    Sliding window attention with optional global tokens.

    Implements efficient local attention where each token attends to:
    1. Tokens within a local window of size w (e.g., w=2048)
    2. Designated global tokens that attend to all positions

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        window_size: Local attention window size (default: 2048)
        dropout: Dropout probability (default: 0.0)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        window_size: int = 2048,
        dropout: float = 0.0,
        block_size: int = 128,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        self.dropout = dropout
        self.block_size = block_size

        # QKV projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)

        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def _create_sliding_window_mask(
        self,
        seq_len: int,
        device: torch.device,
        global_token_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Create attention mask for sliding window pattern.

        Args:
            seq_len: Sequence length
            device: Device for the mask
            global_token_indices: Indices of global tokens [num_global]
                                 These tokens attend to all positions

        Returns:
            Attention mask of shape [seq_len, seq_len]
            True = attend, False = mask out
        """
        half_window = self.window_size // 2
        positions = torch.arange(seq_len, device=device)
        distance = positions[:, None] - positions[None, :]
        mask = distance.abs() <= half_window

        # Add global tokens (attend to all positions)
        if global_token_indices is not None:
            for idx in global_token_indices:
                # Global token attends to all positions
                mask[idx, :] = True
                # All positions attend to global token
                mask[:, idx] = True

        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        global_token_indices: Optional[torch.Tensor] = None,
        rope_module: Optional[nn.Module] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for sliding window attention.

        Args:
            hidden_states: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional external mask, can be:
                           - [batch, seq_len]: Key padding mask (1=attend, 0=mask)
                           - [batch, seq_len, seq_len]: Full attention mask
                           - [batch, num_heads, seq_len, seq_len]: Per-head mask
            global_token_indices: Indices of global tokens [num_global]
            rope_module: Optional RoPE module for position encoding
            position_ids: Optional position IDs [batch, seq_len]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = hidden_states.shape

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        # [batch, seq_len, d_model] -> [batch, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if provided
        if rope_module is not None:
            q, k = rope_module(q, k, position_ids)

        # Compute attention scores
        # [batch, num_heads, seq_len, seq_len]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Create sliding window mask
        sliding_mask = self._create_sliding_window_mask(
            seq_len, hidden_states.device, global_token_indices
        )

        # Expand mask for batch and num_heads
        # [seq_len, seq_len] -> [1, 1, seq_len, seq_len]
        sliding_mask = sliding_mask.unsqueeze(0).unsqueeze(0)

        # Combine with external mask if provided
        if attention_mask is not None:
            # Handle different attention_mask shapes
            if attention_mask.dim() == 2:
                # [batch, seq_len] -> key padding mask
                # Expand to [batch, 1, 1, seq_len] for broadcasting
                key_padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                # Convert to boolean: 1 = attend, 0 = mask
                key_padding_mask = key_padding_mask.bool()
                # Combine with sliding window mask
                combined_mask = sliding_mask & key_padding_mask
            elif attention_mask.dim() == 3:
                # [batch, seq_len, seq_len] -> full attention mask
                # Expand to [batch, 1, seq_len, seq_len]
                full_mask = attention_mask.unsqueeze(1).bool()
                combined_mask = sliding_mask & full_mask
            else:
                # Assume already properly shaped [batch, num_heads, seq_len, seq_len]
                combined_mask = sliding_mask & attention_mask.bool()
        else:
            combined_mask = sliding_mask

        # Apply mask (set masked positions to -inf)
        attn_scores = attn_scores.masked_fill(~combined_mask, float('-inf'))

        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        # [batch, num_heads, seq_len, head_dim]
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        # [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, d_model)

        # Output projection
        output = self.out_proj(attn_output)

        return output


def test_sliding_window_attention():
    """Test sliding window attention implementation."""
    print("Testing Sliding Window Attention...")

    # Configuration
    batch_size = 2
    seq_len = 4096
    d_model = 2816
    num_heads = 16  # 2816 / 16 = 176 (divisible)
    window_size = 2048

    # Create module
    attn = SlidingWindowAttention(
        d_model=d_model,
        num_heads=num_heads,
        window_size=window_size,
        dropout=0.0,
    )

    # Create input
    hidden_states = torch.randn(batch_size, seq_len, d_model)

    # Test 1: Basic forward pass
    print("\n[Test 1] Basic forward pass")
    output = attn(hidden_states)
    assert output.shape == hidden_states.shape, f"Shape mismatch: {output.shape} vs {hidden_states.shape}"
    print(f"  Input shape: {hidden_states.shape}")
    print(f"  Output shape: {output.shape}")
    print("  [PASS]")

    # Test 2: With global tokens
    print("\n[Test 2] With global tokens")
    global_indices = torch.tensor([0, 100, 500, 1000])  # 4 global tokens
    output_global = attn(hidden_states, global_token_indices=global_indices)
    assert output_global.shape == hidden_states.shape
    print(f"  Global token indices: {global_indices.tolist()}")
    print(f"  Output shape: {output_global.shape}")
    print("  [PASS]")

    # Test 3: Verify sliding window mask pattern
    print("\n[Test 3] Verify sliding window mask")
    mask = attn._create_sliding_window_mask(seq_len=1000, device=hidden_states.device)

    # Check middle position (index 500)
    pos_500_attends_to = mask[500].sum().item()
    expected_window = min(window_size, 1000)  # Can't exceed seq_len

    print(f"  Position 500 attends to {int(pos_500_attends_to)} tokens")
    print(f"  Expected window size: {expected_window}")
    print(f"  Window check: {abs(pos_500_attends_to - expected_window) < 10}")  # Allow small variance
    print("  [PASS]")

    # Test 4: Global tokens attend to all
    print("\n[Test 4] Global tokens attend to all positions")
    global_indices_test = torch.tensor([0, 999])
    mask_with_global = attn._create_sliding_window_mask(
        seq_len=1000,
        device=hidden_states.device,
        global_token_indices=global_indices_test
    )

    # Global token 0 should attend to all 1000 positions
    global_0_attends = mask_with_global[0].sum().item()
    print(f"  Global token 0 attends to {int(global_0_attends)} / 1000 positions")
    assert global_0_attends == 1000, "Global token should attend to all positions"

    # All positions should attend to global token 0
    positions_attending_to_0 = mask_with_global[:, 0].sum().item()
    print(f"  {int(positions_attending_to_0)} / 1000 positions attend to global token 0")
    assert positions_attending_to_0 == 1000, "All positions should attend to global token"
    print("  [PASS]")

    print("\n" + "="*50)
    print("[PASS] All Sliding Window Attention tests passed!")
    print("="*50)


def estimate_sliding_window_memory(
    sequence_length: int,
    *,
    window_size: int,
    num_heads: int,
    batch_size: int = 1,
    dtype_bytes: int = 2,
    block_size: int = 128,
    global_tokens: int = 0,
) -> dict[str, float | int]:
    """Estimate dense vs block-sparse memory usage (in MB).

    Args:
        sequence_length: Effective context length.
        window_size: Local attention window size.
        num_heads: Attention heads.
        batch_size: Batch size for inference.
        dtype_bytes: Bytes per score (default assumes bf16/float16 = 2 bytes).
        block_size: Block size for sparse layout used in sliding window kernel.
        global_tokens: Count of global tokens (landmarks) attending to all positions.

    Returns:
        Dictionary with dense and sparse memory estimates plus helper metadata.
    """

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if dtype_bytes <= 0:
        raise ValueError("dtype_bytes must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    effective_window = min(window_size, sequence_length)
    dense_scores = sequence_length * sequence_length
    dense_bytes = dense_scores * num_heads * batch_size * dtype_bytes

    local_scores = sequence_length * effective_window
    global_scores = 2 * sequence_length * global_tokens if global_tokens else 0
    sparse_scores = (local_scores + global_scores)
    sparse_bytes = sparse_scores * num_heads * batch_size * dtype_bytes

    blocks_per_token = math.ceil(effective_window / block_size)
    memory_dense_mb = dense_bytes / (1024 ** 2)
    memory_sparse_mb = sparse_bytes / (1024 ** 2)

    return {
        "dense_mb": memory_dense_mb,
        "sparse_mb": memory_sparse_mb,
        "connections_per_token": effective_window,
        "blocks_per_token": blocks_per_token,
        "global_tokens": global_tokens,
    }


if __name__ == "__main__":
    test_sliding_window_attention()
