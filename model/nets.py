"""
Transformer decoder architecture per plan.md and research.md.

Implements 12-28 layer decoder-only transformer (50-150M params) with
support for auxiliary prediction heads via stop-gradient.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoPEEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) per research.md decision.

    Applies rotation to Q and K matrices for position-aware attention.
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        """
        Args:
            dim: Head dimension (d_model // n_heads)
            max_seq_len: Maximum sequence length
            base: Base for frequency calculation
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Cache for sin/cos values
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update cached sin/cos values if needed."""
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.outer(t, self.inv_freq.to(dtype))
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()[None, :, None, :]
            self._sin_cached = emb.sin()[None, :, None, :]

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query and key tensors.

        Args:
            q: Query tensor [batch, seq_len, n_heads, head_dim]
            k: Key tensor [batch, seq_len, n_heads, head_dim]

        Returns:
            Rotated (q, k) tensors
        """
        batch, seq_len, n_heads, head_dim = q.shape
        self._update_cache(seq_len, q.device, q.dtype)

        # Get cached sin/cos
        cos = self._cos_cached[:, :seq_len, :, :]
        sin = self._sin_cached[:, :seq_len, :, :]

        # Apply rotation
        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin

        return q_rot, k_rot


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with optional RoPE.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_rope: bool = True,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_rope = use_rope

        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # RoPE embedding
        if use_rope:
            self.rope = RoPEEmbedding(self.head_dim, max_seq_len)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with causal masking.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask [batch, seq_len, seq_len]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Apply RoPE if enabled
        if self.use_rope:
            q, k = self.rope(q, k)

        # Transpose for attention: [batch, n_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask
        if mask is None:
            # Create causal mask: [seq_len, seq_len]
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        else:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project: [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(attn_output)

        return output


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network with GELU activation.
    """

    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerDecoderLayer(nn.Module):
    """
    Single transformer decoder layer with pre-norm.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_rope: bool = True,
        max_seq_len: int = 4096,
    ):
        super().__init__()

        # Layer normalization (pre-norm)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Attention and FFN
        self.attention = MultiHeadAttention(
            d_model, n_heads, attention_dropout, use_rope, max_seq_len
        )
        self.ffn = FeedForward(d_model, ffn_dim, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with residual connections.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional attention mask

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Attention block (pre-norm)
        residual = x
        x = self.norm1(x)
        x = self.attention(x, mask)
        x = self.dropout(x)
        x = residual + x

        # FFN block (pre-norm)
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x

        return x


class TransformerDecoder(nn.Module):
    """
    Decoder-only transformer per plan.md.

    12-28 layer architecture with 50-150M parameters.
    Baseline model without auxiliary heads.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 12,
        n_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_rope: bool = True,
        max_seq_len: int = 4096,
    ):
        """
        Initialize transformer decoder.

        Args:
            vocab_size: Vocabulary size (~49180 for StarCoder2 + special tokens)
            d_model: Hidden dimension
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            ffn_mult: FFN dimension multiplier
            dropout: Dropout rate
            attention_dropout: Attention dropout rate
            use_rope: Whether to use RoPE
            max_seq_len: Maximum sequence length
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        ffn_dim = d_model * ffn_mult

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model, n_heads, ffn_dim, dropout, attention_dropout, use_rope, max_seq_len
            )
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

        # LM head (tied with token embedding)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # Weight tying

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights per standard transformer initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through transformer.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            mask: Optional attention mask
            return_hidden: If True, return (logits, hidden_states)

        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
            hidden_states (optional): Final hidden states [batch, seq_len, d_model]
        """
        # Embed tokens
        x = self.token_embedding(input_ids)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        # Final layer norm
        hidden = self.norm(x)

        # LM head
        logits = self.lm_head(hidden)

        if return_hidden:
            return logits, hidden
        return logits

    def estimate_params(self) -> int:
        """
        Estimate total model parameters.

        Returns:
            Parameter count
        """
        return sum(p.numel() for p in self.parameters())


@dataclass
class TransformerOutput:
    """Output from TransformerWithAuxHeads."""
    logits: torch.Tensor  # [batch, seq, vocab]
    hidden: torch.Tensor  # [batch, seq, d_model]
    aux_outputs: dict  # Auxiliary head outputs


class TransformerWithAuxHeads(nn.Module):
    """
    Transformer decoder with auxiliary prediction heads per research.md.

    Attaches 6 auxiliary heads with stop-gradient (detach()) to prevent
    gradient flow to backbone during auxiliary loss computation.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 12,
        n_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_rope: bool = True,
        max_seq_len: int = 4096,
    ):
        """
        Initialize transformer with auxiliary heads.

        Args:
            Same as TransformerDecoder
        """
        super().__init__()

        # Base transformer
        self.backbone = TransformerDecoder(
            vocab_size, d_model, n_layers, n_heads, ffn_mult,
            dropout, attention_dropout, use_rope, max_seq_len
        )

        # Import auxiliary heads (defined in heads.py)
        from .heads import (
            CarryHead, Mult2DHead, DivisionPolicyHead,
            FormatHead, SchemaHead, SelectorHead
        )

        # Auxiliary heads (stop-gradient via detach())
        self.carry_head = CarryHead(d_model)
        self.mult_2d_head = Mult2DHead(d_model)
        self.division_policy_head = DivisionPolicyHead(d_model)
        self.format_head = FormatHead(d_model)
        self.schema_head = SchemaHead(d_model)
        self.selector_head = SelectorHead(d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> TransformerOutput:
        """
        Forward pass with auxiliary heads.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            mask: Optional attention mask

        Returns:
            TransformerOutput with logits, hidden states, and auxiliary outputs
        """
        # Main LM forward pass
        logits, hidden = self.backbone(input_ids, mask, return_hidden=True)

        # Auxiliary heads (stop gradient into backbone)
        hidden_detached = hidden.detach()

        aux_outputs = {
            'carry': self.carry_head(hidden_detached),
            'mult_2d': self.mult_2d_head(hidden_detached),
            'division_policy': self.division_policy_head(hidden_detached),
            'format': self.format_head(hidden_detached),
            'schema': self.schema_head(hidden_detached),
            'selector': self.selector_head(hidden_detached),
        }

        return TransformerOutput(
            logits=logits,
            hidden=hidden,
            aux_outputs=aux_outputs
        )


# Example usage and testing
if __name__ == "__main__":
    print("=== Testing Transformer Architecture ===\n")

    # Test configuration
    vocab_size = 49180
    batch_size = 4
    seq_len = 128
    d_model = 512
    n_layers = 12

    # Create model
    print("Creating TransformerDecoder...")
    model = TransformerDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=8,
        ffn_mult=4,
    )

    print(f"Model parameters: {model.estimate_params() / 1e6:.1f}M")

    # Test forward pass
    print(f"\nTesting forward pass with batch_size={batch_size}, seq_len={seq_len}...")
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits = model(input_ids)

    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected: ({batch_size}, {seq_len}, {vocab_size})")
    assert logits.shape == (batch_size, seq_len, vocab_size), "Shape mismatch!"

    print("\n✓ TransformerDecoder tests passed")

    # Test with auxiliary heads (if heads.py exists)
    try:
        print("\nCreating TransformerWithAuxHeads...")
        model_aux = TransformerWithAuxHeads(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
        )

        with torch.no_grad():
            output = model_aux(input_ids)

        print(f"Main logits shape: {output.logits.shape}")
        print(f"Hidden shape: {output.hidden.shape}")
        print(f"Auxiliary outputs: {list(output.aux_outputs.keys())}")

        print("\n✓ TransformerWithAuxHeads tests passed")
    except ImportError:
        print("\n⚠ heads.py not found, skipping auxiliary head tests")
