"""Dual Encoder for code embedding (T046 - US4).

Small Transformer encoder for embedding code chunks. Trained with contrastive
learning (in-batch negatives) to produce 768-dim embeddings for retrieval.

Architecture from research.md: 6 layers, 768 dim, 12 heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class DualEncoder(nn.Module):
    """
    Dual encoder for code and query embedding.

    A small Transformer encoder (6 layers, 768 dim) trained with contrastive
    learning to embed code chunks and queries into a shared semantic space.
    Uses inner product similarity for fast retrieval.

    Args:
        vocab_size: Vocabulary size (same as main model)
        d_model: Embedding dimension (default: 768)
        n_layers: Number of Transformer layers (default: 6)
        n_heads: Number of attention heads (default: 12)
        d_ff: Feed-forward dimension (default: 3072 = 4 * d_model)
        dropout: Dropout probability (default: 0.1)
        max_seq_len: Maximum sequence length (default: 512 for chunks)

    Examples:
        >>> encoder = DualEncoder(vocab_size=50000)
        >>> input_ids = torch.randint(0, 50000, (2, 128))
        >>> embeddings = encoder(input_ids)
        >>> print(embeddings.shape)  # (2, 768)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 6,
        n_heads: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Position embeddings (learned)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection to normalize embeddings
        self.output_norm = nn.LayerNorm(d_model)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following best practices."""
        # Token embeddings: normal distribution
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

        # Position embeddings: normal distribution
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden: bool = False,
    ) -> torch.Tensor:
        """
        Encode input tokens to embedding vector.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len] (1=attend, 0=mask)
            return_all_hidden: If True, return all token embeddings; else return pooled

        Returns:
            embeddings: [batch, d_model] pooled embeddings (or [batch, seq_len, d_model] if return_all_hidden)

        Examples:
            >>> encoder = DualEncoder(vocab_size=50000)
            >>> input_ids = torch.randint(0, 50000, (4, 128))
            >>> mask = torch.ones(4, 128)
            >>> embeddings = encoder(input_ids, attention_mask=mask)
            >>> print(embeddings.shape)  # (4, 768)
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        token_embeds = self.token_embedding(input_ids)  # [batch, seq_len, d_model]

        # Position embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeds = self.position_embedding(positions)  # [1, seq_len, d_model]

        # Combined embeddings
        hidden_states = token_embeds + position_embeds  # [batch, seq_len, d_model]

        # Create attention mask for Transformer
        # PyTorch Transformer expects: True=mask, False=attend (opposite of BERT)
        if attention_mask is not None:
            # Convert from 1=attend, 0=mask to True=mask, False=attend
            transformer_mask = (attention_mask == 0)
        else:
            transformer_mask = None

        # Transformer encoding
        hidden_states = self.transformer(
            hidden_states,
            src_key_padding_mask=transformer_mask
        )  # [batch, seq_len, d_model]

        if return_all_hidden:
            # Return all token embeddings
            return self.output_norm(hidden_states)

        # Mean pooling over sequence (excluding padding)
        if attention_mask is not None:
            # Masked mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            # Simple mean pooling
            pooled = hidden_states.mean(dim=1)

        # Normalize output
        pooled = self.output_norm(pooled)

        # L2 normalization for inner product similarity
        pooled = F.normalize(pooled, p=2, dim=-1)

        return pooled  # [batch, d_model]

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for dual encoder training.

    Uses in-batch negatives: For each query-code pair in the batch,
    all other codes in the batch are treated as negatives.

    Temperature-scaled softmax with InfoNCE objective.

    Args:
        temperature: Softmax temperature (default: 0.07)

    Examples:
        >>> loss_fn = ContrastiveLoss(temperature=0.07)
        >>> query_embeds = torch.randn(8, 768)
        >>> code_embeds = torch.randn(8, 768)
        >>> loss = loss_fn(query_embeds, code_embeds)
        >>> print(loss.item())  # Scalar loss value
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query_embeddings: torch.Tensor,
        code_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            query_embeddings: [batch, d_model] - normalized query embeddings
            code_embeddings: [batch, d_model] - normalized code embeddings

        Returns:
            loss: Scalar contrastive loss

        Notes:
            - Assumes embeddings are already L2-normalized
            - Uses in-batch negatives (all codes in batch except the positive)
            - Loss is symmetric: query-to-code + code-to-query
        """
        batch_size = query_embeddings.size(0)

        # Compute similarity matrix [batch, batch]
        # Since embeddings are normalized, dot product = cosine similarity
        logits = torch.matmul(query_embeddings, code_embeddings.t()) / self.temperature

        # Labels: positive is on the diagonal
        labels = torch.arange(batch_size, device=logits.device)

        # Cross-entropy loss (query-to-code direction)
        loss_q2c = F.cross_entropy(logits, labels)

        # Symmetric loss (code-to-query direction)
        loss_c2q = F.cross_entropy(logits.t(), labels)

        # Average both directions
        loss = (loss_q2c + loss_c2q) / 2

        return loss


def test_dual_encoder():
    """Test DualEncoder implementation."""
    print("Testing DualEncoder...")

    # Configuration
    vocab_size = 50000
    batch_size = 4
    seq_len = 128
    d_model = 768

    # Create encoder
    print("\n[Test 1] Create encoder")
    encoder = DualEncoder(vocab_size=vocab_size, d_model=d_model)
    num_params = encoder.get_num_params()
    print(f"  Encoder created with {num_params/1e6:.2f}M parameters")
    print(f"  Architecture: {encoder.n_layers} layers, {encoder.n_heads} heads, {d_model} dim")
    print("  [PASS]")

    # Test forward pass
    print("\n[Test 2] Forward pass")
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    embeddings = encoder(input_ids)
    assert embeddings.shape == (batch_size, d_model), f"Shape mismatch: {embeddings.shape}"
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {embeddings.shape}")
    print("  [PASS]")

    # Test normalization
    print("\n[Test 3] L2 normalization")
    norms = torch.norm(embeddings, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Embeddings not normalized"
    print(f"  Embedding norms: {norms.tolist()}")
    print("  All norms ≈ 1.0 ✓")
    print("  [PASS]")

    # Test with attention mask
    print("\n[Test 4] With attention mask")
    attention_mask = torch.ones(batch_size, seq_len)
    # Mask out last 32 tokens
    attention_mask[:, -32:] = 0
    embeddings_masked = encoder(input_ids, attention_mask=attention_mask)
    assert embeddings_masked.shape == (batch_size, d_model)
    print(f"  Masked output shape: {embeddings_masked.shape}")
    print(f"  Mask: first {seq_len-32} tokens attended, last 32 masked")
    print("  [PASS]")

    # Test contrastive loss
    print("\n[Test 5] Contrastive loss")
    loss_fn = ContrastiveLoss(temperature=0.07)

    # Create query and code embeddings
    query_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    code_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    query_embeds = encoder(query_ids)
    code_embeds = encoder(code_ids)

    loss = loss_fn(query_embeds, code_embeds)
    assert loss.ndim == 0, "Loss should be scalar"
    assert loss.item() > 0, "Loss should be positive"
    print(f"  Query embeds: {query_embeds.shape}")
    print(f"  Code embeds: {code_embeds.shape}")
    print(f"  Contrastive loss: {loss.item():.4f}")
    print("  [PASS]")

    # Test similarity computation
    print("\n[Test 6] Similarity computation")
    # Positive pair (same input)
    same_embeds = encoder(input_ids)
    pos_sim = torch.matmul(embeddings, same_embeds.t()).diag().mean()

    # Negative pair (different input)
    diff_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    diff_embeds = encoder(diff_ids)
    neg_sim = torch.matmul(embeddings, diff_embeds.t()).diag().mean()

    print(f"  Positive pair similarity: {pos_sim.item():.4f}")
    print(f"  Negative pair similarity: {neg_sim.item():.4f}")
    print(f"  Note: Positive should be higher after training")
    print("  [PASS]")

    # Test return all hidden states
    print("\n[Test 7] Return all hidden states")
    all_hidden = encoder(input_ids, return_all_hidden=True)
    assert all_hidden.shape == (batch_size, seq_len, d_model)
    print(f"  All hidden states shape: {all_hidden.shape}")
    print("  [PASS]")

    print("\n" + "="*50)
    print("[PASS] All DualEncoder tests passed!")
    print("="*50)
    print(f"\nModel size: {num_params/1e6:.2f}M parameters")
    print(f"Memory footprint (fp32): ~{num_params*4/1e6:.2f}MB")


if __name__ == "__main__":
    test_dual_encoder()
