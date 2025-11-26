"""Landmark Compressor (T050 - US4).

2-layer MLP that compresses code chunk embeddings (768-dim) into landmark tokens
(L × d_model), where L=6 tokens and d_model=2816.

Architecture from research.md and data-model.md.
"""

import torch
import torch.nn as nn
from typing import Optional
import numpy as np


class LandmarkCompressor(nn.Module):
    """
    Compresses code chunk embeddings to landmark tokens (FR-008).

    2-layer MLP architecture:
    - Input: embedding_dim (768) from dual encoder
    - Hidden: hidden_dim (default: 1536 = 2 * embedding_dim)
    - Output: model_dim × num_tokens (2816 × 6 = 16896)
    - Reshape to: (num_tokens, model_dim) = (6, 2816)

    Trained end-to-end with the main model to preserve semantic meaning.

    Args:
        embedding_dim: Input embedding dimension (default: 768)
        model_dim: Model hidden dimension (default: 2816)
        num_tokens: Number of landmark tokens per chunk (default: 6, from FR-008)
        hidden_dim: Hidden layer dimension (default: 2 * embedding_dim)
        dropout: Dropout probability (default: 0.1)

    Examples:
        >>> compressor = LandmarkCompressor(embedding_dim=768, model_dim=2816)
        >>> chunk_embedding = torch.randn(1, 768)  # From dual encoder
        >>> landmark_tokens = compressor(chunk_embedding)
        >>> print(landmark_tokens.shape)  # (1, 6, 2816)
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        model_dim: int = 2816,
        num_tokens: int = 6,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.model_dim = model_dim
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim if hidden_dim is not None else 2 * embedding_dim

        self.output_dim = model_dim * num_tokens  # Total output size

        # 2-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following best practices."""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                # Xavier uniform initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        chunk_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compress chunk embeddings to landmark tokens.

        Args:
            chunk_embeddings: Chunk embeddings [batch, embedding_dim] or [embedding_dim]

        Returns:
            landmark_tokens: [batch, num_tokens, model_dim] or [num_tokens, model_dim]

        Examples:
            >>> compressor = LandmarkCompressor()
            >>> # Single chunk
            >>> emb = torch.randn(768)
            >>> tokens = compressor(emb)
            >>> print(tokens.shape)  # (6, 2816)
            >>>
            >>> # Batch of chunks
            >>> embs = torch.randn(10, 768)
            >>> tokens = compressor(embs)
            >>> print(tokens.shape)  # (10, 6, 2816)
        """
        # Handle both batched and unbatched inputs
        if chunk_embeddings.ndim == 1:
            # Single embedding: [embedding_dim] -> [1, embedding_dim]
            chunk_embeddings = chunk_embeddings.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = chunk_embeddings.size(0)

        # MLP compression: [batch, embedding_dim] -> [batch, model_dim * num_tokens]
        compressed = self.mlp(chunk_embeddings)

        # Reshape to landmark tokens: [batch, model_dim * num_tokens] -> [batch, num_tokens, model_dim]
        landmark_tokens = compressed.view(batch_size, self.num_tokens, self.model_dim)

        # Squeeze if input was unbatched
        if squeeze_output:
            landmark_tokens = landmark_tokens.squeeze(0)  # [num_tokens, model_dim]

        return landmark_tokens

    def compress_batch(
        self,
        chunk_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compress batch of chunk embeddings (convenience method).

        Args:
            chunk_embeddings: [batch, embedding_dim]

        Returns:
            landmark_tokens: [batch, num_tokens, model_dim]

        Examples:
            >>> compressor = LandmarkCompressor()
            >>> embs = torch.randn(32, 768)  # 32 chunks
            >>> tokens = compressor.compress_batch(embs)
            >>> print(tokens.shape)  # (32, 6, 2816)
        """
        return self.forward(chunk_embeddings)

    def compress_numpy(
        self,
        chunk_embedding: np.ndarray,
    ) -> torch.Tensor:
        """
        Compress numpy embedding to landmark tokens.

        Convenience method for working with CodeChunk.embedding (numpy arrays).

        Args:
            chunk_embedding: Numpy array [embedding_dim]

        Returns:
            landmark_tokens: Torch tensor [num_tokens, model_dim]

        Examples:
            >>> compressor = LandmarkCompressor()
            >>> emb_np = np.random.randn(768).astype('float32')
            >>> tokens = compressor.compress_numpy(emb_np)
            >>> print(tokens.shape)  # (6, 2816)
        """
        # Convert to torch tensor
        chunk_tensor = torch.from_numpy(chunk_embedding).float()

        # Compress
        with torch.no_grad():
            landmark_tokens = self.forward(chunk_tensor)

        return landmark_tokens

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def test_landmark_compressor():
    """Test LandmarkCompressor implementation."""
    print("Testing LandmarkCompressor...")

    # Configuration
    embedding_dim = 768
    model_dim = 2816
    num_tokens = 6
    batch_size = 8

    # Create compressor
    print("\n[Test 1] Create compressor")
    compressor = LandmarkCompressor(
        embedding_dim=embedding_dim,
        model_dim=model_dim,
        num_tokens=num_tokens
    )
    num_params = compressor.get_num_params()
    print(f"  Compressor created")
    print(f"  Input dim: {embedding_dim}")
    print(f"  Output: {num_tokens} tokens × {model_dim} dim")
    print(f"  Parameters: {num_params/1e6:.2f}M")
    print("  [PASS]")

    # Test single embedding
    print("\n[Test 2] Compress single embedding")
    single_emb = torch.randn(embedding_dim)
    tokens_single = compressor(single_emb)
    assert tokens_single.shape == (num_tokens, model_dim), \
        f"Shape mismatch: {tokens_single.shape} vs ({num_tokens}, {model_dim})"
    print(f"  Input shape: {single_emb.shape}")
    print(f"  Output shape: {tokens_single.shape}")
    print("  [PASS]")

    # Test batch compression
    print("\n[Test 3] Compress batch")
    batch_emb = torch.randn(batch_size, embedding_dim)
    tokens_batch = compressor(batch_emb)
    assert tokens_batch.shape == (batch_size, num_tokens, model_dim), \
        f"Shape mismatch: {tokens_batch.shape}"
    print(f"  Input shape: {batch_emb.shape}")
    print(f"  Output shape: {tokens_batch.shape}")
    print("  [PASS]")

    # Test numpy compression
    print("\n[Test 4] Compress numpy embedding")
    numpy_emb = np.random.randn(embedding_dim).astype('float32')
    tokens_numpy = compressor.compress_numpy(numpy_emb)
    assert tokens_numpy.shape == (num_tokens, model_dim)
    print(f"  Input type: {type(numpy_emb).__name__}")
    print(f"  Output shape: {tokens_numpy.shape}")
    print("  [PASS]")

    # Test determinism
    print("\n[Test 5] Determinism")
    emb_test = torch.randn(embedding_dim)
    with torch.no_grad():
        output1 = compressor(emb_test)
        output2 = compressor(emb_test)
    assert torch.allclose(output1, output2), "Outputs should be deterministic"
    print("  Same input produces same output ✓")
    print("  [PASS]")

    # Test gradient flow
    print("\n[Test 6] Gradient flow")
    emb_grad = torch.randn(embedding_dim, requires_grad=True)
    tokens_grad = compressor(emb_grad)
    loss = tokens_grad.sum()
    loss.backward()
    assert emb_grad.grad is not None, "Gradients should flow through"
    assert compressor.mlp[0].weight.grad is not None, "MLP should have gradients"
    print("  Gradients flow through compressor ✓")
    print(f"  Input grad norm: {emb_grad.grad.norm().item():.4f}")
    print("  [PASS]")

    # Test different configurations
    print("\n[Test 7] Different configurations")
    configs = [
        {'num_tokens': 4, 'model_dim': 2048},
        {'num_tokens': 8, 'model_dim': 3072},
        {'num_tokens': 6, 'model_dim': 2816, 'hidden_dim': 2048},
    ]

    for i, config in enumerate(configs):
        comp = LandmarkCompressor(embedding_dim=embedding_dim, **config)
        test_emb = torch.randn(embedding_dim)
        test_tokens = comp(test_emb)
        expected_shape = (config['num_tokens'], config['model_dim'])
        assert test_tokens.shape == expected_shape, \
            f"Config {i}: Shape mismatch {test_tokens.shape} vs {expected_shape}"
        print(f"  Config {i+1}: {config['num_tokens']} tokens × {config['model_dim']} dim ✓")

    print("  [PASS]")

    # Test memory and parameter count
    print("\n[Test 8] Memory and parameters")
    compressor_standard = LandmarkCompressor()
    params = compressor_standard.get_num_params()

    # Expected params:
    # Layer 1: 768 * 1536 + 1536 = 1,181,184 + 1,536 = 1,182,720
    # Layer 2: 1536 * 16896 + 16896 = 25,952,256 + 16,896 = 25,969,152
    # Total: ~27.15M
    print(f"  Total parameters: {params/1e6:.2f}M")
    print(f"  Memory (fp32): ~{params*4/1e6:.2f}MB")
    print(f"  Memory (fp16): ~{params*2/1e6:.2f}MB")

    assert params > 20e6 and params < 30e6, "Parameter count should be ~27M"
    print("  [PASS]")

    print("\n" + "="*50)
    print("[PASS] All LandmarkCompressor tests passed!")
    print("="*50)
    print(f"\nCompressor summary:")
    print(f"  Input: {embedding_dim}-dim embeddings from dual encoder")
    print(f"  Output: {num_tokens} landmark tokens of {model_dim} dimensions")
    print(f"  Parameters: {num_params/1e6:.2f}M")
    print(f"  Architecture: 2-layer MLP ({embedding_dim} → {compressor.hidden_dim} → {compressor.output_dim})")


if __name__ == "__main__":
    test_landmark_compressor()
