"""LandmarkToken dataclass (T051 - US4).

Compressed representation of retrieved code chunks inserted into model context.
Each landmark is exactly L=6 tokens (configurable) representing a code chunk.

Schema from data-model.md section 8.
"""

import torch
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...retrieval_index.code_chunk import CodeChunk


@dataclass
class LandmarkToken:
    """
    Compressed representation of a code chunk (FR-008).

    Landmarks are inserted into the model's context as global tokens,
    allowing the model to attend to retrieved code snippets.

    Args:
        chunk_id: Reference to original CodeChunk
        compressed_tokens: Compressed representation [L, d_model] where L=6
        num_tokens: Number of tokens per landmark (default: 6)
        source_chunk: Original CodeChunk (optional, for debugging)
        is_selected: Whether selected by router
        router_score: Router selection score
        bound_position: Position in context (if bound)

    Examples:
        >>> compressed = torch.randn(6, 2816)  # 6 tokens, d_model=2816
        >>> landmark = LandmarkToken(
        ...     chunk_id="main.py:10-25",
        ...     compressed_tokens=compressed
        ... )
        >>> landmark.bind_to_context(position=100)
        >>> assert landmark.is_selected
    """

    # Source
    chunk_id: str
    source_chunk: Optional["CodeChunk"] = None

    # Compressed representation (FR-008: exactly L=6 tokens)
    compressed_tokens: torch.Tensor = None  # (L, d_model) where L=6
    num_tokens: int = 6

    # Selection state (managed by router)
    is_selected: bool = False
    router_score: float = 0.0

    # Position in context
    bound_position: Optional[int] = None

    def __post_init__(self):
        """Validate landmark constraints."""
        if self.compressed_tokens is not None:
            assert self.compressed_tokens.ndim == 2, \
                f"Compressed tokens must be 2D [L, d_model], got shape {self.compressed_tokens.shape}"
            assert self.compressed_tokens.size(0) == self.num_tokens, \
                f"Expected {self.num_tokens} tokens, got {self.compressed_tokens.size(0)}"

    def bind_to_context(self, position: int):
        """
        Bind landmark to specific position in context.

        Marks the landmark as selected and records its position.
        Called by the router when this landmark is chosen.

        Args:
            position: Position in pointer slot (token index)

        Examples:
            >>> landmark.bind_to_context(position=100)
            >>> assert landmark.bound_position == 100
            >>> assert landmark.is_selected
        """
        self.bound_position = position
        self.is_selected = True

    def get_attention_mask(self, seq_len: int) -> torch.Tensor:
        """
        Get attention mask for this landmark.

        Landmarks are global tokens - they attend to all positions
        in the sequence and all positions attend to them.

        Args:
            seq_len: Sequence length for mask

        Returns:
            Attention mask [num_tokens, seq_len] - all True (attend to all)

        Examples:
            >>> mask = landmark.get_attention_mask(seq_len=1024)
            >>> print(mask.shape)  # (6, 1024)
            >>> assert mask.all()  # All True (global token)
        """
        # Landmarks are global tokens - attend to all positions
        return torch.ones(self.num_tokens, seq_len, dtype=torch.bool)

    def get_tokens(self) -> torch.Tensor:
        """
        Get compressed token representations.

        Returns:
            Compressed tokens [L, d_model]

        Examples:
            >>> tokens = landmark.get_tokens()
            >>> print(tokens.shape)  # (6, 2816)
        """
        return self.compressed_tokens

    def to_dict(self) -> dict:
        """
        Serialize landmark to dictionary.

        Returns:
            Dictionary with landmark metadata (without tensors)

        Examples:
            >>> data = landmark.to_dict()
            >>> assert data['chunk_id'] == "main.py:10-25"
        """
        return {
            'chunk_id': self.chunk_id,
            'num_tokens': self.num_tokens,
            'is_selected': self.is_selected,
            'router_score': self.router_score,
            'bound_position': self.bound_position,
            'compressed_shape': list(self.compressed_tokens.shape) if self.compressed_tokens is not None else None,
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        status = "selected" if self.is_selected else "not selected"
        pos_info = f"@{self.bound_position}" if self.bound_position is not None else ""
        score_info = f"(score={self.router_score:.3f})" if self.router_score > 0 else ""

        return (
            f"LandmarkToken(chunk={self.chunk_id}, {status}{pos_info}, "
            f"{self.num_tokens} tokens{score_info})"
        )


def test_landmark_token():
    """Test LandmarkToken implementation."""
    print("Testing LandmarkToken...")

    d_model = 2816
    num_tokens = 6

    # Test 1: Basic creation
    print("\n[Test 1] Basic creation")
    compressed = torch.randn(num_tokens, d_model)
    landmark = LandmarkToken(
        chunk_id="main.py:10-25",
        compressed_tokens=compressed,
    )
    print(f"  Created: {landmark}")
    print(f"  Chunk ID: {landmark.chunk_id}")
    print(f"  Num tokens: {landmark.num_tokens}")
    print(f"  Compressed shape: {landmark.compressed_tokens.shape}")
    print("  [PASS]")

    # Test 2: Bind to context
    print("\n[Test 2] Bind to context")
    assert not landmark.is_selected
    assert landmark.bound_position is None

    landmark.bind_to_context(position=100)
    assert landmark.is_selected
    assert landmark.bound_position == 100
    print(f"  Bound to position: {landmark.bound_position}")
    print(f"  Is selected: {landmark.is_selected}")
    print("  [PASS]")

    # Test 3: Attention mask (global token)
    print("\n[Test 3] Attention mask")
    seq_len = 1024
    mask = landmark.get_attention_mask(seq_len)
    assert mask.shape == (num_tokens, seq_len)
    assert mask.all(), "All positions should be True (global token)"
    print(f"  Mask shape: {mask.shape}")
    print(f"  All True (global token): {mask.all()}")
    print("  [PASS]")

    # Test 4: Get tokens
    print("\n[Test 4] Get tokens")
    tokens = landmark.get_tokens()
    assert tokens.shape == (num_tokens, d_model)
    assert torch.allclose(tokens, compressed)
    print(f"  Tokens shape: {tokens.shape}")
    print("  [PASS]")

    # Test 5: Serialization
    print("\n[Test 5] Serialization")
    data = landmark.to_dict()
    assert data['chunk_id'] == "main.py:10-25"
    assert data['num_tokens'] == num_tokens
    assert data['is_selected'] == True
    assert data['bound_position'] == 100
    assert data['compressed_shape'] == [num_tokens, d_model]
    print(f"  Serialized keys: {list(data.keys())}")
    print(f"  Chunk ID: {data['chunk_id']}")
    print("  [PASS]")

    # Test 6: Validation
    print("\n[Test 6] Validation")
    # Valid landmark
    valid = LandmarkToken(
        chunk_id="test",
        compressed_tokens=torch.randn(6, d_model),
        num_tokens=6
    )
    print("  Valid landmark created successfully")

    # Invalid: wrong number of tokens
    try:
        invalid = LandmarkToken(
            chunk_id="test",
            compressed_tokens=torch.randn(4, d_model),  # Should be 6
            num_tokens=6
        )
        assert False, "Should have raised assertion"
    except AssertionError as e:
        print(f"  Correctly rejected wrong token count: {str(e)[:60]}...")

    # Invalid: wrong shape
    try:
        invalid = LandmarkToken(
            chunk_id="test",
            compressed_tokens=torch.randn(6),  # Should be 2D
            num_tokens=6
        )
        assert False, "Should have raised assertion"
    except AssertionError as e:
        print(f"  Correctly rejected wrong shape: {str(e)[:60]}...")

    print("  [PASS]")

    # Test 7: Router integration
    print("\n[Test 7] Router integration")
    landmark2 = LandmarkToken(
        chunk_id="utils.py:50-75",
        compressed_tokens=torch.randn(num_tokens, d_model),
    )
    landmark2.router_score = 0.85
    landmark2.bind_to_context(position=200)

    print(f"  Landmark with router score: {landmark2}")
    assert landmark2.router_score == 0.85
    print("  [PASS]")

    print("\n" + "="*50)
    print("[PASS] All LandmarkToken tests passed!")
    print("="*50)


if __name__ == "__main__":
    test_landmark_token()
