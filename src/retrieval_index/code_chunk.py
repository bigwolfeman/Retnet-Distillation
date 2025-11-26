"""CodeChunk dataclass for code retrieval (T045 - US4).

Represents a segment of code (file, function, or documentation) that has been
indexed for retrieval (FR-007). Contains original text, embeddings, and metadata.

Schema from data-model.md section 5.
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.retrieval.landmark import LandmarkToken


@dataclass
class CodeChunk:
    """
    Code chunk for retrieval indexing.

    Represents a segment of code with metadata and embeddings for retrieval.
    Each chunk is ≤2048 bytes and has a 768-dim embedding from the dual encoder.

    Args:
        chunk_id: Unique identifier (e.g., "file:path:line_range")
        source_type: Type of source ("file", "function", "class", "doc")
        text: Original code text (up to 2048 bytes)
        language: Programming language (default: "python")
        file_path: Source file path (optional)
        start_line: Starting line number (optional)
        end_line: Ending line number (optional)
        byte_offset: Byte offset in file (optional)
        embedding: 768-dim embedding vector from dual encoder
        embedding_model: Embedding model version
        created_at: Creation timestamp (ISO format)
        last_modified: Last modification timestamp (optional)
        git_commit: Git SHA when indexed (optional)
        retrieval_count: Number of times retrieved (for cache optimization)
        avg_relevance_score: Average retrieval score

    Examples:
        >>> chunk = CodeChunk(
        ...     chunk_id="main.py:10-25",
        ...     source_type="function",
        ...     text="def train_model(config):\\n    ...",
        ...     language="python",
        ...     file_path="src/main.py",
        ...     start_line=10,
        ...     end_line=25,
        ...     embedding=np.random.randn(768)
        ... )
        >>> chunk.validate()
        >>> print(f"Chunk size: {len(chunk.text.encode())} bytes")
    """

    # Identification
    chunk_id: str
    source_type: str  # "file", "function", "class", "doc"

    # Content
    text: str
    language: str = "python"
    file_path: Optional[str] = None

    # Location
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    byte_offset: Optional[int] = None

    # Embeddings (for retrieval)
    embedding: Optional[np.ndarray] = None
    embedding_model: str = "dual-encoder-v1"

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_modified: Optional[str] = None
    git_commit: Optional[str] = None

    # Retrieval statistics
    retrieval_count: int = 0
    avg_relevance_score: float = 0.0

    def to_landmark(self, compressor: Callable) -> "LandmarkToken":
        """
        Compress chunk to landmark tokens (FR-008).

        Converts this code chunk into a compressed landmark representation
        using the provided compressor model. The landmark consists of exactly
        6 tokens (configurable via router_landmark_len_L in ModelConfig).

        Args:
            compressor: Landmark compressor model (from models/retrieval/compressor.py)

        Returns:
            LandmarkToken with compressed representation

        Raises:
            AssertionError: If embedding is None

        Examples:
            >>> # Assuming we have a trained compressor
            >>> landmark = chunk.to_landmark(compressor)
            >>> assert landmark.compressed_tokens.shape == (6, d_model)
            >>> assert landmark.chunk_id == chunk.chunk_id
        """
        from ..models.retrieval.landmark import LandmarkToken

        assert self.embedding is not None, f"Cannot compress chunk without embedding: {self.chunk_id}"

        # Compressor takes embedding and text, returns (L, d_model) tensor
        landmark_embedding = compressor(self.embedding, self.text)

        return LandmarkToken(
            chunk_id=self.chunk_id,
            compressed_tokens=landmark_embedding,
            source_chunk=self
        )

    def validate(self):
        """
        Validate chunk constraints.

        Checks:
        - Text size ≤2048 bytes (as per research.md)
        - Embedding is present (if provided)
        - Embedding dimension is 768 (dual encoder output)

        Raises:
            AssertionError: If any validation fails

        Examples:
            >>> chunk = CodeChunk(
            ...     chunk_id="test",
            ...     source_type="function",
            ...     text="def foo(): pass",
            ...     embedding=np.random.randn(768)
            ... )
            >>> chunk.validate()  # Should pass

            >>> large_chunk = CodeChunk(
            ...     chunk_id="large",
            ...     source_type="file",
            ...     text="x" * 3000  # Too large
            ... )
            >>> large_chunk.validate()  # AssertionError
        """
        # Validate chunk size (FR-007: ≤2048 bytes)
        chunk_bytes = len(self.text.encode('utf-8'))
        assert chunk_bytes <= 2048, \
            f"Chunk exceeds 2048 bytes: {self.chunk_id} ({chunk_bytes} bytes)"

        # Validate embedding if present
        if self.embedding is not None:
            assert isinstance(self.embedding, np.ndarray), \
                f"Embedding must be numpy array: {self.chunk_id}"
            assert self.embedding.shape[0] == 768, \
                f"Invalid embedding dim: {self.embedding.shape} (expected 768) for {self.chunk_id}"
            assert len(self.embedding.shape) == 1, \
                f"Embedding must be 1D vector: {self.embedding.shape} for {self.chunk_id}"

    def update_retrieval_stats(self, relevance_score: float):
        """
        Update retrieval statistics when chunk is retrieved.

        Tracks how often this chunk is retrieved and its average relevance score.
        Useful for cache optimization and quality monitoring.

        Args:
            relevance_score: Relevance score from current retrieval (e.g., cosine similarity)

        Examples:
            >>> chunk = CodeChunk(chunk_id="test", source_type="function", text="code")
            >>> chunk.update_retrieval_stats(0.95)
            >>> print(chunk.retrieval_count)  # 1
            >>> print(chunk.avg_relevance_score)  # 0.95
            >>> chunk.update_retrieval_stats(0.85)
            >>> print(chunk.avg_relevance_score)  # 0.90 (average of 0.95 and 0.85)
        """
        # Update moving average of relevance score
        if self.retrieval_count == 0:
            self.avg_relevance_score = relevance_score
        else:
            # Exponential moving average
            alpha = 0.1
            self.avg_relevance_score = (1 - alpha) * self.avg_relevance_score + alpha * relevance_score

        self.retrieval_count += 1

    def to_dict(self) -> dict:
        """
        Serialize chunk to dictionary.

        Returns:
            Dictionary with all chunk fields

        Examples:
            >>> chunk = CodeChunk(
            ...     chunk_id="test.py:1-10",
            ...     source_type="function",
            ...     text="def test(): pass"
            ... )
            >>> d = chunk.to_dict()
            >>> assert d['chunk_id'] == "test.py:1-10"
            >>> assert d['source_type'] == "function"
        """
        return {
            'chunk_id': self.chunk_id,
            'source_type': self.source_type,
            'text': self.text,
            'language': self.language,
            'file_path': self.file_path,
            'start_line': self.start_line,
            'end_line': self.end_line,
            'byte_offset': self.byte_offset,
            'embedding': self.embedding.tolist() if self.embedding is not None else None,
            'embedding_model': self.embedding_model,
            'created_at': self.created_at,
            'last_modified': self.last_modified,
            'git_commit': self.git_commit,
            'retrieval_count': self.retrieval_count,
            'avg_relevance_score': self.avg_relevance_score,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CodeChunk":
        """
        Deserialize chunk from dictionary.

        Args:
            data: Dictionary with chunk fields

        Returns:
            CodeChunk instance

        Examples:
            >>> data = {
            ...     'chunk_id': 'test.py:1-10',
            ...     'source_type': 'function',
            ...     'text': 'def test(): pass',
            ...     'embedding': [0.1] * 768
            ... }
            >>> chunk = CodeChunk.from_dict(data)
            >>> assert chunk.chunk_id == "test.py:1-10"
        """
        # Convert embedding list back to numpy array
        if data.get('embedding') is not None:
            data['embedding'] = np.array(data['embedding'], dtype=np.float32)

        return cls(**data)

    def __repr__(self) -> str:
        """String representation for debugging."""
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return (
            f"CodeChunk(id={self.chunk_id}, type={self.source_type}, "
            f"lang={self.language}, size={len(self.text.encode())}B, "
            f"text='{text_preview}')"
        )


def test_code_chunk():
    """Test CodeChunk implementation."""
    print("Testing CodeChunk...")

    # Test 1: Basic creation
    print("\n[Test 1] Basic creation")
    chunk = CodeChunk(
        chunk_id="main.py:10-25",
        source_type="function",
        text="def train_model(config):\n    model = Model(config)\n    model.train()\n    return model",
        language="python",
        file_path="src/main.py",
        start_line=10,
        end_line=25,
        embedding=np.random.randn(768).astype(np.float32),
    )
    print(f"  Created: {chunk}")
    print(f"  Chunk ID: {chunk.chunk_id}")
    print(f"  Size: {len(chunk.text.encode())} bytes")
    print("  [PASS]")

    # Test 2: Validation
    print("\n[Test 2] Validation")
    chunk.validate()
    print("  Validation passed for valid chunk")

    # Test invalid chunk (too large)
    try:
        large_chunk = CodeChunk(
            chunk_id="large",
            source_type="file",
            text="x" * 3000,  # Exceeds 2048 bytes
            embedding=np.random.randn(768).astype(np.float32),
        )
        large_chunk.validate()
        assert False, "Should have raised assertion"
    except AssertionError as e:
        print(f"  Correctly rejected large chunk: {str(e)[:60]}...")

    # Test invalid embedding dimension
    try:
        bad_chunk = CodeChunk(
            chunk_id="bad",
            source_type="function",
            text="def foo(): pass",
            embedding=np.random.randn(512).astype(np.float32),  # Wrong dim
        )
        bad_chunk.validate()
        assert False, "Should have raised assertion"
    except AssertionError as e:
        print(f"  Correctly rejected bad embedding dim: {str(e)[:60]}...")

    print("  [PASS]")

    # Test 3: Retrieval stats
    print("\n[Test 3] Retrieval statistics")
    chunk.update_retrieval_stats(0.95)
    assert chunk.retrieval_count == 1
    assert abs(chunk.avg_relevance_score - 0.95) < 1e-6

    chunk.update_retrieval_stats(0.85)
    assert chunk.retrieval_count == 2
    print(f"  After 2 retrievals: count={chunk.retrieval_count}, avg_score={chunk.avg_relevance_score:.3f}")
    print("  [PASS]")

    # Test 4: Serialization
    print("\n[Test 4] Serialization")
    chunk_dict = chunk.to_dict()
    assert chunk_dict['chunk_id'] == "main.py:10-25"
    assert chunk_dict['source_type'] == "function"
    assert len(chunk_dict['embedding']) == 768

    # Round-trip
    chunk_restored = CodeChunk.from_dict(chunk_dict)
    assert chunk_restored.chunk_id == chunk.chunk_id
    assert chunk_restored.source_type == chunk.source_type
    assert chunk_restored.text == chunk.text
    assert np.allclose(chunk_restored.embedding, chunk.embedding)
    print("  Serialization round-trip successful")
    print("  [PASS]")

    # Test 5: Different source types
    print("\n[Test 5] Different source types")
    types_tested = []
    for source_type in ["file", "function", "class", "doc"]:
        test_chunk = CodeChunk(
            chunk_id=f"test_{source_type}",
            source_type=source_type,
            text=f"# {source_type} example",
            embedding=np.random.randn(768).astype(np.float32),
        )
        test_chunk.validate()
        types_tested.append(source_type)
    print(f"  Tested source types: {types_tested}")
    print("  [PASS]")

    print("\n" + "="*50)
    print("[PASS] All CodeChunk tests passed!")
    print("="*50)


if __name__ == "__main__":
    test_code_chunk()
