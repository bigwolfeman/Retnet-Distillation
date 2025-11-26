"""RetrievalIndex wrapper class (T049 - US4).

Unified interface for code retrieval using FAISS (global) or HNSW (workspace) backends.
Implements the RetrievalIndex schema from data-model.md.

Usage:
- Workspace index: Local project code (2-8GB), HNSW backend, fast updates
- Global index: General knowledge (10k-1M chunks), FAISS backend, optimized search
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field
import time

from .code_chunk import CodeChunk
from .faiss_builder import FAISSIndexBuilder
from .hnsw_builder import HNSWIndexBuilder


@dataclass
class RetrievalIndex:
    """
    Unified retrieval index for code chunks.

    Wraps either FAISS (global knowledge) or HNSW (workspace) backend
    for efficient code retrieval.

    Args:
        index_id: Unique identifier for this index
        index_type: "workspace" (HNSW) or "global" (FAISS)
        index_config: Backend-specific configuration
        chunks: Dictionary mapping chunk_id to CodeChunk
        workspace_path: Root path for workspace index (optional)

    Examples:
        >>> # Create workspace index
        >>> index = RetrievalIndex(
        ...     index_id="my-project",
        ...     index_type="workspace"
        ... )
        >>> index.build(chunks)
        >>> results = index.search(query_embedding, k=10)
    """

    # Identification
    index_id: str
    index_type: str  # "workspace" (HNSW) or "global" (FAISS)

    # Index backend (will be FAISSIndexBuilder or HNSWIndexBuilder)
    index_backend: Any = None
    index_config: Dict[str, Any] = field(default_factory=dict)

    # Chunks
    chunks: Dict[str, CodeChunk] = field(default_factory=dict)
    chunk_embeddings: Optional[np.ndarray] = None

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_updated: Optional[str] = None
    num_chunks: int = 0
    index_size_mb: float = 0.0

    # Workspace-specific
    workspace_path: Optional[str] = None
    git_commits_indexed: List[str] = field(default_factory=list)

    def build(self, chunks: List[CodeChunk]):
        """
        Build index from chunks.

        Args:
            chunks: List of CodeChunk objects with embeddings

        Raises:
            AssertionError: If chunks missing embeddings
            ValueError: If index_type is invalid

        Examples:
            >>> chunks = [CodeChunk(...), CodeChunk(...)]
            >>> index.build(chunks)
            >>> print(index.num_chunks)
        """
        print(f"Building {self.index_type} index with {len(chunks)} chunks...")

        # Validate chunks
        for chunk in chunks:
            assert chunk.embedding is not None, \
                f"Chunk {chunk.chunk_id} missing embedding"

        # Store chunks
        self.chunks = {c.chunk_id: c for c in chunks}
        self.num_chunks = len(chunks)

        # Extract embeddings
        embeddings = np.array([c.embedding for c in chunks], dtype=np.float32)
        self.chunk_embeddings = embeddings

        print(f"  Embeddings shape: {embeddings.shape}")

        # Build backend index
        if self.index_type == "workspace":
            self._build_workspace_index(embeddings)
        elif self.index_type == "global":
            self._build_global_index(embeddings)
        else:
            raise ValueError(f"Invalid index_type: {self.index_type}. Must be 'workspace' or 'global'")

        self.last_updated = datetime.utcnow().isoformat()
        print(f"  Index built successfully ({self.num_chunks} chunks)")

    def _build_workspace_index(self, embeddings: np.ndarray):
        """Build HNSW index for workspace."""
        # Get config or use defaults
        M = self.index_config.get('M', 32)
        ef_construction = self.index_config.get('ef_construction', 200)
        ef_search = self.index_config.get('ef_search', 200)
        max_elements = self.index_config.get('max_elements', 100000)

        # Create HNSW builder
        self.index_backend = HNSWIndexBuilder(
            d_model=embeddings.shape[1],
            M=M,
            ef_construction=ef_construction,
            ef_search=ef_search,
            max_elements=max_elements,
            metric='ip'
        )

        # Build index
        self.index_backend.build(embeddings)

        # Update metadata
        stats = self.index_backend.get_stats()
        self.index_size_mb = stats['memory_mb']

    def _build_global_index(self, embeddings: np.ndarray):
        """Build FAISS index for global knowledge."""
        # Get config or use defaults
        n_centroids = self.index_config.get('n_centroids', 1024)
        n_subquantizers = self.index_config.get('n_subquantizers', 64)
        nprobe = self.index_config.get('nprobe', 32)

        # Create FAISS builder
        self.index_backend = FAISSIndexBuilder(
            d_model=embeddings.shape[1],
            n_centroids=n_centroids,
            n_subquantizers=n_subquantizers,
            nprobe=nprobe,
            metric='ip'
        )

        # Train and build index
        self.index_backend.train(embeddings)
        self.index_backend.add(embeddings)

        # Update metadata
        stats = self.index_backend.get_stats()
        self.index_size_mb = stats['memory_mb']

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 32,
        track_latency: bool = True,
    ) -> List[Tuple[CodeChunk, float]]:
        """
        Search index for top-k chunks.

        Args:
            query_embedding: Query vector (d,) float32
            k: Number of results to return
            track_latency: Whether to track search latency

        Returns:
            List of (CodeChunk, score) tuples, sorted by relevance

        Raises:
            RuntimeError: If index not built

        Examples:
            >>> query = np.random.randn(768).astype('float32')
            >>> results = index.search(query, k=10)
            >>> for chunk, score in results:
            ...     print(f"{chunk.chunk_id}: {score:.3f}")
        """
        if self.index_backend is None:
            raise RuntimeError("Index not built. Call build() first.")

        # Ensure query is 2D for backend compatibility
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Time the search (for NFR-003 validation: <500ms for workspace)
        start_time = time.time() if track_latency else None

        # Search backend
        if self.index_type == "workspace":
            indices, distances = self.index_backend.search(query_embedding, k=k)
        else:  # global (FAISS)
            distances, indices = self.index_backend.search(query_embedding, k=k)

        # Record latency
        if track_latency:
            latency_ms = (time.time() - start_time) * 1000
            if self.index_type == "workspace" and latency_ms > 500:
                print(f"Warning: Workspace search took {latency_ms:.1f}ms (target <500ms, NFR-003)")

        # Flatten results (handle 2D output)
        if indices.ndim == 2:
            indices = indices[0]
            distances = distances[0]

        # Retrieve chunks
        results = []
        chunk_list = list(self.chunks.values())

        for idx, score in zip(indices, distances):
            if idx < len(chunk_list):  # Valid index
                chunk = chunk_list[int(idx)]

                # Update retrieval stats
                chunk.update_retrieval_stats(float(score))

                results.append((chunk, float(score)))

        return results

    def refresh(self, new_chunks: List[CodeChunk]):
        """
        Incrementally update index with new chunks (SC-010).

        For workspace: Rebuild HNSW (fast for small indexes)
        For global: Add new chunks incrementally

        Args:
            new_chunks: List of new CodeChunk objects

        Examples:
            >>> new_chunks = [CodeChunk(...), ...]
            >>> index.refresh(new_chunks)
            >>> print(index.num_chunks)  # Updated count
        """
        print(f"Refreshing {self.index_type} index with {len(new_chunks)} new chunks...")

        if self.index_type == "workspace":
            # Workspace: Full rebuild (acceptable for 2-8GB projects)
            all_chunks = list(self.chunks.values()) + new_chunks
            self.build(all_chunks)
            print(f"  Workspace rebuilt with {len(all_chunks)} total chunks")

        else:  # global
            # Global: Incremental add
            for chunk in new_chunks:
                self.chunks[chunk.chunk_id] = chunk

            new_embeddings = np.array([c.embedding for c in new_chunks], dtype=np.float32)
            self.index_backend.add(new_embeddings)
            self.num_chunks += len(new_chunks)

            print(f"  Added {len(new_chunks)} chunks (total: {self.num_chunks})")

        self.last_updated = datetime.utcnow().isoformat()

    def save(self, filepath: str):
        """
        Save index to disk.

        Args:
            filepath: Base path for saving (backend adds extensions)

        Examples:
            >>> index.save("indexes/my-project")
            >>> # Creates: my-project.hnsw + my-project.meta (or .faiss)
        """
        if self.index_backend is None:
            raise RuntimeError("Index not built. Nothing to save.")

        # Save backend index
        self.index_backend.save(filepath)

        # Save chunks separately
        import pickle
        chunks_dict = {cid: chunk.to_dict() for cid, chunk in self.chunks.items()}

        with open(f"{filepath}.chunks.pkl", 'wb') as f:
            pickle.dump(chunks_dict, f)

        # Save index metadata
        metadata = {
            'index_id': self.index_id,
            'index_type': self.index_type,
            'index_config': self.index_config,
            'created_at': self.created_at,
            'last_updated': self.last_updated,
            'num_chunks': self.num_chunks,
            'index_size_mb': self.index_size_mb,
            'workspace_path': self.workspace_path,
            'git_commits_indexed': self.git_commits_indexed,
        }

        with open(f"{filepath}.index_meta.pkl", 'wb') as f:
            pickle.dump(metadata, f)

        print(f"Index saved to {filepath} ({self.num_chunks} chunks, {self.index_size_mb:.2f}MB)")

    def load(self, filepath: str):
        """
        Load index from disk.

        Args:
            filepath: Base path for loading (without extension)

        Examples:
            >>> index = RetrievalIndex(index_id="loaded", index_type="workspace")
            >>> index.load("indexes/my-project")
            >>> print(index.num_chunks)
        """
        import pickle

        # Load index metadata
        with open(f"{filepath}.index_meta.pkl", 'rb') as f:
            metadata = pickle.load(f)

        self.index_id = metadata['index_id']
        self.index_type = metadata['index_type']
        self.index_config = metadata['index_config']
        self.created_at = metadata['created_at']
        self.last_updated = metadata['last_updated']
        self.num_chunks = metadata['num_chunks']
        self.index_size_mb = metadata['index_size_mb']
        self.workspace_path = metadata.get('workspace_path')
        self.git_commits_indexed = metadata.get('git_commits_indexed', [])

        # Load chunks
        with open(f"{filepath}.chunks.pkl", 'rb') as f:
            chunks_dict = pickle.load(f)

        self.chunks = {cid: CodeChunk.from_dict(cdata) for cid, cdata in chunks_dict.items()}

        # Load backend index
        if self.index_type == "workspace":
            self.index_backend = HNSWIndexBuilder()
            self.index_backend.load(filepath)
        else:  # global
            self.index_backend = FAISSIndexBuilder()
            self.index_backend.load(filepath)

        print(f"Index loaded from {filepath} ({self.num_chunks} chunks)")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get index statistics.

        Returns:
            Dictionary with index stats

        Examples:
            >>> stats = index.get_stats()
            >>> print(stats['num_chunks'])
            >>> print(stats['index_size_mb'])
        """
        stats = {
            'index_id': self.index_id,
            'index_type': self.index_type,
            'num_chunks': self.num_chunks,
            'index_size_mb': self.index_size_mb,
            'created_at': self.created_at,
            'last_updated': self.last_updated,
        }

        if self.workspace_path:
            stats['workspace_path'] = self.workspace_path
            stats['git_commits_indexed'] = len(self.git_commits_indexed)

        # Add backend stats
        if self.index_backend:
            backend_stats = self.index_backend.get_stats()
            stats['backend'] = backend_stats

        return stats


def test_retrieval_index():
    """Test RetrievalIndex implementation."""
    print("Testing RetrievalIndex...")

    # Check if dependencies are installed
    try:
        import hnswlib
        import faiss
    except ImportError as e:
        print(f"Skipping tests: {e}")
        print("Install with: pip install hnswlib faiss-cpu")
        return

    # Configuration
    d_model = 768
    n_chunks = 100

    # Create test chunks
    print("\n[Test 1] Create test chunks")
    chunks = []
    for i in range(n_chunks):
        chunk = CodeChunk(
            chunk_id=f"chunk_{i}",
            source_type="function",
            text=f"def function_{i}(): pass",
            embedding=np.random.randn(d_model).astype('float32')
        )
        # Normalize embeddings
        chunk.embedding = chunk.embedding / np.linalg.norm(chunk.embedding)
        chunks.append(chunk)
    print(f"  Created {len(chunks)} test chunks")
    print("  [PASS]")

    # Test workspace index
    print("\n[Test 2] Build workspace index (HNSW)")
    workspace_index = RetrievalIndex(
        index_id="test-workspace",
        index_type="workspace"
    )
    workspace_index.build(chunks)
    assert workspace_index.num_chunks == n_chunks
    print(f"  Built workspace index: {workspace_index.num_chunks} chunks")
    print("  [PASS]")

    # Test global index
    print("\n[Test 3] Build global index (FAISS)")
    global_index = RetrievalIndex(
        index_id="test-global",
        index_type="global",
        index_config={'n_centroids': 16, 'n_subquantizers': 32}  # Smaller for testing
    )
    global_index.build(chunks)
    assert global_index.num_chunks == n_chunks
    print(f"  Built global index: {global_index.num_chunks} chunks")
    print("  [PASS]")

    # Test search
    print("\n[Test 4] Search")
    query = np.random.randn(d_model).astype('float32')
    query = query / np.linalg.norm(query)

    workspace_results = workspace_index.search(query, k=10)
    assert len(workspace_results) == 10
    print(f"  Workspace search returned {len(workspace_results)} results")

    global_results = global_index.search(query, k=10)
    assert len(global_results) == 10
    print(f"  Global search returned {len(global_results)} results")

    # Check result format
    chunk, score = workspace_results[0]
    assert isinstance(chunk, CodeChunk)
    assert isinstance(score, float)
    print(f"  Top result: {chunk.chunk_id} (score: {score:.3f})")
    print("  [PASS]")

    # Test refresh (incremental update)
    print("\n[Test 5] Refresh index")
    new_chunks = []
    for i in range(10):
        chunk = CodeChunk(
            chunk_id=f"new_chunk_{i}",
            source_type="function",
            text=f"def new_function_{i}(): pass",
            embedding=np.random.randn(d_model).astype('float32')
        )
        chunk.embedding = chunk.embedding / np.linalg.norm(chunk.embedding)
        new_chunks.append(chunk)

    workspace_index.refresh(new_chunks)
    assert workspace_index.num_chunks == n_chunks + 10
    print(f"  Workspace refreshed: {workspace_index.num_chunks} total chunks")
    print("  [PASS]")

    # Test save/load
    print("\n[Test 6] Save and load")
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_index")

        # Save
        workspace_index.save(save_path)

        # Load in new index
        loaded_index = RetrievalIndex(index_id="loaded", index_type="workspace")
        loaded_index.load(save_path)

        assert loaded_index.num_chunks == workspace_index.num_chunks
        assert loaded_index.index_id == workspace_index.index_id

        # Search with loaded index
        loaded_results = loaded_index.search(query, k=10)
        assert len(loaded_results) == 10

        print(f"  Saved and loaded successfully")
        print(f"  Loaded index has {loaded_index.num_chunks} chunks")
    print("  [PASS]")

    # Test stats
    print("\n[Test 7] Get statistics")
    stats = workspace_index.get_stats()
    print(f"  Index stats:")
    print(f"    Type: {stats['index_type']}")
    print(f"    Chunks: {stats['num_chunks']}")
    print(f"    Size: {stats['index_size_mb']:.2f} MB")
    print("  [PASS]")

    print("\n" + "="*50)
    print("[PASS] All RetrievalIndex tests passed!")
    print("="*50)


if __name__ == "__main__":
    test_retrieval_index()
