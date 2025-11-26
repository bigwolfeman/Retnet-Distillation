"""HNSW index builder for workspace retrieval (T048 - US4).

Builds HNSW (Hierarchical Navigable Small World) index for fast workspace
code retrieval. Used for local project index (2-8GB codebase).

Index configuration from research.md:
- M=32: Number of bidirectional links per node
- efSearch=200: Size of dynamic candidate list during search
- efConstruction=200: Size of dynamic candidate list during construction
- metric='ip': Inner product for similarity (cosine with normalized vectors)

Fast, in-memory index suitable for incremental updates.
"""

import numpy as np
from typing import List, Tuple, Optional
import os
import pickle


class HNSWIndexBuilder:
    """
    HNSW index builder for workspace code retrieval.

    Builds and manages an HNSW index for fast approximate nearest neighbor
    search. Optimized for workspace-scale retrieval (10k-100k chunks).

    Args:
        d_model: Embedding dimension (default: 768)
        M: Number of bidirectional links (default: 32)
        ef_construction: Size of dynamic list during construction (default: 200)
        ef_search: Size of dynamic list during search (default: 200)
        max_elements: Maximum number of elements (default: 100000)
        metric: Distance metric ("ip" for inner product, "l2" for L2)

    Examples:
        >>> builder = HNSWIndexBuilder(d_model=768)
        >>> embeddings = np.random.randn(10000, 768).astype('float32')
        >>> builder.build(embeddings)
        >>> query = np.random.randn(1, 768).astype('float32')
        >>> indices, distances = builder.search(query, k=10)
    """

    def __init__(
        self,
        d_model: int = 768,
        M: int = 32,
        ef_construction: int = 200,
        ef_search: int = 200,
        max_elements: int = 100000,
        metric: str = "ip",  # "ip" (inner product) or "l2"
    ):
        self.d_model = d_model
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.max_elements = max_elements
        self.metric = metric

        self.index = None
        self.num_vectors = 0

    def build(self, embeddings: np.ndarray):
        """
        Build HNSW index from embeddings.

        Creates index and adds all embeddings. Can be called incrementally
        (but recommend rebuild for best quality).

        Args:
            embeddings: Embeddings to index [n, d_model] (float32)

        Raises:
            AssertionError: If embeddings have wrong shape
            ImportError: If hnswlib is not installed

        Examples:
            >>> builder = HNSWIndexBuilder()
            >>> data = np.random.randn(10000, 768).astype('float32')
            >>> builder.build(data)
            >>> assert builder.num_vectors == 10000
        """
        try:
            import hnswlib
        except ImportError:
            raise ImportError(
                "hnswlib is required. Install with: pip install hnswlib"
            )

        n, d = embeddings.shape
        assert d == self.d_model, f"Embedding dim mismatch: {d} vs {self.d_model}"
        assert embeddings.dtype == np.float32, "Embeddings must be float32"
        assert n <= self.max_elements, f"Too many elements: {n} > {self.max_elements}"

        print(f"Building HNSW index...")
        print(f"  Embeddings: {n} x {d}")
        print(f"  M={self.M}, ef_construction={self.ef_construction}")
        print(f"  Metric: {self.metric}")

        # Create index
        space = 'ip' if self.metric == 'ip' else 'l2'
        self.index = hnswlib.Index(space=space, dim=d)

        # Initialize index
        self.index.init_index(
            max_elements=self.max_elements,
            ef_construction=self.ef_construction,
            M=self.M
        )

        # Set ef for search
        self.index.set_ef(self.ef_search)

        # Add items
        print(f"  Adding {n} vectors...")
        ids = np.arange(n)
        self.index.add_items(embeddings, ids)
        self.num_vectors = n

        print(f"  HNSW index built with {self.num_vectors} vectors")

    def add(self, embeddings: np.ndarray, start_id: Optional[int] = None):
        """
        Add more embeddings to existing index (incremental update).

        Args:
            embeddings: Embeddings to add [n_add, d_model] (float32)
            start_id: Starting ID for new items (default: current num_vectors)

        Examples:
            >>> builder.build(initial_data)
            >>> new_data = np.random.randn(1000, 768).astype('float32')
            >>> builder.add(new_data)
            >>> print(builder.num_vectors)  # initial + 1000
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")

        n_add, d = embeddings.shape
        assert d == self.d_model, f"Embedding dim mismatch: {d} vs {self.d_model}"
        assert embeddings.dtype == np.float32, "Embeddings must be float32"

        if start_id is None:
            start_id = self.num_vectors

        assert start_id + n_add <= self.max_elements, \
            f"Adding {n_add} vectors would exceed max_elements {self.max_elements}"

        print(f"Adding {n_add} vectors to HNSW index...")
        ids = np.arange(start_id, start_id + n_add)
        self.index.add_items(embeddings, ids)
        self.num_vectors += n_add
        print(f"  Total vectors: {self.num_vectors}")

    def search(
        self,
        queries: np.ndarray,
        k: int = 32,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search index for top-k nearest neighbors.

        Args:
            queries: Query embeddings [n_queries, d_model] (float32)
            k: Number of neighbors to return

        Returns:
            indices: [n_queries, k] indices of neighbors
            distances: [n_queries, k] distances/similarities

        Note: Returns (indices, distances) - opposite order from FAISS!

        Examples:
            >>> query = np.random.randn(1, 768).astype('float32')
            >>> indices, distances = builder.search(query, k=10)
            >>> print(indices.shape)    # (1, 10)
            >>> print(distances.shape)  # (1, 10)
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")

        n_queries, d = queries.shape
        assert d == self.d_model, f"Query dim mismatch: {d} vs {self.d_model}"
        assert queries.dtype == np.float32, "Queries must be float32"

        # HNSW returns (indices, distances)
        indices, distances = self.index.knn_query(queries, k=k)
        return indices, distances

    def save(self, filepath: str):
        """
        Save index to disk.

        Args:
            filepath: Path to save index (will create .hnsw file)

        Examples:
            >>> builder.save("workspace_index")
            >>> # Later: builder.load("workspace_index")
        """
        if self.index is None:
            raise RuntimeError("Cannot save unbuilt index")

        # Save HNSW index
        self.index.save_index(f"{filepath}.hnsw")

        # Save metadata
        metadata = {
            'd_model': self.d_model,
            'M': self.M,
            'ef_construction': self.ef_construction,
            'ef_search': self.ef_search,
            'max_elements': self.max_elements,
            'metric': self.metric,
            'num_vectors': self.num_vectors,
        }

        with open(f"{filepath}.meta", 'wb') as f:
            pickle.dump(metadata, f)

        print(f"Index saved to {filepath}.hnsw ({self.num_vectors} vectors)")

    def load(self, filepath: str):
        """
        Load index from disk.

        Args:
            filepath: Path to load index from (without .hnsw extension)

        Examples:
            >>> builder = HNSWIndexBuilder()
            >>> builder.load("workspace_index")
            >>> print(builder.num_vectors)
        """
        try:
            import hnswlib
        except ImportError:
            raise ImportError("hnswlib required")

        # Load metadata
        with open(f"{filepath}.meta", 'rb') as f:
            metadata = pickle.load(f)

        self.d_model = metadata['d_model']
        self.M = metadata['M']
        self.ef_construction = metadata['ef_construction']
        self.ef_search = metadata['ef_search']
        self.max_elements = metadata['max_elements']
        self.metric = metadata['metric']
        self.num_vectors = metadata['num_vectors']

        # Load HNSW index
        space = 'ip' if self.metric == 'ip' else 'l2'
        self.index = hnswlib.Index(space=space, dim=self.d_model)
        self.index.load_index(f"{filepath}.hnsw", max_elements=self.max_elements)
        self.index.set_ef(self.ef_search)

        print(f"Index loaded from {filepath}.hnsw ({self.num_vectors} vectors)")

    def get_stats(self) -> dict:
        """
        Get index statistics.

        Returns:
            Dictionary with index stats

        Examples:
            >>> stats = builder.get_stats()
            >>> print(stats['num_vectors'])
            >>> print(stats['memory_mb'])
        """
        stats = {
            'd_model': self.d_model,
            'M': self.M,
            'ef_construction': self.ef_construction,
            'ef_search': self.ef_search,
            'max_elements': self.max_elements,
            'metric': self.metric,
            'num_vectors': self.num_vectors,
        }

        # Estimate memory usage
        # HNSW memory: roughly M * layers * num_vectors * (4 bytes per link + d * 4 bytes)
        if self.index is not None:
            # Rough estimate: M links per node * 4 bytes (int) + embedding (d * 4 bytes)
            # Plus graph structure overhead
            bytes_per_vector = (self.M * 4) + (self.d_model * 4)
            total_bytes = self.num_vectors * bytes_per_vector
            stats['memory_mb'] = total_bytes / (1024 * 1024)
        else:
            stats['memory_mb'] = 0

        return stats


def test_hnsw_builder():
    """Test HNSWIndexBuilder implementation."""
    print("Testing HNSWIndexBuilder...")

    try:
        import hnswlib
    except ImportError:
        print("hnswlib not installed. Skipping tests.")
        print("Install with: pip install hnswlib")
        return

    # Configuration
    d_model = 768
    n_build = 5000
    n_add = 1000
    n_query = 10

    # Create builder
    print("\n[Test 1] Create builder")
    builder = HNSWIndexBuilder(
        d_model=d_model,
        M=32,
        ef_construction=200,
        ef_search=200,
        max_elements=10000,
        metric="ip"
    )
    print(f"  Builder created:")
    print(f"    d_model={builder.d_model}")
    print(f"    M={builder.M}")
    print(f"    ef_search={builder.ef_search}")
    print("  [PASS]")

    # Build index
    print("\n[Test 2] Build index")
    build_embeddings = np.random.randn(n_build, d_model).astype('float32')
    # Normalize for inner product
    build_embeddings = build_embeddings / np.linalg.norm(build_embeddings, axis=1, keepdims=True)
    builder.build(build_embeddings)
    assert builder.num_vectors == n_build
    print("  [PASS]")

    # Add vectors (incremental)
    print("\n[Test 3] Add vectors incrementally")
    add_embeddings = np.random.randn(n_add, d_model).astype('float32')
    add_embeddings = add_embeddings / np.linalg.norm(add_embeddings, axis=1, keepdims=True)
    builder.add(add_embeddings)
    assert builder.num_vectors == n_build + n_add
    print(f"  Added {n_add} vectors")
    print(f"  Total: {builder.num_vectors}")
    print("  [PASS]")

    # Search
    print("\n[Test 4] Search")
    query_embeddings = np.random.randn(n_query, d_model).astype('float32')
    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    indices, distances = builder.search(query_embeddings, k=10)
    assert indices.shape == (n_query, 10)
    assert distances.shape == (n_query, 10)
    print(f"  Query shape: {query_embeddings.shape}")
    print(f"  Result indices shape: {indices.shape}")
    print(f"  Result distances shape: {distances.shape}")
    print(f"  Sample distances: {distances[0, :5].tolist()}")
    print("  [PASS]")

    # Get stats
    print("\n[Test 5] Get statistics")
    stats = builder.get_stats()
    print(f"  Index stats:")
    print(f"    Vectors: {stats['num_vectors']}")
    print(f"    Memory: {stats['memory_mb']:.2f} MB")
    print(f"    M: {stats['M']}")
    print("  [PASS]")

    # Save and load
    print("\n[Test 6] Save and load")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = os.path.join(tmpdir, "test_index")
        builder.save(index_path)

        # Load in new builder
        builder2 = HNSWIndexBuilder()
        builder2.load(index_path)
        assert builder2.num_vectors == builder.num_vectors

        # Search with loaded index
        indices2, distances2 = builder2.search(query_embeddings, k=10)
        assert np.allclose(distances, distances2, atol=1e-5)
        print(f"  Saved and loaded successfully")
        print(f"  Search results match: {np.allclose(distances, distances2)}")
    print("  [PASS]")

    # Test recall (self-search)
    print("\n[Test 7] Test recall (self-search)")
    # Query with vectors from the index
    test_queries = build_embeddings[:10]  # First 10 vectors
    indices_self, _ = builder.search(test_queries, k=5)
    # First result should be the query itself (ID 0-9)
    recall = sum(indices_self[i, 0] == i for i in range(10)) / 10
    print(f"  Self-search recall@1: {recall*100:.1f}%")
    assert recall >= 0.8, "Self-search recall should be high"
    print("  [PASS]")

    print("\n" + "="*50)
    print("[PASS] All HNSWIndexBuilder tests passed!")
    print("="*50)
    print(f"\nIndex configuration:")
    print(f"  M: {builder.M}")
    print(f"  ef_search: {builder.ef_search}")
    print(f"  Total vectors: {builder.num_vectors}")
    print(f"  Memory: {stats['memory_mb']:.2f} MB")


if __name__ == "__main__":
    test_hnsw_builder()
