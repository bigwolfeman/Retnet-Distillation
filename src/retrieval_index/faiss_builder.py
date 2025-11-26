"""FAISS index builder for global knowledge retrieval (T047 - US4).

Builds FAISS IVF-PQ index for large-scale code retrieval. Used for global
knowledge base (general code patterns, libraries, documentation).

Index configuration from research.md:
- IVF1024: Inverted file index with 1024 centroids
- PQ64: Product quantization with 64 subquantizers
- nprobe=32: Search 32 nearest centroids

Requires ≥30k embeddings for training (FAISS recommendation).
"""

import numpy as np
from typing import List, Tuple, Optional
import os
import pickle


class FAISSIndexBuilder:
    """
    FAISS IVF-PQ index builder for global code retrieval.

    Builds and manages a FAISS index for large-scale retrieval (10k-1M+ chunks).
    Uses IndexIVFPQ for memory-efficient approximate search.

    Args:
        d_model: Embedding dimension (default: 768)
        n_centroids: Number of IVF centroids (default: 1024)
        n_subquantizers: Number of PQ subquantizers (default: 64)
        bits_per_subquantizer: Bits per subquantizer (default: 8)
        nprobe: Number of centroids to search (default: 32)
        metric: Distance metric ("ip" for inner product, "l2" for L2)

    Examples:
        >>> builder = FAISSIndexBuilder(d_model=768)
        >>> embeddings = np.random.randn(50000, 768).astype('float32')
        >>> builder.train(embeddings)
        >>> builder.add(embeddings)
        >>> query = np.random.randn(1, 768).astype('float32')
        >>> indices, distances = builder.search(query, k=10)
    """

    def __init__(
        self,
        d_model: int = 768,
        n_centroids: int = 1024,
        n_subquantizers: int = 64,
        bits_per_subquantizer: int = 8,
        nprobe: int = 32,
        metric: str = "ip",  # "ip" (inner product) or "l2"
    ):
        self.d_model = d_model
        self.n_centroids = n_centroids
        self.n_subquantizers = n_subquantizers
        self.bits_per_subquantizer = bits_per_subquantizer
        self.nprobe = nprobe
        self.metric = metric

        self.index = None
        self.is_trained = False
        self.num_vectors = 0

    def train(self, embeddings: np.ndarray):
        """
        Train FAISS index on embeddings.

        Trains the IVF quantizer and PQ codebook. Requires ≥30k embeddings
        for best quality (FAISS recommendation).

        Args:
            embeddings: Training embeddings [n_train, d_model] (float32)

        Raises:
            AssertionError: If embeddings have wrong shape
            ImportError: If faiss is not installed
            RuntimeError: If training fails

        Examples:
            >>> builder = FAISSIndexBuilder()
            >>> train_data = np.random.randn(50000, 768).astype('float32')
            >>> builder.train(train_data)
            >>> assert builder.is_trained
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu or faiss-gpu is required. Install with: "
                "pip install faiss-cpu (or faiss-gpu for GPU support)"
            )

        n_train, d = embeddings.shape
        assert d == self.d_model, f"Embedding dim mismatch: {d} vs {self.d_model}"
        assert embeddings.dtype == np.float32, "Embeddings must be float32"

        if n_train < 30000:
            print(f"Warning: Training with {n_train} vectors (FAISS recommends ≥30k for best quality)")

        print(f"Training FAISS IVF{self.n_centroids},PQ{self.n_subquantizers} index...")
        print(f"  Embeddings: {n_train} x {d}")
        print(f"  Metric: {self.metric}")

        # Create quantizer (for IVF)
        if self.metric == "ip":
            quantizer = faiss.IndexFlatIP(d)
        else:
            quantizer = faiss.IndexFlatL2(d)

        # Create IVF-PQ index
        self.index = faiss.IndexIVFPQ(
            quantizer,
            d,
            self.n_centroids,
            self.n_subquantizers,
            self.bits_per_subquantizer
        )

        # Train index
        print("  Training IVF quantizer and PQ codebook...")
        self.index.train(embeddings)
        self.is_trained = True

        # Set search parameters
        self.index.nprobe = self.nprobe

        print(f"  Training complete. nprobe={self.nprobe}")

    def add(self, embeddings: np.ndarray):
        """
        Add embeddings to index.

        Args:
            embeddings: Embeddings to add [n_add, d_model] (float32)

        Raises:
            RuntimeError: If index not trained

        Examples:
            >>> builder.train(train_data)
            >>> builder.add(train_data)
            >>> print(builder.num_vectors)  # 50000
        """
        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding vectors")

        n_add, d = embeddings.shape
        assert d == self.d_model, f"Embedding dim mismatch: {d} vs {self.d_model}"
        assert embeddings.dtype == np.float32, "Embeddings must be float32"

        print(f"Adding {n_add} vectors to index...")
        self.index.add(embeddings)
        self.num_vectors += n_add
        print(f"  Total vectors in index: {self.num_vectors}")

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
            distances: [n_queries, k] distances/similarities
            indices: [n_queries, k] indices of neighbors

        Examples:
            >>> query = np.random.randn(1, 768).astype('float32')
            >>> distances, indices = builder.search(query, k=10)
            >>> print(distances.shape)  # (1, 10)
            >>> print(indices.shape)    # (1, 10)
        """
        if not self.is_trained:
            raise RuntimeError("Index must be trained before searching")

        n_queries, d = queries.shape
        assert d == self.d_model, f"Query dim mismatch: {d} vs {self.d_model}"
        assert embeddings.dtype == np.float32, "Queries must be float32"

        distances, indices = self.index.search(queries, k)
        return distances, indices

    def save(self, filepath: str):
        """
        Save index to disk.

        Args:
            filepath: Path to save index (will create .faiss file)

        Examples:
            >>> builder.save("global_index.faiss")
            >>> # Later: builder.load("global_index.faiss")
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained index")

        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu or faiss-gpu required")

        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.faiss")

        # Save metadata
        metadata = {
            'd_model': self.d_model,
            'n_centroids': self.n_centroids,
            'n_subquantizers': self.n_subquantizers,
            'bits_per_subquantizer': self.bits_per_subquantizer,
            'nprobe': self.nprobe,
            'metric': self.metric,
            'num_vectors': self.num_vectors,
            'is_trained': self.is_trained,
        }

        with open(f"{filepath}.meta", 'wb') as f:
            pickle.dump(metadata, f)

        print(f"Index saved to {filepath}.faiss ({self.num_vectors} vectors)")

    def load(self, filepath: str):
        """
        Load index from disk.

        Args:
            filepath: Path to load index from (without .faiss extension)

        Examples:
            >>> builder = FAISSIndexBuilder()
            >>> builder.load("global_index")
            >>> print(builder.num_vectors)
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu or faiss-gpu required")

        # Load metadata
        with open(f"{filepath}.meta", 'rb') as f:
            metadata = pickle.load(f)

        self.d_model = metadata['d_model']
        self.n_centroids = metadata['n_centroids']
        self.n_subquantizers = metadata['n_subquantizers']
        self.bits_per_subquantizer = metadata['bits_per_subquantizer']
        self.nprobe = metadata['nprobe']
        self.metric = metadata['metric']
        self.num_vectors = metadata['num_vectors']
        self.is_trained = metadata['is_trained']

        # Load FAISS index
        self.index = faiss.read_index(f"{filepath}.faiss")
        self.index.nprobe = self.nprobe

        print(f"Index loaded from {filepath}.faiss ({self.num_vectors} vectors)")

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
            'n_centroids': self.n_centroids,
            'n_subquantizers': self.n_subquantizers,
            'nprobe': self.nprobe,
            'metric': self.metric,
            'num_vectors': self.num_vectors,
            'is_trained': self.is_trained,
        }

        # Estimate memory usage
        if self.is_trained:
            # Rough estimate: PQ uses bits_per_subquantizer * n_subquantizers bits per vector
            bytes_per_vector = (self.bits_per_subquantizer * self.n_subquantizers) // 8
            index_size_bytes = self.num_vectors * bytes_per_vector
            # Add overhead for centroids
            centroid_size_bytes = self.n_centroids * self.d_model * 4  # float32
            total_bytes = index_size_bytes + centroid_size_bytes
            stats['memory_mb'] = total_bytes / (1024 * 1024)
        else:
            stats['memory_mb'] = 0

        return stats


def test_faiss_builder():
    """Test FAISSIndexBuilder implementation."""
    print("Testing FAISSIndexBuilder...")

    try:
        import faiss
    except ImportError:
        print("FAISS not installed. Skipping tests.")
        print("Install with: pip install faiss-cpu")
        return

    # Configuration
    d_model = 768
    n_train = 5000  # Smaller for testing (normally ≥30k)
    n_add = 1000
    n_query = 10

    # Create builder
    print("\n[Test 1] Create builder")
    builder = FAISSIndexBuilder(
        d_model=d_model,
        n_centroids=128,  # Smaller for testing (normally 1024)
        n_subquantizers=64,
        nprobe=8,  # Smaller for testing (normally 32)
        metric="ip"
    )
    print(f"  Builder created:")
    print(f"    d_model={builder.d_model}")
    print(f"    n_centroids={builder.n_centroids}")
    print(f"    nprobe={builder.nprobe}")
    print("  [PASS]")

    # Train index
    print("\n[Test 2] Train index")
    train_embeddings = np.random.randn(n_train, d_model).astype('float32')
    # Normalize for inner product
    train_embeddings = train_embeddings / np.linalg.norm(train_embeddings, axis=1, keepdims=True)
    builder.train(train_embeddings)
    assert builder.is_trained
    print("  [PASS]")

    # Add vectors
    print("\n[Test 3] Add vectors")
    add_embeddings = np.random.randn(n_add, d_model).astype('float32')
    add_embeddings = add_embeddings / np.linalg.norm(add_embeddings, axis=1, keepdims=True)
    builder.add(add_embeddings)
    assert builder.num_vectors == n_add
    print(f"  Added {n_add} vectors")
    print("  [PASS]")

    # Search
    print("\n[Test 4] Search")
    query_embeddings = np.random.randn(n_query, d_model).astype('float32')
    query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    distances, indices = builder.search(query_embeddings, k=10)
    assert distances.shape == (n_query, 10)
    assert indices.shape == (n_query, 10)
    print(f"  Query shape: {query_embeddings.shape}")
    print(f"  Result distances shape: {distances.shape}")
    print(f"  Result indices shape: {indices.shape}")
    print(f"  Sample distances: {distances[0, :5].tolist()}")
    print("  [PASS]")

    # Get stats
    print("\n[Test 5] Get statistics")
    stats = builder.get_stats()
    print(f"  Index stats:")
    print(f"    Vectors: {stats['num_vectors']}")
    print(f"    Memory: {stats['memory_mb']:.2f} MB")
    print(f"    Trained: {stats['is_trained']}")
    print("  [PASS]")

    # Save and load
    print("\n[Test 6] Save and load")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = os.path.join(tmpdir, "test_index")
        builder.save(index_path)

        # Load in new builder
        builder2 = FAISSIndexBuilder()
        builder2.load(index_path)
        assert builder2.num_vectors == builder.num_vectors
        assert builder2.is_trained == builder.is_trained

        # Search with loaded index
        distances2, indices2 = builder2.search(query_embeddings, k=10)
        assert np.allclose(distances, distances2, atol=1e-5)
        print(f"  Saved and loaded successfully")
        print(f"  Search results match: {np.allclose(distances, distances2)}")
    print("  [PASS]")

    print("\n" + "="*50)
    print("[PASS] All FAISSIndexBuilder tests passed!")
    print("="*50)
    print(f"\nIndex configuration:")
    print(f"  IVF centroids: {builder.n_centroids}")
    print(f"  PQ subquantizers: {builder.n_subquantizers}")
    print(f"  nprobe: {builder.nprobe}")
    print(f"  Total vectors: {builder.num_vectors}")
    print(f"  Memory: {stats['memory_mb']:.2f} MB")


if __name__ == "__main__":
    test_faiss_builder()
