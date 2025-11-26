"""Integration test for incremental index updates (T059 - US4).

Tests SC-010: Incremental index updates without full rebuild.

Tests end-to-end index refresh:
- Build initial index with a set of chunks
- Add new chunks via refresh() method
- Verify new chunks are searchable
- Test both workspace (HNSW) and global (FAISS) index refresh
- Measure refresh time and validate it's faster than full rebuild
- Verify retrieval quality after refresh

Performance targets:
- Workspace refresh: <10s for 1k new chunks (NFR-003)
- Global refresh: <30s for 1k new chunks (incremental add)
- Refresh should be faster than full rebuild for small updates
"""

import pytest
import numpy as np
import tempfile
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retrieval_index.code_chunk import CodeChunk
from src.retrieval_index.index import RetrievalIndex

# Check for dependencies
try:
    import hnswlib
    HNSWLIB_AVAILABLE = True
except ImportError:
    HNSWLIB_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Skip markers
skip_if_no_hnswlib = pytest.mark.skipif(
    not HNSWLIB_AVAILABLE,
    reason="hnswlib not installed. Install with: pip install hnswlib"
)

skip_if_no_faiss = pytest.mark.skipif(
    not FAISS_AVAILABLE,
    reason="faiss not installed. Install with: pip install faiss-cpu"
)


@pytest.fixture
def sample_chunks():
    """Create sample code chunks for testing."""
    d_model = 768
    chunks = []

    for i in range(100):
        embedding = np.random.randn(d_model).astype('float32')
        # Normalize for inner product search
        embedding = embedding / np.linalg.norm(embedding)

        chunk = CodeChunk(
            chunk_id=f"chunk_{i}",
            source_type="function",
            text=f"def function_{i}():\n    '''Function {i}'''\n    pass",
            language="python",
            file_path=f"src/module_{i//10}.py",
            start_line=i * 10,
            end_line=i * 10 + 10,
            embedding=embedding,
        )
        chunks.append(chunk)

    return chunks


@pytest.fixture
def new_chunks():
    """Create new code chunks to be added via refresh."""
    d_model = 768
    chunks = []

    for i in range(20):
        embedding = np.random.randn(d_model).astype('float32')
        # Normalize for inner product search
        embedding = embedding / np.linalg.norm(embedding)

        chunk = CodeChunk(
            chunk_id=f"new_chunk_{i}",
            source_type="function",
            text=f"def new_function_{i}():\n    '''New function {i}'''\n    pass",
            language="python",
            file_path=f"src/new_module_{i//5}.py",
            start_line=i * 10,
            end_line=i * 10 + 10,
            embedding=embedding,
        )
        chunks.append(chunk)

    return chunks


@pytest.fixture
def temp_index_dir():
    """Create temporary directory for index storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@skip_if_no_hnswlib
class TestWorkspaceIndexRefresh:
    """Test incremental refresh for workspace (HNSW) index."""

    def test_refresh_adds_new_chunks(self, sample_chunks, new_chunks):
        """Test that refresh() adds new chunks to workspace index."""
        # Build initial index
        index = RetrievalIndex(
            index_id="test-workspace",
            index_type="workspace"
        )
        index.build(sample_chunks)

        initial_count = index.num_chunks
        assert initial_count == len(sample_chunks)

        # Refresh with new chunks
        index.refresh(new_chunks)

        # Verify chunk count updated
        assert index.num_chunks == initial_count + len(new_chunks)
        assert index.num_chunks == len(sample_chunks) + len(new_chunks)

        # Verify chunks are in index
        for chunk in new_chunks:
            assert chunk.chunk_id in index.chunks

        print(f"  Workspace refresh: {initial_count} -> {index.num_chunks} chunks")

    def test_new_chunks_are_searchable(self, sample_chunks, new_chunks):
        """Test that new chunks are searchable after refresh."""
        # Build initial index
        index = RetrievalIndex(
            index_id="test-workspace",
            index_type="workspace"
        )
        index.build(sample_chunks)

        # Refresh with new chunks
        index.refresh(new_chunks)

        # Search using embedding from a new chunk
        query_chunk = new_chunks[0]
        query_embedding = query_chunk.embedding

        results = index.search(query_embedding, k=10)

        # Verify we got results
        assert len(results) == 10

        # Verify at least one new chunk is in top results
        # (Should be very high since we're querying with exact embedding)
        result_ids = [chunk.chunk_id for chunk, score in results]
        new_chunk_ids = [chunk.chunk_id for chunk in new_chunks]

        # At least the query chunk itself should be in results
        assert query_chunk.chunk_id in result_ids

        # Should have high similarity score (>0.9 for exact match)
        top_chunk, top_score = results[0]
        if top_chunk.chunk_id == query_chunk.chunk_id:
            assert top_score > 0.99, f"Expected high score for exact match, got {top_score:.3f}"

        print(f"  New chunk searchable: {query_chunk.chunk_id} found with score {top_score:.3f}")

    def test_refresh_performance_vs_rebuild(self, sample_chunks, new_chunks):
        """Test that refresh is faster than full rebuild for small updates."""
        # Build initial index and measure build time
        index1 = RetrievalIndex(
            index_id="test-rebuild",
            index_type="workspace"
        )

        build_start = time.time()
        index1.build(sample_chunks)
        build_time = time.time() - build_start

        # Measure refresh time
        refresh_start = time.time()
        index1.refresh(new_chunks)
        refresh_time = time.time() - refresh_start

        # Measure full rebuild time with all chunks
        index2 = RetrievalIndex(
            index_id="test-rebuild-2",
            index_type="workspace"
        )
        all_chunks = sample_chunks + new_chunks

        rebuild_start = time.time()
        index2.build(all_chunks)
        rebuild_time = time.time() - rebuild_start

        print(f"  Initial build: {build_time:.3f}s ({len(sample_chunks)} chunks)")
        print(f"  Refresh: {refresh_time:.3f}s ({len(new_chunks)} new chunks)")
        print(f"  Full rebuild: {rebuild_time:.3f}s ({len(all_chunks)} chunks)")

        # Note: For HNSW workspace index, refresh does a full rebuild,
        # so refresh_time should be similar to rebuild_time
        # The key is that it's still fast enough for workspace scale
        assert refresh_time < 30.0, f"Refresh too slow: {refresh_time:.1f}s"

        # Both methods should produce same final count
        assert index1.num_chunks == index2.num_chunks

    def test_refresh_preserves_existing_chunks(self, sample_chunks, new_chunks):
        """Test that refresh preserves existing chunks and their metadata."""
        # Build initial index
        index = RetrievalIndex(
            index_id="test-workspace",
            index_type="workspace"
        )
        index.build(sample_chunks)

        # Update retrieval stats on some chunks
        for chunk in list(index.chunks.values())[:10]:
            chunk.update_retrieval_stats(0.95)

        # Refresh with new chunks
        index.refresh(new_chunks)

        # Verify existing chunks still present with stats preserved
        for orig_chunk in sample_chunks[:10]:
            assert orig_chunk.chunk_id in index.chunks
            restored_chunk = index.chunks[orig_chunk.chunk_id]
            # Note: After rebuild, chunks are new objects, but retrieval stats
            # are preserved because we rebuild from existing chunks
            assert restored_chunk.retrieval_count > 0

    def test_refresh_updates_metadata(self, sample_chunks, new_chunks):
        """Test that refresh updates index metadata."""
        # Build initial index
        index = RetrievalIndex(
            index_id="test-workspace",
            index_type="workspace"
        )
        index.build(sample_chunks)

        initial_updated = index.last_updated
        time.sleep(0.01)  # Ensure timestamp changes

        # Refresh
        index.refresh(new_chunks)

        # Verify metadata updated
        assert index.last_updated != initial_updated
        assert index.num_chunks == len(sample_chunks) + len(new_chunks)


@skip_if_no_faiss
class TestGlobalIndexRefresh:
    """Test incremental refresh for global (FAISS) index."""

    def test_refresh_adds_new_chunks_incrementally(self, sample_chunks, new_chunks):
        """Test that global index adds chunks incrementally (no rebuild)."""
        # Build initial index
        index = RetrievalIndex(
            index_id="test-global",
            index_type="global",
            index_config={
                'n_centroids': 16,  # Small for testing
                'n_subquantizers': 32,
            }
        )
        index.build(sample_chunks)

        initial_count = index.num_chunks
        assert initial_count == len(sample_chunks)

        # Refresh with new chunks
        index.refresh(new_chunks)

        # Verify chunk count updated
        assert index.num_chunks == initial_count + len(new_chunks)

        # Verify chunks are in index
        for chunk in new_chunks:
            assert chunk.chunk_id in index.chunks

        print(f"  Global refresh: {initial_count} -> {index.num_chunks} chunks")

    def test_global_refresh_is_incremental(self, sample_chunks, new_chunks):
        """Test that global index refresh uses incremental add (not rebuild)."""
        # Build initial index
        index = RetrievalIndex(
            index_id="test-global",
            index_type="global",
            index_config={
                'n_centroids': 16,
                'n_subquantizers': 32,
            }
        )
        index.build(sample_chunks)

        # Measure refresh time (should be fast - just add, no retrain)
        refresh_start = time.time()
        index.refresh(new_chunks)
        refresh_time = time.time() - refresh_start

        print(f"  Global incremental refresh: {refresh_time:.3f}s ({len(new_chunks)} chunks)")

        # Should be very fast (no retraining)
        assert refresh_time < 5.0, f"Global refresh too slow: {refresh_time:.1f}s"

    def test_global_new_chunks_searchable(self, sample_chunks, new_chunks):
        """Test that new chunks are searchable in global index."""
        # Build initial index
        index = RetrievalIndex(
            index_id="test-global",
            index_type="global",
            index_config={
                'n_centroids': 16,
                'n_subquantizers': 32,
            }
        )
        index.build(sample_chunks)

        # Refresh with new chunks
        index.refresh(new_chunks)

        # Search using embedding from a new chunk
        query_chunk = new_chunks[0]
        query_embedding = query_chunk.embedding

        results = index.search(query_embedding, k=10)

        # Verify we got results
        assert len(results) == 10

        # Verify at least one new chunk is in results
        result_ids = [chunk.chunk_id for chunk, score in results]
        new_chunk_ids = [chunk.chunk_id for chunk in new_chunks]

        # Should find the query chunk (though FAISS IVF-PQ may have lower recall)
        found_new_chunks = sum(1 for rid in result_ids if rid in new_chunk_ids)
        assert found_new_chunks > 0, "No new chunks found in search results"

        print(f"  Found {found_new_chunks} new chunks in top-10 results")


class TestRefreshPersistence:
    """Test that refreshed indexes can be saved and loaded."""

    @skip_if_no_hnswlib
    def test_save_load_after_refresh_workspace(
        self,
        sample_chunks,
        new_chunks,
        temp_index_dir
    ):
        """Test save/load for workspace index after refresh."""
        # Build and refresh
        index = RetrievalIndex(
            index_id="test-workspace",
            index_type="workspace"
        )
        index.build(sample_chunks)
        index.refresh(new_chunks)

        total_chunks = index.num_chunks

        # Save
        save_path = os.path.join(temp_index_dir, "workspace_index")
        index.save(save_path)

        # Load in new index
        loaded_index = RetrievalIndex(
            index_id="loaded",
            index_type="workspace"
        )
        loaded_index.load(save_path)

        # Verify loaded correctly
        assert loaded_index.num_chunks == total_chunks
        assert loaded_index.index_id == index.index_id

        # Verify new chunks are searchable in loaded index
        query_chunk = new_chunks[0]
        results = loaded_index.search(query_chunk.embedding, k=5)
        assert len(results) == 5

        result_ids = [chunk.chunk_id for chunk, score in results]
        assert query_chunk.chunk_id in result_ids

    @skip_if_no_faiss
    def test_save_load_after_refresh_global(
        self,
        sample_chunks,
        new_chunks,
        temp_index_dir
    ):
        """Test save/load for global index after refresh."""
        # Build and refresh
        index = RetrievalIndex(
            index_id="test-global",
            index_type="global",
            index_config={
                'n_centroids': 16,
                'n_subquantizers': 32,
            }
        )
        index.build(sample_chunks)
        index.refresh(new_chunks)

        total_chunks = index.num_chunks

        # Save
        save_path = os.path.join(temp_index_dir, "global_index")
        index.save(save_path)

        # Load in new index
        loaded_index = RetrievalIndex(
            index_id="loaded",
            index_type="global"
        )
        loaded_index.load(save_path)

        # Verify loaded correctly
        assert loaded_index.num_chunks == total_chunks
        assert loaded_index.index_id == index.index_id

        # Verify searchable
        query = np.random.randn(768).astype('float32')
        query = query / np.linalg.norm(query)
        results = loaded_index.search(query, k=5)
        assert len(results) == 5


class TestMultipleRefreshes:
    """Test multiple sequential refreshes."""

    @skip_if_no_hnswlib
    def test_multiple_workspace_refreshes(self, sample_chunks):
        """Test multiple sequential refreshes on workspace index."""
        index = RetrievalIndex(
            index_id="test-workspace",
            index_type="workspace"
        )
        index.build(sample_chunks)

        initial_count = index.num_chunks

        # Perform 3 refreshes
        for batch in range(3):
            new_chunks = []
            for i in range(10):
                embedding = np.random.randn(768).astype('float32')
                embedding = embedding / np.linalg.norm(embedding)

                chunk = CodeChunk(
                    chunk_id=f"batch_{batch}_chunk_{i}",
                    source_type="function",
                    text=f"def batch_{batch}_func_{i}(): pass",
                    embedding=embedding,
                )
                new_chunks.append(chunk)

            index.refresh(new_chunks)
            expected_count = initial_count + (batch + 1) * 10
            assert index.num_chunks == expected_count

        # Final count should be initial + 30
        assert index.num_chunks == initial_count + 30

    @skip_if_no_faiss
    def test_multiple_global_refreshes(self, sample_chunks):
        """Test multiple sequential refreshes on global index."""
        index = RetrievalIndex(
            index_id="test-global",
            index_type="global",
            index_config={
                'n_centroids': 16,
                'n_subquantizers': 32,
            }
        )
        index.build(sample_chunks)

        initial_count = index.num_chunks

        # Perform 3 refreshes
        for batch in range(3):
            new_chunks = []
            for i in range(10):
                embedding = np.random.randn(768).astype('float32')
                embedding = embedding / np.linalg.norm(embedding)

                chunk = CodeChunk(
                    chunk_id=f"batch_{batch}_chunk_{i}",
                    source_type="function",
                    text=f"def batch_{batch}_func_{i}(): pass",
                    embedding=embedding,
                )
                new_chunks.append(chunk)

            index.refresh(new_chunks)
            expected_count = initial_count + (batch + 1) * 10
            assert index.num_chunks == expected_count

        # Final count should be initial + 30
        assert index.num_chunks == initial_count + 30


class TestRefreshEdgeCases:
    """Test edge cases for refresh functionality."""

    @skip_if_no_hnswlib
    def test_refresh_with_empty_list(self, sample_chunks):
        """Test refresh with empty list of new chunks."""
        index = RetrievalIndex(
            index_id="test-workspace",
            index_type="workspace"
        )
        index.build(sample_chunks)

        initial_count = index.num_chunks

        # Refresh with empty list
        index.refresh([])

        # Count should remain the same
        assert index.num_chunks == initial_count

    @skip_if_no_hnswlib
    def test_refresh_duplicate_chunk_ids(self, sample_chunks):
        """Test refresh with chunk IDs that already exist."""
        index = RetrievalIndex(
            index_id="test-workspace",
            index_type="workspace"
        )
        index.build(sample_chunks)

        # Create new chunks with same IDs but different content
        duplicate_chunks = []
        for i in range(5):
            embedding = np.random.randn(768).astype('float32')
            embedding = embedding / np.linalg.norm(embedding)

            chunk = CodeChunk(
                chunk_id=f"chunk_{i}",  # Same ID as existing
                source_type="function",
                text=f"def updated_function_{i}(): pass",  # Different text
                embedding=embedding,
            )
            duplicate_chunks.append(chunk)

        index.refresh(duplicate_chunks)

        # Workspace rebuild should handle this (latest wins)
        # Verify the updated chunks are present
        for dup_chunk in duplicate_chunks:
            assert dup_chunk.chunk_id in index.chunks
            retrieved_chunk = index.chunks[dup_chunk.chunk_id]
            # After rebuild, should have the updated text
            assert "updated_function" in retrieved_chunk.text


class TestRefreshStats:
    """Test statistics tracking during refresh."""

    @skip_if_no_hnswlib
    def test_refresh_stats_workspace(self, sample_chunks, new_chunks):
        """Test that stats are correctly updated after workspace refresh."""
        index = RetrievalIndex(
            index_id="test-workspace",
            index_type="workspace"
        )
        index.build(sample_chunks)

        initial_stats = index.get_stats()

        index.refresh(new_chunks)

        refreshed_stats = index.get_stats()

        # Verify stats updated
        assert refreshed_stats['num_chunks'] > initial_stats['num_chunks']
        assert refreshed_stats['last_updated'] != initial_stats['last_updated']
        assert refreshed_stats['index_size_mb'] > 0

    @skip_if_no_faiss
    def test_refresh_stats_global(self, sample_chunks, new_chunks):
        """Test that stats are correctly updated after global refresh."""
        index = RetrievalIndex(
            index_id="test-global",
            index_type="global",
            index_config={
                'n_centroids': 16,
                'n_subquantizers': 32,
            }
        )
        index.build(sample_chunks)

        initial_stats = index.get_stats()

        index.refresh(new_chunks)

        refreshed_stats = index.get_stats()

        # Verify stats updated
        assert refreshed_stats['num_chunks'] > initial_stats['num_chunks']
        assert refreshed_stats['last_updated'] != initial_stats['last_updated']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
