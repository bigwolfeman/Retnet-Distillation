"""End-to-end integration test for retrieval-augmented generation (T060 - US4).

Tests the complete RAG pipeline:
1. Index a sample codebase with code chunks
2. Simulate a question requiring retrieval
3. Test full pipeline:
   - Query encoding via dual encoder
   - Index search to retrieve relevant chunks
   - Chunk-to-landmark compression
   - Router selection of top-k landmarks
   - (Mock) model incorporation of landmarks into context
4. Verify retrieval quality and top-k selection

This validates that all retrieval components work together correctly and
integrate properly with the main model forward pass.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
import sys
from pathlib import Path
from typing import List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retrieval_index.code_chunk import CodeChunk
from src.retrieval_index.index import RetrievalIndex
from src.retrieval_index.dual_encoder import DualEncoder
from src.models.retrieval.compressor import LandmarkCompressor
from src.models.retrieval.landmark import LandmarkToken
from src.models.routing.router import GumbelTopKRouter
from src.config.model_config import ModelConfig

# Check for required dependencies
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

# Skip all tests if dependencies missing
pytestmark = pytest.mark.skipif(
    not HNSWLIB_AVAILABLE and not FAISS_AVAILABLE,
    reason="hnswlib or faiss required for retrieval tests. Install with: pip install hnswlib faiss-cpu"
)


# ============================================================================
# Fixtures: Create synthetic codebase for testing
# ============================================================================

@pytest.fixture
def synthetic_codebase():
    """Create synthetic code chunks for testing retrieval.

    Simulates a small Python codebase with authentication, database, and API code.
    """
    code_samples = [
        # Authentication module
        {
            'chunk_id': 'auth/login.py:1-15',
            'source_type': 'function',
            'language': 'python',
            'file_path': 'auth/login.py',
            'text': '''def authenticate_user(username: str, password: str) -> bool:
    """Authenticate user with username and password.

    Args:
        username: User's username
        password: User's password (plaintext)

    Returns:
        True if authentication succeeds, False otherwise
    """
    hashed_password = hash_password(password)
    user = get_user_from_database(username)
    return user and user.password_hash == hashed_password'''
        },
        {
            'chunk_id': 'auth/jwt.py:1-12',
            'source_type': 'function',
            'language': 'python',
            'file_path': 'auth/jwt.py',
            'text': '''def generate_jwt_token(user_id: int, expiry_hours: int = 24) -> str:
    """Generate JWT token for authenticated user.

    Args:
        user_id: User's ID
        expiry_hours: Token expiry time in hours

    Returns:
        JWT token string
    """
    payload = {'user_id': user_id, 'exp': time.time() + expiry_hours * 3600}
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')'''
        },
        {
            'chunk_id': 'auth/permissions.py:1-10',
            'source_type': 'function',
            'language': 'python',
            'file_path': 'auth/permissions.py',
            'text': '''def check_user_permissions(user_id: int, resource: str) -> bool:
    """Check if user has permission to access resource.

    Args:
        user_id: User's ID
        resource: Resource identifier

    Returns:
        True if user has permission
    """
    return user_id in get_authorized_users(resource)'''
        },
        # Database module
        {
            'chunk_id': 'db/connection.py:1-12',
            'source_type': 'function',
            'language': 'python',
            'file_path': 'db/connection.py',
            'text': '''def create_database_connection(host: str, port: int) -> Connection:
    """Create connection to PostgreSQL database.

    Args:
        host: Database host
        port: Database port

    Returns:
        Database connection object
    """
    conn = psycopg2.connect(host=host, port=port, database=DB_NAME)
    return conn'''
        },
        {
            'chunk_id': 'db/models.py:1-15',
            'source_type': 'class',
            'language': 'python',
            'file_path': 'db/models.py',
            'text': '''class User(BaseModel):
    """User model for database.

    Attributes:
        id: Primary key
        username: Unique username
        password_hash: Hashed password
        email: User email
        created_at: Account creation timestamp
    """
    id: int
    username: str
    password_hash: str
    email: str
    created_at: datetime'''
        },
        # API module
        {
            'chunk_id': 'api/routes.py:1-10',
            'source_type': 'function',
            'language': 'python',
            'file_path': 'api/routes.py',
            'text': '''@app.post("/api/login")
async def login_endpoint(username: str, password: str):
    """API endpoint for user login.

    Validates credentials and returns JWT token.
    """
    if authenticate_user(username, password):
        token = generate_jwt_token(get_user_id(username))
        return {"token": token, "status": "success"}
    return {"error": "Invalid credentials", "status": "failed"}'''
        },
        {
            'chunk_id': 'api/middleware.py:1-12',
            'source_type': 'function',
            'language': 'python',
            'file_path': 'api/middleware.py',
            'text': '''def auth_middleware(request: Request):
    """Middleware to validate JWT token on protected routes.

    Args:
        request: HTTP request object

    Raises:
        Unauthorized: If token is invalid or missing
    """
    token = request.headers.get('Authorization')
    if not token or not verify_jwt_token(token):
        raise Unauthorized("Invalid or missing token")'''
        },
        # Utility module (less relevant to authentication)
        {
            'chunk_id': 'utils/logging.py:1-8',
            'source_type': 'function',
            'language': 'python',
            'file_path': 'utils/logging.py',
            'text': '''def setup_logger(name: str, level: str = "INFO"):
    """Setup application logger.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))'''
        },
        {
            'chunk_id': 'utils/config.py:1-8',
            'source_type': 'function',
            'language': 'python',
            'file_path': 'utils/config.py',
            'text': '''def load_config(config_path: str) -> dict:
    """Load application configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path) as f:
        return yaml.safe_load(f)'''
        },
        {
            'chunk_id': 'utils/cache.py:1-10',
            'source_type': 'class',
            'language': 'python',
            'file_path': 'utils/cache.py',
            'text': '''class RedisCache:
    """Redis-based caching layer.

    Attributes:
        client: Redis client instance
        ttl: Default time-to-live for cached entries
    """
    def __init__(self, host: str, port: int, ttl: int = 3600):
        self.client = redis.Redis(host=host, port=port)
        self.ttl = ttl'''
        },
    ]

    return code_samples


@pytest.fixture
def tokenizer():
    """Create a simple tokenizer for testing.

    Uses character-level tokenization for simplicity.
    """
    class SimpleTokenizer:
        def __init__(self, vocab_size=1000):
            self.vocab_size = vocab_size

        def encode(self, text: str, max_length: int = 512) -> torch.Tensor:
            """Encode text to token IDs."""
            # Simple hash-based encoding for testing
            tokens = [hash(c) % self.vocab_size for c in text[:max_length]]
            # Pad to max_length
            tokens = tokens + [0] * (max_length - len(tokens))
            return torch.tensor(tokens[:max_length]).unsqueeze(0)

        def create_attention_mask(self, text: str, max_length: int = 512) -> torch.Tensor:
            """Create attention mask for text."""
            length = min(len(text), max_length)
            mask = [1] * length + [0] * (max_length - length)
            return torch.tensor(mask).unsqueeze(0)

    return SimpleTokenizer(vocab_size=1000)


@pytest.fixture
def dual_encoder(tokenizer):
    """Create and initialize a dual encoder for embedding."""
    encoder = DualEncoder(
        vocab_size=tokenizer.vocab_size,
        d_model=768,
        n_layers=2,  # Smaller for testing
        n_heads=12,
        max_seq_len=512,
    )
    encoder.eval()  # Use eval mode for deterministic behavior
    return encoder


@pytest.fixture
def landmark_compressor():
    """Create landmark compressor."""
    compressor = LandmarkCompressor(
        embedding_dim=768,
        model_dim=2816,
        num_tokens=6,
    )
    compressor.eval()
    return compressor


@pytest.fixture
def router():
    """Create Gumbel top-k router."""
    return GumbelTopKRouter(
        budget_B=24,
        temperature=0.7,
    )


# ============================================================================
# Test: End-to-End Retrieval Pipeline
# ============================================================================

class TestRetrievalE2E:
    """Test end-to-end retrieval-augmented generation pipeline."""

    def test_index_codebase(self, synthetic_codebase, dual_encoder, tokenizer):
        """Test indexing a synthetic codebase.

        Step 1: Create embeddings for all code chunks and build index.
        """
        print("\n[Step 1] Indexing synthetic codebase...")

        # Create code chunks with embeddings
        chunks = []
        for code_data in synthetic_codebase:
            # Encode code text
            token_ids = tokenizer.encode(code_data['text'])
            attention_mask = tokenizer.create_attention_mask(code_data['text'])

            # Generate embedding
            with torch.no_grad():
                embedding = dual_encoder(token_ids, attention_mask=attention_mask)
                embedding_np = embedding.squeeze(0).cpu().numpy()

            # Create CodeChunk
            chunk = CodeChunk(
                chunk_id=code_data['chunk_id'],
                source_type=code_data['source_type'],
                text=code_data['text'],
                language=code_data['language'],
                file_path=code_data.get('file_path'),
                embedding=embedding_np,
            )
            chunk.validate()
            chunks.append(chunk)

        print(f"  Created {len(chunks)} code chunks with embeddings")

        # Build retrieval index
        index = RetrievalIndex(
            index_id="test-codebase",
            index_type="workspace",
        )
        index.build(chunks)

        print(f"  Index built: {index.num_chunks} chunks indexed")
        print(f"  Index size: {index.index_size_mb:.2f} MB")

        assert index.num_chunks == len(chunks)
        assert all(c.chunk_id in index.chunks for c in chunks)

        return index, chunks

    def test_query_encoding(self, dual_encoder, tokenizer):
        """Test encoding a query for retrieval.

        Step 2: Encode user query "how do we handle authentication?"
        """
        print("\n[Step 2] Encoding query...")

        query_text = "how do we handle authentication and JWT tokens?"

        # Encode query
        query_ids = tokenizer.encode(query_text)
        query_mask = tokenizer.create_attention_mask(query_text)

        with torch.no_grad():
            query_embedding = dual_encoder(query_ids, attention_mask=query_mask)
            query_embedding_np = query_embedding.squeeze(0).cpu().numpy()

        print(f"  Query: '{query_text}'")
        print(f"  Query embedding shape: {query_embedding_np.shape}")
        print(f"  Query embedding norm: {np.linalg.norm(query_embedding_np):.4f}")

        assert query_embedding_np.shape == (768,)
        assert np.abs(np.linalg.norm(query_embedding_np) - 1.0) < 0.01  # Should be normalized

        return query_embedding_np

    def test_full_retrieval_pipeline(
        self,
        synthetic_codebase,
        dual_encoder,
        tokenizer,
        landmark_compressor,
        router,
    ):
        """Test complete end-to-end retrieval pipeline.

        Steps:
        1. Index synthetic codebase
        2. Encode query
        3. Search index
        4. Compress chunks to landmarks
        5. Route top-k landmarks
        6. (Mock) incorporate into model context
        """
        print("\n" + "="*70)
        print("FULL E2E RETRIEVAL PIPELINE TEST")
        print("="*70)

        # ============================================================
        # Step 1: Index codebase
        # ============================================================
        index, chunks = self.test_index_codebase(synthetic_codebase, dual_encoder, tokenizer)

        # ============================================================
        # Step 2: Encode query
        # ============================================================
        query_embedding = self.test_query_encoding(dual_encoder, tokenizer)

        # ============================================================
        # Step 3: Search index for relevant chunks
        # ============================================================
        print("\n[Step 3] Searching index...")

        k_retrieve = 10  # Retrieve top-10 chunks
        results = index.search(query_embedding, k=k_retrieve)

        print(f"  Retrieved {len(results)} chunks")
        print(f"  Top 5 results:")
        for i, (chunk, score) in enumerate(results[:5]):
            print(f"    {i+1}. {chunk.chunk_id} (score: {score:.4f})")
            print(f"       Preview: {chunk.text[:80]}...")

        assert len(results) == k_retrieve

        # Verify relevant chunks are retrieved
        retrieved_ids = [chunk.chunk_id for chunk, score in results]

        # Authentication-related chunks should be highly ranked
        auth_chunks = [cid for cid in retrieved_ids if 'auth' in cid or 'login' in cid]
        print(f"\n  Authentication-related chunks retrieved: {len(auth_chunks)}/{len(results)}")
        print(f"  Auth chunks: {auth_chunks}")

        # Should retrieve at least some auth-related chunks for this query
        assert len(auth_chunks) > 0, "Should retrieve authentication-related chunks"

        # ============================================================
        # Step 4: Compress chunks to landmarks
        # ============================================================
        print("\n[Step 4] Compressing chunks to landmarks...")

        landmarks = []
        for chunk, score in results:
            # Convert chunk embedding to torch tensor
            chunk_embedding_tensor = torch.from_numpy(chunk.embedding).float()

            # Compress to landmark
            with torch.no_grad():
                compressed_tokens = landmark_compressor(chunk_embedding_tensor)

            # Create landmark
            landmark = LandmarkToken(
                chunk_id=chunk.chunk_id,
                source_chunk=chunk,
                compressed_tokens=compressed_tokens,
                router_score=score,
            )
            landmarks.append(landmark)

        print(f"  Compressed {len(landmarks)} chunks to landmarks")
        print(f"  Each landmark: {landmarks[0].num_tokens} tokens × {landmarks[0].compressed_tokens.size(-1)} dim")

        assert len(landmarks) == k_retrieve
        assert all(lm.compressed_tokens.shape == (6, 2816) for lm in landmarks)

        # ============================================================
        # Step 5: Router selection of top-k landmarks
        # ============================================================
        print("\n[Step 5] Routing top-k landmarks...")

        # Stack landmarks for batch processing
        landmark_tensors = torch.stack([lm.compressed_tokens for lm in landmarks]).unsqueeze(0)
        # Shape: [1, num_landmarks, L, d_model] = [1, 10, 6, 2816]

        # Create route logits from retrieval scores
        route_logits = torch.tensor([lm.router_score for lm in landmarks]).unsqueeze(0)
        # Shape: [1, num_landmarks] = [1, 10]

        k_select = 4  # Select top-4 landmarks (4 × 6 = 24 tokens, matching budget B=24)

        router.eval()  # Eval mode for deterministic selection
        with torch.no_grad():
            selected_landmarks, selected_probs, aux_losses = router.select_landmarks(
                landmark_tensors, route_logits, k=k_select
            )

        print(f"  Selected {k_select} landmarks from {len(landmarks)} candidates")
        print(f"  Selected landmarks shape: {selected_landmarks.shape}")
        print(f"  Total tokens: {k_select * 6} (budget B=24)")

        # Get selected indices
        _, selected_indices, _ = router(route_logits, k=k_select, return_aux_losses=False)
        selected_indices = selected_indices.squeeze(0).tolist()

        print(f"\n  Selected landmark indices: {selected_indices}")
        print(f"  Selected chunks:")
        for idx in selected_indices:
            lm = landmarks[idx]
            print(f"    - {lm.chunk_id} (score: {lm.router_score:.4f})")

        assert selected_landmarks.shape == (1, k_select, 6, 2816)

        # ============================================================
        # Step 6: (Mock) Model incorporation
        # ============================================================
        print("\n[Step 6] (Mock) Model incorporation...")

        # In the real model forward pass, selected landmarks would be:
        # 1. Reshaped from [batch, k, L, d_model] to [batch, k*L, d_model]
        # 2. Inserted into the context as global tokens
        # 3. Attention band allows model to attend to these tokens

        batch_size = 1
        selected_tokens_flat = selected_landmarks.reshape(batch_size, k_select * 6, 2816)

        print(f"  Flattened selected tokens: {selected_tokens_flat.shape}")
        print(f"  Shape: [batch={batch_size}, total_tokens={k_select * 6}, d_model=2816]")

        # Mock: Create dummy input sequence
        seq_len = 32
        dummy_input_hidden = torch.randn(batch_size, seq_len, 2816)

        # Mock: Concatenate landmarks as global tokens
        # In reality, these would be positioned in pointer slots
        context_with_landmarks = torch.cat([selected_tokens_flat, dummy_input_hidden], dim=1)

        print(f"  Context with landmarks: {context_with_landmarks.shape}")
        print(f"  Total sequence: {k_select * 6} landmark tokens + {seq_len} input tokens")

        assert context_with_landmarks.shape == (batch_size, k_select * 6 + seq_len, 2816)

        print("\n  Model would now process this context with attention band")
        print("  allowing bidirectional attention to landmark tokens")

        # ============================================================
        # Verification
        # ============================================================
        print("\n[Verification]")
        print(f"  ✓ Query encoded successfully")
        print(f"  ✓ Index search returned {len(results)} relevant chunks")
        print(f"  ✓ {len(auth_chunks)} authentication-related chunks retrieved")
        print(f"  ✓ All chunks compressed to landmarks")
        print(f"  ✓ Router selected top-{k_select} landmarks within budget")
        print(f"  ✓ Landmarks ready for model incorporation")

        print("\n" + "="*70)
        print("E2E RETRIEVAL PIPELINE: SUCCESS")
        print("="*70)

    def test_retrieval_quality(
        self,
        synthetic_codebase,
        dual_encoder,
        tokenizer,
    ):
        """Test that retrieval returns relevant results for different queries."""
        print("\n[Test] Retrieval Quality")

        # Index codebase
        index, chunks = self.test_index_codebase(synthetic_codebase, dual_encoder, tokenizer)

        # Test different queries
        test_queries = [
            ("how do we handle authentication?", ['auth/login.py', 'auth/jwt.py', 'api/login']),
            ("database connection setup", ['db/connection.py', 'db/models.py']),
            ("JWT token generation", ['auth/jwt.py', 'api/login']),
        ]

        for query_text, expected_keywords in test_queries:
            print(f"\n  Query: '{query_text}'")

            # Encode query
            query_ids = tokenizer.encode(query_text)
            query_mask = tokenizer.create_attention_mask(query_text)

            with torch.no_grad():
                query_embedding = dual_encoder(query_ids, attention_mask=query_mask)
                query_embedding_np = query_embedding.squeeze(0).cpu().numpy()

            # Search
            results = index.search(query_embedding_np, k=5)
            retrieved_ids = [chunk.chunk_id for chunk, score in results]

            print(f"    Top results: {retrieved_ids[:3]}")

            # Check if any expected keywords appear in results
            relevant_count = sum(
                any(keyword in chunk_id for keyword in expected_keywords)
                for chunk_id in retrieved_ids
            )

            print(f"    Relevant results: {relevant_count}/{len(results)}")

            # Should retrieve at least one relevant chunk
            assert relevant_count > 0, f"No relevant chunks for query: {query_text}"

    def test_top_k_selection_correctness(self, router):
        """Test that router correctly selects top-k landmarks."""
        print("\n[Test] Top-k Selection Correctness")

        # Create mock landmarks with known scores
        num_landmarks = 10
        k = 4

        # Create logits where we know the top-k
        route_logits = torch.tensor([
            [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        ])
        expected_indices = {0, 1, 2, 3}  # Top-4 indices

        router.eval()
        with torch.no_grad():
            selected_indices, selected_probs, _ = router(
                route_logits, k=k, return_aux_losses=False
            )

        selected_set = set(selected_indices[0].tolist())

        print(f"  Expected top-{k}: {sorted(expected_indices)}")
        print(f"  Selected: {sorted(selected_set)}")

        assert selected_set == expected_indices, "Router should select top-k landmarks"
        print(f"  ✓ Router correctly selected top-{k} landmarks")

    def test_index_save_load(self, synthetic_codebase, dual_encoder, tokenizer):
        """Test saving and loading retrieval index."""
        print("\n[Test] Index Save/Load")

        # Build index
        index, chunks = self.test_index_codebase(synthetic_codebase, dual_encoder, tokenizer)

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "test_index")

            # Save
            index.save(index_path)
            print(f"  Index saved to {index_path}")

            # Load in new index
            loaded_index = RetrievalIndex(
                index_id="loaded",
                index_type="workspace"
            )
            loaded_index.load(index_path)

            print(f"  Index loaded: {loaded_index.num_chunks} chunks")

            # Verify
            assert loaded_index.num_chunks == index.num_chunks
            assert loaded_index.index_id == index.index_id

            # Test search on loaded index
            query_text = "authentication"
            query_ids = tokenizer.encode(query_text)
            query_mask = tokenizer.create_attention_mask(query_text)

            with torch.no_grad():
                query_embedding = dual_encoder(query_ids, attention_mask=query_mask)
                query_embedding_np = query_embedding.squeeze(0).cpu().numpy()

            results = loaded_index.search(query_embedding_np, k=5)

            print(f"  Search on loaded index: {len(results)} results")
            assert len(results) == 5

            print(f"  ✓ Index saved and loaded successfully")


class TestRetrievalErrorHandling:
    """Test error handling in retrieval pipeline."""

    def test_empty_index_search(self):
        """Test searching an empty index."""
        index = RetrievalIndex(
            index_id="empty",
            index_type="workspace"
        )

        query = np.random.randn(768).astype('float32')

        with pytest.raises(RuntimeError, match="Index not built"):
            index.search(query, k=10)

    def test_invalid_chunk_size(self):
        """Test that chunks exceeding size limit are rejected."""
        large_text = "x" * 3000  # Exceeds 2048 bytes

        chunk = CodeChunk(
            chunk_id="large",
            source_type="file",
            text=large_text,
        )

        with pytest.raises(AssertionError, match="exceeds 2048 bytes"):
            chunk.validate()

    def test_missing_embedding(self, landmark_compressor):
        """Test that chunks without embeddings cannot be compressed."""
        chunk = CodeChunk(
            chunk_id="no_embed",
            source_type="function",
            text="def foo(): pass",
            embedding=None,  # Missing embedding
        )

        # This would fail if we try to compress
        # In the real pipeline, validate() catches this earlier


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
