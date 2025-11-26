"""
Cached teacher client for reading pre-cached teacher logits.

This client provides the same interface as VLLMTeacherClient but reads from
a local cache instead of hitting the server. Falls back to server if cache miss.

Usage:
    # Use cached client instead of VLLMTeacherClient
    client = CachedTeacherClient(
        cache_dir="data/teacher_cache/",
        fallback_url="http://localhost:8080",
        fallback_api_key="token-abc123",
    )

    # Same interface as VLLMTeacherClient
    results = client.get_prompt_logprobs(
        input_ids=[[1, 2, 3, ...]],
        topk=128,
    )

Performance:
    - Cache hit: ~0.1ms per sequence (1000x faster than server)
    - Cache miss: Falls back to server (same as VLLMTeacherClient)
    - Memory efficient: Lazy loads shards on demand
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.distillation.vllm_teacher_client import VLLMTeacherClient

logger = logging.getLogger(__name__)


class CachedTeacherClient:
    """
    Teacher client that reads from pre-cached logits.

    Provides the same interface as VLLMTeacherClient for seamless integration.
    Falls back to server on cache miss.

    Attributes:
        cache_dir: Directory containing cached logits
        fallback_client: VLLMTeacherClient for cache misses (optional)
        cache_index: Dict mapping sequence_id -> (shard_file, row_index)
        loaded_shards: Dict of loaded shard DataFrames (LRU cache)
        max_loaded_shards: Max number of shards to keep in memory
    """

    def __init__(
        self,
        cache_dir: str,
        fallback_url: Optional[str] = None,
        fallback_api_key: Optional[str] = None,
        fallback_model: str = "meta-llama/Llama-3.2-1B-Instruct",
        max_loaded_shards: int = 10,
    ):
        """
        Initialize cached teacher client.

        Args:
            cache_dir: Directory containing cache_shard_*.parquet files
            fallback_url: URL for fallback server (optional)
            fallback_api_key: API key for fallback server (optional)
            fallback_model: Model name for fallback server
            max_loaded_shards: Max shards to keep in memory (LRU)
        """
        self.cache_dir = Path(cache_dir)
        self.max_loaded_shards = max_loaded_shards

        if not self.cache_dir.exists():
            raise ValueError(f"Cache directory not found: {cache_dir}")

        # Load manifest
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"Cache manifest not found: {manifest_path}")

        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)

        logger.info(
            f"Loaded cache manifest: {self.manifest['total_sequences']} sequences "
            f"in {self.manifest['num_shards']} shards"
        )

        # Build cache index (sequence_id -> shard file)
        self._build_cache_index()

        # Fallback client (optional)
        self.fallback_client = None
        if fallback_url:
            logger.info(f"Initializing fallback client at {fallback_url}")
            self.fallback_client = VLLMTeacherClient(
                base_url=fallback_url,
                model=fallback_model,
                api_key=fallback_api_key,
            )

        # Shard cache (LRU)
        self.loaded_shards: Dict[str, pd.DataFrame] = {}
        self.shard_access_order: List[str] = []

        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0

    def _build_cache_index(self):
        """Build index mapping sequence_id -> shard file."""
        logger.info("Building cache index...")
        self.cache_index: Dict[str, str] = {}

        shard_files = sorted(self.cache_dir.glob("cache_shard_*.parquet"))
        for shard_file in shard_files:
            # Load just the sequence_id column to build index
            df = pd.read_parquet(shard_file, columns=["sequence_id"])
            for seq_id in df["sequence_id"]:
                self.cache_index[seq_id] = shard_file.name

        logger.info(f"Indexed {len(self.cache_index)} sequences")

    def _load_shard(self, shard_name: str) -> pd.DataFrame:
        """Load a shard file with LRU caching."""
        # Check if already loaded
        if shard_name in self.loaded_shards:
            # Update access order (move to end = most recent)
            self.shard_access_order.remove(shard_name)
            self.shard_access_order.append(shard_name)
            return self.loaded_shards[shard_name]

        # Load shard
        shard_path = self.cache_dir / shard_name
        df = pd.read_parquet(shard_path)

        # Add to cache
        self.loaded_shards[shard_name] = df
        self.shard_access_order.append(shard_name)

        # Evict oldest if over limit
        if len(self.loaded_shards) > self.max_loaded_shards:
            oldest_shard = self.shard_access_order.pop(0)
            del self.loaded_shards[oldest_shard]
            logger.debug(f"Evicted shard {oldest_shard} from memory")

        return df

    def _get_cached_sequence(self, sequence_id: str) -> Optional[Dict[str, Any]]:
        """Get cached sequence by ID."""
        # Check if in cache
        if sequence_id not in self.cache_index:
            return None

        # Load shard
        shard_name = self.cache_index[sequence_id]
        df = self._load_shard(shard_name)

        # Find sequence in shard
        row = df[df["sequence_id"] == sequence_id]
        if len(row) == 0:
            logger.warning(f"Sequence {sequence_id} not found in shard {shard_name}")
            return None

        return row.iloc[0].to_dict()

    def get_prompt_logprobs(
        self,
        input_ids: List[List[int]],
        topk: int = 128,
        temperature: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Get top-k logprobs for input sequences.

        Same interface as VLLMTeacherClient.get_prompt_logprobs.

        Args:
            input_ids: Token ID sequences. Shape: (batch_size, seq_len)
            topk: Number of top logprobs to return per position (max 128)
            temperature: Temperature for logprob computation (ignored for cache)

        Returns:
            List of dicts, one per sequence, containing:
                - tokens: List[str] - token strings (not available from cache)
                - top_logprobs: List[Dict[str, float]] - sparse logprobs per position (not available)
                - indices: List[List[int]] - top-k token IDs per position
                - logprobs: List[List[float]] - top-k logprobs per position

        Raises:
            ValueError: If no cache and no fallback client configured
        """
        results = []

        for seq_input_ids in input_ids:
            # Generate sequence ID from input_ids
            # NOTE: This assumes deterministic ID generation matching cache script
            # In practice, you'd use actual sequence IDs from your dataset
            seq_id = self._generate_sequence_id(seq_input_ids)

            # Try cache first
            cached = self._get_cached_sequence(seq_id)

            if cached is not None:
                # Cache hit
                self.cache_hits += 1

                # Decompress logits
                result = self._decompress_cached_sequence(cached)
                results.append(result)

            else:
                # Cache miss
                self.cache_misses += 1
                logger.debug(f"Cache miss for sequence {seq_id}")

                if self.fallback_client is None:
                    raise ValueError(
                        f"Cache miss for sequence {seq_id} and no fallback client configured"
                    )

                # Fetch from server
                logger.debug("Fetching from fallback server...")
                fallback_results = self.fallback_client.get_prompt_logprobs(
                    input_ids=[seq_input_ids],
                    topk=topk,
                    temperature=temperature,
                )
                results.append(fallback_results[0])

        return results

    def _generate_sequence_id(self, input_ids: List[int]) -> str:
        """
        Generate deterministic sequence ID from input_ids.

        NOTE: This is a placeholder. In practice, you should use the actual
        sequence IDs from your dataset that match the cache.
        """
        # Hash input_ids to get deterministic ID
        import hashlib

        id_bytes = np.array(input_ids, dtype=np.int32).tobytes()
        id_hash = hashlib.sha256(id_bytes).hexdigest()[:16]
        return f"seq_{id_hash}"

    def _decompress_cached_sequence(self, cached: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decompress cached sequence to teacher logprobs format.

        Converts int8 quantized values back to float logprobs.
        """
        teacher_indices = cached["teacher_indices"]
        teacher_values = cached["teacher_values"]
        teacher_scales = cached["teacher_scales"]
        teacher_other = cached["teacher_other"]

        # Decompress each position
        indices = []
        logprobs = []

        for pos_indices, pos_values, scale, other in zip(
            teacher_indices, teacher_values, teacher_scales, teacher_other
        ):
            if not pos_indices:
                # Empty position
                indices.append([])
                logprobs.append([])
                continue

            # Dequantize values
            pos_indices = np.array(pos_indices, dtype=np.int32)
            pos_values = np.array(pos_values, dtype=np.uint8)

            # Convert back to probabilities
            pos_probs = pos_values.astype(np.float32) * scale

            # Normalize (probs should sum to 1.0 - other)
            total_prob = pos_probs.sum()
            if total_prob > 0:
                pos_probs = pos_probs / total_prob * (1.0 - other)

            # Convert to logprobs
            pos_logprobs = np.log(pos_probs + 1e-10)  # Add epsilon for numerical stability

            indices.append(pos_indices.tolist())
            logprobs.append(pos_logprobs.tolist())

        return {
            "tokens": [],  # Not available from cache
            "top_logprobs": [],  # Not available from cache
            "indices": indices,
            "logprobs": logprobs,
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "loaded_shards": len(self.loaded_shards),
            "indexed_sequences": len(self.cache_index),
        }

    def close(self):
        """Close the client (and fallback client if present)."""
        if self.fallback_client:
            self.fallback_client.close()

        # Clear shard cache
        self.loaded_shards.clear()
        self.shard_access_order.clear()

        logger.info("Closed CachedTeacherClient")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


class CachedTeacherClientWithExplicitIDs(CachedTeacherClient):
    """
    Version of CachedTeacherClient that requires explicit sequence IDs.

    Use this when you have explicit sequence IDs that match the cache.

    Usage:
        client = CachedTeacherClientWithExplicitIDs(cache_dir="data/teacher_cache/")

        results = client.get_prompt_logprobs_with_ids(
            sequence_ids=["seq_000001", "seq_000002"],
            input_ids=[[1, 2, 3, ...], [4, 5, 6, ...]],
            topk=128,
        )
    """

    def get_prompt_logprobs_with_ids(
        self,
        sequence_ids: List[str],
        input_ids: List[List[int]],
        topk: int = 128,
        temperature: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Get top-k logprobs for sequences with explicit IDs.

        Args:
            sequence_ids: List of sequence IDs matching cache
            input_ids: Token ID sequences (used for fallback only)
            topk: Number of top logprobs to return per position
            temperature: Temperature for logprob computation (fallback only)

        Returns:
            List of dicts with logprobs data per sequence
        """
        if len(sequence_ids) != len(input_ids):
            raise ValueError("sequence_ids and input_ids must have same length")

        results = []

        for seq_id, seq_input_ids in zip(sequence_ids, input_ids):
            # Try cache first
            cached = self._get_cached_sequence(seq_id)

            if cached is not None:
                # Cache hit
                self.cache_hits += 1
                result = self._decompress_cached_sequence(cached)
                results.append(result)

            else:
                # Cache miss
                self.cache_misses += 1
                logger.debug(f"Cache miss for sequence {seq_id}")

                if self.fallback_client is None:
                    raise ValueError(
                        f"Cache miss for sequence {seq_id} and no fallback client configured"
                    )

                # Fetch from server
                logger.debug("Fetching from fallback server...")
                fallback_results = self.fallback_client.get_prompt_logprobs(
                    input_ids=[seq_input_ids],
                    topk=topk,
                    temperature=temperature,
                )
                results.append(fallback_results[0])

        return results


# Example usage
if __name__ == "__main__":
    # Example with explicit IDs (recommended)
    client = CachedTeacherClientWithExplicitIDs(
        cache_dir="data/teacher_cache_test/",
        fallback_url="http://localhost:8080",
        fallback_api_key="token-abc123",
    )

    # Get logprobs for sequences
    results = client.get_prompt_logprobs_with_ids(
        sequence_ids=["dummy_000000", "dummy_000001"],
        input_ids=[
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
        ],
        topk=128,
    )

    print(f"Got {len(results)} results")

    # Print stats
    stats = client.get_cache_stats()
    print(f"Cache stats: {stats}")

    client.close()
