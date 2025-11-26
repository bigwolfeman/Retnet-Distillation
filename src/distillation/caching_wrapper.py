"""
Caching wrapper that adds logit caching to any teacher client.

This wrapper can wrap ANY teacher client (Direct, VLLM, network) and add
transparent caching. The wrapped client behaves identically but also writes
logits to disk in the background.

Use cases:
- Cache while training: Train with DirectTeacherClient + CachingWrapper
- Cache from network: Use VLLMTeacherClient + CachingWrapper to cache remote logits
- One-time caching: Cache all logits, then switch to CachedTeacherClient

Example:
    # Wrap DirectTeacherClient
    teacher = DirectTeacherClient("meta-llama/Llama-3.2-1B")
    teacher = CachingTeacherWrapper(teacher, "data/cache/")

    # Now every get_top_k_logits() call will also write to cache
    results = teacher.get_top_k_logits(input_ids=[[1, 2, 3, ...]])

    # Wrap VLLMTeacherClient
    teacher = VLLMTeacherClient("http://server:8080", "llama-1b")
    teacher = CachingTeacherWrapper(teacher, "data/cache/")

Architecture:
    CachingTeacherWrapper wraps any client with get_top_k_logits() or get_prompt_logprobs().
    It calls the wrapped client, gets results, writes to cache, and returns results.

    The cache format matches cache_teacher_logits.py for compatibility with
    CachedTeacherClient.
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CachingTeacherWrapper:
    """
    Wrapper that adds caching to any teacher client.

    Wraps any teacher client with get_top_k_logits() or get_prompt_logprobs()
    and transparently caches results to disk.

    Attributes:
        teacher: Wrapped teacher client
        cache_dir: Directory to write cache files
        shard_size: Number of sequences per shard file
        current_shard: Current shard number
        buffer: Buffer of sequences to write
        total_cached: Total sequences cached
        manifest: Cache manifest
    """

    def __init__(
        self,
        teacher_client: Any,
        cache_dir: str,
        shard_size: int = 1000,
        auto_flush: bool = True,
    ):
        """
        Initialize caching wrapper.

        Args:
            teacher_client: Teacher client to wrap (must have get_top_k_logits or get_prompt_logprobs)
            cache_dir: Directory to write cache files
            shard_size: Number of sequences per shard file
            auto_flush: Auto-flush buffer when it reaches shard_size

        Raises:
            ValueError: If teacher_client doesn't have required methods
        """
        # Validate teacher client has required methods
        if not (hasattr(teacher_client, 'get_top_k_logits') or
                hasattr(teacher_client, 'get_prompt_logprobs')):
            raise ValueError(
                "teacher_client must have get_top_k_logits() or get_prompt_logprobs() method"
            )

        self.teacher = teacher_client
        self.cache_dir = Path(cache_dir)
        self.shard_size = shard_size
        self.auto_flush = auto_flush

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize state
        self.current_shard = 0
        self.buffer = []
        self.total_cached = 0

        # Load existing manifest if present
        self.manifest_path = self.cache_dir / "manifest.json"
        if self.manifest_path.exists():
            with open(self.manifest_path, "r") as f:
                self.manifest = json.load(f)
            self.current_shard = self.manifest.get("num_shards", 0)
            self.total_cached = self.manifest.get("total_sequences", 0)
            logger.info(f"Resuming cache: {self.total_cached} sequences, {self.current_shard} shards")
        else:
            self.manifest = {
                "total_sequences": 0,
                "num_shards": 0,
                "shard_files": [],
                "topk": 128,
                "timestamp": time.time(),
            }

        logger.info(f"CachingTeacherWrapper initialized: {cache_dir}")
        logger.info(f"  Shard size: {shard_size}")
        logger.info(f"  Auto flush: {auto_flush}")

    @property
    def device(self):
        """Pass-through device attribute from wrapped client (for async gating logic)."""
        return getattr(self.teacher, 'device', None)

    def get_top_k_logits(
        self,
        input_ids: List[List[int]],
        topk: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get top-k logits and cache them.

        Args:
            input_ids: List of token ID sequences
            topk: Number of top logits to return

        Returns:
            List of dicts with logprobs data per sequence
        """
        # Get logits from wrapped teacher
        if hasattr(self.teacher, 'get_top_k_logits'):
            results = self.teacher.get_top_k_logits(input_ids=input_ids, topk=topk)
        elif hasattr(self.teacher, 'get_prompt_logprobs'):
            # Fall back to get_prompt_logprobs if get_top_k_logits not available
            topk = topk or 128
            results = self.teacher.get_prompt_logprobs(input_ids=input_ids, topk=topk)
        else:
            raise RuntimeError("Wrapped teacher has no get_top_k_logits or get_prompt_logprobs")

        # Cache results
        self._cache_batch(input_ids, results)

        return results

    def get_prompt_logprobs(
        self,
        input_ids: List[List[int]],
        topk: int = 128,
        temperature: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Get prompt logprobs and cache them.

        Alias for get_top_k_logits() to match VLLMTeacherClient API.

        Args:
            input_ids: List of token ID sequences
            topk: Number of top logits to return
            temperature: Temperature (passed to wrapped client)

        Returns:
            List of dicts with logprobs data per sequence
        """
        # Get logits from wrapped teacher
        if hasattr(self.teacher, 'get_prompt_logprobs'):
            results = self.teacher.get_prompt_logprobs(
                input_ids=input_ids,
                topk=topk,
                temperature=temperature,
            )
        elif hasattr(self.teacher, 'get_top_k_logits'):
            # Fall back to get_top_k_logits (ignore temperature)
            if temperature != 1.0:
                logger.warning(
                    f"Temperature={temperature} ignored by wrapped client. "
                    "Apply temperature in loss function."
                )
            results = self.teacher.get_top_k_logits(input_ids=input_ids, topk=topk)
        else:
            raise RuntimeError("Wrapped teacher has no get_prompt_logprobs or get_top_k_logits")

        # Cache results
        self._cache_batch(input_ids, results)

        return results

    def _cache_batch(
        self,
        input_ids: List[List[int]],
        results: List[Dict[str, Any]],
    ):
        """
        Cache a batch of results.

        Args:
            input_ids: Original input IDs
            results: Results from teacher
        """
        # Convert each result to cache format
        for seq_input_ids, result in zip(input_ids, results):
            # Generate sequence ID (hash of input_ids)
            seq_id = self._generate_sequence_id(seq_input_ids)

            # Convert to cache format
            cached_seq = self._convert_to_cache_format(seq_id, seq_input_ids, result)

            # Add to buffer
            self.buffer.append(cached_seq)

        # Auto-flush if buffer is full
        if self.auto_flush and len(self.buffer) >= self.shard_size:
            self.flush()

    def _generate_sequence_id(self, input_ids: List[int]) -> str:
        """
        Generate deterministic sequence ID from input_ids.

        Args:
            input_ids: Input token IDs

        Returns:
            Sequence ID string
        """
        id_bytes = np.array(input_ids, dtype=np.int32).tobytes()
        id_hash = hashlib.sha256(id_bytes).hexdigest()[:16]
        return f"seq_{id_hash}"

    def _convert_to_cache_format(
        self,
        seq_id: str,
        input_ids: List[int],
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Convert result to cache format with int8 quantization.

        Args:
            seq_id: Sequence ID
            input_ids: Input token IDs
            result: Result from teacher

        Returns:
            Dict in cache format
        """
        indices = result["indices"]
        logprobs = result["logprobs"]

        teacher_indices = []
        teacher_values = []
        teacher_scales = []
        teacher_other = []

        for position_indices, position_logprobs in zip(indices, logprobs):
            if not position_indices:
                # Empty position
                teacher_indices.append([])
                teacher_values.append([])
                teacher_scales.append(0.0)
                teacher_other.append(0.0)
                continue

            # Convert to numpy
            position_indices = np.array(position_indices, dtype=np.int32)
            position_logprobs = np.array(position_logprobs, dtype=np.float32)

            # Convert to probabilities
            position_probs = np.exp(position_logprobs)
            total_prob = position_probs.sum()

            # Other mass (probability not in top-k)
            other_mass = max(0.0, 1.0 - total_prob)

            # Normalize to sum to 1.0
            if total_prob > 0:
                position_probs = position_probs / total_prob * (1.0 - other_mass)

            # Quantize to int8 (0-255)
            max_prob = position_probs.max()
            if max_prob > 0:
                scale_factor = max_prob / 255.0
                quantized_values = np.round(position_probs / scale_factor).astype(np.uint8)
            else:
                scale_factor = 1.0
                quantized_values = np.zeros(len(position_probs), dtype=np.uint8)

            teacher_indices.append(position_indices.tolist())
            teacher_values.append(quantized_values.tolist())
            teacher_scales.append(float(scale_factor))
            teacher_other.append(float(other_mass))

        return {
            "sequence_id": seq_id,
            "input_ids": input_ids,
            "teacher_indices": teacher_indices,
            "teacher_values": teacher_values,
            "teacher_scales": teacher_scales,
            "teacher_other": teacher_other,
        }

    def flush(self):
        """
        Flush buffer to disk as a shard file.

        Saves current buffer as a parquet file and clears buffer.
        """
        if not self.buffer:
            logger.debug("Buffer empty, nothing to flush")
            return

        # Save shard
        shard_path = self.cache_dir / f"cache_shard_{self.current_shard:04d}.parquet"
        logger.info(f"Flushing buffer to {shard_path} ({len(self.buffer)} sequences)...")

        try:
            # Convert buffer to DataFrame
            df = pd.DataFrame(self.buffer)

            # Save with compression
            df.to_parquet(
                shard_path,
                engine="pyarrow",
                compression="snappy",
                index=False,
            )

            # Update state
            self.total_cached += len(self.buffer)
            self.current_shard += 1
            self.buffer = []

            # Update manifest
            self.manifest["total_sequences"] = self.total_cached
            self.manifest["num_shards"] = self.current_shard
            if shard_path.name not in self.manifest["shard_files"]:
                self.manifest["shard_files"].append(shard_path.name)
            self.manifest["timestamp"] = time.time()

            # Save manifest
            with open(self.manifest_path, "w") as f:
                json.dump(self.manifest, f, indent=2)

            logger.info(f"Flushed shard {self.current_shard - 1}: {len(df)} sequences")
            logger.info(f"Total cached: {self.total_cached} sequences")

        except Exception as e:
            logger.error(f"Failed to flush buffer: {e}")
            raise

    def get_top_k_logits_tensors(self, input_ids, topk: Optional[int] = None):
        """
        GPU-native tensor API pass-through to wrapped client.

        FIX #2: This method exposes the tensor API of the wrapped client through
        the caching wrapper, ensuring the trainer can use the fast GPU-native path.

        NOTE: This method does NOT cache results (caching requires CPU conversion).
        Use get_top_k_logits() or get_prompt_logprobs() for cached results.

        Args:
            input_ids: Token ID tensor on GPU. Shape: [batch_size, seq_len]
            topk: Number of top logits to return (default: wrapped client default)

        Returns:
            tuple of 3 tensors (all on GPU):
                - topk_indices: [batch, seq_len, topk]
                - topk_logprobs: [batch, seq_len, topk]
                - other_mass: [batch, seq_len, 1]

        Raises:
            NotImplementedError: If wrapped client doesn't support tensor API
        """
        # Check if wrapped client has tensor API
        if not hasattr(self.teacher, 'get_top_k_logits_tensors'):
            raise NotImplementedError(
                f"Wrapped client {type(self.teacher).__name__} does not support "
                "get_top_k_logits_tensors(). This method requires DirectTeacherClient "
                "or another client implementing the GPU-native tensor API."
            )

        # Proxy directly to wrapped client (no caching for tensor API)
        # NOTE: Caching would require .cpu() conversion which defeats the purpose
        # of the tensor API. Use get_top_k_logits() for cached results.
        return self.teacher.get_top_k_logits_tensors(input_ids=input_ids, topk=topk)

    def health_check(self) -> bool:
        """
        Check if wrapped teacher is healthy.

        Returns:
            True if teacher is healthy, False otherwise
        """
        if hasattr(self.teacher, 'health_check'):
            return self.teacher.health_check()
        else:
            # Assume healthy if no health_check method
            return True

    def close(self):
        """
        Close the wrapper and wrapped teacher.

        Flushes any remaining buffered sequences and closes wrapped teacher.
        """
        logger.info("Closing CachingTeacherWrapper...")

        try:
            # Flush remaining buffer
            if self.buffer:
                logger.info(f"Flushing final buffer: {len(self.buffer)} sequences")
                self.flush()

            # Close wrapped teacher
            if hasattr(self.teacher, 'close'):
                self.teacher.close()

            logger.info("CachingTeacherWrapper closed")

        except Exception as e:
            logger.warning(f"Error closing CachingTeacherWrapper: {e}")

    def __del__(self):
        """Finalizer to ensure buffer is flushed."""
        try:
            if self.buffer:
                self.flush()
        except Exception:
            pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        try:
            self.close()
        except Exception as e:
            logger.warning(f"Error in __exit__ while closing: {e}")
        return False

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get caching statistics.

        Returns:
            Dict with cache statistics
        """
        return {
            "total_cached": self.total_cached,
            "current_shard": self.current_shard,
            "buffer_size": len(self.buffer),
            "cache_dir": str(self.cache_dir),
            "shard_size": self.shard_size,
        }
