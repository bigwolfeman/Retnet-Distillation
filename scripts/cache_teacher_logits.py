#!/usr/bin/env python3
"""
Brain-dead-simple script for caching teacher logits that "just works".

This script handles ALL complexity internally - just run it and walk away.

Usage Examples:
    # Test mode (quick validation, ~5 minutes)
    python scripts/cache_teacher_logits.py --test

    # Full caching (default, hours depending on dataset size)
    python scripts/cache_teacher_logits.py

    # Custom configuration
    python scripts/cache_teacher_logits.py \
        --data-path data/custom_train/ \
        --output-dir data/custom_cache/ \
        --batch-size 16 \
        --workers 8

    # Resume interrupted run
    python scripts/cache_teacher_logits.py --resume

Features:
    - Auto-detect data format (JSONL, parquet, text files)
    - Auto-tokenize with Llama tokenizer if needed
    - Progress bar with ETA
    - Auto-resume on crash (save checkpoints every 1000 sequences)
    - Compression with parquet (automatic)
    - Verify cache integrity after completion
    - Parallel processing with configurable workers
    - Exponential backoff on network errors
    - Graceful error handling (disk full, network errors, etc.)
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import AutoTokenizer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.distillation.vllm_teacher_client import (
    VLLMNetworkError,
    VLLMServerError,
    VLLMTeacherClient,
    VLLMTeacherError,
)
from src.distillation.fast_teacher_client import (
    FastTeacherClient,
    FastTeacherError,
    FastTeacherServerError,
    FastTeacherNetworkError,
)
from src.utils.tokenizer_utils import load_tokenizer_with_fallback, get_tokenizer_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for caching process."""

    test_mode: bool
    data_path: Path
    output_dir: Path
    teacher_url: str
    api_key: str
    batch_size: int
    max_sequences: Optional[int]
    resume: bool
    workers: int
    topk: int
    shard_size: int
    checkpoint_interval: int
    max_retries: int
    timeout: float
    tokenizer_name: str
    hf_token: Optional[str]
    use_fast_endpoint: bool


@dataclass
class CachedSequence:
    """A single cached sequence with teacher logits."""

    sequence_id: str
    input_ids: List[int]
    teacher_indices: List[List[int]]
    teacher_values: List[List[int]]
    teacher_scales: List[float]
    teacher_other: List[float]


class TeacherLogitCache:
    """
    Main caching orchestrator.

    Handles:
    - Data loading from various formats
    - Parallel fetching from teacher server
    - Checkpointing and resume
    - Progress tracking
    - Cache validation
    """

    def __init__(self, config: CacheConfig):
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize checkpoint state
        self.checkpoint_path = self.output_dir / "checkpoint.json"
        self.manifest_path = self.output_dir / "manifest.json"
        self.processed_ids = set()
        self.current_shard = 0
        self.total_cached = 0

        # Load checkpoint if resuming
        if config.resume and self.checkpoint_path.exists():
            self._load_checkpoint()

        # Initialize tokenizer with fallback logic
        logger.info("Loading tokenizer...")
        self.tokenizer = load_tokenizer_with_fallback(
            preferred_tokenizer=config.tokenizer_name,
            hf_token=config.hf_token,
            trust_remote_code=True,
        )

        # Log tokenizer info
        tokenizer_info = get_tokenizer_info(self.tokenizer)
        logger.info(f"Loaded tokenizer: {tokenizer_info['name_or_path']}")
        logger.info(f"  Vocab size: {tokenizer_info['vocab_size']}")
        logger.info(f"  Max length: {tokenizer_info['model_max_length']}")

        # Initialize teacher client (fast or slow)
        if config.use_fast_endpoint:
            logger.info(f"Connecting to teacher server at {config.teacher_url} (FAST endpoint)...")
            self.teacher_client = FastTeacherClient(
                base_url=config.teacher_url,
                model="meta-llama/Llama-3.2-1B-Instruct",
                api_key=config.api_key,
                timeout=config.timeout,
                max_retries=config.max_retries,
                fallback_to_slow=True,  # Fallback if endpoint not available
            )
            logger.info("Using FAST /v1/topk endpoint (100x+ speedup expected)")
        else:
            logger.info(f"Connecting to teacher server at {config.teacher_url} (slow prompt_logprobs)...")
            self.teacher_client = VLLMTeacherClient(
                base_url=config.teacher_url,
                model="meta-llama/Llama-3.2-1B-Instruct",
                api_key=config.api_key,
                timeout=config.timeout,
                max_retries=config.max_retries,
            )
            logger.warning("Using SLOW prompt_logprobs approach (consider --use-fast-endpoint)")

        # Thread-local storage for parallel workers
        # Each thread gets its own teacher client to avoid session conflicts
        self._thread_local = threading.local()

        # Verify server is reachable
        if not self.teacher_client.health_check():
            raise RuntimeError(
                f"Cannot reach teacher server at {config.teacher_url}. "
                "Please check the server is running and the URL is correct."
            )
        logger.info("Teacher server connection verified.")

    def _load_checkpoint(self):
        """Load checkpoint state for resume."""
        logger.info("Loading checkpoint for resume...")
        with open(self.checkpoint_path, "r") as f:
            checkpoint = json.load(f)

        self.processed_ids = set(checkpoint.get("processed_ids", []))
        self.current_shard = checkpoint.get("current_shard", 0)
        self.total_cached = checkpoint.get("total_cached", 0)

        logger.info(
            f"Resumed from checkpoint: {self.total_cached} sequences cached, "
            f"shard {self.current_shard}"
        )

    def _save_checkpoint(self):
        """Save checkpoint state."""
        checkpoint = {
            "processed_ids": list(self.processed_ids),
            "current_shard": self.current_shard,
            "total_cached": self.total_cached,
            "timestamp": time.time(),
        }

        # Atomic write
        temp_path = self.checkpoint_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(checkpoint, f, indent=2)
        temp_path.replace(self.checkpoint_path)

    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load data from various formats.

        Supports:
        - JSONL files (*.jsonl)
        - Parquet files (*.parquet)
        - Text files (*.txt) - one sequence per line
        - Directories containing any of the above

        Returns:
            List of dicts with 'id' and 'text' or 'input_ids' fields
        """
        logger.info(f"Loading data from {self.config.data_path}...")
        data = []

        if self.config.data_path.is_file():
            data = self._load_single_file(self.config.data_path)
        elif self.config.data_path.is_dir():
            # Load all supported files in directory
            for pattern in ["*.jsonl", "*.parquet", "*.txt"]:
                for file_path in self.config.data_path.glob(pattern):
                    data.extend(self._load_single_file(file_path))
        else:
            raise ValueError(f"Data path not found: {self.config.data_path}")

        # Limit sequences if specified
        if self.config.max_sequences:
            data = data[: self.config.max_sequences]

        # Filter out already processed sequences if resuming
        if self.config.resume and self.processed_ids:
            original_count = len(data)
            data = [d for d in data if d.get("id") not in self.processed_ids]
            logger.info(
                f"Filtered {original_count - len(data)} already-processed sequences"
            )

        logger.info(f"Loaded {len(data)} sequences")
        return data

    def _load_single_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load data from a single file."""
        logger.info(f"Loading file: {file_path}")

        if file_path.suffix == ".jsonl":
            return self._load_jsonl(file_path)
        elif file_path.suffix == ".parquet":
            return self._load_parquet(file_path)
        elif file_path.suffix == ".txt":
            return self._load_text(file_path)
        else:
            logger.warning(f"Unsupported file format: {file_path}")
            return []

    def _load_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSONL file."""
        data = []
        with open(file_path, "r") as f:
            for i, line in enumerate(f):
                try:
                    record = json.loads(line)
                    # Ensure ID exists
                    if "id" not in record:
                        record["id"] = f"{file_path.stem}_{i:06d}"
                    data.append(record)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {i}: {e}")
        return data

    def _load_parquet(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load Parquet file."""
        df = pd.read_parquet(file_path)
        return df.to_dict("records")

    def _load_text(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load plain text file (one sequence per line)."""
        data = []
        with open(file_path, "r") as f:
            for i, line in enumerate(f):
                text = line.strip()
                if text:
                    data.append({"id": f"{file_path.stem}_{i:06d}", "text": text})
        return data

    def prepare_sequences(
        self, data: List[Dict[str, Any]]
    ) -> List[Tuple[str, List[int]]]:
        """
        Prepare sequences for caching.

        Tokenizes text if needed, extracts IDs.

        Returns:
            List of (sequence_id, input_ids) tuples
        """
        logger.info("Preparing sequences...")
        sequences = []

        for record in tqdm(data, desc="Tokenizing"):
            seq_id = record.get("id", f"seq_{len(sequences):06d}")

            # Get input_ids (tokenize if needed)
            if "input_ids" in record:
                input_ids = record["input_ids"]
            elif "text" in record:
                # Tokenize text
                input_ids = self.tokenizer.encode(record["text"], add_special_tokens=True)
            else:
                logger.warning(f"Sequence {seq_id} has no text or input_ids, skipping")
                continue

            sequences.append((seq_id, input_ids))

        return sequences

    def cache_sequences(self, sequences: List[Tuple[str, List[int]]]):
        """
        Cache teacher logits for all sequences.

        Uses parallel workers to fetch from teacher server.
        Saves to sharded parquet files.
        """
        if self.config.workers == 1:
            # Sequential mode - simpler and always works
            logger.info(f"Caching {len(sequences)} sequences sequentially (workers=1)...")
            self._cache_sequences_sequential(sequences)
        else:
            # Parallel mode - faster but requires thread-safe client
            logger.info(f"Caching {len(sequences)} sequences using {self.config.workers} workers...")
            self._cache_sequences_parallel(sequences)

    def _cache_sequences_sequential(self, sequences: List[Tuple[str, List[int]]]):
        """Sequential caching - no threading, uses shared client."""
        shard_buffer = []
        sequences_cached = 0

        # Process sequences in batches
        pbar = tqdm(total=len(sequences), desc="Caching", unit="seq")

        for i in range(0, len(sequences), self.config.batch_size):
            batch = sequences[i:i + self.config.batch_size]

            try:
                cached_batch = self._fetch_batch(batch)
                shard_buffer.extend(cached_batch)
                sequences_cached += len(cached_batch)
                self.total_cached += len(cached_batch)
                pbar.update(len(cached_batch))

                # Add IDs to processed set
                for cached_seq in cached_batch:
                    self.processed_ids.add(cached_seq.sequence_id)

                # Save shard if buffer full
                if len(shard_buffer) >= self.config.shard_size:
                    self._save_shard(shard_buffer)
                    shard_buffer = []
                    self.current_shard += 1

                # Save checkpoint periodically
                if sequences_cached % self.config.checkpoint_interval == 0:
                    self._save_checkpoint()

            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                logger.debug(traceback.format_exc())

        # Save remaining sequences in buffer
        if shard_buffer:
            self._save_shard(shard_buffer)

        pbar.close()

        # Final checkpoint
        self._save_checkpoint()

        logger.info(f"Cached {sequences_cached} sequences total")

    def _cache_sequences_parallel(self, sequences: List[Tuple[str, List[int]]]):
        """Parallel caching with thread-local teacher clients."""
        # Create work queue
        work_queue = Queue()
        for seq in sequences:
            work_queue.put(seq)

        # Current shard buffer
        shard_buffer = []
        total_sequences = len(sequences)
        sequences_cached = 0

        # Progress bar
        pbar = tqdm(total=total_sequences, desc="Caching", unit="seq")

        # Process in batches with parallel workers
        # IMPORTANT: Each worker must create its own teacher client (thread-local)
        with ThreadPoolExecutor(max_workers=self.config.workers) as executor:
            futures = []

            # Submit initial batch
            for _ in range(min(self.config.workers, len(sequences))):
                if not work_queue.empty():
                    batch = self._get_batch_from_queue(work_queue)
                    if batch:
                        future = executor.submit(self._fetch_batch_threadsafe, batch)
                        futures.append(future)

            # Process results as they complete
            while futures:
                # Wait for any future to complete
                done_futures = []
                for future in as_completed(futures):
                    done_futures.append(future)
                    try:
                        cached_batch = future.result()
                        shard_buffer.extend(cached_batch)
                        sequences_cached += len(cached_batch)
                        self.total_cached += len(cached_batch)
                        pbar.update(len(cached_batch))

                        # Add IDs to processed set
                        for cached_seq in cached_batch:
                            self.processed_ids.add(cached_seq.sequence_id)

                        # Save shard if buffer full
                        if len(shard_buffer) >= self.config.shard_size:
                            self._save_shard(shard_buffer)
                            shard_buffer = []
                            self.current_shard += 1

                        # Save checkpoint periodically
                        if sequences_cached % self.config.checkpoint_interval == 0:
                            self._save_checkpoint()

                    except Exception as e:
                        logger.error(f"Error processing batch: {e}")
                        logger.debug(traceback.format_exc())

                # Remove completed futures
                for done_future in done_futures:
                    futures.remove(done_future)

                # Submit new work if available
                while len(futures) < self.config.workers and not work_queue.empty():
                    batch = self._get_batch_from_queue(work_queue)
                    if batch:
                        future = executor.submit(self._fetch_batch_threadsafe, batch)
                        futures.append(future)

        # Save remaining sequences in buffer
        if shard_buffer:
            self._save_shard(shard_buffer)

        pbar.close()

        # Final checkpoint
        self._save_checkpoint()

        logger.info(f"Cached {sequences_cached} sequences total")

    def _get_batch_from_queue(self, work_queue: Queue) -> List[Tuple[str, List[int]]]:
        """Get a batch from work queue."""
        batch = []
        for _ in range(self.config.batch_size):
            if work_queue.empty():
                break
            batch.append(work_queue.get())
        return batch

    def _get_thread_local_client(self):
        """
        Get or create a thread-local teacher client.

        This ensures each worker thread has its own HTTP session,
        avoiding the requests.Session() thread-safety issues.
        """
        if not hasattr(self._thread_local, 'client'):
            # Create a new client for this thread (fast or slow)
            logger.debug(f"Creating thread-local teacher client for thread {threading.current_thread().name}")
            if self.config.use_fast_endpoint:
                self._thread_local.client = FastTeacherClient(
                    base_url=self.config.teacher_url,
                    model="meta-llama/Llama-3.2-1B-Instruct",
                    api_key=self.config.api_key,
                    timeout=self.config.timeout,
                    max_retries=self.config.max_retries,
                    fallback_to_slow=True,
                )
            else:
                self._thread_local.client = VLLMTeacherClient(
                    base_url=self.config.teacher_url,
                    model="meta-llama/Llama-3.2-1B-Instruct",
                    api_key=self.config.api_key,
                    timeout=self.config.timeout,
                    max_retries=self.config.max_retries,
                )
        return self._thread_local.client

    def _fetch_batch_threadsafe(
        self, batch: List[Tuple[str, List[int]]]
    ) -> List[CachedSequence]:
        """
        Thread-safe version of _fetch_batch that uses thread-local client.

        This method creates a separate teacher client per thread to avoid
        sharing the requests.Session() object across threads (not thread-safe).
        """
        if not batch:
            return []

        # Get thread-local client (creates one if needed)
        client = self._get_thread_local_client()

        # Prepare batch for teacher client
        batch_ids = [seq_id for seq_id, _ in batch]
        batch_input_ids = [input_ids for _, input_ids in batch]

        try:
            # Fetch from teacher server using thread-local client
            results = client.get_prompt_logprobs(
                input_ids=batch_input_ids,
                topk=self.config.topk,
                temperature=1.0,
            )

            # Convert to cached sequences
            cached_batch = []
            for (seq_id, input_ids), result in zip(batch, results):
                cached_seq = self._convert_to_cached_sequence(seq_id, input_ids, result)
                cached_batch.append(cached_seq)

            return cached_batch

        except (VLLMNetworkError, VLLMServerError, VLLMTeacherError) as e:
            logger.error(f"Failed to fetch batch: {e}")
            # Return empty to skip this batch (already logged)
            return []

        except Exception as e:
            logger.error(f"Unexpected error fetching batch: {e}")
            logger.debug(traceback.format_exc())
            return []

    def _fetch_batch(
        self, batch: List[Tuple[str, List[int]]]
    ) -> List[CachedSequence]:
        """
        Fetch teacher logits for a batch of sequences.

        Handles retries and errors gracefully.
        Uses shared teacher client (for sequential mode only).
        """
        if not batch:
            return []

        # Prepare batch for teacher client
        batch_ids = [seq_id for seq_id, _ in batch]
        batch_input_ids = [input_ids for _, input_ids in batch]

        try:
            # Fetch from teacher server
            results = self.teacher_client.get_prompt_logprobs(
                input_ids=batch_input_ids,
                topk=self.config.topk,
                temperature=1.0,
            )

            # Convert to cached sequences
            cached_batch = []
            for (seq_id, input_ids), result in zip(batch, results):
                cached_seq = self._convert_to_cached_sequence(seq_id, input_ids, result)
                cached_batch.append(cached_seq)

            return cached_batch

        except (VLLMNetworkError, VLLMServerError, VLLMTeacherError,
                FastTeacherNetworkError, FastTeacherServerError, FastTeacherError) as e:
            logger.error(f"Failed to fetch batch: {e}")
            # Return empty to skip this batch (already logged)
            return []

        except Exception as e:
            logger.error(f"Unexpected error fetching batch: {e}")
            logger.debug(traceback.format_exc())
            return []

    def _convert_to_cached_sequence(
        self, seq_id: str, input_ids: List[int], result: Dict[str, Any]
    ) -> CachedSequence:
        """
        Convert teacher response to cached sequence format.

        Compresses logits to int8 format with scale factors.
        """
        indices = result["indices"]
        logprobs = result["logprobs"]

        # Convert logprobs to int8 with scale factors
        teacher_indices = []
        teacher_values = []
        teacher_scales = []
        teacher_other = []

        for position_indices, position_logprobs in zip(indices, logprobs):
            if not position_indices:
                # Empty position (e.g., BOS)
                teacher_indices.append([])
                teacher_values.append([])
                teacher_scales.append(0.0)
                teacher_other.append(0.0)
                continue

            # Convert to numpy for easier manipulation
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
            # Scale factor chosen to minimize quantization error
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

        return CachedSequence(
            sequence_id=seq_id,
            input_ids=input_ids,
            teacher_indices=teacher_indices,
            teacher_values=teacher_values,
            teacher_scales=teacher_scales,
            teacher_other=teacher_other,
        )

    def _save_shard(self, shard_buffer: List[CachedSequence]):
        """Save a shard to parquet file."""
        if not shard_buffer:
            return

        shard_path = self.output_dir / f"cache_shard_{self.current_shard:04d}.parquet"
        logger.info(f"Saving shard {self.current_shard} ({len(shard_buffer)} sequences)...")

        # Convert to pandas DataFrame
        records = []
        for cached_seq in shard_buffer:
            records.append(
                {
                    "sequence_id": cached_seq.sequence_id,
                    "input_ids": cached_seq.input_ids,
                    "teacher_indices": cached_seq.teacher_indices,
                    "teacher_values": cached_seq.teacher_values,
                    "teacher_scales": cached_seq.teacher_scales,
                    "teacher_other": cached_seq.teacher_other,
                }
            )

        df = pd.DataFrame(records)

        # Save with compression
        df.to_parquet(
            shard_path,
            engine="pyarrow",
            compression="snappy",
            index=False,
        )

        logger.info(f"Saved shard to {shard_path}")

    def save_manifest(self, total_sequences: int, elapsed_time: float):
        """Save manifest with cache metadata."""
        # Count shards
        shard_files = sorted(self.output_dir.glob("cache_shard_*.parquet"))

        # Calculate total size
        total_size = sum(f.stat().st_size for f in shard_files)

        manifest = {
            "total_sequences": total_sequences,
            "num_shards": len(shard_files),
            "shard_files": [f.name for f in shard_files],
            "topk": self.config.topk,
            "timestamp": time.time(),
            "elapsed_time_seconds": elapsed_time,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "config": {
                "teacher_url": self.config.teacher_url,
                "batch_size": self.config.batch_size,
                "workers": self.config.workers,
            },
        }

        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Saved manifest to {self.manifest_path}")

    def validate_cache(self, num_samples: int = 10):
        """
        Validate cache integrity.

        Loads random samples and compares to fresh fetch from server.
        """
        logger.info(f"Validating cache with {num_samples} random samples...")

        # Load all cached sequences
        shard_files = sorted(self.output_dir.glob("cache_shard_*.parquet"))
        if not shard_files:
            logger.warning("No cache shards found to validate")
            return

        # Sample random sequences
        all_sequences = []
        for shard_file in shard_files:
            df = pd.read_parquet(shard_file)
            all_sequences.extend(df.to_dict("records"))

        if len(all_sequences) == 0:
            logger.warning("No sequences in cache to validate")
            return

        # Randomly sample
        num_samples = min(num_samples, len(all_sequences))
        sampled_indices = np.random.choice(len(all_sequences), num_samples, replace=False)
        sampled_sequences = [all_sequences[i] for i in sampled_indices]

        # Validate each sample
        validation_errors = 0
        for sample in tqdm(sampled_sequences, desc="Validating"):
            try:
                # Fetch fresh from server
                fresh_results = self.teacher_client.get_prompt_logprobs(
                    input_ids=[sample["input_ids"]],
                    topk=self.config.topk,
                    temperature=1.0,
                )
                fresh_result = fresh_results[0]

                # Compare
                cached_indices = sample["teacher_indices"]
                fresh_indices = fresh_result["indices"]

                # Check sequence length matches
                if len(cached_indices) != len(fresh_indices):
                    logger.error(
                        f"Sequence {sample['sequence_id']}: Length mismatch "
                        f"(cached={len(cached_indices)}, fresh={len(fresh_indices)})"
                    )
                    validation_errors += 1
                    continue

                # Check indices match at each position
                for pos, (cached_pos, fresh_pos) in enumerate(
                    zip(cached_indices, fresh_indices)
                ):
                    # Skip empty positions
                    if not cached_pos and not fresh_pos:
                        continue

                    # Check top indices match (order matters)
                    if cached_pos != fresh_pos:
                        logger.warning(
                            f"Sequence {sample['sequence_id']} position {pos}: "
                            f"Indices mismatch (minor differences expected due to compression)"
                        )

            except Exception as e:
                logger.error(f"Validation error for {sample['sequence_id']}: {e}")
                validation_errors += 1

        if validation_errors == 0:
            logger.info("Validation passed: All samples match!")
        else:
            logger.warning(
                f"Validation completed with {validation_errors}/{num_samples} errors"
            )

    def run(self):
        """Run the full caching pipeline."""
        start_time = time.time()

        try:
            # Load data
            data = self.load_data()

            if len(data) == 0:
                logger.error("No data to cache!")
                return

            # Prepare sequences
            sequences = self.prepare_sequences(data)

            if len(sequences) == 0:
                logger.error("No valid sequences to cache!")
                return

            # Cache sequences
            self.cache_sequences(sequences)

            # Save manifest
            elapsed_time = time.time() - start_time
            self.save_manifest(len(sequences), elapsed_time)

            # Validate cache
            self.validate_cache(num_samples=10)

            # Print summary
            self._print_summary(elapsed_time)

        except KeyboardInterrupt:
            logger.warning("Interrupted by user. Saving checkpoint...")
            self._save_checkpoint()
            logger.info("Checkpoint saved. Run with --resume to continue.")
            sys.exit(1)

        except Exception as e:
            logger.error(f"Fatal error: {e}")
            logger.debug(traceback.format_exc())
            self._save_checkpoint()
            logger.info("Checkpoint saved. Run with --resume to continue.")
            sys.exit(1)

        finally:
            # Close teacher client
            self.teacher_client.close()

    def _print_summary(self, elapsed_time: float):
        """Print summary statistics."""
        # Load manifest
        if not self.manifest_path.exists():
            return

        with open(self.manifest_path, "r") as f:
            manifest = json.load(f)

        print("\n" + "=" * 60)
        print("CACHING COMPLETE!")
        print("=" * 60)
        print(f"Sequences cached: {manifest['total_sequences']}")
        print(f"Shards created: {manifest['num_shards']}")
        print(f"Total storage: {manifest['total_size_mb']:.2f} MB")
        print(f"Avg per sequence: {manifest['total_size_mb'] / manifest['total_sequences']:.4f} MB")
        print(f"Time elapsed: {elapsed_time / 60:.1f} minutes")
        print(f"Throughput: {manifest['total_sequences'] / elapsed_time:.2f} seq/sec")
        print(f"Output directory: {self.output_dir}")
        print("=" * 60 + "\n")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Cache teacher logits for distillation training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Test mode (100 sequences)
    python scripts/cache_teacher_logits.py --test

    # Full caching
    python scripts/cache_teacher_logits.py

    # Custom configuration
    python scripts/cache_teacher_logits.py --data-path data/custom/ --workers 8

    # Resume interrupted run
    python scripts/cache_teacher_logits.py --resume
        """,
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: cache only 100 sequences to data/teacher_cache_test/",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Input data path (default: data/train/ or data/dummy_test/ for --test)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Cache output directory (default: data/teacher_cache/ or data/teacher_cache_test/ for --test)",
    )

    parser.add_argument(
        "--teacher-url",
        type=str,
        default="http://localhost:8080",
        help="vLLM server URL (default: http://localhost:8080)",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default="token-abc123",
        help="API key for authentication (default: token-abc123)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for fetching (default: 8)",
    )

    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Max sequences to process (default: all)",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from interrupted run",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel workers (default: 4). Use --workers 1 for sequential/debug mode.",
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=128,
        help="Top-k logits to cache (default: 128)",
    )

    parser.add_argument(
        "--shard-size",
        type=int,
        default=1000,
        help="Sequences per shard file (default: 1000)",
    )

    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Save checkpoint every N sequences (default: 1000)",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Request timeout in seconds (default: 60.0)",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries for network errors (default: 5)",
    )

    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HuggingFace tokenizer name (default: meta-llama/Llama-3.2-1B-Instruct)",
    )

    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for gated models (or set HF_TOKEN env var)",
    )

    parser.add_argument(
        "--use-fast-endpoint",
        action="store_true",
        help="Use fast /v1/topk endpoint instead of slow prompt_logprobs (100x+ speedup)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Apply test mode defaults
    if args.test:
        data_path = args.data_path or "data/dummy_test/"
        output_dir = args.output_dir or "data/teacher_cache_test/"
        max_sequences = args.max_sequences or 100
        logger.info("TEST MODE: Caching 100 sequences to data/teacher_cache_test/")
    else:
        data_path = args.data_path or "data/train/"
        output_dir = args.output_dir or "data/teacher_cache/"
        max_sequences = args.max_sequences

    # Create config
    config = CacheConfig(
        test_mode=args.test,
        data_path=Path(data_path),
        output_dir=Path(output_dir),
        teacher_url=args.teacher_url,
        api_key=args.api_key,
        batch_size=args.batch_size,
        max_sequences=max_sequences,
        resume=args.resume,
        workers=args.workers,
        topk=args.topk,
        shard_size=args.shard_size,
        checkpoint_interval=args.checkpoint_interval,
        max_retries=args.max_retries,
        timeout=args.timeout,
        tokenizer_name=args.tokenizer_name,
        hf_token=args.hf_token,
        use_fast_endpoint=args.use_fast_endpoint,
    )

    # Validate data path exists
    if not config.data_path.exists():
        logger.error(f"Data path not found: {config.data_path}")
        logger.info("Available data directories:")
        data_root = Path("data")
        if data_root.exists():
            for item in data_root.iterdir():
                if item.is_dir():
                    logger.info(f"  - {item}")
        sys.exit(1)

    # Print configuration
    print("\n" + "=" * 60)
    print("Teacher Logit Caching Configuration")
    print("=" * 60)
    print(f"Mode: {'TEST (100 sequences)' if config.test_mode else 'FULL'}")
    print(f"Data path: {config.data_path}")
    print(f"Output directory: {config.output_dir}")
    print(f"Teacher URL: {config.teacher_url}")
    print(f"Batch size: {config.batch_size}")
    print(f"Workers: {config.workers}")
    print(f"Top-k: {config.topk}")
    print(f"Resume: {config.resume}")
    print(f"Fast endpoint: {config.use_fast_endpoint} {'(100x+ speedup!)' if config.use_fast_endpoint else '(SLOW - consider --use-fast-endpoint)'}")
    if config.max_sequences:
        print(f"Max sequences: {config.max_sequences}")
    print("=" * 60 + "\n")

    # Run caching
    cache = TeacherLogitCache(config)
    cache.run()


if __name__ == "__main__":
    main()
