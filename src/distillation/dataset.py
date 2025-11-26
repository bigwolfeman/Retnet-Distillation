"""
Data pipeline for distillation training.

Implements data loading, tokenization, and batching for 4k context sequences.
This module provides:
- SimpleDataLoader: Basic data loading from JSONL/text files
- Llama tokenizer integration (vocab_size=128256)
- Sequence truncation/padding to 4k max
- Batch iteration for PyTorch training loops

Tasks implemented: T032, T033
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Iterator, Any
from collections import OrderedDict
import torch
from torch.utils.data import IterableDataset, Dataset
from transformers import AutoTokenizer
import pyarrow.parquet as pq
import glob


logger = logging.getLogger(__name__)


class SimpleDataLoader(Dataset):
    """Simple batch data loader for distillation training.

    Supports loading from:
    - JSONL files (one JSON object per line)
    - Text files (one sequence per line)
    - Pre-tokenized format (optional, for efficiency)

    Features:
    - Automatic sequence truncation/padding to max_length (default 4096)
    - Llama-3.2-1B tokenizer integration
    - Support for both raw text and pre-tokenized inputs
    - Efficient batching for PyTorch DataLoader

    Example:
        >>> loader = SimpleDataLoader(
        ...     data_path="data/train.jsonl",
        ...     max_length=4096,
        ...     tokenizer_name="meta-llama/Llama-3.2-1B"
        ... )
        >>> batch = loader[0]
        >>> batch.keys()
        dict_keys(['input_ids', 'attention_mask', 'labels'])
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        max_length: int = 4096,
        tokenizer_name: str = "meta-llama/Llama-3.2-1B",
        tokenizer: Optional[AutoTokenizer] = None,
        text_field: str = "text",
        use_pretokenized: bool = False,
        pretokenized_field: str = "input_ids",
        return_labels: bool = True,
        padding: str = "max_length",
        truncation: bool = True,
    ):
        """Initialize SimpleDataLoader.

        Args:
            data_path: Path to data file (JSONL or text)
            max_length: Maximum sequence length (default 4096 for v1)
            tokenizer_name: HuggingFace tokenizer name
            tokenizer: Pre-loaded tokenizer (optional, loads if None)
            text_field: Field name for text in JSONL files
            use_pretokenized: Whether data is already tokenized
            pretokenized_field: Field name for pre-tokenized data
            return_labels: Whether to return labels (for causal LM)
            padding: Padding strategy ("max_length" or "longest")
            truncation: Whether to truncate sequences
        """
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.text_field = text_field
        self.use_pretokenized = use_pretokenized
        self.pretokenized_field = pretokenized_field
        self.return_labels = return_labels
        self.padding = padding
        self.truncation = truncation

        # Load tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = self._load_tokenizer(tokenizer_name)

        # Load data
        self.data = self._load_data()

    def _load_tokenizer(self, tokenizer_name: str) -> AutoTokenizer:
        """Load Llama tokenizer.

        Args:
            tokenizer_name: HuggingFace model name

        Returns:
            Loaded tokenizer with vocab_size=128256
        """
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=True,
        )

        # Verify vocab size (Llama models typically have 128000 or 128256)
        if tokenizer.vocab_size not in [128000, 128256]:
            print(f"Warning: Unexpected vocab_size={tokenizer.vocab_size}, expected 128000 or 128256")
            print("Continuing anyway, but model may need adjustment")

        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def _load_data(self) -> List[Dict]:
        """Load data from file.

        Returns:
            List of data records (dicts or strings)
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        data = []

        # Determine file type
        is_jsonl = self.data_path.suffix in ['.jsonl', '.json']

        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    if is_jsonl:
                        # Parse JSON
                        record = json.loads(line)
                        data.append(record)
                    else:
                        # Raw text
                        data.append({"text": line})
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
                    continue

        if len(data) == 0:
            raise ValueError(f"No valid data found in {self.data_path}")

        return data

    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single tokenized example.

        Args:
            idx: Index of example

        Returns:
            Dictionary with:
            - input_ids: [seq_len] token IDs
            - attention_mask: [seq_len] attention mask (1 for real tokens, 0 for padding)
            - labels: [seq_len] labels for causal LM (same as input_ids, or custom)
        """
        record = self.data[idx]

        if self.use_pretokenized:
            # Use pre-tokenized data
            if isinstance(record, dict) and self.pretokenized_field in record:
                input_ids = record[self.pretokenized_field]

                # Convert to tensor if needed
                if not isinstance(input_ids, torch.Tensor):
                    input_ids = torch.tensor(input_ids, dtype=torch.long)

                # Truncate if needed
                if len(input_ids) > self.max_length:
                    input_ids = input_ids[:self.max_length]

                # Pad if needed
                if self.padding == "max_length" and len(input_ids) < self.max_length:
                    padding_length = self.max_length - len(input_ids)
                    input_ids = torch.cat([
                        input_ids,
                        torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=torch.long)
                    ])

                # Create labels (no manual shift; HF models shift internally)
                if self.return_labels:
                    labels = input_ids.clone()
                    # Ignore last position (no target) and any trailing padding
                    labels[-1] = -100
                    if self.tokenizer.pad_token_id is not None:
                        pad_id = self.tokenizer.pad_token_id
                        # mask only trailing pads, not all occurrences of pad_id
                        seq_len = len(labels)
                        while seq_len > 0 and labels[seq_len - 1] == pad_id:
                            labels[seq_len - 1] = -100
                            seq_len -= 1
                else:
                    labels = None

                # Attention mask (no shift; mask only real pads)
                if self.tokenizer.pad_token_id is not None:
                    attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
                else:
                    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

            else:
                raise ValueError(
                    f"Pre-tokenized data expected but field '{self.pretokenized_field}' not found"
                )
        else:
            # Tokenize raw text
            if isinstance(record, dict):
                text = record.get(self.text_field, "")
            else:
                text = str(record)

            if not text:
                raise ValueError(f"Empty text at index {idx}")

            # Tokenize with special tokens
            encoded = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding=self.padding,
                truncation=self.truncation,
                return_tensors="pt",
            )

            # Extract tensors (remove batch dimension)
            input_ids = encoded['input_ids'].squeeze(0)

            # Create labels
            if self.return_labels:
                labels = input_ids.clone()
                # Note: Model handles shifting internally, so we don't shift here.
                # Mask padding tokens in labels
                labels[labels == self.tokenizer.pad_token_id] = -100
            else:
                labels = None

            # Create attention mask (aligned with input_ids)
            attention_mask = encoded['attention_mask'].squeeze(0)

        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

        if labels is not None:
            result['labels'] = labels

        return result

    def get_batch(self, indices: List[int]) -> Dict[str, torch.Tensor]:
        """Get a batch of examples.

        Args:
            indices: List of example indices

        Returns:
            Batched tensors with shape [batch_size, seq_len]
        """
        batch = [self[idx] for idx in indices]
        return self.collate_fn(batch)

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function for DataLoader.

        Args:
            batch: List of examples from __getitem__

        Returns:
            Batched tensors
        """
        # Stack all tensors
        result = {}

        for key in batch[0].keys():
            tensors = [item[key] for item in batch]
            result[key] = torch.stack(tensors, dim=0)

        return result


class LRUCache:
    """Least Recently Used (LRU) cache for sequence data.

    Features:
    - Tracks access order using OrderedDict (most recent at end)
    - Evicts least-recently-used items when memory limit reached
    - O(1) cache hit, O(1) cache miss with eviction
    - Tracks cache hits/misses for performance monitoring

    Example:
        >>> cache = LRUCache(max_size_mb=100)
        >>> cache.put(0, {'input_ids': [1, 2, 3]}, size_bytes=12)
        >>> item = cache.get(0)  # Cache hit
        >>> stats = cache.get_stats()
        >>> print(f"Hit rate: {stats['hit_rate']:.2%}")
    """

    def __init__(self, max_size_mb: int):
        """Initialize LRU cache.

        Args:
            max_size_mb: Maximum cache size in megabytes
        """
        self.cache = OrderedDict()  # key: idx, value: (data, size_bytes)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.current_size_bytes = 0
        self.hits = 0
        self.misses = 0

    def get(self, key: int) -> Optional[Dict]:
        """Get item from cache, marking it as recently used.

        Args:
            key: Index/key to retrieve

        Returns:
            Cached item if found, None otherwise
        """
        if key in self.cache:
            self.hits += 1
            # Move to end (most recent)
            self.cache.move_to_end(key)
            data, _ = self.cache[key]
            return data

        self.misses += 1
        return None

    def put(self, key: int, value: Dict, size_bytes: int):
        """Add item to cache, evicting LRU items if needed.

        Args:
            key: Index/key to store
            value: Data to cache
            size_bytes: Size of data in bytes
        """
        # If key already exists, remove old entry first
        if key in self.cache:
            _, old_size = self.cache[key]
            self.current_size_bytes -= old_size
            del self.cache[key]

        # Evict least recently used items until we have space
        while self.current_size_bytes + size_bytes > self.max_size_bytes:
            if not self.cache:
                # Cache is empty but item still too large - skip caching
                return

            # Remove least recently used (first item in OrderedDict)
            lru_key, (lru_data, lru_size) = self.cache.popitem(last=False)
            self.current_size_bytes -= lru_size

        # Add new item (at end, marking as most recent)
        self.cache[key] = (value, size_bytes)
        self.current_size_bytes += size_bytes

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache performance metrics
        """
        total_accesses = self.hits + self.misses
        return {
            'hits': self.hits,
            'misses': self.misses,
            'total_accesses': total_accesses,
            'cache_size': len(self.cache),
            'cache_size_mb': self.current_size_bytes / (1024 * 1024),
            'hit_rate': self.hits / total_accesses if total_accesses > 0 else 0.0
        }

    def clear(self):
        """Clear all cached items."""
        self.cache.clear()
        self.current_size_bytes = 0


class ParquetDataLoader(Dataset):
    """Data loader for preprocessed parquet shards.

    Loads pre-tokenized data from parquet files created by preprocessing scripts.
    Expects parquet files with columns: input_ids, attention_mask, labels.

    Features:
    - Lazy loading: Opens parquet files on-demand, doesn't load entire dataset into RAM
    - Memory-efficient: Uses PyArrow's memory-mapped reads
    - No tokenization needed (data is pre-tokenized)
    - Compatible with PyTorch DataLoader

    Example:
        >>> loader = ParquetDataLoader(data_path="data/preprocessed/train")
        >>> batch = loader[0]
        >>> batch.keys()
        dict_keys(['input_ids', 'attention_mask', 'labels'])
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        ram_cache_mb: int = 0,
        max_length: Optional[int] = None,
    ):
        """Initialize ParquetDataLoader.

        Args:
            data_path: Path to directory containing parquet shards (*.parquet)
            ram_cache_mb: If > 0, pre-load up to this many MB of data into RAM.
                         If 0 (default), use lazy loading from disk.
            max_length: Maximum sequence length. If specified, sequences longer
                       than this will be truncated. If None, no truncation.
        """
        self.data_path = Path(data_path)
        self.ram_cache_mb = ram_cache_mb
        self.max_length = max_length

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")

        # Find all parquet shards (RECURSIVE search)
        if self.data_path.is_dir():
            self.shard_files = sorted(glob.glob(str(self.data_path / "**/*.parquet"), recursive=True))
        else:
            self.shard_files = [str(self.data_path)]

        if not self.shard_files:
            raise ValueError(f"No parquet files found in: {self.data_path}")

        # Build index: map global index -> (shard_idx, row_within_shard)
        # This is lightweight - just counts rows in each shard
        self.index_map = []
        self.total_rows = 0

        for shard_idx, shard_file in enumerate(self.shard_files):
            # Get row count without loading full table (fast)
            metadata = pq.read_metadata(shard_file)
            num_rows = metadata.num_rows

            for row_idx in range(num_rows):
                self.index_map.append((shard_idx, row_idx))

            self.total_rows += num_rows

        print(f"Indexed {self.total_rows} sequences from {len(self.shard_files)} parquet shard(s)")

        # Cache for opened parquet files (lazy loading)
        self._file_cache = {}

        # Initialize LRU cache if ram_cache_mb > 0
        self._lru_cache = None
        if self.ram_cache_mb > 0:
            print(f"Initializing LRU cache (limit: {self.ram_cache_mb} MB)...")
            self._lru_cache = LRUCache(max_size_mb=self.ram_cache_mb)

            # Estimate cache capacity
            # Each sequence: 3 arrays * length * 4 bytes/int
            estimated_length = self.max_length if self.max_length else 4096
            item_size = 3 * estimated_length * 4
            estimated_capacity = (self.ram_cache_mb * 1024 * 1024) // item_size
            print(f"LRU cache initialized (estimated capacity: ~{estimated_capacity:,} sequences)")
            print(f"Cache will dynamically load and evict sequences during training")
        else:
            print("LRU caching disabled - using lazy loading from disk")

    def __getstate__(self):
        """Prepare dataset for pickling (for DataLoader workers).

        PyArrow file handles are not fork-safe and cannot be shared across
        processes. Clear the file cache and LRU cache stats to avoid issues.
        Each worker will build its own LRU cache as sequences are accessed.
        """
        state = self.__dict__.copy()
        state['_file_cache'] = {}  # Clear file cache - not fork-safe
        # Reset LRU cache but keep the configuration
        if state['_lru_cache'] is not None:
            # Create a fresh LRU cache with same settings
            state['_lru_cache'] = LRUCache(max_size_mb=self.ram_cache_mb)
        return state

    def __setstate__(self, state):
        """Restore dataset after unpickling in worker process."""
        self.__dict__.update(state)

    def __len__(self) -> int:
        """Return total number of examples."""
        return self.total_rows

    def _get_table(self, shard_idx: int):
        """Get parquet table for a shard (with caching)."""
        if shard_idx not in self._file_cache:
            # Open parquet file and cache it (memory-mapped, efficient)
            self._file_cache[shard_idx] = pq.read_table(self.shard_files[shard_idx])
        return self._file_cache[shard_idx]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example by index (from LRU cache or lazy load).

        Args:
            idx: Index of example to retrieve

        Returns:
            Dictionary with input_ids, attention_mask, and labels as tensors
        """
        if idx >= self.total_rows:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_rows}")

        # Try LRU cache first (if enabled)
        if self._lru_cache is not None:
            cached_item = self._lru_cache.get(idx)
            if cached_item is not None:
                # Cache hit - convert to tensors and return
                return self._to_tensors(cached_item)

        # Cache miss - load from disk
        shard_idx, row_idx = self.index_map[idx]
        table = self._get_table(shard_idx)
        item = {
            'input_ids': table['input_ids'][row_idx].as_py(),
            'attention_mask': table['attention_mask'][row_idx].as_py(),
            'labels': table['labels'][row_idx].as_py(),
        }

        # Truncate if max_length is specified
        if self.max_length is not None:
            for key in ['input_ids', 'attention_mask', 'labels']:
                if len(item[key]) > self.max_length:
                    item[key] = item[key][:self.max_length]

        # Add to LRU cache (if enabled)
        if self._lru_cache is not None:
            # Each sequence: 3 arrays * length * 4 bytes/int
            actual_length = self.max_length if self.max_length else len(item['input_ids'])
            item_size = 3 * actual_length * 4
            self._lru_cache.put(idx, item, item_size)

        # Convert to tensors
        return self._to_tensors(item)

    def _to_tensors(self, item: Dict) -> Dict[str, torch.Tensor]:
        """Convert item dictionary to tensors.

        Args:
            item: Dictionary with list values

        Returns:
            Dictionary with tensor values
        """
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(item['labels'], dtype=torch.long),
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get LRU cache statistics for debugging and monitoring.

        Returns:
            Dictionary with cache performance metrics
        """
        if self._lru_cache is None:
            return {
                'cache_enabled': False,
                'ram_cache_mb': self.ram_cache_mb,
            }

        stats = self._lru_cache.get_stats()
        return {
            'cache_enabled': True,
            'ram_cache_mb': self.ram_cache_mb,
            **stats
        }

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function for DataLoader.

        Args:
            batch: List of examples from __getitem__

        Returns:
            Batched tensors with shape [batch_size, seq_length]
        """
        result = {}
        for key in batch[0].keys():
            tensors = [item[key] for item in batch]
            result[key] = torch.stack(tensors, dim=0)
        return result


class StreamingDataLoader(IterableDataset):
    """Streaming data loader with lazy loading to prevent RAM overflow.

    For large datasets that don't fit in RAM, this loader:
    1. Loads data in chunks (ring buffer)
    2. Yields examples on-demand
    3. Refills buffer when exhausted

    This is useful for very large datasets (>10GB) but adds complexity.
    For v1 (4k context, moderate dataset size), SimpleDataLoader is sufficient.

    Example:
        >>> loader = StreamingDataLoader(
        ...     data_path="data/train.jsonl",
        ...     buffer_size=10000,
        ...     max_length=4096
        ... )
        >>> for batch in loader:
        ...     # Train on batch
        ...     pass
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        buffer_size: int = 10000,
        max_length: int = 4096,
        tokenizer_name: str = "meta-llama/Llama-3.2-1B",
        tokenizer: Optional[AutoTokenizer] = None,
        text_field: str = "text",
        use_pretokenized: bool = False,
        pretokenized_field: str = "input_ids",
        shuffle_buffer: bool = True,
        seed: Optional[int] = None,
    ):
        """Initialize StreamingDataLoader.

        Args:
            data_path: Path to data file (JSONL)
            buffer_size: Number of examples to keep in memory
            max_length: Maximum sequence length
            tokenizer_name: HuggingFace tokenizer name
            tokenizer: Pre-loaded tokenizer (optional)
            text_field: Field name for text in JSONL
            use_pretokenized: Whether data is already tokenized
            pretokenized_field: Field name for pre-tokenized data
            shuffle_buffer: Whether to shuffle buffer
            seed: Random seed for shuffling
        """
        self.data_path = Path(data_path)
        self.buffer_size = buffer_size
        self.max_length = max_length
        self.text_field = text_field
        self.use_pretokenized = use_pretokenized
        self.pretokenized_field = pretokenized_field
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

        # Load tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                use_fast=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # Validate file exists
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over dataset with streaming.

        Yields:
            Tokenized examples
        """
        import random

        if self.seed is not None:
            random.seed(self.seed)

        buffer = []

        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Add to buffer
                buffer.append(record)

                # Yield from buffer when full
                if len(buffer) >= self.buffer_size:
                    if self.shuffle_buffer:
                        random.shuffle(buffer)

                    for record in buffer:
                        yield self._process_record(record)

                    buffer = []

            # Yield remaining buffer
            if buffer:
                if self.shuffle_buffer:
                    random.shuffle(buffer)

                for record in buffer:
                    yield self._process_record(record)

    def _process_record(self, record: Dict) -> Dict[str, torch.Tensor]:
        """Process a single record (tokenize and format).

        Args:
            record: Data record

        Returns:
            Tokenized example
        """
        if self.use_pretokenized:
            # Use pre-tokenized data
            input_ids = torch.tensor(record[self.pretokenized_field], dtype=torch.long)

            # Truncate
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]

            # Pad
            if len(input_ids) < self.max_length:
                padding_length = self.max_length - len(input_ids)
                input_ids = torch.cat([
                    input_ids,
                    torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=torch.long)
                ])

        else:
            # Tokenize raw text
            text = record.get(self.text_field, "")

            encoded = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            input_ids = encoded['input_ids'].squeeze(0)

        # Create labels (no manual shift; HF models shift internally)
        labels = input_ids.clone()
        labels[-1] = -100  # Ignore last position (no target)
        if self.tokenizer.pad_token_id is not None:
            pad_id = self.tokenizer.pad_token_id
            # mask only trailing padding, not valid eos tokens
            seq_len = len(labels)
            while seq_len > 0 and labels[seq_len - 1] == pad_id:
                labels[seq_len - 1] = -100
                seq_len -= 1

        # Create attention mask (aligned with input_ids, NOT shifted)
        # FIX: Don't shift attention mask - keep it aligned with input positions
        # For next-token prediction: position i predicts i+1, we compute loss at all non-pad positions
        # Original mask: [1, 1, 1, 0, 0] for [BOS, tok1, tok2, PAD, PAD]
        # We want:       [1, 1, 1, 0, 0] (compute loss on all valid tokens including last)
        if self.tokenizer.pad_token_id is not None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        else:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


def load_llama_tokenizer(
    model_name: str = "meta-llama/Llama-3.2-1B",
    cache_dir: Optional[str] = None,
    use_fast: bool = True,
    adapter_path: Optional[str] = None,
    hf_token: Optional[str] = None,
    trust_remote_code: bool = True,
) -> AutoTokenizer:
    """Load Llama tokenizer for distillation.

    This is a convenience function that:
    - Loads Llama-3.2-1B tokenizer (vocab_size=128256)
    - Verifies vocab size
    - Sets pad_token if needed (Llama models don't have pad token by default)
    - Handles BOS/EOS/PAD tokens correctly

    Args:
        model_name: HuggingFace model name (default: meta-llama/Llama-3.2-1B)
        cache_dir: Optional cache directory for tokenizer files
        use_fast: Whether to use fast tokenizer (recommended)

    Returns:
        Loaded tokenizer

    Raises:
        ValueError: If tokenizer cannot be loaded or vocab size is wrong

    Example:
        >>> tokenizer = load_llama_tokenizer()
        >>> tokenizer.vocab_size
        128256
        >>> tokens = tokenizer.encode("Hello world", add_special_tokens=True)
        >>> len(tokens)
        4
    """

    candidate_sources: List[str] = []

    # Prefer tokenizer files that ship with the adapter (if provided)
    if adapter_path:
        adapter_dir = Path(adapter_path)
        if adapter_dir.is_file():
            adapter_dir = adapter_dir.parent

        token_files = ["tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json"]
        if adapter_dir.exists() and any((adapter_dir / f).exists() for f in token_files):
            candidate_sources.append(str(adapter_dir.resolve()))
        else:
            logger.debug(
                "Adapter tokenizer files not found in %s (files checked: %s)",
                adapter_dir,
                token_files,
            )

    # Fallback to the HuggingFace model name
    candidate_sources.append(model_name)

    last_error: Optional[Exception] = None

    for source in candidate_sources:
        try:
            logger.info(f"Loading tokenizer from: {source}")

            load_kwargs = {
                "use_fast": use_fast,
                "trust_remote_code": trust_remote_code,
            }

            if cache_dir:
                load_kwargs["cache_dir"] = cache_dir

            # If source is a local path, avoid accidental network calls
            source_path = Path(source)
            if source_path.exists():
                load_kwargs["local_files_only"] = True
            elif hf_token:
                load_kwargs["token"] = hf_token

            tokenizer = AutoTokenizer.from_pretrained(source, **load_kwargs)

            # Verify vocab size (Llama models typically have 128000 or 128256)
            if tokenizer.vocab_size not in [128000, 128256]:
                print(f"Warning: Unexpected vocab_size={tokenizer.vocab_size}, expected 128000 or 128256")
                print("Continuing anyway, but model may need adjustment")

            # Set pad token if not set (Llama models often don't have pad token)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                print(f"Set pad_token to eos_token: {tokenizer.eos_token}")

            # Print special tokens for debugging
            print(f"Tokenizer loaded: {source}")
            print(f"  vocab_size: {tokenizer.vocab_size}")
            print(f"  bos_token: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")
            print(f"  eos_token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
            print(f"  pad_token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")

            return tokenizer

        except Exception as e:
            last_error = e
            logger.warning(f"Failed to load tokenizer from {source}: {e}")
            continue

    raise ValueError(
        f"Failed to load tokenizer. Tried sources: {candidate_sources}. Last error: {last_error}"
    )


def streaming_collate_fn(batch):
    """
    Convert batch items to tensors on-demand for streaming dataloaders.

    This function is defined at module level (not inside create_streaming_dataloaders)
    to allow pickling for multiprocessing workers on Windows.

    Args:
        batch: List of dict items from dataset, each with keys:
            - 'input_ids': List[int] or torch.Tensor
            - 'attention_mask': List[int] or torch.Tensor
            - 'labels': List[int] or torch.Tensor

    Returns:
        dict: Batch with torch.long tensors for each key
    """
    return {
        'input_ids': torch.stack([
            item['input_ids'] if isinstance(item['input_ids'], torch.Tensor)
            else torch.tensor(item['input_ids'], dtype=torch.long)
            for item in batch
        ]),
        'attention_mask': torch.stack([
            item['attention_mask'] if isinstance(item['attention_mask'], torch.Tensor)
            else torch.tensor(item['attention_mask'], dtype=torch.long)
            for item in batch
        ]),
        'labels': torch.stack([
            item['labels'] if isinstance(item['labels'], torch.Tensor)
            else torch.tensor(item['labels'], dtype=torch.long)
            for item in batch
        ]),
    }


def create_streaming_dataloaders(
    train_dataset,
    val_dataset: Optional[Any],
    batch_size: int,
    num_workers: int = 2,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    shuffle_train: bool = True,
    drop_last: bool = True,
):
    """
    Create memory-efficient streaming PyTorch DataLoaders for training and validation.

    This function implements a streaming approach that converts data to tensors on-demand
    rather than upfront, significantly reducing memory usage. Key differences from
    standard DataLoader creation:

    1. No upfront conversion: Removes set_format(type='torch') calls that convert
       the entire dataset to tensors at once
    2. On-demand conversion: Uses custom collate_fn to convert batches to tensors
       only when needed during iteration
    3. Reduced workers: Defaults to 2 workers (vs 4) to minimize memory overhead
    4. Explicit prefetch: Sets prefetch_factor=2 to limit memory from prefetching
    5. Worker reuse: Uses persistent_workers=True to avoid worker recreation overhead

    Memory benefits:
    - Original approach: Entire dataset converted to tensors in memory
    - Streaming approach: Only active batches (batch_size * prefetch_factor * num_workers)
      are converted to tensors at any time
    - For large datasets (millions of sequences), this can reduce memory by 10-100x

    Args:
        train_dataset: Training dataset (HuggingFace Dataset or SimpleDataLoader)
        val_dataset: Validation dataset (optional)
        batch_size: Batch size for DataLoader
        num_workers: Number of DataLoader worker processes (default: 2, reduced from 4)
        pin_memory: Pin memory for faster GPU transfer (default: True)
        prefetch_factor: Number of batches to prefetch per worker (default: 2)
        shuffle_train: Whether to shuffle training data (default: True)
        drop_last: Drop incomplete batches (default: True)

    Returns:
        Tuple[DataLoader, Optional[DataLoader]]: (train_loader, val_loader)
            Each loader yields batches with torch.long tensors:
            - input_ids: Shape [batch_size, seq_length]
            - attention_mask: Shape [batch_size, seq_length]
            - labels: Shape [batch_size, seq_length]

    Example:
        >>> # Memory-efficient loading for large datasets
        >>> train_loader, val_loader = create_streaming_dataloaders(
        ...     train_dataset, val_dataset, batch_size=32, num_workers=2
        ... )
        >>> for batch in train_loader:
        ...     input_ids = batch['input_ids']  # Shape: [32, 4096]
        ...     labels = batch['labels']        # Shape: [32, 4096]
        ...     # Only this batch is converted to tensors, not entire dataset

    Performance notes:
        - Use num_workers=0 for debugging (single-process, easier to debug)
        - Use num_workers=2-4 for training (parallel data loading)
        - Increase prefetch_factor if GPU is starved waiting for data
        - Decrease prefetch_factor if running out of memory
    """
    from torch.utils.data import DataLoader

    # Note: collate_fn is defined at module level (streaming_collate_fn)
    # to allow pickling for multiprocessing workers

    # Create DataLoaders with streaming configuration
    # Note: We do NOT call set_format(type='torch') - this is intentional
    # to avoid converting the entire dataset to tensors upfront

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=streaming_collate_fn,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            collate_fn=streaming_collate_fn,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False,
        )

    return train_loader, val_loader


class PretokenizedShardDataset(Dataset):
    """Data loader for pretokenized parquet shards with manifest-based split filtering.

    Designed for consuming large pretokenized corpora organized as:
    - manifest.json (metadata about all splits)
    - split_name/shard_0000.parquet, shard_0001.parquet, ...

    Features:
    - Manifest detection: Automatically detects manifest.json in data_path
    - Split filtering: Optional filtering to specific splits via config
    - Lazy loading: Uses PyArrow to stream shards without loading into RAM
    - Memory-efficient: Only loads individual rows on-demand
    - Deterministic length: Calculates total sequences from manifest
    - Compatible with PyTorch DataLoader

    Example manifest.json:
        {
          "tokenizer": "meta-llama/Llama-3.2-1B",
          "max_seq_length": 8192,
          "splits": {
            "openhermes": {
              "path": "data/distillation_preprocessed/openhermes",
              "shards": 2,
              "sequences": 105222,
              "tokens": 866206327
            }
          }
        }

    Example:
        >>> dataset = PretokenizedShardDataset(
        ...     data_path="data/distillation_preprocessed",
        ...     max_length=4096,
        ...     splits=["openhermes", "numina_cot"]
        ... )
        >>> batch = dataset[0]
        >>> batch.keys()
        dict_keys(['input_ids', 'attention_mask', 'labels'])
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        max_length: int = 4096,
        splits: Optional[List[str]] = None,
        tokenizer_pad_token_id: Optional[int] = None,
        tokenizer_eos_token_id: Optional[int] = None,
    ):
        """Initialize PretokenizedShardDataset.

        Args:
            data_path: Path to directory containing manifest.json
            max_length: Maximum sequence length for truncation (default 4096)
            splits: Optional list of split names to load (None = all splits)
            tokenizer_pad_token_id: Pad token ID for masking
            tokenizer_eos_token_id: EOS token ID for detecting padding boundaries
        """
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.splits_filter = splits
        
        # Validate data_path is a directory
        if not self.data_path.is_dir():
            raise ValueError(f"data_path must be a directory, got: {self.data_path}")

        # Load and parse manifest
        manifest_path = self.data_path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"manifest.json not found in {self.data_path}. "
                "This dataset requires a manifest file. "
                "Use ParquetDataLoader for non-manifest parquet data."
            )

        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
            
        # Resolve token IDs
        self.pad_token_id = tokenizer_pad_token_id if tokenizer_pad_token_id is not None else 128001
        self.eos_token_id = tokenizer_eos_token_id  # Can be None if not provided

        logger.info(f"Loaded manifest from {manifest_path}")
        logger.info(f"  Tokenizer: {self.manifest.get('tokenizer', 'unknown')}")
        logger.info(f"  Max seq length: {self.manifest.get('max_seq_length', 'unknown')}")
        logger.info(f"  Total splits: {len(self.manifest['splits'])}")

        # Filter splits if specified
        available_splits = list(self.manifest['splits'].keys())
        if self.splits_filter is not None:
            # Validate requested splits exist
            invalid_splits = set(self.splits_filter) - set(available_splits)
            if invalid_splits:
                raise ValueError(
                    f"Requested splits not found in manifest: {invalid_splits}. "
                    f"Available splits: {available_splits}"
                )
            selected_splits = self.splits_filter
            logger.info(f"  Filtering to {len(selected_splits)} splits: {selected_splits}")
        else:
            selected_splits = available_splits
            logger.info(f"  Using all {len(selected_splits)} splits")

        # Build index map: global_idx -> (split_name, shard_idx, row_idx)
        # This is lightweight - just counts, doesn't load data
        self.index_map = []
        self.total_sequences = 0

        for split_name in selected_splits:
            split_info = self.manifest['splits'][split_name]
            split_path_str = split_info['path']

            # FIX: Robust path resolution
            # 1. Try absolute path
            split_path = Path(split_path_str)
            if not split_path.is_absolute():
                # 2. Try relative to manifest directory (most likely)
                candidate_1 = self.data_path / split_path_str
                # 3. Try relative to project root (fallback)
                candidate_2 = Path(split_path_str)
                # 4. Try relative to data_path parent (legacy behavior)
                candidate_3 = self.data_path.parent / split_path_str
                
                if candidate_1.exists():
                    split_path = candidate_1
                elif candidate_2.exists():
                    split_path = candidate_2
                elif candidate_3.exists():
                    split_path = candidate_3
                else:
                    # Default to candidate_1 but warn
                    logger.warning(f"Could not resolve path for split {split_name}: {split_path_str}")
                    logger.warning(f"  Tried: {candidate_1}, {candidate_2}, {candidate_3}")
                    split_path = candidate_1

            num_shards = split_info['shards']
            num_sequences = split_info['sequences']

            if num_shards == 0 or num_sequences == 0:
                logger.warning(f"Skipping empty split: {split_name} (0 shards or sequences)")
                continue

            # Find all shard files for this split
            shard_files = sorted(glob.glob(str(split_path / "shard_*.parquet")))
            if len(shard_files) != num_shards:
                logger.warning(
                    f"Split {split_name}: manifest claims {num_shards} shards, "
                    f"but found {len(shard_files)} shard files. Using actual files."
                )

            # Index each shard
            for shard_idx, shard_file in enumerate(shard_files):
                # Get row count from parquet metadata (fast, no data loading)
                try:
                    metadata = pq.read_metadata(shard_file)
                    num_rows = metadata.num_rows
                except Exception as e:
                    logger.error(f"Failed to read metadata from {shard_file}: {e}")
                    continue

                # Add entries to index map
                for row_idx in range(num_rows):
                    self.index_map.append((split_name, shard_file, row_idx))

                self.total_sequences += num_rows

        if self.total_sequences == 0:
            raise ValueError(
                f"No sequences found in any splits at {self.data_path}. "
                "Check if manifest paths are correct and parquet files exist."
            )

        logger.info(f"Indexed {self.total_sequences:,} sequences from {len(selected_splits)} split(s)")

        # Cache for opened parquet files (lazy loading)
        self._file_cache = {}

    def __len__(self) -> int:
        """Return total number of sequences."""
        return self.total_sequences

    def _get_table(self, shard_file: str):
        """Get parquet table for a shard (with caching)."""
        if shard_file not in self._file_cache:
            # Open parquet file and cache it (memory-mapped, efficient)
            self._file_cache[shard_file] = pq.read_table(shard_file)
        return self._file_cache[shard_file]

    def cleanup_file_cache(self, keep_recent: int = 5):
        """Close old file handles to prevent memory leak.

        CRITICAL FIX: PyArrow memory-maps parquet files and never releases them,
        causing 241GB memory accumulation over time. This method periodically
        clears the cache to prevent OOM kills.

        Args:
            keep_recent: Keep this many most recently used files open (ignored for now,
                        we do full clear for simplicity)
        """
        if len(self._file_cache) == 0:
            return

        logger.info(f"Clearing parquet file cache ({len(self._file_cache)} files open)")
        self._file_cache.clear()

        # Force garbage collection to release memory-mapped regions
        import gc
        gc.collect()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sequence by global index.

        Args:
            idx: Global index across all splits and shards

        Returns:
            Dictionary with:
            - input_ids: [seq_len] token IDs
            - attention_mask: [seq_len] attention mask (shifted for next-token prediction)
            - labels: [seq_len] labels (shifted input_ids, -100 for padding/last token)
        """
        if idx >= self.total_sequences:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_sequences}")

        # Lookup split, shard, and row from index map
        split_name, shard_file, row_idx = self.index_map[idx]

        # Load shard table (cached)
        table = self._get_table(shard_file)

        # Extract input_ids from parquet
        # Parquet stores as fixed_size_list<int32>[N], convert to Python list
        input_ids_list = table['input_ids'][row_idx].as_py()

        # Convert to tensor
        input_ids = torch.tensor(input_ids_list, dtype=torch.long)

        # Truncate if needed
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]

        seq_len = len(input_ids)

        # Attention mask (1 = real token, 0 = padding)
        if self.pad_token_id is not None:
            attention_mask = (input_ids != self.pad_token_id).long()
            valid_length = attention_mask.sum().item()
        else:
            attention_mask = torch.ones(seq_len, dtype=torch.long)
            valid_length = seq_len

        # Ensure we have at least one valid token
        valid_length = max(1, valid_length)

        # Create labels with left-shift (predict next token)
        labels = input_ids.clone()

        if valid_length > 1:
            labels[:valid_length - 1] = input_ids[1:valid_length]

        # No target after the last valid token (or padding positions)
        labels[valid_length - 1:] = -100

        # Mask padding tokens explicitly (in case padding exists before truncation)
        labels[attention_mask == 0] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

    def __getstate__(self):
        """Prepare dataset for pickling (for DataLoader workers).

        PyArrow file handles are not fork-safe. Clear the file cache
        to avoid issues. Each worker will build its own cache.
        """
        state = self.__dict__.copy()
        state['_file_cache'] = {}  # Clear file cache - not fork-safe
        return state

    def __setstate__(self, state):
        """Restore dataset after unpickling in worker process."""
        self.__dict__.update(state)

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function for DataLoader.

        Args:
            batch: List of examples from __getitem__

        Returns:
            Batched tensors with shape [batch_size, seq_length]
        """
        result = {}
        for key in batch[0].keys():
            tensors = [item[key] for item in batch]
            result[key] = torch.stack(tensors, dim=0)
        return result


# Export public API
__all__ = [
    'SimpleDataLoader',
    'StreamingDataLoader',
    'ParquetDataLoader',
    'PretokenizedShardDataset',
    'load_llama_tokenizer',
    'streaming_collate_fn',
    'create_streaming_dataloaders',
]
