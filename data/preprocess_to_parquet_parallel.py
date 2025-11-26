#!/usr/bin/env python3
"""
Parallel preprocessing of JSONL data to tokenized parquet shards.
Uses multiprocessing for fast tokenization on multi-core systems.
"""

import json
import os
import argparse
from pathlib import Path
from typing import List, Dict, Any
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial

# Configuration
TOKENIZER_NAME = "meta-llama/Llama-3.2-1B"
MAX_SEQ_LENGTH = 8192
SHARD_SIZE = 2 * 1024 * 1024 * 1024  # 2GB per shard
DEFAULT_INPUT = Path("distillation")
DEFAULT_OUTPUT = Path("distillation_preprocessed")

def tokenize_batch(lines: List[str], tokenizer_name: str) -> List[List[int]]:
    """Tokenize a batch of lines in a worker process."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    results = []
    for line in lines:
        try:
            sample = json.loads(line)
            text = sample.get('text', '')
            tokens = tokenizer.encode(text, add_special_tokens=True)
            results.append(tokens)
        except Exception as e:
            # Skip malformed lines
            results.append([])
    return results

def pack_sequences(tokens_list: List[List[int]], max_length: int) -> List[np.ndarray]:
    """Pack token sequences into fixed-length sequences."""
    all_tokens = []
    for tokens in tokens_list:
        all_tokens.extend(tokens)

    sequences = []
    for i in range(0, len(all_tokens), max_length):
        seq = all_tokens[i:i + max_length]
        if len(seq) == max_length:
            sequences.append(np.array(seq, dtype=np.int32))

    return sequences

def save_shard(sequences: List[np.ndarray], output_dir: Path, split_name: str, shard_idx: int, max_seq_length: int):
    """Save a batch of sequences as a parquet shard."""
    sequences_array = np.stack(sequences)
    attention_mask = np.ones_like(sequences_array)
    labels = np.zeros_like(sequences_array)
    labels[:, :-1] = sequences_array[:, 1:]
    labels[:, -1] = -100

    table = pa.Table.from_arrays(
        [
            pa.array(sequences_array.tolist(), type=pa.list_(pa.int32(), max_seq_length)),
            pa.array(attention_mask.tolist(), type=pa.list_(pa.int32(), max_seq_length)),
            pa.array(labels.tolist(), type=pa.list_(pa.int32(), max_seq_length))
        ],
        names=['input_ids', 'attention_mask', 'labels']
    )

    shard_file = output_dir / f"shard_{shard_idx:04d}.parquet"
    pq.write_table(table, shard_file, compression='snappy')

    file_size_mb = shard_file.stat().st_size / (1024 * 1024)
    print(f"  Saved {split_name} shard {shard_idx}: {len(sequences):,} sequences, {file_size_mb:.1f} MB")

def process_split_parallel(
    input_file: Path,
    output_dir: Path,
    split_name: str,
    tokenizer_name: str,
    max_seq_length: int,
    shard_size: int,
    num_workers: int
):
    """Process a single split with parallel tokenization."""
    print(f"\n{'='*60}")
    print(f"Processing {split_name} split (parallel)")
    print(f"{'='*60}")
    print(f"Input: {input_file}")
    print(f"Output: {output_dir}")
    print(f"Workers: {num_workers}")

    # Count total lines
    print("Counting samples...")
    with open(input_file, 'r') as f:
        total_samples = sum(1 for _ in f)
    print(f"Total samples: {total_samples:,}")

    # Statistics
    stats = {
        'total_samples': 0,
        'total_tokens': 0,
        'total_sequences': 0,
        'shards_created': 0
    }

    # Batching parameters
    batch_size = max(1000, num_workers * 100)  # Larger batches for efficiency
    sequence_buffer = []
    shard_idx = 0

    # Read file in batches and process in parallel
    print(f"Tokenizing with {num_workers} workers (batch size: {batch_size})...")

    with open(input_file, 'r') as f:
        batch = []
        with Pool(processes=num_workers) as pool:
            with tqdm(total=total_samples, desc=f"Processing {split_name}") as pbar:
                for line in f:
                    batch.append(line)

                    if len(batch) >= batch_size:
                        # Process batch in parallel
                        tokenize_fn = partial(tokenize_batch, tokenizer_name=tokenizer_name)

                        # Split batch into chunks for workers
                        chunk_size = max(1, len(batch) // num_workers)
                        chunks = [batch[i:i + chunk_size] for i in range(0, len(batch), chunk_size)]

                        # Parallel tokenization
                        results = pool.map(tokenize_fn, chunks)

                        # Flatten results
                        all_tokens = []
                        for chunk_tokens in results:
                            all_tokens.extend(chunk_tokens)

                        # Update stats
                        stats['total_samples'] += len(all_tokens)
                        for tokens in all_tokens:
                            stats['total_tokens'] += len(tokens)

                        # Pack sequences
                        packed = pack_sequences(all_tokens, max_seq_length)
                        sequence_buffer.extend(packed)

                        # Save shard if buffer is large enough
                        buffer_size_bytes = len(sequence_buffer) * max_seq_length * 4
                        if buffer_size_bytes >= shard_size:
                            save_shard(sequence_buffer, output_dir, split_name, shard_idx, max_seq_length)
                            stats['total_sequences'] += len(sequence_buffer)
                            stats['shards_created'] += 1
                            sequence_buffer = []
                            shard_idx += 1

                        pbar.update(len(batch))
                        batch = []

                # Process remaining batch
                if batch:
                    tokenize_fn = partial(tokenize_batch, tokenizer_name=tokenizer_name)
                    chunk_size = max(1, len(batch) // num_workers)
                    chunks = [batch[i:i + chunk_size] for i in range(0, len(batch), chunk_size)]
                    results = pool.map(tokenize_fn, chunks)

                    all_tokens = []
                    for chunk_tokens in results:
                        all_tokens.extend(chunk_tokens)

                    stats['total_samples'] += len(all_tokens)
                    for tokens in all_tokens:
                        stats['total_tokens'] += len(tokens)

                    packed = pack_sequences(all_tokens, max_seq_length)
                    sequence_buffer.extend(packed)
                    pbar.update(len(batch))

    # Save final shard
    if sequence_buffer:
        save_shard(sequence_buffer, output_dir, split_name, shard_idx, max_seq_length)
        stats['total_sequences'] += len(sequence_buffer)
        stats['shards_created'] += 1

    # Save stats
    stats_file = output_dir / f"{split_name}_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print(f"\n{split_name.upper()} Processing Complete:")
    print(f"  Samples processed: {stats['total_samples']:,}")
    print(f"  Total tokens: {stats['total_tokens']:,}")
    print(f"  Sequences created: {stats['total_sequences']:,}")
    print(f"  Shards created: {stats['shards_created']}")
    if stats['total_samples'] > 0:
        print(f"  Avg tokens/sample: {stats['total_tokens'] / stats['total_samples']:.1f}")
        print(f"  Avg sequences/sample: {stats['total_sequences'] / stats['total_samples']:.3f}")
    else:
        print(f"  ⚠️  Empty file - no data to process")
    print(f"  Stats saved to: {stats_file}")

    return stats

def find_jsonl_files(input_path: Path, recursive: bool = False) -> List[Path]:
    """Find all JSONL files in the input path."""
    jsonl_files = []

    if input_path.is_file() and input_path.suffix == '.jsonl':
        jsonl_files.append(input_path)
    elif input_path.is_dir():
        pattern = "**/*.jsonl" if recursive else "*.jsonl"
        jsonl_files = sorted(input_path.glob(pattern))

    return jsonl_files

def is_already_processed(output_subdir: Path) -> bool:
    """Check if a file has already been processed successfully.

    Args:
        output_subdir: Path to the output subdirectory for this file

    Returns:
        True if the directory exists, contains parquet files, AND has a valid stats file with non-zero samples
    """
    if not output_subdir.exists():
        return False

    # Check if directory has any parquet files
    parquet_files = list(output_subdir.glob("*.parquet"))
    if len(parquet_files) == 0:
        return False

    # Check if stats file exists and has valid data
    file_stem = output_subdir.name
    stats_file = output_subdir / f"{file_stem}_stats.json"

    if not stats_file.exists():
        print(f"    Warning: No stats file found for {file_stem}, will re-process")
        return False

    try:
        with open(stats_file, 'r') as f:
            stats = json.load(f)

        # Verify we have actual data (not an empty/failed processing)
        total_samples = stats.get('total_samples', 0)
        total_sequences = stats.get('total_sequences', 0)
        shards_created = stats.get('shards_created', 0)

        if total_samples == 0 or total_sequences == 0 or shards_created == 0:
            print(f"    Warning: {file_stem} has empty stats (samples={total_samples}, sequences={total_sequences}, shards={shards_created})")
            return False

        return True

    except (json.JSONDecodeError, KeyError) as e:
        print(f"    Warning: Invalid stats file for {file_stem}: {e}, will re-process")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Parallel preprocessing of JSONL data to tokenized parquet shards",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument('--tokenizer', type=str, default=TOKENIZER_NAME)
    parser.add_argument('--max-seq-length', type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument('--shard-size', type=int, default=SHARD_SIZE // (1024**3))
    parser.add_argument('--workers', type=int, default=cpu_count() // 2,
                       help='Number of parallel workers (default: half of CPU cores)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip files that have already been processed (check if output directory exists with valid data)')
    parser.add_argument('--force', action='store_true',
                       help='Force re-processing of all files, even if already processed (overrides --skip-existing)')

    args = parser.parse_args()
    shard_size = args.shard_size * 1024**3

    print("="*60)
    print("Parallel Data Preprocessing to Parquet")
    print("="*60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Max sequence length: {args.max_seq_length}")
    print(f"Shard size: {args.shard_size} GB")
    print(f"Workers: {args.workers}")
    print(f"CPU cores available: {cpu_count()}")
    print("="*60)

    # Find files
    jsonl_files = find_jsonl_files(args.input, args.recursive)

    if not jsonl_files:
        print(f"\nERROR: No JSONL files found in {args.input}")
        return

    print(f"\nFound {len(jsonl_files)} JSONL file(s)")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Process all files
    all_stats = []
    skipped_files = []

    for idx, jsonl_file in enumerate(jsonl_files):
        file_stem = jsonl_file.stem
        output_subdir = args.output / file_stem

        # Check if already processed (unless --force is specified)
        if args.skip_existing and not args.force and is_already_processed(output_subdir):
            print(f"\n{'='*60}")
            print(f"SKIPPING file {idx+1}/{len(jsonl_files)}: {file_stem}")
            print(f"  Already processed (found valid parquet files in {output_subdir})")
            print(f"  Use --force to re-process")
            print(f"{'='*60}")
            skipped_files.append(file_stem)

            # Load existing stats if available
            stats_file = output_subdir / f"{file_stem}_stats.json"
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                    all_stats.append((file_stem, stats))
            continue

        output_subdir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Processing file {idx+1}/{len(jsonl_files)}: {file_stem}")
        print(f"{'='*60}")

        stats = process_split_parallel(
            jsonl_file,
            output_subdir,
            file_stem,
            args.tokenizer,
            args.max_seq_length,
            shard_size,
            args.workers
        )
        all_stats.append((file_stem, stats))

    # Create manifest
    manifest = {
        'tokenizer': args.tokenizer,
        'max_seq_length': args.max_seq_length,
        'shard_size_gb': args.shard_size,
        'workers': args.workers,
        'splits': {}
    }

    total_sequences = 0
    total_shards = 0
    total_samples = 0
    total_tokens = 0

    for split_name, stats in all_stats:
        manifest['splits'][split_name] = {
            'path': str(args.output / split_name),
            'shards': stats['shards_created'],
            'sequences': stats['total_sequences'],
            'samples': stats['total_samples'],
            'tokens': stats['total_tokens']
        }
        total_sequences += stats['total_sequences']
        total_shards += stats['shards_created']
        total_samples += stats['total_samples']
        total_tokens += stats['total_tokens']

    manifest_file = args.output / "manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Final summary
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    print(f"\nDataset Summary:")
    print(f"  Files found: {len(jsonl_files)}")
    print(f"  Files processed: {len(all_stats)}")
    if skipped_files:
        print(f"  Files skipped (already processed): {len(skipped_files)}")
        print(f"    Skipped: {', '.join(skipped_files[:5])}")
        if len(skipped_files) > 5:
            print(f"    ... and {len(skipped_files) - 5} more")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Total sequences: {total_sequences:,}")
    print(f"  Total shards: {total_shards}")
    if total_samples > 0:
        print(f"  Avg tokens/sample: {total_tokens / total_samples:.1f}")

    print(f"\nTraining Estimates:")
    print(f"  Steps per epoch (batch=16): {total_sequences / 16:,.0f}")
    print(f"  Steps per epoch (batch=32): {total_sequences / 32:,.0f}")

    print(f"\nOutput directory: {args.output}")
    print(f"Manifest: {manifest_file}")
    print("\nReady for training!")

if __name__ == "__main__":
    main()
