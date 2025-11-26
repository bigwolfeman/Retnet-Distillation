#!/usr/bin/env python3
"""
Preprocess downloaded data into tokenized parquet shards.
Creates efficient parquet files with tokenized sequences for training.
Supports both single-file and directory-based datasets.
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

# Configuration
TOKENIZER_NAME = "meta-llama/Llama-3.2-1B"
MAX_SEQ_LENGTH = 8192  # 8K tokens - good balance for 1B model
SHARD_SIZE = 2 * 1024 * 1024 * 1024  # 2GB per shard

# Default Input/Output paths (can be overridden by CLI args)
DEFAULT_INPUT = Path("/mnt/BigAssDrive/00projects/00DeepNet/000Distill-Titan-Retnet-HRM/data/fineweb_large")
DEFAULT_OUTPUT = Path("/mnt/BigAssDrive/00projects/00DeepNet/000Distill-Titan-Retnet-HRM/data/fineweb_large_preprocessed")


def pack_sequences(tokens_list: List[List[int]], max_length: int) -> List[np.ndarray]:
    """
    Pack token sequences into fixed-length sequences.
    Concatenates all tokens and splits into max_length chunks.
    """
    # Flatten all tokens into one long sequence
    all_tokens = []
    for tokens in tokens_list:
        all_tokens.extend(tokens)

    # Split into max_length sequences
    sequences = []
    for i in range(0, len(all_tokens), max_length):
        seq = all_tokens[i:i + max_length]
        if len(seq) == max_length:  # Only keep full sequences
            sequences.append(np.array(seq, dtype=np.int32))

    return sequences

def process_split(
    input_file: Path,
    output_dir: Path,
    split_name: str,
    tokenizer,
    max_seq_length: int,
    shard_size: int
):
    """Process a single split (train or val) and save as parquet shards."""
    print(f"\n{'='*60}")
    print(f"Processing {split_name} split")
    print(f"{'='*60}")
    print(f"Input: {input_file}")
    print(f"Output: {output_dir}")

    # Count total lines for progress bar
    print("Counting samples...")
    with open(input_file, 'r') as f:
        total_samples = sum(1 for _ in f)
    print(f"Total samples: {total_samples:,}")

    # Load tokenizer
    print("Loading tokenizer...")

    # Statistics
    stats = {
        'total_samples': 0,
        'total_tokens': 0,
        'total_sequences': 0,
        'shards_created': 0
    }

    # Buffer for accumulating sequences
    sequence_buffer = []
    token_buffer = []
    buffer_size_bytes = 0
    shard_idx = 0

    # Process samples
    print("Tokenizing and packing sequences...")
    with open(input_file, 'r') as f:
        for line in tqdm(f, total=total_samples, desc=f"Processing {split_name}"):
            sample = json.loads(line)
            text = sample['text']

            # Tokenize
            tokens = tokenizer.encode(text, add_special_tokens=True)
            token_buffer.append(tokens)

            stats['total_samples'] += 1
            stats['total_tokens'] += len(tokens)

            # Pack sequences when buffer gets large enough (every 1000 samples)
            if len(token_buffer) >= 1000:
                packed = pack_sequences(token_buffer, max_seq_length)
                sequence_buffer.extend(packed)
                token_buffer = []

                # Estimate buffer size (4 bytes per token, max_seq_length tokens per sequence)
                buffer_size_bytes = len(sequence_buffer) * max_seq_length * 4

                # Save shard if buffer is large enough
                if buffer_size_bytes >= shard_size:
                    save_shard(sequence_buffer, output_dir, split_name, shard_idx)
                    stats['total_sequences'] += len(sequence_buffer)
                    stats['shards_created'] += 1
                    sequence_buffer = []
                    buffer_size_bytes = 0
                    shard_idx += 1

    # Pack remaining tokens
    if token_buffer:
        packed = pack_sequences(token_buffer, max_seq_length)
        sequence_buffer.extend(packed)

    # Save final shard if there's data
    if sequence_buffer:
        save_shard(sequence_buffer, output_dir, split_name, shard_idx)
        stats['total_sequences'] += len(sequence_buffer)
        stats['shards_created'] += 1

    # Save statistics
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
    print(f"  Stats saved to: {stats_file}")

    return stats

def save_shard(sequences: List[np.ndarray], output_dir: Path, split_name: str, shard_idx: int):
    """Save a batch of sequences as a parquet shard."""
    # Stack sequences into a 2D array
    sequences_array = np.stack(sequences)  # Shape: (num_sequences, max_seq_length)

    # Create attention masks (all 1s for packed sequences)
    attention_mask = np.ones_like(sequences_array)

    # Create labels (shifted input_ids for causal LM)
    # labels[i] = input_ids[i+1], last token label = -100
    labels = np.zeros_like(sequences_array)
    labels[:, :-1] = sequences_array[:, 1:]  # Shift left
    labels[:, -1] = -100  # Ignore last token

    # Create PyArrow table with all three columns
    table = pa.Table.from_arrays(
        [
            pa.array(sequences_array.tolist(), type=pa.list_(pa.int32(), MAX_SEQ_LENGTH)),
            pa.array(attention_mask.tolist(), type=pa.list_(pa.int32(), MAX_SEQ_LENGTH)),
            pa.array(labels.tolist(), type=pa.list_(pa.int32(), MAX_SEQ_LENGTH))
        ],
        names=['input_ids', 'attention_mask', 'labels']
    )

    # Save as parquet
    shard_file = output_dir / f"shard_{shard_idx:04d}.parquet"
    pq.write_table(table, shard_file, compression='snappy')

    # Print progress
    file_size_mb = shard_file.stat().st_size / (1024 * 1024)
    print(f"  Saved {split_name} shard {shard_idx}: {len(sequences):,} sequences, {file_size_mb:.1f} MB")


def find_jsonl_files(input_path: Path, recursive: bool = False) -> List[Path]:
    """Find all JSONL files in the input path."""
    jsonl_files = []

    if input_path.is_file() and input_path.suffix == '.jsonl':
        # Single file
        jsonl_files.append(input_path)
    elif input_path.is_dir():
        # Directory - search for JSONL files
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
        description="Preprocess JSONL data into tokenized parquet shards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process FineWeb (default):
  python preprocess_to_parquet.py

  # Process curriculum datasets recursively (first time):
  python preprocess_to_parquet.py \\
    --input data/distillation \\
    --output data/distillation_preprocessed \\
    --recursive --skip-existing

  # Add new data and re-run (skips already-processed files):
  python preprocess_to_parquet.py \\
    --input data/distillation \\
    --output data/distillation_preprocessed \\
    --recursive --skip-existing

  # Force re-process everything (e.g., after changing tokenizer):
  python preprocess_to_parquet.py \\
    --input data/distillation \\
    --output data/distillation_preprocessed \\
    --recursive --force --max-seq-length 4096

  # Process single file:
  python preprocess_to_parquet.py \\
    --input data/my_data.jsonl \\
    --output data/my_data_preprocessed
        """
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=DEFAULT_INPUT,
        help='Input path (directory or JSONL file)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_OUTPUT,
        help='Output directory for parquet shards'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Recursively search for JSONL files in subdirectories'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default=TOKENIZER_NAME,
        help=f'Tokenizer to use (default: {TOKENIZER_NAME})'
    )
    parser.add_argument(
        '--max-seq-length',
        type=int,
        default=MAX_SEQ_LENGTH,
        help=f'Maximum sequence length (default: {MAX_SEQ_LENGTH})'
    )
    parser.add_argument(
        '--shard-size',
        type=int,
        default=SHARD_SIZE // (1024**3),  # Convert to GB for input
        help='Target shard size in GB (default: 2)'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip files that have already been processed (check if output directory exists with parquet files)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-processing of all files, even if already processed (overrides --skip-existing)'
    )

    args = parser.parse_args()

    # Convert shard size back to bytes
    shard_size = args.shard_size * 1024**3

    print("="*60)
    print("Data Preprocessing to Parquet")
    print("="*60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Max sequence length: {args.max_seq_length}")
    print(f"Target shard size: {args.shard_size} GB")
    print(f"Recursive search: {args.recursive}")
    print("="*60)

    # Find all JSONL files
    jsonl_files = find_jsonl_files(args.input, args.recursive)

    if not jsonl_files:
        print(f"\nERROR: No JSONL files found in {args.input}")
        if not args.recursive:
            print("Tip: Use --recursive to search subdirectories")
        return

    print(f"\nFound {len(jsonl_files)} JSONL file(s):")
    for f in jsonl_files:
        print(f"  - {f}")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {args.output}")

    # Load tokenizer once
    print(f"\nLoading tokenizer: {args.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size:,}")

    # Process all files
    all_stats = []
    skipped_files = []

    for idx, jsonl_file in enumerate(jsonl_files):
        # Create a subdirectory for each source file
        file_stem = jsonl_file.stem
        output_subdir = args.output / file_stem

        # Check if already processed (unless --force is specified)
        if args.skip_existing and not args.force and is_already_processed(output_subdir):
            print(f"\n{'='*60}")
            print(f"SKIPPING file {idx+1}/{len(jsonl_files)}: {file_stem}")
            print(f"  Already processed (found parquet files in {output_subdir})")
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

        stats = process_split(
            jsonl_file,
            output_subdir,
            file_stem,
            tokenizer,
            args.max_seq_length,
            shard_size
        )
        all_stats.append((file_stem, stats))

    # Create overall manifest
    manifest = {
        'tokenizer': args.tokenizer,
        'max_seq_length': args.max_seq_length,
        'shard_size_gb': args.shard_size,
        'input_path': str(args.input),
        'output_path': str(args.output),
        'files_processed': len(jsonl_files),
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
    print(f"  Avg tokens/sample: {total_tokens / total_samples:.1f}" if total_samples > 0 else "")

    print(f"\nTraining Estimates:")
    print(f"  Steps per epoch (batch=16): {total_sequences / 16:,.0f}")
    print(f"  Steps per epoch (batch=32): {total_sequences / 32:,.0f}")

    print(f"\nOutput directory: {args.output}")
    print(f"Manifest: {manifest_file}")
    print("\nReady for training!")

if __name__ == "__main__":
    main()
