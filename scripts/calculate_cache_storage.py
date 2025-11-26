#!/usr/bin/env python3
"""
Storage Calculator for Teacher Logit Caching

Calculate storage requirements for caching teacher logits in knowledge distillation.
Supports different compression ratios, top-k values, and dataset sizes.

Usage:
    python calculate_cache_storage.py --num-sequences 100000 --seq-len 4096 --top-k 128
    python calculate_cache_storage.py --help
"""

import argparse
import json
from typing import Dict, Any


def bytes_to_human(bytes_val: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def calculate_per_sequence_storage(seq_len: int, top_k: int, vocab_size: int = 128256) -> Dict[str, Any]:
    """
    Calculate storage requirements per sequence.

    Args:
        seq_len: Sequence length in tokens
        top_k: Number of top logits to store
        vocab_size: Vocabulary size (default: Llama-3.2 128256)

    Returns:
        Dict with storage breakdown
    """
    # Top-k indices: uint16 (2 bytes) for vocab < 65536, uint32 (4 bytes) for larger
    # Llama-3.2 vocab_size = 128256 requires uint32
    index_bytes = 2 if vocab_size < 65536 else 4
    indices_size = seq_len * top_k * index_bytes

    # Top-k values: k × 1 byte (int8 quantized)
    values_size = seq_len * top_k * 1

    # Scale factors: seq_len × 4 bytes (float32)
    scale_size = seq_len * 4

    # Other_mass: seq_len × 4 bytes (float32)
    other_mass_size = seq_len * 4

    # Total per sequence
    total_size = indices_size + values_size + scale_size + other_mass_size

    return {
        'indices_bytes': indices_size,
        'values_bytes': values_size,
        'scale_bytes': scale_size,
        'other_mass_bytes': other_mass_size,
        'total_bytes': total_size,
        'total_kb': total_size / 1024,
        'total_mb': total_size / (1024 ** 2),
    }


def calculate_dataset_storage(
    num_sequences: int,
    seq_len: int,
    top_k: int,
    compression_ratio: float = 3.0,
    vocab_size: int = 128256
) -> Dict[str, Any]:
    """
    Calculate storage requirements for entire dataset.

    Args:
        num_sequences: Number of unique sequences
        seq_len: Sequence length in tokens
        top_k: Number of top logits to store
        compression_ratio: Compression ratio (default: 3.0 for parquet)
        vocab_size: Vocabulary size

    Returns:
        Dict with dataset-level storage breakdown
    """
    per_seq = calculate_per_sequence_storage(seq_len, top_k, vocab_size)

    raw_bytes = per_seq['total_bytes'] * num_sequences
    compressed_bytes = raw_bytes / compression_ratio

    return {
        'num_sequences': num_sequences,
        'seq_len': seq_len,
        'top_k': top_k,
        'compression_ratio': compression_ratio,
        'per_sequence': per_seq,
        'raw_bytes': raw_bytes,
        'raw_gb': raw_bytes / (1024 ** 3),
        'compressed_bytes': compressed_bytes,
        'compressed_gb': compressed_bytes / (1024 ** 3),
        'raw_human': bytes_to_human(raw_bytes),
        'compressed_human': bytes_to_human(compressed_bytes),
    }


def estimate_training_time(
    num_sequences: int,
    optimizer_steps: int,
    grad_accum: int,
    epochs: int = 1
) -> Dict[str, Any]:
    """
    Estimate training time requirements.

    Args:
        num_sequences: Number of unique sequences in dataset
        optimizer_steps: Total optimizer steps needed
        grad_accum: Gradient accumulation steps
        epochs: Number of epochs over dataset

    Returns:
        Dict with training time estimates
    """
    # Total forward passes needed
    total_forward_passes = optimizer_steps * grad_accum

    # Forward passes available from dataset
    passes_per_epoch = num_sequences
    total_passes_available = passes_per_epoch * epochs

    # How many epochs needed?
    epochs_needed = total_forward_passes / passes_per_epoch

    # Can we complete training?
    can_complete = total_passes_available >= total_forward_passes

    return {
        'optimizer_steps': optimizer_steps,
        'grad_accum': grad_accum,
        'total_forward_passes': total_forward_passes,
        'passes_per_epoch': passes_per_epoch,
        'epochs_provided': epochs,
        'epochs_needed': epochs_needed,
        'can_complete': can_complete,
        'utilization_pct': min(100.0, (total_forward_passes / total_passes_available) * 100),
    }


def calculate_breakeven(
    num_sequences: int,
    seq_len: int,
    top_k: int,
    cache_time_hours: float,
    training_time_cached_hours: float,
    training_time_online_hours: float,
    compression_ratio: float = 3.0
) -> Dict[str, Any]:
    """
    Calculate break-even analysis for caching vs. online fetching.

    Args:
        num_sequences: Number of unique sequences
        seq_len: Sequence length
        top_k: Top-k value
        cache_time_hours: Time to pre-cache all logits (hours)
        training_time_cached_hours: Training time with cached logits (hours)
        training_time_online_hours: Training time with online fetching (hours)
        compression_ratio: Compression ratio

    Returns:
        Dict with break-even analysis
    """
    storage = calculate_dataset_storage(num_sequences, seq_len, top_k, compression_ratio)

    total_time_cached = cache_time_hours + training_time_cached_hours
    total_time_online = training_time_online_hours

    time_saved = total_time_online - total_time_cached
    speedup = total_time_online / total_time_cached if total_time_cached > 0 else float('inf')

    is_worth_it = time_saved > 0

    return {
        'storage': storage,
        'cache_time_hours': cache_time_hours,
        'training_time_cached_hours': training_time_cached_hours,
        'training_time_online_hours': training_time_online_hours,
        'total_time_cached_hours': total_time_cached,
        'total_time_online_hours': total_time_online,
        'time_saved_hours': time_saved,
        'speedup': speedup,
        'is_worth_it': is_worth_it,
        'recommendation': 'CACHE' if is_worth_it else 'ONLINE',
    }


def generate_decision_matrix(
    seq_len: int = 4096,
    top_k: int = 128,
    optimizer_steps: int = 70000,
    grad_accum: int = 256,
    compression_ratio: float = 3.0,
    steps_per_sec_cached: float = 5.89,
    steps_per_sec_online: float = 0.136
) -> str:
    """
    Generate a decision matrix for different dataset sizes.

    Args:
        seq_len: Sequence length
        top_k: Top-k value
        optimizer_steps: Total optimizer steps
        grad_accum: Gradient accumulation
        compression_ratio: Compression ratio
        steps_per_sec_cached: Training speed with cached logits
        steps_per_sec_online: Training speed with online fetching

    Returns:
        Formatted decision matrix string
    """
    scenarios = [
        ('10k', 10_000),
        ('100k', 100_000),
        ('1M', 1_000_000),
        ('10M', 10_000_000),
    ]

    lines = []
    lines.append("=" * 140)
    lines.append("TEACHER LOGIT CACHING DECISION MATRIX")
    lines.append("=" * 140)
    lines.append(f"Configuration: seq_len={seq_len}, top_k={top_k}, optimizer_steps={optimizer_steps:,}, grad_accum={grad_accum}")
    lines.append(f"Training speed: {steps_per_sec_cached:.2f} steps/sec (cached), {steps_per_sec_online:.3f} steps/sec (online)")
    lines.append("=" * 140)
    lines.append("")

    header = f"{'Dataset':<12} {'Raw Cache':<12} {'Compressed':<12} {'Cache Time':<12} {'Train (cached)':<15} {'Train (online)':<15} {'Speedup':<10} {'Rec':<8}"
    lines.append(header)
    lines.append("-" * 140)

    for name, num_seq in scenarios:
        storage = calculate_dataset_storage(num_seq, seq_len, top_k, compression_ratio)

        # Training time estimates
        training_time_cached_hours = optimizer_steps / steps_per_sec_cached / 3600
        training_time_online_hours = optimizer_steps / steps_per_sec_online / 3600

        # Cache time estimate (assume same speed as online for fetching)
        # Pre-caching needs to fetch logits for all sequences once
        cache_time_hours = num_seq / (steps_per_sec_online * 3600)

        total_cached = cache_time_hours + training_time_cached_hours
        total_online = training_time_online_hours

        speedup = total_online / total_cached if total_cached > 0 else 0

        # Recommendation
        if speedup > 1.5:
            rec = "CACHE"
        elif speedup > 1.0:
            rec = "MAYBE"
        else:
            rec = "ONLINE"

        # Format time
        def format_time(hours):
            if hours < 1:
                return f"{hours * 60:.1f}m"
            elif hours < 24:
                return f"{hours:.1f}h"
            else:
                return f"{hours / 24:.1f}d"

        row = (
            f"{name:<12} "
            f"{storage['raw_human']:<12} "
            f"{storage['compressed_human']:<12} "
            f"{format_time(cache_time_hours):<12} "
            f"{format_time(training_time_cached_hours):<15} "
            f"{format_time(training_time_online_hours):<15} "
            f"{speedup:.2f}x{'':<6} "
            f"{rec:<8}"
        )
        lines.append(row)

    lines.append("=" * 140)
    lines.append("")
    lines.append("Recommendations:")
    lines.append("  - CACHE: Significant speedup, caching is highly recommended")
    lines.append("  - MAYBE: Marginal benefit, consider dataset reuse across experiments")
    lines.append("  - ONLINE: Minimal benefit, online fetching may be simpler")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Calculate storage requirements for teacher logit caching',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calculate for 100k sequences
  python calculate_cache_storage.py --num-sequences 100000

  # Calculate for 1M sequences with higher compression
  python calculate_cache_storage.py --num-sequences 1000000 --compression-ratio 4.0

  # Generate decision matrix
  python calculate_cache_storage.py --decision-matrix

  # Break-even analysis
  python calculate_cache_storage.py --num-sequences 100000 --cache-time 10 --train-cached 5 --train-online 150
        """
    )

    parser.add_argument('--num-sequences', type=int, default=100_000,
                        help='Number of unique sequences (default: 100,000)')
    parser.add_argument('--seq-len', type=int, default=4096,
                        help='Sequence length in tokens (default: 4096)')
    parser.add_argument('--top-k', type=int, default=128,
                        help='Number of top logits to store (default: 128)')
    parser.add_argument('--compression-ratio', type=float, default=3.0,
                        help='Compression ratio (default: 3.0 for parquet)')
    parser.add_argument('--vocab-size', type=int, default=128256,
                        help='Vocabulary size (default: 128256 for Llama-3.2)')

    # Training parameters
    parser.add_argument('--optimizer-steps', type=int, default=70000,
                        help='Total optimizer steps (default: 70,000)')
    parser.add_argument('--grad-accum', type=int, default=256,
                        help='Gradient accumulation steps (default: 256)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs over dataset (default: 1)')

    # Break-even analysis
    parser.add_argument('--cache-time', type=float, default=None,
                        help='Time to pre-cache all logits (hours)')
    parser.add_argument('--train-cached', type=float, default=None,
                        help='Training time with cached logits (hours)')
    parser.add_argument('--train-online', type=float, default=None,
                        help='Training time with online fetching (hours)')

    # Output options
    parser.add_argument('--decision-matrix', action='store_true',
                        help='Generate decision matrix for multiple scenarios')
    parser.add_argument('--json', action='store_true',
                        help='Output results as JSON')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: stdout)')

    args = parser.parse_args()

    if args.decision_matrix:
        # Generate decision matrix
        matrix = generate_decision_matrix(
            seq_len=args.seq_len,
            top_k=args.top_k,
            optimizer_steps=args.optimizer_steps,
            grad_accum=args.grad_accum,
            compression_ratio=args.compression_ratio
        )
        print(matrix)
        return

    # Calculate storage
    storage = calculate_dataset_storage(
        num_sequences=args.num_sequences,
        seq_len=args.seq_len,
        top_k=args.top_k,
        compression_ratio=args.compression_ratio,
        vocab_size=args.vocab_size
    )

    # Training time estimates
    training = estimate_training_time(
        num_sequences=args.num_sequences,
        optimizer_steps=args.optimizer_steps,
        grad_accum=args.grad_accum,
        epochs=args.epochs
    )

    result = {
        'storage': storage,
        'training': training,
    }

    # Break-even analysis if requested
    if args.cache_time is not None and args.train_cached is not None and args.train_online is not None:
        breakeven = calculate_breakeven(
            num_sequences=args.num_sequences,
            seq_len=args.seq_len,
            top_k=args.top_k,
            cache_time_hours=args.cache_time,
            training_time_cached_hours=args.train_cached,
            training_time_online_hours=args.train_online,
            compression_ratio=args.compression_ratio
        )
        result['breakeven'] = breakeven

    if args.json:
        output = json.dumps(result, indent=2)
    else:
        # Human-readable output
        lines = []
        lines.append("=" * 80)
        lines.append("TEACHER LOGIT CACHING STORAGE CALCULATOR")
        lines.append("=" * 80)
        lines.append("")

        lines.append("Storage Requirements:")
        lines.append(f"  Dataset size: {storage['num_sequences']:,} sequences")
        lines.append(f"  Sequence length: {storage['seq_len']:,} tokens")
        lines.append(f"  Top-k: {storage['top_k']}")
        lines.append(f"  Compression ratio: {storage['compression_ratio']:.1f}x")
        lines.append("")

        lines.append("Per-Sequence Storage:")
        per_seq = storage['per_sequence']
        lines.append(f"  Indices: {bytes_to_human(per_seq['indices_bytes'])}")
        lines.append(f"  Values: {bytes_to_human(per_seq['values_bytes'])}")
        lines.append(f"  Scale factors: {bytes_to_human(per_seq['scale_bytes'])}")
        lines.append(f"  Other mass: {bytes_to_human(per_seq['other_mass_bytes'])}")
        lines.append(f"  Total: {bytes_to_human(per_seq['total_bytes'])} ({per_seq['total_mb']:.2f} MB)")
        lines.append("")

        lines.append("Dataset-Level Storage:")
        lines.append(f"  Raw: {storage['raw_human']} ({storage['raw_gb']:.2f} GB)")
        lines.append(f"  Compressed: {storage['compressed_human']} ({storage['compressed_gb']:.2f} GB)")
        lines.append("")

        lines.append("Training Requirements:")
        lines.append(f"  Optimizer steps: {training['optimizer_steps']:,}")
        lines.append(f"  Gradient accumulation: {training['grad_accum']}")
        lines.append(f"  Total forward passes needed: {training['total_forward_passes']:,}")
        lines.append(f"  Forward passes per epoch: {training['passes_per_epoch']:,}")
        lines.append(f"  Epochs needed: {training['epochs_needed']:.2f}")
        lines.append(f"  Can complete training: {'YES' if training['can_complete'] else 'NO'}")
        lines.append(f"  Dataset utilization: {training['utilization_pct']:.1f}%")
        lines.append("")

        if 'breakeven' in result:
            be = result['breakeven']
            lines.append("Break-Even Analysis:")
            lines.append(f"  Cache time: {be['cache_time_hours']:.2f} hours")
            lines.append(f"  Training time (cached): {be['training_time_cached_hours']:.2f} hours")
            lines.append(f"  Training time (online): {be['training_time_online_hours']:.2f} hours")
            lines.append(f"  Total time (cached): {be['total_time_cached_hours']:.2f} hours")
            lines.append(f"  Total time (online): {be['total_time_online_hours']:.2f} hours")
            lines.append(f"  Time saved: {be['time_saved_hours']:.2f} hours")
            lines.append(f"  Speedup: {be['speedup']:.2f}x")
            lines.append(f"  Recommendation: {be['recommendation']}")
            lines.append("")

        lines.append("=" * 80)

        output = "\n".join(lines)

    # Write output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Results written to {args.output}")
    else:
        print(output)


if __name__ == '__main__':
    main()
