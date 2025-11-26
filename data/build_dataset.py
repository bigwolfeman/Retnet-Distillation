"""
Dataset builder CLI for curriculum training data generation.

Per tasks.md T027: Accept --bands, --samples, --out, --seed flags.
Calls generators and emits JSONL shards to data/shards/{band}/{split}.{shard}.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.generators.gen_format import generate_format_batch
from data.generators.gen_a0_a1 import generate_a0_a1_batch
from data.generators.gen_a2 import generate_a2_batch
from data.generators.gen_a3_a4 import generate_a3_a4_batch
from data.generators.gen_a5_a6 import generate_a5_a6_batch
from data.generators.gen_a7 import generate_a7_batch
from data.generators.gen_a8 import generate_a8_batch
from data.generators.gen_a9 import generate_a9_batch
from data.generators.gen_a10 import generate_a10_batch
from data.generators.gen_a11 import generate_a11_batch
from data.generators.gen_a12 import generate_a12_batch
from data.generators.gen_mbpp_lite import generate_mbpp_lite_batch
from data.schema import DataRecord


# Generator registry
GENERATORS = {
    "FORMAT": generate_format_batch,
    "A0": generate_a0_a1_batch,
    "A1": generate_a0_a1_batch,
    "A2": generate_a2_batch,
    "A3": generate_a3_a4_batch,
    "A4": generate_a3_a4_batch,
    "A5": generate_a5_a6_batch,
    "A6": generate_a5_a6_batch,
    "A7": generate_a7_batch,
    "A8": generate_a8_batch,
    "A9": generate_a9_batch,
    "A10": generate_a10_batch,
    "A11": generate_a11_batch,
    "A12": generate_a12_batch,
    "MBPP_LITE": generate_mbpp_lite_batch,
}

# Default sample counts per band (from plan.md and curriculum.yaml)
DEFAULT_SAMPLE_COUNTS = {
    "FORMAT": {"train": 10000, "val": 1000, "test": 1000},
    "A0": {"train": 10000, "val": 1000, "test": 1000},
    "A1": {"train": 10000, "val": 1000, "test": 1000},
    "A2": {"train": 400000, "val": 20000, "test": 20000},
    "A3": {"train": 300000, "val": 15000, "test": 15000},
    "A4": {"train": 300000, "val": 15000, "test": 15000},
    # ... more bands
}


def save_jsonl(records: List[DataRecord], output_path: Path):
    """
    Save records to JSONL format.

    Args:
        records: List of DataRecords
        output_path: Path to output file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            json_str = json.dumps(record.to_dict(), ensure_ascii=False)
            f.write(json_str + '\n')


def build_band_dataset(
    band: str,
    num_samples: int,
    output_dir: Path,
    seed: int,
    shard_size: int = 10000,
):
    """
    Build dataset for a single band.

    Args:
        band: Band ID (A0, A1, A2, etc.)
        num_samples: Total number of samples to generate
        output_dir: Output directory for JSONL files
        seed: Random seed base
        shard_size: Number of samples per shard file
    """
    print(f"\nGenerating {band} dataset...")
    print(f"  Samples: {num_samples}")
    print(f"  Seed: {seed}")

    # Get generator
    generator = GENERATORS.get(band)
    if generator is None:
        print(f"  ✗ No generator found for {band}")
        return

    # Calculate split sizes (70% train, 15% val, 15% test)
    train_size = int(num_samples * 0.70)
    val_size = int(num_samples * 0.15)
    test_size = num_samples - train_size - val_size

    # Generate each split
    splits = [
        ("train", train_size, seed),
        ("val", val_size, seed + train_size),
        ("test", test_size, seed + train_size + val_size),
    ]

    total_generated = 0
    total_verified = 0

    for split_name, split_size, split_seed in splits:
        print(f"\n  {split_name.upper()}: Generating {split_size} samples...")

        # Generate samples
        records = generator(
            num_samples=split_size,
            seed_start=split_seed,
            split=split_name,
            band=band,
        )

        # Verify all pass
        passed = sum(1 for r in records if r.verifier["ok"])
        failed = len(records) - passed

        total_generated += len(records)
        total_verified += passed

        if failed > 0:
            print(f"    ✗ Verifier failures: {failed}/{len(records)}")
            print(f"    First failure:")
            for r in records:
                if not r.verifier["ok"]:
                    print(f"      {r.to_dict()}")
                    break
            return

        # Save to shards
        shard_num = 0
        for start_idx in range(0, len(records), shard_size):
            end_idx = min(start_idx + shard_size, len(records))
            shard_records = records[start_idx:end_idx]

            # Output path: data/shards/{band}/{split}.{shard_num:04d}.jsonl
            output_path = output_dir / band / f"{split_name}.{shard_num:04d}.jsonl"

            save_jsonl(shard_records, output_path)

            print(f"    Saved {len(shard_records)} samples to {output_path.name}")
            shard_num += 1

    # Summary
    print(f"\n  ✓ Generated {total_generated} samples")
    print(f"  ✓ Verifier pass rate: {total_verified}/{total_generated} (100%)")


def main():
    """Main CLI entry point."""
    # Fix Unicode encoding on Windows
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    parser = argparse.ArgumentParser(
        description="Build curriculum training dataset"
    )

    parser.add_argument(
        "--bands",
        nargs="+",
        required=True,
        help="Bands to generate (e.g., A0 A1 A2)",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of samples per band (default: use band-specific defaults)",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/shards"),
        help="Output directory for JSONL shards (default: data/shards)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help="Random seed base (default: 1000)",
    )

    parser.add_argument(
        "--shard-size",
        type=int,
        default=10000,
        help="Samples per shard file (default: 10000)",
    )

    args = parser.parse_args()

    # Print configuration
    print("=== Dataset Builder ===")
    print(f"Bands: {args.bands}")
    print(f"Output directory: {args.out}")
    print(f"Seed: {args.seed}")
    print(f"Shard size: {args.shard_size}")

    # Build datasets
    for band in args.bands:
        # Get sample count
        if args.samples:
            num_samples = args.samples
        else:
            # Use default for this band
            num_samples = sum(DEFAULT_SAMPLE_COUNTS.get(band, {
                "train": 10000,
                "val": 1000,
                "test": 1000
            }).values())

        build_band_dataset(
            band=band,
            num_samples=num_samples,
            output_dir=args.out,
            seed=args.seed,
            shard_size=args.shard_size,
        )

    print("\n✓ All datasets generated successfully!")


if __name__ == "__main__":
    main()
