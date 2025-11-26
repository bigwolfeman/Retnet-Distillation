#!/usr/bin/env python3
"""CLI for forecasting attention memory usage (US4)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core.attention import AttentionForecastService


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate attention memory usage and recommended mode.",
    )
    parser.add_argument("--sequence-length", type=int, required=True, help="Sequence length to evaluate.")
    parser.add_argument("--window-size", type=int, default=2048, help="Sliding window size (default: 2048).")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1).")
    parser.add_argument("--num-heads", type=int, default=16, help="Attention heads (default: 16).")
    parser.add_argument("--dtype-bytes", type=int, default=2, help="Bytes per score (default: 2 for bf16/fp16).")
    parser.add_argument("--block-size", type=int, default=128, help="Block size for sparse window (default: 128).")
    parser.add_argument("--memory-ceiling-mb", type=float, default=1024.0, help="Ceiling before switching modes.")
    parser.add_argument("--mode", choices=["DENSE", "SPARSE"], default="DENSE", help="Current attention mode.")
    parser.add_argument("--global-tokens", type=int, default=0, help="Count of global tokens (landmarks).")
    parser.add_argument("--json", action="store_true", help="Emit result as JSON.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    service = AttentionForecastService(
        num_heads=args.num_heads,
        block_size=args.block_size,
        dtype_bytes=args.dtype_bytes,
        memory_ceiling_mb=args.memory_ceiling_mb,
    )

    result = service.forecast(
        args.sequence_length,
        window_size=args.window_size,
        batch_size=args.batch_size,
        current_mode=args.mode,
        global_tokens=args.global_tokens,
    )

    payload = result.to_dict()

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(f"Sequence length: {args.sequence_length}")
    print(f"Window size: {args.window_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Dense usage (MB): {payload['denseUsageMb']:.3f}")
    print(f"Sparse usage (MB): {payload['sparseUsageMb']:.3f}")
    print(f"Recommendation: {payload['recommendation']}")
    print(f"Projected usage (MB): {payload['projectedUsageMb']:.3f}")
    if "triggersSparseAtSequenceLength" in payload:
        print(f"Dense mode reaches ceiling at seq_len ≈ {payload['triggersSparseAtSequenceLength']}")
    if "notes" in payload:
        print(f"Notes: {payload['notes']}")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
