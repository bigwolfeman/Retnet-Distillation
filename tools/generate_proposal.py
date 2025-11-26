#!/usr/bin/env python3
"""
CLI for generating proposals using decoding profiles.

This utility is intended for operator sandboxes and smoke tests. It loads the
decoding profile catalog, constructs a SimpleLEngine instance, and generates a
proposal for the provided prompt using the selected profile.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch

from src.core.api.proposals import ProposalService, solution_proposal_to_dict
from src.models.titans.decoding_profiles import DecodingProfileCatalog, load_catalog


DEFAULT_PROFILE_PATH = Path("configs/decoding_profiles.yaml")


def _build_engine(device: str):
    try:
        from src.models.retnet.backbone import RetNetBackbone
        from src.models.titans.l_engine import SimpleLEngine
        from src.data.tokenizer import RetNetTokenizer
    except ImportError as exc:  # pragma: no cover - requires optional dependencies
        raise RuntimeError(
            "Required inference dependencies are missing. Ensure TorchScale and transformers "
            "are installed before running proposal generation."
        ) from exc

    tokenizer = RetNetTokenizer()
    backbone = RetNetBackbone(vocab_size=tokenizer.vocab_size)
    engine = SimpleLEngine(
        engine_id="cli-engine",
        backbone=backbone,
        tokenizer=tokenizer,
        device=device,
    )
    return engine, tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Titan-HRM proposals with decoding profiles.",
    )
    parser.add_argument(
        "--profile-id",
        required=True,
        help="Decoding profile UUID to use for generation.",
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=DEFAULT_PROFILE_PATH,
        help="Path to decoding profiles configuration.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt text to generate from.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="File containing prompt text.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Override the profile's max_new_tokens parameter.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device (default: cuda if available else cpu).",
    )
    return parser.parse_args()


def _resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt and args.prompt_file:
        raise SystemExit("Provide either --prompt or --prompt-file, not both.")
    if args.prompt_file:
        return args.prompt_file.read_text(encoding="utf-8")
    if args.prompt:
        return args.prompt
    raise SystemExit("Prompt text is required. Use --prompt or --prompt-file.")


def main() -> None:
    args = parse_args()
    prompt = _resolve_prompt(args)

    try:
        catalog = load_catalog(args.profiles)
    except Exception as exc:  # pragma: no cover - configuration errors
        sys.exit(f"Failed to load decoding profiles: {exc}")

    try:
        profile = catalog.get(args.profile_id)
    except KeyError as exc:
        sys.exit(str(exc))

    try:
        engine, tokenizer = _build_engine(args.device)
    except RuntimeError as exc:
        sys.exit(str(exc))

    service = ProposalService(
        engine=engine,
        tokenizer=tokenizer,
        profiles_path=args.profiles,
    )

    try:
        proposal, profile = service.generate_from_prompt(
            problem_id=f"cli_{torch.randint(0, 1_000_000, (1,)).item()}",
            prompt=prompt,
            profile_id=profile.profile_id,
            max_new_tokens=args.max_new_tokens,
        )
    except Exception as exc:  # pragma: no cover - runtime errors
        sys.exit(f"Generation failed: {exc}")

    payload = solution_proposal_to_dict(proposal)
    payload["profile"] = profile.to_dict()

    print("=== Proposal Summary ===")
    print(f"profile: {profile.name} ({profile.profile_id})")
    print(f"latency_ms: {payload['latencyMs']}")
    print(f"raw_confidence: {payload['rawConfidence']:.3f}")
    print(f"calibrated_confidence: {payload['calibratedConfidence']:.3f}")
    print("--- content ---")
    print(payload["contentText"])


if __name__ == "__main__":
    main()

