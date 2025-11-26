"""Checkpoint loading for RetNet-HRM.

Provides functionality to load pretrained model checkpoints.
Uses safetensors for fast, safe model serialization.
"""

import os
import json
from typing import Dict, Any, Optional
import torch
from safetensors.torch import load_file as st_load_file


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    allow_mismatch: bool = False,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Load checkpoint with architecture validation.

    Implements permissive loading: warns on mismatch but allows proceeding
    if allow_mismatch=True.

    Args:
        model: Target model
        checkpoint_path: Path to checkpoint directory or .safetensors file
        optimizer: Optional optimizer to load state into
        allow_mismatch: Allow loading despite architecture mismatch
        device: Device to load checkpoint to

    Returns:
        Metadata dict from checkpoint

    Raises:
        RuntimeError: If architecture mismatch and allow_mismatch=False
        FileNotFoundError: If checkpoint not found
    """
    # Handle both directory and file paths
    if os.path.isdir(checkpoint_path):
        # Try to find model.safetensors in directory
        model_path = os.path.join(checkpoint_path, "model.safetensors")
        meta_path = os.path.join(checkpoint_path, "ckpt.json")
    else:
        model_path = checkpoint_path
        meta_path = checkpoint_path.replace(".safetensors", ".json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found at {model_path}")

    # Load metadata if available
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)

    # Load model weights
    model_sd = st_load_file(model_path, device=device)
    missing, unexpected = model.load_state_dict(model_sd, strict=False)

    # Check for architecture mismatch
    warn = bool(missing or unexpected)
    if warn:
        print("\n⚠️  ARCHITECTURE MISMATCH DETECTED")
        if missing:
            print(f"\nMissing keys (first 10): {missing[:10]}")
        if unexpected:
            print(f"\nUnexpected keys (first 10): {unexpected[:10]}")

        if not allow_mismatch:
            raise RuntimeError(
                "Architecture mismatch detected. Reload with allow_mismatch=True to force loading."
            )
        else:
            print("\n✅ Proceeding with permissive loading (allow_mismatch=True)")

    # Load optimizer state if provided and exists
    if optimizer is not None and os.path.isdir(checkpoint_path):
        opath = os.path.join(checkpoint_path, "optimizer.pt")
        if os.path.exists(opath):
            try:
                optimizer.load_state_dict(torch.load(opath, map_location=device))
            except Exception as e:
                print(f"\n⚠️  Optimizer load failed: {e}")
                print("Optimizer state not loaded.")

    return meta
