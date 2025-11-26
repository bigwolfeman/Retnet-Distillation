#!/usr/bin/env python3
"""
CLI for exporting planner session artifacts.

Loads a serialized `HLayerState`, captures the associated neural memory tensors,
and writes a zip bundle containing the artifact manifest plus tensor payloads.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Optional

import torch

from src.core.planner import build_planner_session_artifact
from src.models.titans.data_model import HLayerState, RoutingDecision


def _default_state_path(session_id: str) -> Path:
    return Path("sessions") / f"{session_id}.state.pt"


def _load_state(path: Path) -> HLayerState:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, HLayerState):
        raise TypeError(f"Expected HLayerState at {path}, found {type(obj).__name__}")
    return obj


def _load_decision(path: Path) -> RoutingDecision:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, RoutingDecision):
        raise TypeError(f"Expected RoutingDecision at {path}, found {type(obj).__name__}")
    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a planner session artifact for pause-and-resume workflows.",
    )
    parser.add_argument(
        "--session-id",
        required=True,
        help="Planner session identifier.",
    )
    parser.add_argument(
        "--state",
        type=Path,
        help="Path to serialized HLayerState (defaults to sessions/<session-id>.state.pt).",
    )
    parser.add_argument(
        "--decision",
        type=Path,
        help="Optional path to serialized RoutingDecision to capture routing summary.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output zip file for the session artifact.",
    )
    parser.add_argument(
        "--saved-by",
        type=str,
        help="Identifier for the operator saving the session.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        help="Optional resume notes to embed in the artifact manifest.",
    )
    parser.add_argument(
        "--profile-id",
        type=str,
        help="Optional decoding profile identifier associated with the session.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    state_path = args.state or _default_state_path(args.session_id)
    if not state_path.exists():
        sys.exit(f"State file not found: {state_path}")

    try:
        state = _load_state(state_path)
    except TypeError as exc:
        sys.exit(str(exc))

    last_decision: Optional[RoutingDecision] = None
    if args.decision:
        if not args.decision.exists():
            sys.exit(f"Decision file not found: {args.decision}")
        try:
            last_decision = _load_decision(args.decision)
        except TypeError as exc:
            sys.exit(str(exc))

    artifact, tensor_payloads = build_planner_session_artifact(
        session_id=args.session_id,
        state=state,
        saved_by=args.saved_by,
        resume_notes=args.notes,
        last_decision=last_decision,
        profile_id=args.profile_id,
    )

    artifact_dict = artifact.to_dict()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, mode="w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("artifact.json", json.dumps(artifact_dict, indent=2) + "\n")
        for storage_path, data in tensor_payloads.items():
            bundle.writestr(storage_path, data)

    print(f"Artifact {artifact.artifact_id} saved to {args.out}")
    print(f"context_snapshot_hash={artifact.context_snapshot_hash}")
    print(f"tensor_count={len(artifact.tensors)}")


if __name__ == "__main__":
    main()

