"""Utilities for exporting and restoring planner session artifacts."""

from __future__ import annotations

import hashlib
import io
import json
import uuid
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import torch

from src.models.planning.session_artifact import (
    DEFAULT_ARTIFACT_VERSION,
    PlannerSessionArtifact,
    RoutingSnapshot,
    TensorManifestEntry,
    TensorResolutionError,
    compute_snapshot_hash,
)
try:
    from src.models.titans.data_model import HLayerState, RoutingDecision
except (ModuleNotFoundError, ImportError):  # pragma: no cover - lightweight fallback
    from dataclasses import dataclass, field
    from typing import Any, Dict, List, Optional

    @dataclass
    class HLayerState:  # type: ignore[override]
        episode_id: str
        step: int
        memory_state: Any
        engine_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
        threshold_history: List[float] = field(default_factory=list)
        cached_context: Optional[torch.Tensor] = None

    @dataclass
    class RoutingDecision:  # Minimal placeholder
        selected_engines: List[str]
        confidence_threshold: float
        predicted_risk: float
try:  # Optional heavy dependency (einops) may not be available in lightweight test environments.
    from src.models.titans.neural_memory import NeuralMemState, TensorDict
except (ModuleNotFoundError, ImportError):  # pragma: no cover - fallback for minimal test environments
    from collections import namedtuple

    class TensorDict(dict):
        """Minimal TensorDict fallback for testing environments without optional deps."""

    NeuralMemState = namedtuple(
        "NeuralMemState",
        ["seq_index", "weights", "cache_store_segment", "states", "updates"],
    )


def _sanitize_segment(segment: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in segment)
    return safe or "tensor"


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(tensor.detach().cpu(), buffer)
    return buffer.getvalue()


def _tensor_checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _serialize_mapping(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _collect_state_tensors(state: HLayerState) -> Sequence[Tuple[str, torch.Tensor]]:
    if state.memory_state is None:
        return ()

    mem_state = state.memory_state
    tensors: list[Tuple[str, torch.Tensor]] = []

    for name, tensor in mem_state.weights.items():
        tensors.append((f"memory/weights/{name}", tensor))

    if mem_state.cache_store_segment is not None:
        tensors.append(("memory/cache_store_segment", mem_state.cache_store_segment))

    if mem_state.states is not None:
        last_update, last_momentum = mem_state.states
        if last_update is not None:
            for name, tensor in last_update.items():
                tensors.append((f"memory/states/last_update/{name}", tensor))
        if last_momentum is not None:
            for name, tensor in last_momentum.items():
                tensors.append((f"memory/states/last_momentum/{name}", tensor))

    if mem_state.updates is not None:
        for name, tensor in mem_state.updates.items():
            tensors.append((f"memory/updates/{name}", tensor))

    if state.cached_context is not None:
        tensors.append(("planner/cached_context", state.cached_context))

    return tensors


def _build_additional_blobs(
    *,
    artifact_meta: Mapping[str, Any],
    routing_snapshot: RoutingSnapshot,
    memory_metadata: Mapping[str, Any],
) -> Dict[str, bytes]:
    return {
        "artifact_metadata": _serialize_mapping(artifact_meta),
        "routing_metadata": _serialize_mapping(routing_snapshot.to_dict()),
        "memory_metadata": _serialize_mapping(memory_metadata),
    }


def build_planner_session_artifact(
    *,
    session_id: str,
    state: HLayerState,
    saved_by: str | None = None,
    resume_notes: str | None = None,
    last_decision: RoutingDecision | None = None,
    profile_id: str | None = None,
) -> Tuple[PlannerSessionArtifact, Dict[str, bytes]]:
    """Create a planner session artifact and serialized tensor payloads."""
    if state.memory_state is None:
        raise ValueError("HLayerState.memory_state is required to export a planner session")

    tensor_pairs = list(_collect_state_tensors(state))
    if not tensor_pairs:
        raise ValueError("No tensors available to export for the planner session")

    manifest_entries: list[TensorManifestEntry] = []
    payloads: Dict[str, bytes] = {}
    path_counts: Dict[str, int] = {}

    for key, tensor in tensor_pairs:
        if not isinstance(tensor, torch.Tensor):
            continue

        sanitized_segments = [_sanitize_segment(part) for part in key.split("/")]
        if not sanitized_segments:
            sanitized_segments = ["tensor"]

        base_path = PurePosixPath(*sanitized_segments).as_posix()
        occurrence = path_counts.get(base_path, 0)
        path_counts[base_path] = occurrence + 1
        if occurrence:
            sanitized_segments[-1] = f"{sanitized_segments[-1]}_{occurrence}"

        storage_parts = sanitized_segments[:-1] + [f"{sanitized_segments[-1]}.pt"]
        storage_path = PurePosixPath("tensors", *storage_parts).as_posix()

        data = _tensor_to_bytes(tensor)
        manifest_entries.append(
            TensorManifestEntry(
                name=key,
                shape=tuple(int(dim) for dim in tensor.shape),
                dtype=str(tensor.dtype),
                storage_path=storage_path,
                checksum=_tensor_checksum(data),
            )
        )
        payloads[storage_path] = data

    cached_context_checksum = next(
        (entry.checksum for entry in manifest_entries if entry.name == "planner/cached_context"),
        None,
    )

    if last_decision is not None:
        last_selected_engines: Sequence[str] = tuple(last_decision.selected_engines)
        last_confidence = last_decision.confidence_threshold
        predicted_risk = last_decision.predicted_risk
    else:
        last_selected_engines = ()
        last_confidence = state.threshold_history[-1] if state.threshold_history else None
        predicted_risk = None

    mem_state = state.memory_state
    memory_metadata: Dict[str, Any] = {
        "episode_id": state.episode_id,
        "step": state.step,
        "threshold_history_length": len(state.threshold_history),
        "seq_index": int(mem_state.seq_index),
        "weight_names": sorted(str(name) for name in mem_state.weights.keys()),
        "has_cache_store_segment": mem_state.cache_store_segment is not None,
        "has_updates": mem_state.updates is not None,
    }

    has_last_update = False
    has_last_momentum = False
    if mem_state.states is not None:
        last_update, last_momentum = mem_state.states
        has_last_update = last_update is not None
        has_last_momentum = last_momentum is not None
    memory_metadata["has_last_update"] = has_last_update
    memory_metadata["has_last_momentum"] = has_last_momentum

    routing_snapshot = RoutingSnapshot(
        step=state.step,
        threshold_history=list(state.threshold_history),
        engine_stats=state.engine_stats,
        last_selected_engines=last_selected_engines,
        last_confidence=last_confidence,
        predicted_risk=predicted_risk,
        cached_context_checksum=cached_context_checksum,
        profile_id=profile_id,
        additional_metadata={
            "episode_id": state.episode_id,
            "memory": memory_metadata,
        },
    )

    artifact = PlannerSessionArtifact(
        artifact_id=str(uuid.uuid4()),
        session_id=session_id,
        created_at=datetime.now(timezone.utc),
        tensors=manifest_entries,
        routing_metadata=routing_snapshot,
        context_snapshot_hash="0" * 64,
        version=DEFAULT_ARTIFACT_VERSION,
        saved_by=saved_by,
        resume_notes=resume_notes,
    )

    artifact_meta = {
        "artifact_id": artifact.artifact_id,
        "session_id": artifact.session_id,
        "version": artifact.version,
        "saved_by": artifact.saved_by or "",
        "resume_notes": artifact.resume_notes or "",
    }

    additional_blobs = _build_additional_blobs(
        artifact_meta=artifact_meta,
        routing_snapshot=routing_snapshot,
        memory_metadata=memory_metadata,
    )

    def resolver(entry: TensorManifestEntry) -> bytes:
        try:
            return payloads[entry.storage_path]
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise TensorResolutionError(entry.storage_path) from exc

    context_hash = compute_snapshot_hash(
        artifact.tensors,
        resolver,
        additional_blobs=additional_blobs,
    )
    artifact = artifact.with_context_snapshot_hash(context_hash)

    return artifact, payloads


def restore_state_from_artifact(
    artifact: PlannerSessionArtifact,
    tensor_loader: Callable[[str], bytes],
    *,
    device: torch.device | str | None = None,
    target_session_id: Optional[str] = None,
    strict_version_check: bool = True,
) -> HLayerState:
    """Rehydrate an `HLayerState` from a persisted planner session artifact."""
    if strict_version_check and artifact.version != DEFAULT_ARTIFACT_VERSION:
        raise ValueError(
            f"Incompatible artifact version '{artifact.version}'. "
            f"Expected '{DEFAULT_ARTIFACT_VERSION}'."
        )

    resolved_device = torch.device(device) if device is not None else torch.device("cpu")

    memory_metadata = artifact.routing_metadata.additional_metadata.get("memory", {})
    artifact_meta = {
        "artifact_id": artifact.artifact_id,
        "session_id": artifact.session_id,
        "version": artifact.version,
        "saved_by": artifact.saved_by or "",
        "resume_notes": artifact.resume_notes or "",
    }

    additional_blobs = _build_additional_blobs(
        artifact_meta=artifact_meta,
        routing_snapshot=artifact.routing_metadata,
        memory_metadata=memory_metadata,
    )

    def resolver(entry: TensorManifestEntry) -> bytes:
        try:
            return tensor_loader(entry.storage_path)
        except FileNotFoundError as exc:
            raise TensorResolutionError(entry.storage_path) from exc

    artifact.verify_snapshot_hash(resolver, additional_blobs=additional_blobs)

    weights: Dict[str, torch.Tensor] = {}
    last_update_dict: Dict[str, torch.Tensor] = {}
    last_momentum_dict: Dict[str, torch.Tensor] = {}
    updates_dict: Dict[str, torch.Tensor] = {}
    cache_store_segment: torch.Tensor | None = None
    cached_context: torch.Tensor | None = None

    for entry in artifact.tensors:
        data = resolver(entry)
        tensor = torch.load(io.BytesIO(data), map_location=resolved_device)
        parts = entry.name.split("/") if entry.name else []
        if not parts:
            continue

        category = parts[0]

        if category == "memory":
            if len(parts) == 1:
                continue
            subcategory = parts[1]
            if subcategory == "weights":
                param_name = "/".join(parts[2:]) if len(parts) > 2 else "weight"
                weights[param_name] = tensor
            elif subcategory == "cache_store_segment":
                cache_store_segment = tensor
            elif subcategory == "states" and len(parts) >= 3:
                state_kind = parts[2]
                param_name = "/".join(parts[3:]) if len(parts) > 3 else "state"
                if state_kind == "last_update":
                    last_update_dict[param_name] = tensor
                elif state_kind == "last_momentum":
                    last_momentum_dict[param_name] = tensor
            elif subcategory == "updates":
                param_name = "/".join(parts[2:]) if len(parts) > 2 else "update"
                updates_dict[param_name] = tensor
        elif category == "planner" and len(parts) >= 2 and parts[1] == "cached_context":
            cached_context = tensor

    if not weights:
        raise ValueError("Session artifact does not contain neural memory weights")

    weights_td = TensorDict({k: v for k, v in weights.items()})
    last_update_td = TensorDict(last_update_dict) if last_update_dict else TensorDict({})
    last_momentum_td = TensorDict(last_momentum_dict) if last_momentum_dict else TensorDict({})
    updates_td = TensorDict(updates_dict) if updates_dict else None

    seq_index = int(memory_metadata.get("seq_index", 0)) if memory_metadata else 0
    memory_state = NeuralMemState(
        seq_index,
        weights_td,
        cache_store_segment,
        (last_update_td, last_momentum_td),
        updates_td,
    )

    engine_stats = {k: dict(v) for k, v in artifact.routing_metadata.engine_stats.items()}
    threshold_history = list(artifact.routing_metadata.threshold_history)

    restored_state = HLayerState(
        episode_id=target_session_id or artifact.session_id,
        step=artifact.routing_metadata.step,
        memory_state=memory_state,
        engine_stats=engine_stats,
        threshold_history=threshold_history,
        cached_context=cached_context,
    )

    return restored_state
