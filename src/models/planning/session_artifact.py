"""Planner session artifact models and helpers.

Implements the persistence structures defined in
`specs/002-stabilize-titan-hrm/data-model.md` for User Story 2. A planner
session artifact captures the tensors required to restore neural memory as
well as routing metadata so a paused session can resume without divergence.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Sequence
import hashlib
import string
import uuid


DEFAULT_ARTIFACT_VERSION = "1.0.0"


class SnapshotIntegrityError(ValueError):
    """Raised when an artifact's snapshot hash does not match the payload."""


class TensorResolutionError(FileNotFoundError):
    """Raised when tensor bytes cannot be resolved during hashing or restore."""

    def __init__(self, storage_path: str) -> None:
        super().__init__(f"Tensor data '{storage_path}' could not be resolved")
        self.storage_path = storage_path


def _ensure_hex(value: str, *, field_name: str) -> str:
    if not value:
        raise ValueError(f"{field_name} must be provided")
    if len(value) % 2 != 0:
        raise ValueError(f"{field_name} must have an even number of characters")
    if any(ch not in string.hexdigits for ch in value):
        raise ValueError(f"{field_name} must be hexadecimal")
    return value.lower()


def _normalise_rel_path(path: str) -> str:
    pure = PurePosixPath(path)
    if pure.is_absolute():
        raise ValueError(f"storage_path must be relative, got '{path}'")
    normalised = pure.as_posix()
    if normalised.startswith("../"):
        raise ValueError(f"storage_path cannot escape artifact root: '{path}'")
    return normalised


@dataclass(frozen=True)
class TensorManifestEntry:
    """Metadata describing a persisted tensor within an artifact bundle."""

    name: str
    shape: Sequence[int]
    dtype: str
    storage_path: str
    checksum: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("TensorManifestEntry.name must be provided")
        shape_tuple = tuple(int(dim) for dim in self.shape)
        if any(dim < 0 for dim in shape_tuple):
            raise ValueError(f"TensorManifestEntry.shape must be non-negative, got {shape_tuple}")
        object.__setattr__(self, "shape", shape_tuple)

        if not self.dtype:
            raise ValueError("TensorManifestEntry.dtype must be provided")

        normalised_path = _normalise_rel_path(self.storage_path)
        object.__setattr__(self, "storage_path", normalised_path)

        if self.checksum is not None:
            checksum = _ensure_hex(self.checksum, field_name="TensorManifestEntry.checksum")
            object.__setattr__(self, "checksum", checksum)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "storage_path": self.storage_path,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TensorManifestEntry":
        return cls(
            name=payload["name"],
            shape=payload["shape"],
            dtype=payload["dtype"],
            storage_path=payload["storage_path"],
            checksum=payload.get("checksum"),
        )


@dataclass(frozen=True)
class RoutingSnapshot:
    """Routing metadata required to resume planner activity deterministically."""

    step: int
    threshold_history: Sequence[float] = field(default_factory=tuple)
    engine_stats: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    last_selected_engines: Sequence[str] = field(default_factory=tuple)
    last_confidence: float | None = None
    predicted_risk: float | None = None
    cached_context_checksum: str | None = None
    profile_id: str | None = None
    additional_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError(f"RoutingSnapshot.step must be >= 0, got {self.step}")

        thresholds = tuple(float(val) for val in self.threshold_history)
        for value in thresholds:
            if value < 0.0 or value > 1.0:
                raise ValueError("RoutingSnapshot.threshold_history values must be within [0, 1]")
        object.__setattr__(self, "threshold_history", thresholds)

        engines = tuple(str(engine) for engine in self.last_selected_engines)
        object.__setattr__(self, "last_selected_engines", engines)

        normalised_stats: Dict[str, Dict[str, float]] = {}
        for name, metrics in self.engine_stats.items():
            metric_dict = dict(metrics)
            for metric_name, value in metric_dict.items():
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"RoutingSnapshot.engine_stats values must be numeric, "
                        f"got {type(value)} for '{name}.{metric_name}'"
                    )
            normalised_stats[str(name)] = metric_dict
        object.__setattr__(self, "engine_stats", normalised_stats)

        if self.last_confidence is not None and not (0.0 <= self.last_confidence <= 1.0):
            raise ValueError("RoutingSnapshot.last_confidence must be within [0, 1]")

        if self.predicted_risk is not None and not (0.0 <= self.predicted_risk <= 1.0):
            raise ValueError("RoutingSnapshot.predicted_risk must be within [0, 1]")

        if self.cached_context_checksum is not None:
            checksum = _ensure_hex(
                self.cached_context_checksum,
                field_name="RoutingSnapshot.cached_context_checksum",
            )
            object.__setattr__(self, "cached_context_checksum", checksum)

        object.__setattr__(self, "additional_metadata", dict(self.additional_metadata))

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "step": self.step,
            "threshold_history": list(self.threshold_history),
            "engine_stats": {k: dict(v) for k, v in self.engine_stats.items()},
            "last_selected_engines": list(self.last_selected_engines),
            "last_confidence": self.last_confidence,
            "predicted_risk": self.predicted_risk,
            "cached_context_checksum": self.cached_context_checksum,
            "profile_id": self.profile_id,
            "additional_metadata": dict(self.additional_metadata),
        }
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RoutingSnapshot":
        return cls(
            step=int(payload["step"]),
            threshold_history=payload.get("threshold_history", ()),
            engine_stats=payload.get("engine_stats", {}),
            last_selected_engines=payload.get("last_selected_engines", ()),
            last_confidence=payload.get("last_confidence"),
            predicted_risk=payload.get("predicted_risk"),
            cached_context_checksum=payload.get("cached_context_checksum"),
            profile_id=payload.get("profile_id"),
            additional_metadata=payload.get("additional_metadata", {}),
        )


@dataclass(frozen=True)
class PlannerSessionArtifact:
    """Serialized planner session metadata and tensor manifest."""

    artifact_id: str
    session_id: str
    created_at: datetime
    tensors: Sequence[TensorManifestEntry]
    routing_metadata: RoutingSnapshot
    context_snapshot_hash: str
    version: str = DEFAULT_ARTIFACT_VERSION
    saved_by: str | None = None
    resume_notes: str | None = None

    def __post_init__(self) -> None:
        try:
            uuid.UUID(self.artifact_id)
        except (ValueError, AttributeError) as exc:
            raise ValueError(f"artifact_id must be a valid UUID string, got '{self.artifact_id}'") from exc

        if not self.session_id:
            raise ValueError("session_id must be provided")

        if self.created_at.tzinfo is None:
            object.__setattr__(self, "created_at", self.created_at.replace(tzinfo=timezone.utc))
        else:
            object.__setattr__(self, "created_at", self.created_at.astimezone(timezone.utc))

        tensor_entries = tuple(self.tensors)
        if not tensor_entries:
            raise ValueError("PlannerSessionArtifact requires at least one tensor entry")
        object.__setattr__(self, "tensors", tensor_entries)

        if not isinstance(self.routing_metadata, RoutingSnapshot):
            raise TypeError("routing_metadata must be a RoutingSnapshot instance")

        snapshot_hash = _ensure_hex(self.context_snapshot_hash, field_name="context_snapshot_hash")
        object.__setattr__(self, "context_snapshot_hash", snapshot_hash)

        if not self.version:
            raise ValueError("version must be provided")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "saved_by": self.saved_by,
            "tensors": [entry.to_dict() for entry in self.tensors],
            "routing_metadata": self.routing_metadata.to_dict(),
            "context_snapshot_hash": self.context_snapshot_hash,
            "version": self.version,
            "resume_notes": self.resume_notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PlannerSessionArtifact":
        created_at = datetime.fromisoformat(payload["created_at"])
        tensors = [
            TensorManifestEntry.from_dict(entry)
            for entry in payload.get("tensors", [])
        ]
        routing_metadata = RoutingSnapshot.from_dict(payload["routing_metadata"])
        return cls(
            artifact_id=payload["artifact_id"],
            session_id=payload["session_id"],
            created_at=created_at,
            tensors=tensors,
            routing_metadata=routing_metadata,
            context_snapshot_hash=payload["context_snapshot_hash"],
            version=payload.get("version", DEFAULT_ARTIFACT_VERSION),
            saved_by=payload.get("saved_by"),
            resume_notes=payload.get("resume_notes"),
        )

    def with_context_snapshot_hash(self, context_snapshot_hash: str) -> "PlannerSessionArtifact":
        """Return a copy with an updated context snapshot hash."""
        snapshot_hash = _ensure_hex(context_snapshot_hash, field_name="context_snapshot_hash")
        return replace(self, context_snapshot_hash=snapshot_hash)

    def verify_snapshot_hash(
        self,
        resolver: Callable[[TensorManifestEntry], bytes],
        *,
        additional_blobs: Mapping[str, bytes] | None = None,
    ) -> None:
        """Validate the artifact's snapshot hash against current tensor bytes."""
        computed = compute_snapshot_hash(
            self.tensors,
            resolver,
            additional_blobs=additional_blobs,
        )
        if computed != self.context_snapshot_hash:
            raise SnapshotIntegrityError(
                f"Snapshot hash mismatch: expected {self.context_snapshot_hash}, got {computed}"
            )


def compute_snapshot_hash(
    tensors: Sequence[TensorManifestEntry],
    resolver: Callable[[TensorManifestEntry], bytes],
    *,
    additional_blobs: Mapping[str, bytes] | None = None,
) -> str:
    """
    Compute a deterministic SHA-256 digest for the provided tensor entries.

    Args:
        tensors: Manifest entries describing persisted tensors.
        resolver: Callable returning raw bytes for a tensor entry.
        additional_blobs: Optional mapping of extra metadata names to bytes that
            should contribute to the hash (e.g., routing metadata JSON).

    Returns:
        Hex-encoded SHA-256 digest representing the artifact contents.
    """
    digest = hashlib.sha256()
    for entry in sorted(tensors, key=lambda item: item.storage_path):
        digest.update(entry.storage_path.encode("utf-8"))
        digest.update(entry.name.encode("utf-8"))
        digest.update(",".join(str(dim) for dim in entry.shape).encode("utf-8"))
        digest.update(entry.dtype.encode("utf-8"))
        try:
            blob = resolver(entry)
        except FileNotFoundError as exc:
            raise TensorResolutionError(entry.storage_path) from exc
        if not isinstance(blob, (bytes, bytearray)):
            raise TypeError("resolver must return bytes-like objects")
        data = bytes(blob)
        digest.update(len(data).to_bytes(8, byteorder="big"))
        digest.update(data)

    if additional_blobs:
        for name in sorted(additional_blobs):
            data = additional_blobs[name]
            if not isinstance(data, (bytes, bytearray)):
                raise TypeError("additional_blobs values must be bytes-like")
            digest.update(name.encode("utf-8"))
            digest.update(len(data).to_bytes(8, byteorder="big"))
            digest.update(bytes(data))

    return digest.hexdigest()

