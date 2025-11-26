"""In-memory planner session service for artifact management."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch

from src.core.planner.artifacts import (
    build_planner_session_artifact,
    restore_state_from_artifact,
)
from src.models.planning.session_artifact import PlannerSessionArtifact
from src.models.titans.data_model import HLayerState, RoutingDecision


def _camel_to_snake(value: str) -> str:
    result = []
    for char in value:
        if char.isupper():
            if result:
                result.append("_")
            result.append(char.lower())
        else:
            result.append(char)
    return "".join(result)


def _normalize_tensor_hint(entry: Mapping[str, Any]) -> Tuple[str, str]:
    try:
        name = entry["name"]
        storage_path = entry.get("storagePath") or entry.get("storage_path")
    except KeyError as exc:  # pragma: no cover - defensive blocking
        raise ValueError("Tensor manifest entries must include 'name' and 'storagePath'") from exc
    if storage_path is None:
        raise ValueError("Tensor manifest entries must include 'storagePath'")
    return str(name), str(storage_path)


def _validate_tensor_manifest(
    expected: Sequence[Mapping[str, Any]],
    artifact: PlannerSessionArtifact,
) -> None:
    expected_pairs = sorted(_normalize_tensor_hint(entry) for entry in expected)
    actual_pairs = sorted((entry.name, entry.storage_path) for entry in artifact.tensors)
    if expected_pairs != actual_pairs:
        raise ValueError("Provided tensor manifest does not match generated artifact tensors")


def _normalize_routing_hint(payload: Mapping[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in payload.items():
        normalized[_camel_to_snake(key)] = value
    return normalized


def _validate_routing_hint(
    expected: Mapping[str, Any],
    artifact: PlannerSessionArtifact,
) -> None:
    expected_normalized = _normalize_routing_hint(expected)
    actual = artifact.routing_metadata.to_dict()
    mismatches = []
    for key, expected_value in expected_normalized.items():
        if key not in actual:
            continue
        if actual[key] != expected_value:
            mismatches.append(key)
    if mismatches:
        raise ValueError(f"Routing metadata mismatch for keys: {', '.join(mismatches)}")


class PlannerSessionService:
    """Manage planner session states, artifacts, and restores in-memory."""

    def __init__(self) -> None:
        self._states: MutableMapping[str, HLayerState] = {}
        self._decisions: MutableMapping[str, RoutingDecision] = {}
        self._artifacts: MutableMapping[str, Tuple[PlannerSessionArtifact, Dict[str, bytes]]] = {}
        self._restored_states: MutableMapping[str, HLayerState] = {}

    def register_state(self, session_id: str, state: HLayerState) -> None:
        self._states[session_id] = state

    def register_decision(self, session_id: str, decision: RoutingDecision) -> None:
        self._decisions[session_id] = decision

    def get_state(self, session_id: str) -> HLayerState:
        try:
            return self._states[session_id]
        except KeyError as exc:
            raise KeyError(f"Session '{session_id}' not found") from exc

    def save_artifact(
        self,
        session_id: str,
        payload: Mapping[str, Any],
    ) -> PlannerSessionArtifact:
        state = self.get_state(session_id)
        last_decision = self._decisions.get(session_id)

        resume_notes = payload.get("resumeNotes")
        saved_by = payload.get("savedBy")
        profile_id = payload.get("profileId")

        artifact, tensor_payloads = build_planner_session_artifact(
            session_id=session_id,
            state=state,
            saved_by=saved_by,
            resume_notes=resume_notes,
            last_decision=last_decision,
            profile_id=profile_id,
        )

        if tensors_hint := payload.get("tensors"):
            if not isinstance(tensors_hint, Sequence):
                raise ValueError("tensors must be a sequence of manifest entries")
            _validate_tensor_manifest(tensors_hint, artifact)

        if routing_hint := payload.get("routingMetadata"):
            if not isinstance(routing_hint, Mapping):
                raise ValueError("routingMetadata must be an object")
            _validate_routing_hint(routing_hint, artifact)

        self._artifacts[artifact.artifact_id] = (artifact, tensor_payloads)
        return artifact

    def get_artifact(self, artifact_id: str) -> PlannerSessionArtifact:
        try:
            artifact, _ = self._artifacts[artifact_id]
        except KeyError as exc:
            raise KeyError(f"Artifact '{artifact_id}' not found") from exc
        return artifact

    def restore_artifact(
        self,
        artifact_id: str,
        *,
        target_session_id: Optional[str] = None,
        strict_version_check: bool = True,
    ) -> Tuple[HLayerState, Dict[str, Any], Sequence[str]]:
        try:
            artifact, tensor_payloads = self._artifacts[artifact_id]
        except KeyError as exc:
            raise KeyError(f"Artifact '{artifact_id}' not found") from exc

        def loader(storage_path: str) -> bytes:
            try:
                return tensor_payloads[storage_path]
            except KeyError as exc:
                raise KeyError(f"Tensor payload '{storage_path}' missing for artifact '{artifact_id}'") from exc

        restored_state = restore_state_from_artifact(
            artifact,
            loader,
            device=torch.device("cpu"),
            target_session_id=target_session_id,
            strict_version_check=strict_version_check,
        )

        resolved_session_id = restored_state.episode_id
        self._restored_states[resolved_session_id] = restored_state

        routing = artifact.routing_metadata
        summary = {
            "step": routing.step,
            "thresholdHistory": list(routing.threshold_history),
            "lastSelectedEngines": list(routing.last_selected_engines),
            "lastConfidence": routing.last_confidence,
            "predictedRisk": routing.predicted_risk,
            "cachedContextChecksum": routing.cached_context_checksum,
        }

        warnings: list[str] = []
        if routing.last_confidence is None:
            warnings.append("No confidence value recorded for last planner decision.")

        return restored_state, summary, warnings

