import torch

from src.core.planner.artifacts import (
    HLayerState,
    build_planner_session_artifact,
    restore_state_from_artifact,
)
from src.core.planner.service import PlannerSessionService


def _sample_state() -> HLayerState:
    class MemoryState:
        def __init__(self) -> None:
            self.seq_index = 3
            self.weights = {"core": torch.arange(6, dtype=torch.float32).reshape(1, 6)}
            self.cache_store_segment = torch.zeros(1, 1, dtype=torch.float32)
            self.states = (
                {"core": torch.ones(1, 6, dtype=torch.float32)},
                {"core": torch.full((1, 6), 0.5, dtype=torch.float32)},
            )
            self.updates = {"core": torch.randn(1, 3, 6)}

    cached_context = torch.linspace(0.0, 1.0, steps=8).reshape(1, 8)

    return HLayerState(
        episode_id="session-xyz",
        step=5,
        memory_state=MemoryState(),
        engine_stats={"engine_0": {"successes": 2.0, "attempts": 3.0}},
        threshold_history=[0.6, 0.7, 0.75],
        cached_context=cached_context,
    )


def test_planner_artifact_round_trip():
    state = _sample_state()
    artifact, tensor_payloads = build_planner_session_artifact(
        session_id="session-xyz",
        state=state,
        saved_by="tester",
        resume_notes="checkpoint before handoff",
        profile_id="profile-123",
    )

    restored = restore_state_from_artifact(
        artifact,
        lambda storage_path: tensor_payloads[storage_path],
        device=torch.device("cpu"),
    )

    assert restored.step == state.step
    assert restored.threshold_history == state.threshold_history
    assert torch.equal(
        restored.memory_state.weights["core"],
        state.memory_state.weights["core"],
    )


def test_planner_session_service_save_and_restore():
    state = _sample_state()

    # Build a reference artifact to generate validation hints
    reference_artifact, _ = build_planner_session_artifact(
        session_id="session-xyz",
        state=state,
        saved_by="tester",
        resume_notes="checkpoint before handoff",
        profile_id="profile-123",
    )

    service = PlannerSessionService()
    service.register_state("session-xyz", state)

    request_payload = {
        "tensors": [
            {
                "name": entry.name,
                "shape": list(entry.shape),
                "dtype": entry.dtype,
                "storagePath": entry.storage_path,
                "checksum": entry.checksum,
            }
            for entry in reference_artifact.tensors
        ],
        "routingMetadata": {
            "step": reference_artifact.routing_metadata.step,
            "thresholdHistory": list(reference_artifact.routing_metadata.threshold_history),
            "lastSelectedEngines": list(reference_artifact.routing_metadata.last_selected_engines),
            "lastConfidence": reference_artifact.routing_metadata.last_confidence,
            "predictedRisk": reference_artifact.routing_metadata.predicted_risk,
            "cachedContextChecksum": reference_artifact.routing_metadata.cached_context_checksum,
        },
        "resumeNotes": "checkpoint before handoff",
        "savedBy": "tester",
        "profileId": "profile-123",
    }

    stored_artifact = service.save_artifact("session-xyz", request_payload)
    assert sorted(entry.name for entry in stored_artifact.tensors) == sorted(
        entry.name for entry in reference_artifact.tensors
    )

    restored_state, routing_summary, warnings = service.restore_artifact(
        stored_artifact.artifact_id,
        target_session_id="session-xyz-resume",
    )

    assert restored_state.episode_id == "session-xyz-resume"
    assert restored_state.threshold_history == state.threshold_history
    assert routing_summary["step"] == state.step
    assert warnings == []
