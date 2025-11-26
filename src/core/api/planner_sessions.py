"""
API helpers for planner session artifact workflows.

Provides FastAPI router utilities and in-memory orchestration for exporting and
restoring planner session artifacts, including deterministic verification of
tensor manifests and routing metadata.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

try:  # Optional FastAPI dependency
    from fastapi import APIRouter, HTTPException, status
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    APIRouter = None  # type: ignore
    HTTPException = None  # type: ignore
    status = None  # type: ignore

from src.core.planner import PlannerSessionService


def create_fastapi_router(
    service: PlannerSessionService,
) -> "APIRouter":  # type: ignore[name-defined]
    """Return a FastAPI router exposing planner session endpoints."""
    if APIRouter is None or HTTPException is None or status is None:
        raise RuntimeError(
            "FastAPI is not installed. Install fastapi to enable session endpoints."
        )

    router = APIRouter(prefix="/planner/sessions", tags=["planner-sessions"])

    @router.post(
        "/{session_id}/artifacts",
        status_code=status.HTTP_201_CREATED,
    )
    def save_artifact(session_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            artifact = service.save_artifact(session_id, payload)
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        return artifact.to_dict()

    @router.post("/restore")
    def restore_artifact(payload: Mapping[str, Any]) -> Dict[str, Any]:
        artifact_id = payload.get("artifactId")
        if not artifact_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="artifactId is required",
            )
        target_session_id: Optional[str] = payload.get("targetSessionId")
        strict_version_check = payload.get("strictVersionCheck", True)

        try:
            restored_state, routing_summary, warnings = service.restore_artifact(
                artifact_id,
                target_session_id=target_session_id,
                strict_version_check=bool(strict_version_check),
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc

        response = {
            "sessionId": restored_state.episode_id,
            "restoredAt": datetime.now(timezone.utc).isoformat(),
            "routingSummary": routing_summary,
            "warnings": list(warnings),
        }
        return response

    return router

