"""FastAPI bindings for retrieval registry operations."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from src.core.retrieval import RetrievalValidationService


try:  # Optional FastAPI dependency for runtime APIs
    from fastapi import APIRouter, HTTPException, status
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    APIRouter = None  # type: ignore
    HTTPException = None  # type: ignore
    status = None  # type: ignore


def _serialize_source(service: RetrievalValidationService, source) -> Dict[str, Any]:
    return source.to_dict(base_path=service.registry.base_path)


def _serialize_validation(result) -> Dict[str, Any]:
    payload = result.to_dict()
    return payload


def create_fastapi_router(
    service: RetrievalValidationService,
) -> "APIRouter":  # type: ignore[name-defined]
    if APIRouter is None or HTTPException is None or status is None:
        raise RuntimeError(
            "FastAPI is not installed. Install fastapi to enable retrieval endpoints."
        )

    router = APIRouter(prefix="/retrieval", tags=["retrieval"])

    @router.get("/sources")
    def list_sources() -> Sequence[Dict[str, Any]]:
        sources = service.list_sources()
        return [_serialize_source(service, source) for source in sources]

    @router.post("/sources/validate")
    def validate_source(payload: Mapping[str, Any]) -> Dict[str, Any]:
        source_id = payload.get("sourceId")
        if not source_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="sourceId is required",
            )

        asset_manifest = payload.get("assetManifest")
        force_refresh = bool(payload.get("forceRefresh", False))

        try:
            result = service.validate_source(
                source_id,
                asset_manifest=asset_manifest,
                force_refresh=force_refresh,
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc

        response = _serialize_validation(result)
        return response

    return router
