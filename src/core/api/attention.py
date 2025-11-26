"""FastAPI router for attention memory forecast endpoints."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from src.core.attention import AttentionForecastService


try:  # Optional dependency
    from fastapi import APIRouter, HTTPException, status
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    APIRouter = None  # type: ignore
    HTTPException = None  # type: ignore
    status = None  # type: ignore


def create_fastapi_router(
    service: AttentionForecastService,
) -> "APIRouter":  # type: ignore[name-defined]
    if APIRouter is None or HTTPException is None or status is None:
        raise RuntimeError(
            "FastAPI is not installed. Install fastapi to enable attention endpoints."
        )

    router = APIRouter(prefix="/attention", tags=["attention"])

    @router.post("/band/forecast")
    def forecast_attention(payload: Mapping[str, Any]) -> Dict[str, Any]:
        seq_len = payload.get("sequenceLength")
        if seq_len is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="sequenceLength is required",
            )

        window_size = payload.get("windowSize", 2048)
        batch_size = payload.get("batchSize", 1)
        current_mode = payload.get("currentMode", "DENSE")
        dtype_bytes = payload.get("stepBytesEstimate")
        global_tokens = payload.get("globalTokens", 0)

        try:
            result = service.forecast(
                int(seq_len),
                window_size=int(window_size),
                batch_size=int(batch_size),
                current_mode=str(current_mode),
                global_tokens=int(global_tokens),
                step_bytes_estimate=int(dtype_bytes) if dtype_bytes is not None else None,
            )
        except (ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

        return result.to_dict()

    return router
