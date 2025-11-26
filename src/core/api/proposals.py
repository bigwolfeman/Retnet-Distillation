"""
API helpers for decoding profiles and proposal generation.

Provides a service abstraction plus optional FastAPI router bindings that align
with specs/002-stabilize-titan-hrm/contracts/openapi.yaml.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Protocol, Sequence
import uuid

import torch

from src.models.titans.data_model import Problem, SolutionProposal
from src.models.titans.decoding_profiles import (
    DecodingProfile,
    DecodingProfileCatalog,
    DecodingProfileMode,
    DecodingProfileParameters,
    enable_profile,
    load_catalog,
)
from src.core.telemetry.latency_logger import LatencyLogger


class TokenizerLike(Protocol):
    """Minimal tokenizer protocol expected by the proposal service."""

    def encode(self, text: str, add_special_tokens: bool = True, max_length: Optional[int] = None) -> Sequence[int]: ...

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str: ...


class PlannerEngine(Protocol):
    """Protocol abstraction for SimpleLEngine to ease testing."""

    def solve(
        self,
        problem: Problem,
        blackboard: Optional[Any] = None,
        *,
        profile: Optional[DecodingProfile] = None,
        parameters: Optional[DecodingProfileParameters] = None,
    ) -> SolutionProposal: ...


class ProposalService:
    """Coordinate decoding profiles and proposal generation."""

    def __init__(
        self,
        engine: PlannerEngine,
        *,
        tokenizer: TokenizerLike,
        profiles_path: Path | str = Path("configs/decoding_profiles.yaml"),
        latency_logger: LatencyLogger | None = None,
    ) -> None:
        self._engine = engine
        self._tokenizer = tokenizer
        self._profiles_path = Path(profiles_path)
        self._catalog = load_catalog(self._profiles_path)
        self._latency_logger = latency_logger

    def list_profiles(self, include_disabled: bool = False) -> Iterable[DecodingProfile]:
        profiles = list(self._catalog)
        if include_disabled:
            return profiles
        return [profile for profile in profiles if profile.enabled]

    def get_profile(self, profile_id: str) -> DecodingProfile:
        return self._catalog.get(profile_id)

    def reload_profiles(self) -> None:
        self._catalog = load_catalog(self._profiles_path)

    def set_profile_enabled(self, profile_id: str, enabled: bool) -> None:
        self._catalog = enable_profile(self._catalog, profile_id, enabled)

    def generate_from_prompt(
        self,
        *,
        problem_id: str,
        prompt: str,
        profile_id: str,
        domain: str = "text",
        max_new_tokens: Optional[int] = None,
    ) -> tuple[SolutionProposal, DecodingProfile]:
        if not prompt.strip():
            raise ValueError("Prompt must not be empty")

        profile = self.get_profile(profile_id)
        parameters = profile.parameters
        if max_new_tokens is not None and max_new_tokens > 0:
            parameters = DecodingProfileParameters(
                temperature=parameters.temperature,
                top_p=parameters.top_p,
                top_k=parameters.top_k,
                repetition_penalty=parameters.repetition_penalty,
                max_new_tokens=max_new_tokens,
                reuse_state=parameters.reuse_state,
            )

        token_ids = self._tokenizer.encode(prompt, add_special_tokens=True)
        input_tokens = torch.tensor(token_ids, dtype=torch.long)

        problem = Problem(
            problem_id=problem_id,
            domain=domain,
            input_text=prompt,
            input_tokens=input_tokens,
        )

        proposal = self._engine.solve(problem, profile=profile, parameters=parameters)
        if self._latency_logger is not None:
            self._latency_logger.record(
                event="proposal.generate",
                latency_ms=proposal.latency * 1000,
                metadata={
                    "profile_id": profile.profile_id,
                    "profile_mode": profile.mode.value,
                },
            )
        return proposal, profile


def solution_proposal_to_dict(proposal: SolutionProposal) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "proposalId": proposal.proposal_id,
        "problemId": proposal.problem_id,
        "engineId": proposal.engine_id,
        "contentText": proposal.content_text,
        "tokens": proposal.content.tolist() if isinstance(proposal.content, torch.Tensor) else proposal.content,
        "rawConfidence": proposal.raw_confidence,
        "calibratedConfidence": proposal.calibrated_confidence,
        "latencyMs": int(proposal.latency * 1000),
        "cost": proposal.cost,
        "timestamp": datetime.fromtimestamp(proposal.timestamp, tz=timezone.utc).isoformat(),
        "reasoningTrace": proposal.reasoning_trace or [],
    }
    return payload


def decoding_profile_to_dict(profile: DecodingProfile) -> Dict[str, Any]:
    payload = profile.to_dict()
    payload["mode"] = profile.mode.value
    return payload


try:  # Optional FastAPI dependency
    from fastapi import APIRouter, HTTPException, status
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    APIRouter = None  # type: ignore
    HTTPException = None  # type: ignore
    status = None  # type: ignore


def create_fastapi_router(
    service: ProposalService,
) -> "APIRouter":  # type: ignore[name-defined]
    """Return a FastAPI router exposing decoding profile and proposal endpoints."""
    if APIRouter is None or HTTPException is None or status is None:
        raise RuntimeError(
            "FastAPI is not installed. Install fastapi to enable proposal endpoints."
        )

    router = APIRouter(prefix="/proposals", tags=["proposals"])

    @router.get("/decoding-profiles")
    def list_profiles(includeDisabled: bool = False) -> Iterable[Dict[str, Any]]:
        try:
            return [
                decoding_profile_to_dict(profile)
                for profile in service.list_profiles(include_disabled=includeDisabled)
            ]
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

    @router.post("/generate")
    def generate_proposal(payload: Mapping[str, Any]) -> Dict[str, Any]:
        profile_id = payload.get("profileId")
        prompt = payload.get("prompt")
        session_id = payload.get("sessionId") or f"proposal_{uuid.uuid4().hex[:8]}"
        domain = payload.get("domain", "text")
        max_tokens = payload.get("maxTokens")

        if not profile_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="profileId is required",
            )
        if not prompt:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="prompt is required",
            )

        try:
            proposal, profile = service.generate_from_prompt(
                problem_id=session_id,
                prompt=prompt,
                profile_id=profile_id,
                domain=domain,
                max_new_tokens=max_tokens,
            )
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

        response = solution_proposal_to_dict(proposal)
        response["profile"] = decoding_profile_to_dict(profile)
        return response

    return router
