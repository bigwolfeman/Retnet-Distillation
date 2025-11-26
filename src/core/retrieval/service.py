"""Services for retrieval registry validation (US4)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from src.models.retrieval import (
    AssetEntry,
    RetrievalRegistry,
    RetrievalSource,
    RetrievalSourceStatus,
)


@dataclass(frozen=True)
class ValidationResult:
    """Container for retrieval validation outcomes."""

    source: RetrievalSource
    status: RetrievalSourceStatus
    missing_assets: Sequence[str]
    remediation_actions: Sequence[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sourceId": self.source.source_id,
            "status": self.status.value,
            "missingAssets": list(self.missing_assets),
            "remediationActions": list(self.remediation_actions),
        }


class RetrievalValidationService:
    """High-level orchestration for retrieval registry operations."""

    def __init__(self, registry: RetrievalRegistry) -> None:
        self._registry = registry

    @property
    def registry(self) -> RetrievalRegistry:
        return self._registry

    def list_sources(self) -> Sequence[RetrievalSource]:
        return list(self._registry.evaluate().values())

    def validate_source(
        self,
        source_id: str,
        *,
        asset_manifest: Sequence[Mapping[str, Any]] | None = None,
        force_refresh: bool = False,
    ) -> ValidationResult:
        try:
            source = self._registry.get(source_id)
        except KeyError as exc:
            raise KeyError(f"Retrieval source '{source_id}' not found") from exc

        if asset_manifest:
            override_assets = [AssetEntry.from_mapping(entry) for entry in asset_manifest]
            target_source = replace(source, assets=tuple(override_assets))
        else:
            target_source = source

        evaluated = target_source.evaluate(self._registry.base_path)
        status = evaluated.status
        missing_assets = list(evaluated.missing_assets)
        remediation: list[str] = []

        if status is RetrievalSourceStatus.MISSING_ASSETS:
            remediation = [f"Restore or rebuild asset: {path}" for path in missing_assets]
        elif status is RetrievalSourceStatus.STALE and not force_refresh:
            remediation = [
                "Refresh source assets or rerun validation pipeline to update timestamps."
            ]

        if status is RetrievalSourceStatus.STALE and force_refresh:
            status = RetrievalSourceStatus.READY
            remediation = []

        adjusted_source = replace(evaluated, status=status, missing_assets=tuple(missing_assets))

        return ValidationResult(
            source=adjusted_source,
            status=status,
            missing_assets=missing_assets,
            remediation_actions=remediation,
        )
