"""Retrieval registry models and validation utilities (US4).

Implements the retrieval source schema defined in
`specs/002-stabilize-titan-hrm/data-model.md` and powers the
fail-fast checks required by FR-007/FR-008 before enabling retrieval.

The registry is designed to load from a YAML or JSON manifest, validate
asset availability, enforce freshness SLAs, and surface actionable error
messages when preconditions are not met.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence
import json


try:  # PyYAML is optional but preferred for manifest readability
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback to JSON manifests
    yaml = None


__all__ = [
    "AssetEntry",
    "RetrievalRegistry",
    "RetrievalSource",
    "RetrievalSourceStatus",
    "RetrievalSourceType",
    "LandmarkCachePolicy",
]


class RetrievalSourceType(str, Enum):
    """Supported retrieval source categories."""

    DOCUMENT_INDEX = "DOCUMENT_INDEX"
    EMBEDDING_STORE = "EMBEDDING_STORE"
    KNOWLEDGE_GRAPH = "KNOWLEDGE_GRAPH"


class RetrievalSourceStatus(str, Enum):
    """Lifecycle status for retrieval assets."""

    READY = "READY"
    STALE = "STALE"
    MISSING_ASSETS = "MISSING_ASSETS"


class LandmarkCachePolicy(str, Enum):
    """Cache policy for landmark tokens referenced by retrieval sources."""

    REUSE = "REUSE"
    REFRESH = "REFRESH"
    DISABLE = "DISABLE"


def _parse_iso_timestamp(value: Optional[str]) -> Optional[datetime]:
    """Parse ISO-8601 timestamps, normalising to UTC."""

    if not value:
        return None

    text = value.strip()
    if not text:
        return None

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _ensure_uuid(value: str, *, field_name: str) -> str:
    import uuid

    try:
        uuid.UUID(str(value))
    except (ValueError, AttributeError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"{field_name} must be a valid UUID string") from exc
    return str(value)


@dataclass(frozen=True)
class AssetEntry:
    """Entry describing an artefact required by a retrieval source."""

    path: str
    checksum: Optional[str] = None
    freshness_timestamp: Optional[datetime] = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AssetEntry":
        return cls(
            path=str(payload["path"]),
            checksum=payload.get("checksum"),
            freshness_timestamp=_parse_iso_timestamp(payload.get("freshnessTimestamp") or payload.get("freshness_timestamp")),
        )

    def resolved_path(self, base_path: Path) -> Path:
        candidate = Path(self.path).expanduser()
        if not candidate.is_absolute():
            candidate = (base_path / candidate).resolve()
        return candidate

    def exists(self, base_path: Path) -> bool:
        return self.resolved_path(base_path).exists()


@dataclass
class RetrievalSource:
    """Domain model for retrieval source registry entries."""

    source_id: str
    display_name: str
    source_type: RetrievalSourceType
    assets: Sequence[AssetEntry] = field(default_factory=tuple)
    checkpoint_version: Optional[str] = None
    validated_at: Optional[datetime] = None
    freshness_ttl_hours: Optional[int] = None
    status: RetrievalSourceStatus = RetrievalSourceStatus.STALE
    missing_assets: Sequence[str] = field(default_factory=tuple)
    landmark_cache_policy: LandmarkCachePolicy = LandmarkCachePolicy.REUSE

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RetrievalSource":
        source_id = _ensure_uuid(payload["source_id"] if "source_id" in payload else payload["sourceId"], field_name="source_id")
        display_name = str(payload.get("display_name") or payload.get("displayName") or source_id)
        source_type = RetrievalSourceType(payload.get("source_type") or payload.get("sourceType"))

        # Asset metadata may arrive as raw paths or structured entries
        assets: list[AssetEntry] = []
        raw_assets = payload.get("assets") or payload.get("asset_manifest") or payload.get("assetManifest")
        if raw_assets is None:
            raw_paths = payload.get("asset_paths") or payload.get("assetPaths") or []
            assets = [AssetEntry(path=str(path)) for path in raw_paths]
        else:
            assets = [AssetEntry.from_mapping(entry) for entry in raw_assets]

        validated_at = _parse_iso_timestamp(payload.get("validated_at") or payload.get("validatedAt"))
        ttl_hours = payload.get("freshness_ttl_hours") or payload.get("freshnessTtlHours")
        freshness_ttl_hours = int(ttl_hours) if ttl_hours is not None else None

        status_value = payload.get("status")
        status = RetrievalSourceStatus(status_value) if status_value else RetrievalSourceStatus.STALE

        missing_assets = tuple(str(item) for item in payload.get("missing_assets") or payload.get("missingAssets") or ())

        cache_policy_value = payload.get("landmark_cache_policy") or payload.get("landmarkCachePolicy")
        cache_policy = LandmarkCachePolicy(cache_policy_value) if cache_policy_value else LandmarkCachePolicy.REUSE

        return cls(
            source_id=source_id,
            display_name=display_name,
            source_type=source_type,
            assets=tuple(assets),
            checkpoint_version=payload.get("checkpoint_version") or payload.get("checkpointVersion"),
            validated_at=validated_at,
            freshness_ttl_hours=freshness_ttl_hours,
            status=status,
            missing_assets=missing_assets,
            landmark_cache_policy=cache_policy,
        )

    def with_status(
        self,
        *,
        status: RetrievalSourceStatus,
        missing_assets: Sequence[str],
    ) -> "RetrievalSource":
        return replace(self, status=status, missing_assets=tuple(missing_assets))

    def evaluate(self, base_path: Path, *, now: Optional[datetime] = None) -> "RetrievalSource":
        """Compute current status based on asset presence and freshness."""

        resolved_missing: list[str] = []
        for asset in self.assets:
            path = asset.resolved_path(base_path)
            if not path.exists():
                resolved_missing.append(str(path))

        if resolved_missing:
            return self.with_status(status=RetrievalSourceStatus.MISSING_ASSETS, missing_assets=resolved_missing)

        stale = False
        if self.validated_at is not None and self.freshness_ttl_hours:
            expires_at = self.validated_at + timedelta(hours=self.freshness_ttl_hours)
            current = now or datetime.now(timezone.utc)
            stale = current > expires_at

        status = RetrievalSourceStatus.STALE if stale else RetrievalSourceStatus.READY
        return self.with_status(status=status, missing_assets=())

    def to_dict(self, *, base_path: Optional[Path] = None) -> Dict[str, Any]:
        """Serialize source metadata for API responses."""

        asset_paths: list[str] = []
        for asset in self.assets:
            if base_path is None:
                asset_paths.append(asset.path)
            else:
                asset_paths.append(str(asset.resolved_path(base_path)))

        return {
            "sourceId": self.source_id,
            "displayName": self.display_name,
            "sourceType": self.source_type.value,
            "assetPaths": asset_paths,
            "checkpointVersion": self.checkpoint_version,
            "validatedAt": self.validated_at.isoformat() if self.validated_at else None,
            "freshnessTtlHours": self.freshness_ttl_hours,
            "status": self.status.value,
            "missingAssets": list(self.missing_assets),
            "landmarkCachePolicy": self.landmark_cache_policy.value,
        }


class RetrievalRegistry:
    """Collection of retrieval sources plus validation helpers."""

    def __init__(self, sources: Iterable[RetrievalSource], *, manifest_path: Path) -> None:
        sources_list = list(sources)
        seen: MutableMapping[str, str] = {}
        for source in sources_list:
            if source.source_id in seen:
                raise ValueError(
                    f"Duplicate retrieval source id '{source.source_id}' defined for "
                    f"'{source.display_name}' and '{seen[source.source_id]}'"
                )
            seen[source.source_id] = source.display_name

        self._sources: Dict[str, RetrievalSource] = {source.source_id: source for source in sources_list}
        self._manifest_path = manifest_path
        self._base_path = manifest_path.parent.resolve()

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    @property
    def base_path(self) -> Path:
        return self._base_path

    def list_sources(self) -> Sequence[RetrievalSource]:
        return list(self._sources.values())

    def get(self, source_id: str) -> RetrievalSource:
        return self._sources[source_id]

    def evaluate(self, *, now: Optional[datetime] = None) -> Dict[str, RetrievalSource]:
        evaluated: Dict[str, RetrievalSource] = {}
        for source_id, source in self._sources.items():
            evaluated[source_id] = source.evaluate(self._base_path, now=now)
        return evaluated

    def ensure_all_ready(self, *, now: Optional[datetime] = None) -> Dict[str, RetrievalSource]:
        evaluated = self.evaluate(now=now)
        failures = [src for src in evaluated.values() if src.status is not RetrievalSourceStatus.READY]
        if failures:
            lines = []
            for source in failures:
                if source.status is RetrievalSourceStatus.MISSING_ASSETS:
                    lines.append(
                        f"{source.display_name}: missing assets -> {', '.join(source.missing_assets)}"
                    )
                elif source.status is RetrievalSourceStatus.STALE:
                    ttl = source.freshness_ttl_hours or 0
                    timestamp = source.validated_at.isoformat() if source.validated_at else "unknown"
                    lines.append(
                        f"{source.display_name}: validation expired (validated_at={timestamp}, ttl={ttl}h)"
                    )
            raise RuntimeError(
                "Retrieval registry validation failed:\n - " + "\n - ".join(lines)
            )
        return evaluated

    def to_dict(self) -> Dict[str, Any]:
        evaluated = self.evaluate()
        return {
            "manifestPath": str(self._manifest_path),
            "sources": [src.to_dict(base_path=self._base_path) for src in evaluated.values()],
        }

    @classmethod
    def load(cls, manifest_path: Path | str) -> "RetrievalRegistry":
        path = Path(manifest_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Retrieval manifest not found: {path}")

        suffix = path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            if yaml is None:
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:  # pragma: no cover - requires PyYAML on invalid JSON
                    raise RuntimeError(
                        "PyYAML is required to load YAML retrieval manifests. Install with 'pip install pyyaml'."
                    ) from exc
            else:
                raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            raw = json.loads(path.read_text(encoding="utf-8"))

        if raw is None:
            raise ValueError(f"Retrieval manifest {path} is empty")

        if isinstance(raw, Mapping) and "sources" in raw:
            entries = raw["sources"]
        else:
            entries = raw

        if not isinstance(entries, Sequence):
            raise ValueError("Retrieval manifest must contain a sequence of sources")

        sources = [RetrievalSource.from_mapping(entry) for entry in entries]
        return cls(sources, manifest_path=path)
