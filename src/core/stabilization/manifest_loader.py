"""
Utilities for loading and validating Titan-HRM runtime readiness manifests.

The manifest encodes mandatory dependencies and optional accelerators that the
readiness workflow consumes. Loader helpers centralise schema validation and
normalise YAML/JSON ingestion for downstream components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
import json

MANIFEST_SCHEMA_PATH = Path("configs/stabilization/runtime_manifest.schema.json")


def _load_yaml_module():
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - dependent on environment
        raise RuntimeError(
            "PyYAML is required to parse runtime manifests with .yaml extension. "
            "Install with `pip install pyyaml`."
        ) from exc
    return yaml


def _read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Runtime manifest file not found: {path}")
    return path.read_text(encoding="utf-8")


def _parse_manifest_text(path: Path) -> Dict[str, Any]:
    text = _read_text(path)
    if path.suffix in {".yaml", ".yml"}:
        yaml = _load_yaml_module()
        return yaml.safe_load(text) or {}
    return json.loads(text or "{}")
class ManifestValidationError(ValueError):
    """Raised when a runtime manifest fails validation."""


@dataclass(frozen=True)
class EnvironmentConfig:
    id: str
    label: str | None = None
    description: str | None = None


@dataclass(frozen=True)
class MandatoryDependencyConfig:
    name: str
    min_version: str
    detection_command: str
    detection_env: Mapping[str, str] = field(default_factory=dict)
    remediation: str | None = None


@dataclass(frozen=True)
class OptionalModuleConfig:
    name: str
    performance_impact: str
    detection_command: str | None = None
    detection_env: Mapping[str, str] = field(default_factory=dict)
    acknowledgement_required: bool = True


@dataclass(frozen=True)
class RuntimeManifest:
    schema_version: str
    environment: EnvironmentConfig
    mandatory_dependencies: Sequence[MandatoryDependencyConfig]
    optional_modules: Sequence[OptionalModuleConfig] = field(default_factory=list)
    global_remediation: Sequence[str] = field(default_factory=list)


def validate_manifest(raw_manifest: Mapping[str, Any], schema_path: Optional[Path] = None) -> None:
    """Validate raw manifest data using lightweight structural checks."""

    errors: List[str] = []
    _ = schema_path  # Retained for compatibility with previous interface.

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(isinstance(raw_manifest, Mapping), "Manifest must be a mapping/dictionary")
    if not isinstance(raw_manifest, Mapping):
        raise ManifestValidationError("Manifest must be a mapping/dictionary")

    schema_version = raw_manifest.get("schemaVersion")
    require(isinstance(schema_version, str) and bool(schema_version), "schemaVersion must be a non-empty string")

    environment = raw_manifest.get("environment")
    require(isinstance(environment, Mapping), "environment must be a mapping")
    if isinstance(environment, Mapping):
        require(
            isinstance(environment.get("id"), str) and bool(environment.get("id")),
            "environment.id must be a non-empty string",
        )

    mandatory = raw_manifest.get("mandatoryDependencies")
    require(isinstance(mandatory, list) and mandatory, "mandatoryDependencies must be a non-empty list")
    if isinstance(mandatory, list):
        for idx, dependency in enumerate(mandatory):
            require(isinstance(dependency, Mapping), f"mandatoryDependencies[{idx}] must be a mapping")
            if not isinstance(dependency, Mapping):
                continue
            require(
                isinstance(dependency.get("name"), str) and dependency["name"],
                f"Dependency {idx} missing name",
            )
            require(
                isinstance(dependency.get("minVersion"), str) and dependency["minVersion"],
                f"Dependency {dependency.get('name', idx)} missing minVersion",
            )
            detection = dependency.get("detection")
            require(
                isinstance(detection, Mapping) and isinstance(detection.get("command"), str) and detection["command"],
                f"Dependency {dependency.get('name', idx)} missing detection.command",
            )

    optional_modules = raw_manifest.get("optionalModules", [])
    if optional_modules is not None:
        require(isinstance(optional_modules, list), "optionalModules must be a list when provided")
        if isinstance(optional_modules, list):
            for idx, module in enumerate(optional_modules):
                require(isinstance(module, Mapping), f"optionalModules[{idx}] must be a mapping")
                if not isinstance(module, Mapping):
                    continue
                require(
                    isinstance(module.get("name"), str) and module["name"],
                    f"optionalModules[{idx}] missing name",
                )
                performance = module.get("performanceImpact")
                if performance is not None:
                    require(
                        isinstance(performance, str),
                        f"optionalModules[{idx}].performanceImpact must be a string",
                    )
                detection = module.get("detection")
                if detection is not None:
                    require(
                        isinstance(detection, Mapping),
                        f"optionalModules[{idx}].detection must be a mapping",
                    )
                    if isinstance(detection, Mapping) and detection.get("command") is not None:
                        require(
                            isinstance(detection.get("command"), str) and detection["command"],
                            f"optionalModules[{idx}].detection.command must be a non-empty string",
                        )

    if errors:
        raise ManifestValidationError("Runtime manifest validation failed:\n" + "\n".join(f"- {msg}" for msg in errors))


def build_runtime_manifest(raw: Mapping[str, Any]) -> RuntimeManifest:
    """Construct a RuntimeManifest object from validated raw data."""
    environment_data = raw["environment"]
    environment = EnvironmentConfig(
        id=environment_data["id"],
        label=environment_data.get("label"),
        description=environment_data.get("description"),
    )

    mandatory_dependencies = [
        MandatoryDependencyConfig(
            name=dep["name"],
            min_version=dep["minVersion"],
            detection_command=dep["detection"]["command"],
            detection_env=dep["detection"].get("env", {}),
            remediation=dep.get("remediation"),
        )
        for dep in raw["mandatoryDependencies"]
    ]

    optional_modules_raw: Iterable[Mapping[str, Any]] = raw.get("optionalModules", [])
    optional_modules: List[OptionalModuleConfig] = []
    for module in optional_modules_raw:
        detection = module.get("detection") or {}
        optional_modules.append(
            OptionalModuleConfig(
                name=module["name"],
                performance_impact=module["performanceImpact"],
                detection_command=detection.get("command"),
                detection_env=detection.get("env", {}),
                acknowledgement_required=module.get("acknowledgementRequired", True),
            )
        )

    return RuntimeManifest(
        schema_version=raw["schemaVersion"],
        environment=environment,
        mandatory_dependencies=mandatory_dependencies,
        optional_modules=optional_modules,
        global_remediation=raw.get("globalRemediation", []),
    )


def load_runtime_manifest(path: Path | str, schema_path: Optional[Path | str] = None) -> RuntimeManifest:
    """Load, validate, and normalize a runtime readiness manifest."""
    manifest_path = Path(path)
    raw = _parse_manifest_text(manifest_path)
    validate_manifest(
        raw,
        schema_path=Path(schema_path) if schema_path else None,
    )
    return build_runtime_manifest(raw)
