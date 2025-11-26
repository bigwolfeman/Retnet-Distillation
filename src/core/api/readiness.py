"""
API helpers for Titan-HRM runtime readiness workflows.

Provides data mappers and service orchestration to expose readiness scans via
HTTP handlers or other integration layers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

try:  # FastAPI is optional; router helpers work only when installed.
    from fastapi import APIRouter, HTTPException, status
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    APIRouter = None  # type: ignore
    HTTPException = None  # type: ignore
    status = None  # type: ignore

from src.core.readiness.auditor import DetectionError, RuntimeAuditor
from src.core.stabilization.manifest_loader import (
    MANIFEST_SCHEMA_PATH,
    RuntimeManifest,
    build_runtime_manifest,
    validate_manifest,
)
from src.models.runtime_readiness.report import RuntimeReadinessReport
from src.models.runtime_readiness.serialization import report_to_dict

DEFAULT_SCHEMA_VERSION = "1.0.0"


def _serialize_report_internal(report: RuntimeReadinessReport) -> Dict[str, Any]:
    """Serialize a readiness report for API responses."""
    return report_to_dict(report)


def _prepare_manifest_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert client payload into a manifest-like structure for validation."""
    if "environmentId" not in payload:
        raise ValueError("environmentId is required")
    mandatory = payload.get("mandatoryDependencies") or []
    if not mandatory:
        raise ValueError("mandatoryDependencies must contain at least one entry")

    environment = {"id": payload["environmentId"]}
    if label := payload.get("environmentLabel"):
        environment["label"] = label
    if description := payload.get("environmentDescription"):
        environment["description"] = description

    manifest: Dict[str, Any] = {
        "schemaVersion": payload.get("schemaVersion", DEFAULT_SCHEMA_VERSION),
        "environment": environment,
        "mandatoryDependencies": [],
    }

    for dependency in mandatory:
        command = dependency.get("detectionCommand")
        if not command:
            raise ValueError(
                f"detectionCommand missing for dependency '{dependency.get('name')}'"
            )
        manifest["mandatoryDependencies"].append(
            {
                "name": dependency["name"],
                "minVersion": dependency["minVersion"],
                "detection": {"command": command},
                "remediation": dependency.get("remediation"),
            }
        )

    optional_modules = []
    for module in payload.get("optionalModules", []):
        entry: Dict[str, Any] = {
            "name": module["name"],
            "performanceImpact": module.get(
                "performanceImpact", "Impact not documented."
            ),
        }
        if module.get("detectionCommand"):
            entry["detection"] = {"command": module["detectionCommand"]}
        if "acknowledgementRequired" in module:
            entry["acknowledgementRequired"] = module["acknowledgementRequired"]
        optional_modules.append(entry)

    if optional_modules:
        manifest["optionalModules"] = optional_modules

    if payload.get("globalRemediation"):
        manifest["globalRemediation"] = payload["globalRemediation"]

    return manifest


class ReadinessService:
    """In-memory orchestration layer for readiness scans."""

    def __init__(
        self,
        *,
        schema_path: Path | str = MANIFEST_SCHEMA_PATH,
    ) -> None:
        self._schema_path = Path(schema_path)
        self._reports: MutableMapping[str, RuntimeReadinessReport] = {}

    def _build_manifest(self, payload: Mapping[str, Any]) -> RuntimeManifest:
        manifest_payload = _prepare_manifest_payload(payload)
        validate_manifest(manifest_payload, schema_path=self._schema_path)
        return build_runtime_manifest(manifest_payload)

    def create_scan(
        self,
        payload: Mapping[str, Any],
        *,
        operator_acknowledged: bool,
        notes: Optional[str] = None,
    ) -> RuntimeReadinessReport:
        manifest = self._build_manifest(payload)
        auditor = RuntimeAuditor(manifest)
        report = auditor.generate_report(
            operator_acknowledged=operator_acknowledged,
            notes=notes,
        )
        self._reports[report.report_id] = report
        return report

    def get_report(self, report_id: str) -> RuntimeReadinessReport:
        try:
            return self._reports[report_id]
        except KeyError as exc:
            raise KeyError(f"Report '{report_id}' not found") from exc

    def list_reports(self) -> Iterable[RuntimeReadinessReport]:
        return list(self._reports.values())


def create_fastapi_router(
    service: ReadinessService,
) -> "APIRouter":  # type: ignore[name-defined]
    """Return a FastAPI router exposing runtime readiness endpoints."""
    if APIRouter is None or HTTPException is None or status is None:
        raise RuntimeError(
            "FastAPI is not installed. Install fastapi to enable HTTP endpoints."
        )

    router = APIRouter(prefix="/runtime-readiness", tags=["runtime-readiness"])

    @router.post(
        "/scans",
        status_code=status.HTTP_201_CREATED,
    )
    def create_scan_endpoint(payload: Dict[str, Any]) -> Dict[str, Any]:
        operator_ack = payload.get("operatorAcknowledged", False)
        notes = payload.get("notes")
        try:
            report = service.create_scan(
                payload,
                operator_acknowledged=operator_ack,
                notes=notes,
            )
        except (ValueError, DetectionError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        return _serialize_report_internal(report)

    @router.get("/scans/{report_id}")
    def get_report_endpoint(report_id: str) -> Dict[str, Any]:
        try:
            report = service.get_report(report_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc
        return _serialize_report_internal(report)

    return router
