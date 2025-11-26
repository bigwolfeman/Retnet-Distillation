"""Serialization helpers for runtime readiness domain models."""

from __future__ import annotations

from typing import Any, Dict

from .report import (
    DependencyCheckResult,
    OptionalModuleCheckResult,
    RuntimeReadinessReport,
)


def dependency_to_dict(dep: DependencyCheckResult) -> Dict[str, Any]:
    return {
        "name": dep.name,
        "version_required": dep.version_required,
        "status": dep.status.value,
        "detected_version": dep.detected_version,
        "remediation": dep.remediation,
    }


def optional_module_to_dict(module: OptionalModuleCheckResult) -> Dict[str, Any]:
    return {
        "name": module.name,
        "status": module.status.value,
        "impact_summary": module.impact_summary,
        "acknowledged_by": module.acknowledged_by,
    }


def report_to_dict(report: RuntimeReadinessReport) -> Dict[str, Any]:
    return {
        "report_id": report.report_id,
        "environment_id": report.environment_id,
        "generated_at": report.generated_at.isoformat(),
        "status": report.status.value,
        "mandatory_checks": [dependency_to_dict(dep) for dep in report.mandatory_checks],
        "optional_modules": [optional_module_to_dict(module) for module in report.optional_modules],
        "remediation_actions": list(report.remediation_actions),
        "operator_acknowledgement": report.operator_acknowledgement,
        "notes": report.notes,
    }
