"""
Runtime readiness auditing workflow.

Consumes the runtime manifest, executes dependency detection commands, and
produces a structured `RuntimeReadinessReport` for operators and downstream
automation.
"""

from __future__ import annotations

import json
import re
import subprocess
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional

from src.core.stabilization.hash_utils import hash_text
from src.core.stabilization.manifest_loader import (
    MANIFEST_SCHEMA_PATH,
    MandatoryDependencyConfig,
    OptionalModuleConfig,
    RuntimeManifest,
    load_runtime_manifest,
)
from src.models.runtime_readiness.report import (
    CheckStatus,
    DependencyCheckResult,
    OptionalModuleCheckResult,
    OptionalModuleStatus,
    RuntimeReadinessReport,
    RuntimeReadinessStatus,
)

try:
    from packaging import version as packaging_version
except ModuleNotFoundError:  # pragma: no cover - runtime dependency check
    packaging_version = None


class DetectionError(Exception):
    """Raised when dependency detection fails unexpectedly."""


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class CommandRunner:
    """Executes shell commands with optional environment overrides."""

    def run(self, command: str, env: Optional[Mapping[str, str]] = None) -> CommandResult:
        merged_env = None
        if env is not None:
            merged_env = os.environ.copy()
            merged_env.update(env)
        completed = subprocess.run(
            command,
            shell=True,
            env=merged_env,
            capture_output=True,
            text=True,
            check=False,
        )
        return CommandResult(
            returncode=completed.returncode,
            stdout=completed.stdout.strip(),
            stderr=completed.stderr.strip(),
        )


def _parse_detected_version(output: str) -> Optional[str]:
    if not output:
        return None
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        parsed = output
    if isinstance(parsed, dict) and "version" in parsed:
        return str(parsed["version"]).strip()
    if isinstance(parsed, str):
        token = parsed.strip().split()[0]
        return token or None
    return None


def _compare_versions(detected: str, minimum: str) -> bool:
    if packaging_version:
        return packaging_version.parse(detected) >= packaging_version.parse(minimum)

    def normalize(value: str) -> tuple:
        parts = re.split(r"[._-]", value)
        normalized = []
        for part in parts:
            if part.isdigit():
                normalized.append(int(part))
            else:
                normalized.append(part)
        return tuple(normalized)

    return normalize(detected) >= normalize(minimum)


def _run_dependency_check(
    dependency: MandatoryDependencyConfig,
    runner: CommandRunner,
) -> DependencyCheckResult:
    result = runner.run(dependency.detection_command, env=dependency.detection_env)
    if result.returncode != 0:
        status = CheckStatus.MISSING
        detected_version = None
    else:
        detected_version = _parse_detected_version(result.stdout)
        if not detected_version:
            status = CheckStatus.FAILED
        elif _compare_versions(detected_version, dependency.min_version):
            status = CheckStatus.PASSED
        else:
            status = CheckStatus.FAILED
    remediation = dependency.remediation
    if status == CheckStatus.FAILED and not remediation:
        remediation = (
            f"Upgrade {dependency.name} to at least version {dependency.min_version}"
        )
    if status == CheckStatus.MISSING and not remediation:
        remediation = f"Install {dependency.name} (>= {dependency.min_version})"
    return DependencyCheckResult(
        name=dependency.name,
        version_required=dependency.min_version,
        status=status,
        detected_version=detected_version,
        remediation=remediation,
    )


def _run_optional_module_check(
    module: OptionalModuleConfig,
    runner: CommandRunner,
    *,
    operator_acknowledged: bool,
) -> OptionalModuleCheckResult:
    status = OptionalModuleStatus.AVAILABLE
    acknowledged_by = None
    # Attempt detection when a command is provided; missing command implies manual check.
    if module.detection_command:
        result = runner.run(module.detection_command, env=module.detection_env)
        if result.returncode != 0:
            status = OptionalModuleStatus.MISSING
            if operator_acknowledged:
                acknowledged_by = "operator"
    return OptionalModuleCheckResult(
        name=module.name,
        status=status,
        impact_summary=module.performance_impact,
        acknowledged_by=acknowledged_by,
    )


class RuntimeAuditor:
    """
    Evaluate runtime manifests and produce readiness reports.

    The auditor coordinates dependency detection, optional module assessment,
    and remediation rollups, returning a structured `RuntimeReadinessReport`.
    """

    def __init__(
        self,
        manifest: RuntimeManifest,
        command_runner: Optional[CommandRunner] = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._manifest = manifest
        self._command_runner = command_runner or CommandRunner()
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    @classmethod
    def from_path(
        cls,
        manifest_path: Path | str,
        *,
        schema_path: Path | str | None = None,
        command_runner: Optional[CommandRunner] = None,
        clock: Callable[[], datetime] | None = None,
    ) -> "RuntimeAuditor":
        manifest = load_runtime_manifest(manifest_path, schema_path=schema_path)
        return cls(manifest, command_runner=command_runner, clock=clock)

    def _check_dependencies(self) -> Iterable[DependencyCheckResult]:
        for dependency in self._manifest.mandatory_dependencies:
            yield _run_dependency_check(dependency, self._command_runner)

    def _check_optional_modules(
        self, *, operator_acknowledged: bool
    ) -> Iterable[OptionalModuleCheckResult]:
        for module in self._manifest.optional_modules:
            yield _run_optional_module_check(
                module,
                self._command_runner,
                operator_acknowledged=operator_acknowledged,
            )

    def generate_report(
        self,
        *,
        operator_acknowledged: bool,
        notes: Optional[str] = None,
        supplemental_remediation: Optional[Iterable[str]] = None,
    ) -> RuntimeReadinessReport:
        generated_at = self._clock()
        mandatory_results = list(self._check_dependencies())
        optional_results = list(
            self._check_optional_modules(operator_acknowledged=operator_acknowledged)
        )

        missing_or_failed = [
            result
            for result in mandatory_results
            if result.status in {CheckStatus.FAILED, CheckStatus.MISSING}
        ]
        status = (
            RuntimeReadinessStatus.PASS
            if not missing_or_failed
            else RuntimeReadinessStatus.BLOCKED
        )

        remediation_steps = list(self._manifest.global_remediation)
        if supplemental_remediation:
            remediation_steps.extend(supplemental_remediation)

        # `operator_acknowledgement` must be true if any optional modules missing.
        optional_missing = [
            module
            for module in optional_results
            if module.status == OptionalModuleStatus.MISSING
        ]
        if optional_missing and not operator_acknowledged:
            raise DetectionError(
                "Optional modules are missing but operator acknowledgement not provided."
            )

        return RuntimeReadinessReport(
            report_id=hash_text(
                f"{self._manifest.environment.id}:{generated_at.isoformat()}"
            ),
            environment_id=self._manifest.environment.id,
            generated_at=generated_at,
            status=status,
            mandatory_checks=mandatory_results,
            optional_modules=optional_results,
            remediation_actions=remediation_steps,
            operator_acknowledgement=operator_acknowledged,
            notes=notes,
        )


def validate_manifest_file(path: Path | str) -> None:
    """Validate a manifest file on disk, raising if it is invalid."""
    load_runtime_manifest(path, schema_path=MANIFEST_SCHEMA_PATH)
