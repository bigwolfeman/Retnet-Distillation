"""
Domain models representing Titan-HRM runtime readiness checks.

These structures codify the invariants captured in the stabilization data
model: dependency checks, optional module acknowledgements, and the overall
readiness report that gates environment launches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable, List, Sequence


class CheckStatus(str, Enum):
    """Possible outcomes for a dependency verification."""

    PASSED = "PASSED"
    FAILED = "FAILED"
    MISSING = "MISSING"


class OptionalModuleStatus(str, Enum):
    """Availability of an optional accelerator or performance enhancer."""

    AVAILABLE = "AVAILABLE"
    MISSING = "MISSING"


class RuntimeReadinessStatus(str, Enum):
    """Pass/fail state for the complete runtime readiness assessment."""

    PASS = "PASS"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class DependencyCheckResult:
    """
    Result of validating a mandatory runtime dependency.

    Attributes mirror the dependency check value object defined in the data
    model. Validation ensures we do not accidentally mark a dependency as
    passed without capturing the detected version.
    """

    name: str
    version_required: str
    status: CheckStatus
    detected_version: str | None = None
    remediation: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("DependencyCheckResult.name must be provided")
        if not self.version_required:
            raise ValueError("DependencyCheckResult.version_required must be provided")
        if self.status == CheckStatus.PASSED and not self.detected_version:
            raise ValueError(
                f"{self.name} marked as PASSED but detected_version is missing"
            )


@dataclass(frozen=True)
class OptionalModuleCheckResult:
    """
    Result of auditing an optional module or accelerator.

    Optional modules can be missing, but missing modules require acknowledgement
    from the operator and a concise impact summary.
    """

    name: str
    status: OptionalModuleStatus
    impact_summary: str
    acknowledged_by: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("OptionalModuleCheckResult.name must be provided")
        if not self.impact_summary:
            raise ValueError("OptionalModuleCheckResult.impact_summary must be provided")
        if self.status == OptionalModuleStatus.MISSING and not self.acknowledged_by:
            raise ValueError(
                f"{self.name} missing but acknowledged_by is not recorded"
            )


def _ensure_all_passed(checks: Sequence[DependencyCheckResult]) -> bool:
    return all(check.status == CheckStatus.PASSED for check in checks)


def _ensure_all_provided(items: Iterable[str | None], field_name: str) -> None:
    for value in items:
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError(f"RuntimeReadinessReport.{field_name} values must be set")


@dataclass(frozen=True)
class RuntimeReadinessReport:
    """
    Aggregated readiness data used to gate Titan-HRM model launches.

    The report couples the status flag with the individual dependency outcomes,
    optional module acknowledgements, and remediation guidance. Validation is
    enforced in `__post_init__` to catch configuration errors early.
    """

    report_id: str
    environment_id: str
    generated_at: datetime
    status: RuntimeReadinessStatus
    mandatory_checks: Sequence[DependencyCheckResult] = field(default_factory=list)
    optional_modules: Sequence[OptionalModuleCheckResult] = field(default_factory=list)
    remediation_actions: Sequence[str] = field(default_factory=list)
    operator_acknowledgement: bool = False
    notes: str | None = None

    def __post_init__(self) -> None:
        _ensure_all_provided(
            [self.report_id, self.environment_id],
            field_name="identifier",
        )
        if self.generated_at.tzinfo is None:
            object.__setattr__(self, "generated_at", self.generated_at.replace(tzinfo=timezone.utc))

        if not self.mandatory_checks:
            raise ValueError("RuntimeReadinessReport requires at least one mandatory check")

        if self.status == RuntimeReadinessStatus.PASS and not _ensure_all_passed(
            self.mandatory_checks
        ):
            raise ValueError(
                "RuntimeReadinessReport cannot be PASS when any mandatory check failed"
            )

        if self.status == RuntimeReadinessStatus.BLOCKED and _ensure_all_passed(
            self.mandatory_checks
        ):
            raise ValueError(
                "RuntimeReadinessReport marked BLOCKED but all mandatory checks passed"
            )

        missing_optional = [
            module
            for module in self.optional_modules
            if module.status == OptionalModuleStatus.MISSING
        ]
        if missing_optional and not self.operator_acknowledgement:
            raise ValueError(
                "Optional modules are missing but operator_acknowledgement is False"
            )

    def is_pass(self) -> bool:
        """Return True when the report status indicates the environment is ready."""
        return self.status == RuntimeReadinessStatus.PASS

    def missing_dependencies(self) -> List[DependencyCheckResult]:
        """Return all mandatory dependencies that failed or are missing."""
        return [
            check
            for check in self.mandatory_checks
            if check.status in {CheckStatus.FAILED, CheckStatus.MISSING}
        ]

    def missing_optional_modules(self) -> List[OptionalModuleCheckResult]:
        """Return optional modules that are not currently available."""
        return [
            module
            for module in self.optional_modules
            if module.status == OptionalModuleStatus.MISSING
        ]

    def remediation_plan(self) -> List[str]:
        """
        Merge remediation steps written on individual dependency checks with
        top-level guidance, ensuring the operator receives a consolidated plan.
        """
        steps: List[str] = list(self.remediation_actions)
        for check in self.mandatory_checks:
            if check.remediation:
                steps.append(check.remediation)
        return steps
