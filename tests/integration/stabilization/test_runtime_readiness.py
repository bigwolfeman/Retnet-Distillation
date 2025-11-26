import pytest

from src.core.readiness.auditor import CommandResult, RuntimeAuditor
from src.core.stabilization.manifest_loader import build_runtime_manifest, validate_manifest
from src.models.runtime_readiness.report import CheckStatus, RuntimeReadinessStatus


class FakeRunner:
    def __init__(self, outcomes):
        self._outcomes = outcomes

    def run(self, command, env=None):
        result = self._outcomes.get(command)
        if result is None:
            raise AssertionError(f"Unexpected command executed: {command}")
        return result


def _build_manifest(overrides=None):
    payload = {
        "schemaVersion": "1.0.0",
        "environment": {"id": "test-env"},
        "mandatoryDependencies": [
            {
                "name": "torch",
                "minVersion": "2.0.0",
                "detection": {"command": "check torch"},
                "remediation": "pip install torch>=2.0.0",
            }
        ],
        "optionalModules": [
            {
                "name": "xformers",
                "performanceImpact": "Improves attention throughput",
                "detection": {"command": "check xformers"},
            }
        ],
        "globalRemediation": ["Consult readiness docs"],
    }
    if overrides:
        payload.update(overrides)
    validate_manifest(payload)
    return build_runtime_manifest(payload)


def test_runtime_readiness_passes_when_all_dependencies_succeed():
    manifest = _build_manifest()
    runner = FakeRunner(
        {
            "check torch": CommandResult(0, '{"version": "2.1.0"}', ""),
            "check xformers": CommandResult(0, "", ""),
        }
    )
    auditor = RuntimeAuditor(manifest, command_runner=runner)
    report = auditor.generate_report(operator_acknowledged=True)

    assert report.status == RuntimeReadinessStatus.PASS
    assert not report.missing_dependencies()
    assert report.operator_acknowledgement is True


def test_runtime_readiness_blocks_when_dependency_missing():
    manifest = _build_manifest()
    runner = FakeRunner(
        {
            "check torch": CommandResult(1, "", "module not found"),
            "check xformers": CommandResult(1, "", "module not found"),
        }
    )
    auditor = RuntimeAuditor(manifest, command_runner=runner)
    report = auditor.generate_report(operator_acknowledged=True)

    assert report.status == RuntimeReadinessStatus.BLOCKED
    missing = report.missing_dependencies()
    assert missing
    assert missing[0].status in {CheckStatus.MISSING, CheckStatus.FAILED}

    optional_missing = report.missing_optional_modules()
    assert optional_missing and optional_missing[0].name == "xformers"
