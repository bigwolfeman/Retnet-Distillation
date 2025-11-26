#!/usr/bin/env python3
"""
CLI entrypoint for Titan-HRM runtime readiness checks.

Example usage:

    python -m tools.runtime_check --manifest configs/stabilization/runtime_manifest.example.yaml --output readiness.json --ack-optional
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from src.core.readiness.auditor import RuntimeAuditor
from src.core.stabilization.manifest_loader import MANIFEST_SCHEMA_PATH
from src.models.runtime_readiness.serialization import report_to_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Titan-HRM runtime readiness audit and emit a readiness report.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to the runtime manifest (YAML or JSON).",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=MANIFEST_SCHEMA_PATH,
        help="Optional path to a JSON schema for manifest validation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write report JSON to this file instead of stdout.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        help="Optional notes to include in the readiness report.",
    )
    parser.add_argument(
        "--ack-optional",
        action="store_true",
        help="Set this flag if an operator has acknowledged missing optional modules.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    auditor = RuntimeAuditor.from_path(
        args.manifest,
        schema_path=args.schema,
    )
    report = auditor.generate_report(
        operator_acknowledged=args.ack_optional,
        notes=args.notes,
    )
    payload = report_to_dict(report)
    output = json.dumps(payload, indent=2)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
        print(f"Readiness report written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
