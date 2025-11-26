#!/usr/bin/env python3
"""CLI for validating retrieval manifests and sources (US4)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from src.core.retrieval import RetrievalValidationService
from src.models.retrieval import RetrievalRegistry, RetrievalSource, RetrievalSourceStatus


DEFAULT_MANIFEST_PATH = Path("configs/retrieval_manifest.yaml")


def _format_source(source: RetrievalSource) -> str:
    status = source.status.value
    name = source.display_name
    return f"{source.source_id}  {name}  {status}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Titan-HRM retrieval assets against the stabilization manifest.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=f"Path to retrieval manifest (default: {DEFAULT_MANIFEST_PATH})",
    )
    parser.add_argument(
        "--source-id",
        help="Validate a single retrieval source by UUID.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit validation summary as JSON.",
    )
    parser.add_argument(
        "--allow-stale",
        action="store_true",
        help="Treat stale sources as warnings instead of failures.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore freshness TTL when validating a single source.",
    )
    return parser


def _filter_sources(
    sources: Sequence[RetrievalSource],
    *,
    source_id: str | None,
) -> list[RetrievalSource]:
    if source_id is None:
        return list(sources)
    for source in sources:
        if source.source_id == source_id:
            return [source]
    raise SystemExit(f"Source '{source_id}' not found in manifest")


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        registry = RetrievalRegistry.load(args.manifest)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    service = RetrievalValidationService(registry)
    sources = service.list_sources()
    targets = _filter_sources(sources, source_id=args.source_id)

    failures: list[RetrievalSource] = []
    summaries: list[dict[str, object]] = []

    for source in targets:
        result = service.validate_source(
            source.source_id,
            force_refresh=args.force_refresh,
        )
        evaluated = result.source
        summaries.append({
            "sourceId": evaluated.source_id,
            "displayName": evaluated.display_name,
            "status": result.status.value,
            "missingAssets": list(result.missing_assets),
            "remediationActions": list(result.remediation_actions),
        })

        is_failure = result.status is RetrievalSourceStatus.MISSING_ASSETS
        if result.status is RetrievalSourceStatus.STALE and not args.allow_stale:
            is_failure = True

        if is_failure:
            failures.append(evaluated)

    if args.json:
        print(json.dumps({"sources": summaries}, indent=2))
    else:
        for summary in summaries:
            status = summary["status"]
            print(f"{summary['sourceId']}  {summary['displayName']}  {status}")
            if summary["missingAssets"]:
                for asset in summary["missingAssets"]:
                    print(f"    missing asset: {asset}")
            if summary["remediationActions"]:
                for action in summary["remediationActions"]:
                    print(f"    remediation: {action}")

    if failures:
        names = ", ".join(f.source_id for f in failures)
        print(
            f"Validation failed for: {names}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
