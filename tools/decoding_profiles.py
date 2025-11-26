#!/usr/bin/env python3
"""
CLI for managing Titan-HRM decoding profiles.

Supports listing, inspecting, and enabling/disabling profiles stored in
`configs/decoding_profiles.yaml`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.models.titans.decoding_profiles import (
    DecodingProfile,
    DecodingProfileCatalog,
    enable_profile,
    load_catalog,
    dump_catalog,
)


DEFAULT_CONFIG_PATH = Path("configs/decoding_profiles.yaml")


def _load_catalog(path: Path) -> DecodingProfileCatalog:
    return load_catalog(path)


def _print_profile(profile: DecodingProfile, *, as_json: bool = False) -> None:
    payload = profile.to_dict()
    if as_json:
        print(json.dumps(payload, indent=2))
        return

    lines = [
        f"profile_id: {payload['profile_id']}",
        f"name: {payload['name']}",
        f"mode: {payload['mode']}",
        f"enabled: {payload['enabled']}",
        f"latency_budget_ms: {payload['latency_budget_ms']}",
        f"quality_notes: {payload.get('quality_notes') or '—'}",
        "parameters:",
    ]
    for key, value in payload["parameters"].items():
        lines.append(f"  {key}: {value}")
    print("\n".join(lines))


def cmd_list(args: argparse.Namespace) -> None:
    catalog = _load_catalog(args.config)
    for profile in catalog:
        status = "ENABLED " if profile.enabled else "disabled"
        print(f"{profile.profile_id}  {profile.name:24}  {profile.mode.value:12}  {status}")


def cmd_show(args: argparse.Namespace) -> None:
    catalog = _load_catalog(args.config)
    profile = catalog.get(args.profile_id)
    _print_profile(profile, as_json=args.json)


def cmd_enable_disable(args: argparse.Namespace, enabled: bool) -> None:
    catalog = _load_catalog(args.config)
    updated = enable_profile(catalog, args.profile_id, enabled=enabled)
    dump_catalog(updated, args.config)
    action = "Enabled" if enabled else "Disabled"
    print(f"{action} profile {args.profile_id}")


def cmd_validate(args: argparse.Namespace) -> None:
    _ = _load_catalog(args.config)
    print(f"Decoding profile configuration {args.config} is valid.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage Titan-HRM decoding profiles.",
    )
    parser.set_defaults(func=None)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to decoding profiles configuration (default: configs/decoding_profiles.yaml).",
    )

    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser("list", help="List available decoding profiles.")
    list_parser.set_defaults(func=cmd_list)

    show_parser = subparsers.add_parser("show", help="Show details for a profile.")
    show_parser.add_argument("profile_id", help="Profile UUID to inspect.")
    show_parser.add_argument("--json", action="store_true", help="Print profile as JSON.")
    show_parser.set_defaults(func=cmd_show)

    enable_parser = subparsers.add_parser("enable", help="Enable a profile by UUID.")
    enable_parser.add_argument("profile_id", help="Profile UUID to enable.")
    enable_parser.set_defaults(func=lambda args: cmd_enable_disable(args, True))

    disable_parser = subparsers.add_parser("disable", help="Disable a profile by UUID.")
    disable_parser.add_argument("profile_id", help="Profile UUID to disable.")
    disable_parser.set_defaults(func=lambda args: cmd_enable_disable(args, False))

    validate_parser = subparsers.add_parser("validate", help="Validate the profile configuration.")
    validate_parser.set_defaults(func=cmd_validate)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.func is None:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()

