from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from src.core.attention import AttentionForecastService
from src.core.retrieval import RetrievalValidationService
from src.models.retrieval import RetrievalRegistry, RetrievalSourceStatus


def _write_manifest(tmp_path: Path, sources: list[dict[str, object]]) -> Path:
    manifest_path = tmp_path / "retrieval_manifest.json"
    manifest_path.write_text(json.dumps({"sources": sources}, indent=2), encoding="utf-8")
    return manifest_path


def test_retrieval_validation_service_and_cli(tmp_path: Path) -> None:
    ready_asset = tmp_path / "workspace.idx"
    ready_asset.write_text("ok", encoding="utf-8")

    stale_asset = tmp_path / "stale.idx"
    stale_asset.write_text("ok", encoding="utf-8")

    missing_asset = tmp_path / "missing.idx"

    now = datetime.now(timezone.utc)

    manifest = [
        {
            "sourceId": "11111111-2222-3333-4444-555555555555",
            "displayName": "Ready Source",
            "sourceType": "DOCUMENT_INDEX",
            "assetPaths": [str(ready_asset)],
            "validatedAt": now.isoformat(),
            "freshnessTtlHours": 24,
            "status": "READY",
            "landmarkCachePolicy": "REUSE",
        },
        {
            "sourceId": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "displayName": "Missing Source",
            "sourceType": "EMBEDDING_STORE",
            "assetPaths": [str(missing_asset)],
            "validatedAt": now.isoformat(),
            "freshnessTtlHours": 24,
            "status": "READY",
            "landmarkCachePolicy": "REFRESH",
        },
        {
            "sourceId": "99999999-8888-7777-6666-555555555555",
            "displayName": "Stale Source",
            "sourceType": "DOCUMENT_INDEX",
            "assetPaths": [str(stale_asset)],
            "validatedAt": (now - timedelta(hours=5)).isoformat(),
            "freshnessTtlHours": 1,
            "status": "READY",
            "landmarkCachePolicy": "REUSE",
        },
    ]

    manifest_path = _write_manifest(tmp_path, manifest)

    registry = RetrievalRegistry.load(manifest_path)
    service = RetrievalValidationService(registry)

    sources = {source.display_name: source for source in service.list_sources()}

    assert sources["Ready Source"].status is RetrievalSourceStatus.READY
    assert sources["Missing Source"].status is RetrievalSourceStatus.MISSING_ASSETS
    assert sources["Stale Source"].status is RetrievalSourceStatus.STALE

    ready_result = service.validate_source("11111111-2222-3333-4444-555555555555")
    assert ready_result.status is RetrievalSourceStatus.READY

    missing_result = service.validate_source("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
    assert missing_result.status is RetrievalSourceStatus.MISSING_ASSETS
    assert missing_asset.resolve().as_posix() in missing_result.missing_assets

    stale_result = service.validate_source(
        "99999999-8888-7777-6666-555555555555",
        force_refresh=True,
    )
    assert stale_result.status is RetrievalSourceStatus.READY

    # CLI: validate ready source succeeds
    proc_ready = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.retrieval_validate",
            "--manifest",
            str(manifest_path),
            "--source-id",
            "11111111-2222-3333-4444-555555555555",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_ready.returncode == 0

    # CLI: full manifest fails due to missing assets and stale entry
    proc_all = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.retrieval_validate",
            "--manifest",
            str(manifest_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_all.returncode == 1
    assert "missing asset" in proc_all.stdout.lower()

    # CLI: allow stale + force-refresh suppresses failure for stale source
    proc_force = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.retrieval_validate",
            "--manifest",
            str(manifest_path),
            "--source-id",
            "99999999-8888-7777-6666-555555555555",
            "--force-refresh",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_force.returncode == 0

    assert "Ready Source" in proc_ready.stdout or proc_ready.stderr

    # Attention forecast: dense exceeds ceiling, sparse recommendation expected
    attention_service = AttentionForecastService(
        num_heads=16,
        block_size=128,
        dtype_bytes=2,
        memory_ceiling_mb=512.0,
    )
    attention_result = attention_service.forecast(
        sequence_length=32000,
        window_size=2048,
        batch_size=1,
        current_mode="DENSE",
        global_tokens=4,
    )
    assert attention_result.recommendation in {"SWITCH_TO_SPARSE", "REDUCE_CONTEXT"}
    assert attention_result.dense_usage_mb > 512.0

    # CLI integration for attention forecast (JSON mode for stability)
    proc_attention = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.attention_forecast",
            "--sequence-length",
            "32000",
            "--window-size",
            "2048",
            "--batch-size",
            "1",
            "--num-heads",
            "16",
            "--memory-ceiling-mb",
            "512",
            "--mode",
            "DENSE",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_attention.returncode == 0
    payload = json.loads(proc_attention.stdout)
    assert payload["recommendation"] in {"SWITCH_TO_SPARSE", "REDUCE_CONTEXT"}
