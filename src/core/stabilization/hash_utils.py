"""
Hashing helpers shared across stabilization workflows.

Provides convenience functions for computing SHA-256 digests on files,
directories, and in-memory payloads with consistent encoding.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

BUFFER_SIZE = 1024 * 1024  # 1 MiB


def _sha256() -> hashlib._Hash:
    return hashlib.sha256()


def hash_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest for the provided bytes."""
    digest = _sha256()
    digest.update(data)
    return digest.hexdigest()


def hash_text(text: str, encoding: str = "utf-8") -> str:
    """Return the SHA-256 hex digest for the provided string."""
    return hash_bytes(text.encode(encoding))


def hash_file(path: Path | str) -> str:
    """Return the SHA-256 hex digest for the file at `path`."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot hash missing file: {file_path}")

    digest = _sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(BUFFER_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def hash_paths(paths: Iterable[Path | str]) -> str:
    """
    Compute a stable SHA-256 digest for a collection of paths.

    File contents are hashed individually and combined in sorted order by path
    name to produce a deterministic aggregate hash.
    """
    digest = _sha256()
    for path in sorted(Path(p) for p in paths):
        digest.update(str(path).encode("utf-8"))
        digest.update(hash_file(path).encode("utf-8"))
    return digest.hexdigest()
