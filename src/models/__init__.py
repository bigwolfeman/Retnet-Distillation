"""Model package with lazy imports for heavy dependencies."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "RetNetHRMModel",
    "ModelOutput",
    "InferenceOutput",
    "RetNetBackbone",
    "RetNetOutputHead",
    "ACT",
]


_DEFERS = {
    "RetNetHRMModel": ("src.models.core", "RetNetHRMModel"),
    "ModelOutput": ("src.models.core", "ModelOutput"),
    "InferenceOutput": ("src.models.core", "InferenceOutput"),
    "RetNetBackbone": ("src.models.retnet", "RetNetBackbone"),
    "RetNetOutputHead": ("src.models.retnet", "RetNetOutputHead"),
    "ACT": ("src.models.hrm", "ACT"),
}


def __getattr__(name: str) -> Any:  # pragma: no cover - thin import shim
    try:
        module_name, attr = _DEFERS[name]
    except KeyError as exc:  # pragma: no cover
        raise AttributeError(f"module 'src.models' has no attribute '{name}'") from exc
    module = importlib.import_module(module_name)
    value = getattr(module, attr)
    globals()[name] = value
    return value

