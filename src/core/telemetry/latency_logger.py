"""
Latency logging utilities for proposal generation telemetry.

Captures per-event latency measurements and exposes percentile summaries for
reporting in stabilization dashboards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List, Mapping, Optional, Sequence


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = pct * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


@dataclass
class LatencyRecord:
    event: str
    latency_ms: float
    metadata: Mapping[str, str] = field(default_factory=dict)


class LatencyLogger:
    """Capture and summarise latency measurements."""

    def __init__(self) -> None:
        self._records: List[LatencyRecord] = []

    def record(self, event: str, latency_ms: float, metadata: Optional[Mapping[str, str]] = None) -> None:
        self._records.append(
            LatencyRecord(
                event=event,
                latency_ms=float(latency_ms),
                metadata=dict(metadata or {}),
            )
        )

    def reset(self) -> None:
        self._records.clear()

    def events(self) -> Sequence[LatencyRecord]:
        return list(self._records)

    def summary(self) -> Dict[str, float]:
        latencies = [record.latency_ms for record in self._records]
        if not latencies:
            return {"count": 0, "avg_ms": 0.0, "p50_ms": 0.0, "p90_ms": 0.0, "p99_ms": 0.0}
        return {
            "count": len(latencies),
            "avg_ms": mean(latencies),
            "p50_ms": _percentile(latencies, 0.5),
            "p90_ms": _percentile(latencies, 0.9),
            "p99_ms": _percentile(latencies, 0.99),
        }

