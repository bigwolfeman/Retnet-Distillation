"""Attention memory forecasting utilities (US4)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

from src.models.attention import estimate_sliding_window_memory


RECOMMEND_KEEP_DENSE = "KEEP_DENSE"
RECOMMEND_SWITCH_TO_SPARSE = "SWITCH_TO_SPARSE"
RECOMMEND_REDUCE_CONTEXT = "REDUCE_CONTEXT"


@dataclass(frozen=True)
class AttentionForecastResult:
    projected_usage_mb: float
    recommendation: str
    dense_usage_mb: float
    sparse_usage_mb: float
    connections_per_token: int
    blocks_per_token: int
    triggers_sparse_at_sequence_length: int | None
    notes: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "projectedUsageMb": round(self.projected_usage_mb, 3),
            "recommendation": self.recommendation,
            "denseUsageMb": round(self.dense_usage_mb, 3),
            "sparseUsageMb": round(self.sparse_usage_mb, 3),
            "connectionsPerToken": self.connections_per_token,
            "blocksPerToken": self.blocks_per_token,
        }
        if self.triggers_sparse_at_sequence_length is not None:
            payload["triggersSparseAtSequenceLength"] = self.triggers_sparse_at_sequence_length
        if self.notes:
            payload["notes"] = self.notes
        return payload


class AttentionForecastService:
    """Provide memory usage projections for attention configurations."""

    def __init__(
        self,
        *,
        num_heads: int = 16,
        block_size: int = 128,
        dtype_bytes: int = 2,
        memory_ceiling_mb: float = 1024.0,
    ) -> None:
        self._num_heads = num_heads
        self._block_size = block_size
        self._dtype_bytes = dtype_bytes
        self._memory_ceiling_mb = memory_ceiling_mb

    def forecast(
        self,
        sequence_length: int,
        *,
        window_size: int,
        batch_size: int = 1,
        current_mode: str = "DENSE",
        global_tokens: int = 0,
        step_bytes_estimate: int | None = None,
    ) -> AttentionForecastResult:
        dtype_bytes = step_bytes_estimate or self._dtype_bytes
        metrics = estimate_sliding_window_memory(
            sequence_length,
            window_size=window_size,
            num_heads=self._num_heads,
            batch_size=batch_size,
            dtype_bytes=dtype_bytes,
            block_size=self._block_size,
            global_tokens=global_tokens,
        )

        dense_mb = metrics["dense_mb"]
        sparse_mb = metrics["sparse_mb"]
        connections_per_token = metrics["connections_per_token"]
        blocks_per_token = metrics["blocks_per_token"]

        active_mode = current_mode.upper() if current_mode else "DENSE"
        active_usage = sparse_mb if active_mode == "SPARSE" else dense_mb

        triggers_sparse = self._compute_dense_threshold(
            batch_size=batch_size,
            dtype_bytes=dtype_bytes,
        )

        if dense_mb <= self._memory_ceiling_mb:
            recommendation = RECOMMEND_KEEP_DENSE
            notes = "Dense attention fits under the configured memory ceiling."
        elif sparse_mb <= self._memory_ceiling_mb:
            recommendation = RECOMMEND_SWITCH_TO_SPARSE
            active_usage = sparse_mb
            notes = "Switch to sparse sliding window to stay within memory budget."
        else:
            recommendation = RECOMMEND_REDUCE_CONTEXT
            notes = "Even sparse attention exceeds the memory ceiling; reduce sequence length."

        if recommendation == RECOMMEND_KEEP_DENSE:
            projected_usage = dense_mb if active_mode == "DENSE" else sparse_mb
        elif recommendation == RECOMMEND_SWITCH_TO_SPARSE:
            projected_usage = sparse_mb
        else:
            projected_usage = sparse_mb

        return AttentionForecastResult(
            projected_usage_mb=projected_usage,
            recommendation=recommendation,
            dense_usage_mb=dense_mb,
            sparse_usage_mb=sparse_mb,
            connections_per_token=connections_per_token,
            blocks_per_token=blocks_per_token,
            triggers_sparse_at_sequence_length=triggers_sparse,
            notes=notes,
        )

    def _compute_dense_threshold(
        self,
        *,
        batch_size: int,
        dtype_bytes: int,
    ) -> int | None:
        bytes_per_token_pair = self._num_heads * batch_size * dtype_bytes
        if bytes_per_token_pair <= 0:
            return None
        ceiling_bytes = self._memory_ceiling_mb * (1024 ** 2)
        seq_limit = math.floor(math.sqrt(ceiling_bytes / bytes_per_token_pair))
        return max(seq_limit, 0)
