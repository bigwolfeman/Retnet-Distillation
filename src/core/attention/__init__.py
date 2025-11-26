"""Attention forecasting services."""

from .service import (
    AttentionForecastResult,
    AttentionForecastService,
    RECOMMEND_KEEP_DENSE,
    RECOMMEND_REDUCE_CONTEXT,
    RECOMMEND_SWITCH_TO_SPARSE,
)

__all__ = [
    "AttentionForecastResult",
    "AttentionForecastService",
    "RECOMMEND_KEEP_DENSE",
    "RECOMMEND_REDUCE_CONTEXT",
    "RECOMMEND_SWITCH_TO_SPARSE",
]
