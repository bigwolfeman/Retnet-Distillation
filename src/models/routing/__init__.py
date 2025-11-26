"""Routing components for RetNet-HRM model.

This module provides budgeted routing for landmark token selection.
"""

from .router import GumbelTopKRouter

__all__ = ["GumbelTopKRouter"]
