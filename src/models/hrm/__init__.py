"""HRM (Hierarchical Recurrent Memory) and ACT (Adaptive Computation Time) components."""

from .controller import HRMController, HRMSummarizer
from .halting import ACTHaltingHead, HaltingOutput, ACTManager

# Legacy import (if act.py exists)
try:
    from .act import ACT
    __all__ = ["HRMController", "HRMSummarizer", "ACTHaltingHead", "HaltingOutput", "ACTManager", "ACT"]
except ImportError:
    __all__ = ["HRMController", "HRMSummarizer", "ACTHaltingHead", "HaltingOutput", "ACTManager"]
