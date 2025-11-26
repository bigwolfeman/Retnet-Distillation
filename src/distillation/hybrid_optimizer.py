"""
Hybrid optimizer wrapper for combining multiple optimizers.

Allows using different optimizers for different parameter groups,
e.g., Muon for 2D+ parameters and AdamW for <2D parameters.
"""

from typing import List, Dict, Any
import torch
from torch.optim import Optimizer


class HybridOptimizer(Optimizer):
    """Wraps multiple optimizers to act as a single optimizer.

    Useful for combining optimizers like Muon (for 2D+ parameters)
    and AdamW (for <2D parameters).

    Args:
        optimizers: List of optimizer instances to wrap

    Example:
        >>> muon_opt = Muon(model_2d_params, lr=1e-3)
        >>> adamw_opt = AdamW(model_1d_params, lr=1e-3)
        >>> optimizer = HybridOptimizer([muon_opt, adamw_opt])
        >>> optimizer.step()
        >>> optimizer.zero_grad()
    """

    def __init__(self, optimizers: List[Optimizer]):
        """Initialize hybrid optimizer.

        Args:
            optimizers: List of optimizer instances
        """
        # Store child optimizers
        self.optimizers = optimizers

        # Initialize defaults (required by scheduler)
        self.defaults = {}

        # Don't call super().__init__() since we manage everything through child optimizers
        # The @property param_groups will provide access to all param_groups

    def step(self, closure=None):
        """Perform a single optimization step for all optimizers.

        Args:
            closure: Optional closure to reevaluate the model
        """
        loss = None
        for optimizer in self.optimizers:
            loss = optimizer.step(closure)
        return loss

    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients for all optimizers.

        Args:
            set_to_none: Set gradients to None instead of zero
        """
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict from all optimizers.

        Returns:
            Dictionary containing state from all optimizers
        """
        return {
            f'optimizer_{i}': opt.state_dict()
            for i, opt in enumerate(self.optimizers)
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict into all optimizers.

        Args:
            state_dict: Dictionary containing state for all optimizers
        """
        for i, opt in enumerate(self.optimizers):
            key = f'optimizer_{i}'
            if key in state_dict:
                opt.load_state_dict(state_dict[key])

    @property
    def param_groups(self):
        """Get all parameter groups from all optimizers.

        This property ensures the scheduler always sees the current
        parameter groups from all child optimizers.
        """
        groups = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups

