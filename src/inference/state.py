"""Context state management for recurrent inference.

Implements ContextState from data-model.md.
Supports O(1) memory per layer (FR-005) and 64k+ sequences (FR-002).
"""

import torch
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ContextState:
    """Running state during recurrent inference.

    Implements data-model.md ContextState schema.
    Enables incremental processing with O(1) memory per layer (FR-005).

    Attributes:
        retnet_states: List of per-layer retention states (O(1) each)
        attention_kv_cache: KV cache for attention band (future US2)
        hrm_hidden: HRM controller state (future US3)
        hrm_ponder_accumulator: ACT halting accumulator (future US3)
        hrm_step_count: Current ACT step (future US3)
        current_landmarks: Active landmark tokens (future US4)
        retrieval_cache: Query cache for retrieval (future US4)
        current_position: Token position in stream
        segment_id: Segment ID for chunk-recurrent mode
        peak_memory_mb: Peak memory usage tracking
    """

    # RetNet recurrent state (MVP US1)
    retnet_states: List[torch.Tensor] = field(default_factory=list)

    # Attention KV cache (US2 - future)
    attention_kv_cache: Dict[int, tuple] = field(default_factory=dict)

    # HRM controller state (US3 - future)
    hrm_hidden: Optional[torch.Tensor] = None
    hrm_ponder_accumulator: Optional[torch.Tensor] = None
    hrm_step_count: int = 0

    # Retrieval state (US4 - future)
    current_landmarks: Optional[torch.Tensor] = None
    retrieval_cache: Dict[str, Any] = field(default_factory=dict)

    # Position tracking
    current_position: int = 0
    segment_id: int = 0

    # Memory tracking (FR-003)
    peak_memory_mb: float = 0.0

    def reset(self):
        """Reset state for new sequence.

        Clears all caches and resets counters.
        """
        self.retnet_states = []
        self.attention_kv_cache = {}
        self.hrm_hidden = None
        self.hrm_ponder_accumulator = None
        self.hrm_step_count = 0
        self.current_landmarks = None
        self.retrieval_cache = {}
        self.current_position = 0
        self.segment_id = 0
        self.peak_memory_mb = 0.0

    def update_position(self, new_tokens: int):
        """Advance position counter.

        Args:
            new_tokens: Number of new tokens processed
        """
        self.current_position += new_tokens

    def check_memory_limit(self, limit_gb: float = 32.0):
        """Validate memory constraint (FR-003).

        Args:
            limit_gb: Memory limit in GB (default 32.0)

        Raises:
            RuntimeError: If memory usage exceeds limit
        """
        if torch.cuda.is_available():
            current_mb = torch.cuda.memory_allocated() / 1024 / 1024
            self.peak_memory_mb = max(self.peak_memory_mb, current_mb)

            current_gb = current_mb / 1024
            if current_gb > limit_gb:
                raise RuntimeError(
                    f"Memory limit exceeded: {current_gb:.2f}GB > {limit_gb}GB (FR-003)\n"
                    f"Try reducing sequence length or using smaller chunks"
                )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state metadata (for debugging or caching).

        Returns:
            Dict with state metadata (not full tensors)
        """
        return {
            'current_position': self.current_position,
            'segment_id': self.segment_id,
            'hrm_step_count': self.hrm_step_count,
            'peak_memory_mb': self.peak_memory_mb,
            'num_retnet_states': len(self.retnet_states),
            'has_hrm_hidden': self.hrm_hidden is not None,
            'has_landmarks': self.current_landmarks is not None,
        }

    @classmethod
    def initialize(
        cls,
        batch_size: int,
        device: torch.device,
        n_layers: int = 28,
        d_model: int = 2816,
        n_heads: int = 12,
    ) -> "ContextState":
        """Initialize fresh context state.

        Args:
            batch_size: Batch size
            device: Device for state tensors
            n_layers: Number of RetNet layers
            d_model: Model hidden dimension
            n_heads: Number of retention heads

        Returns:
            Initialized ContextState
        """
        # Initialize RetNet states (zero states)
        retnet_states = []
        d_head = d_model // n_heads

        for _ in range(n_layers):
            # Per-layer retention state
            # Shape: (batch_size, n_heads, d_head)
            layer_state = torch.zeros(
                batch_size, n_heads, d_head,
                device=device,
                dtype=torch.float32,  # RetNet uses fp32 for states
            )
            retnet_states.append(layer_state)

        return cls(retnet_states=retnet_states)

    def get_retnet_state(self) -> List[torch.Tensor]:
        """Get RetNet recurrent state.

        Returns:
            List of per-layer retention states
        """
        return self.retnet_states

    def update_retnet_state(self, new_states: List[torch.Tensor]):
        """Update RetNet recurrent state.

        Args:
            new_states: New per-layer retention states
        """
        self.retnet_states = new_states

    def validate_memory(self):
        """Check current memory usage and update peak.

        Raises:
            RuntimeError: If memory exceeds 32GB limit
        """
        self.check_memory_limit(limit_gb=32.0)


class InferenceStateManager:
    """Manager for inference state across multiple sequences.

    Handles state initialization, caching, and memory monitoring.
    """

    def __init__(
        self,
        batch_size: int = 1,
        device: str = "cuda",
        n_layers: int = 28,
        d_model: int = 2816,
        n_heads: int = 12,
    ):
        """Initialize state manager.

        Args:
            batch_size: Batch size for inference
            device: Device for state tensors
            n_layers: Number of RetNet layers
            d_model: Model hidden dimension
            n_heads: Number of retention heads
        """
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads

        # Current active state
        self.state: Optional[ContextState] = None

    def create_new_state(self) -> ContextState:
        """Create and initialize new context state.

        Returns:
            Initialized ContextState
        """
        self.state = ContextState.initialize(
            batch_size=self.batch_size,
            device=self.device,
            n_layers=self.n_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
        )
        return self.state

    def reset(self):
        """Reset current state."""
        if self.state is not None:
            self.state.reset()
        else:
            self.create_new_state()

    def get_state(self) -> ContextState:
        """Get current state (create if needed).

        Returns:
            Current ContextState
        """
        if self.state is None:
            self.create_new_state()
        return self.state

    def validate_state_memory(self):
        """Validate current state memory usage.

        Raises:
            RuntimeError: If memory exceeds limit
        """
        if self.state is not None:
            self.state.validate_memory()
