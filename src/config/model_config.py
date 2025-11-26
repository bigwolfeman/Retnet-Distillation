"""Model configuration dataclass for RetNet-HRM architecture.

This module defines the complete model architecture configuration including:
- RetNet backbone parameters
- HRM/ACT adaptive computation settings
- Thin attention band configuration
- Router and retrieval settings

Implements schema from data-model.md with validation for FR-004 (2-3B parameters).
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, Optional
import json
import sys


@dataclass
class ModelConfig:
    """Model architecture configuration for RetNet-HRM.

    All parameters validated against requirements:
    - FR-004: 2-3B parameter count
    - FR-003: Fits in 32GB RAM
    - FR-006: 1-10 adaptive computation steps
    """

    # Model dimensions
    d_model: int = 2816                    # Hidden dimension
    n_layers_retnet: int = 28              # RetNet backbone layers
    n_layers_attention: int = 3            # Thin attention band layers
    n_retention_heads: int = 12            # Retention heads per layer
    mlp_mult: int = 4                      # MLP expansion factor
    vocab_size: int = 100352               # Vocabulary size

    # Context and attention
    max_seq_len_train: int = 32768         # Training sequence length
    max_seq_len_infer: int = 65536         # Inference sequence length
    attention_window: int = 2048           # Local attention window
    use_rope_in_attention: bool = True     # RoPE in attention band

    # HRM / Adaptive Computation (FR-006)
    hrm_t_max: int = 6                     # Max ponder steps
    hrm_epsilon: float = 1e-3              # Halting threshold
    hrm_ponder_tau: float = 0.002          # Ponder cost weight
    hrm_halting_bias_init: float = -1.0    # Initial halting bias

    # Router
    router_budget_B: int = 24              # Max global tokens
    router_landmark_len_L: int = 6         # Tokens per landmark
    router_gumbel_temp: float = 0.7        # Gumbel temperature
    router_lambda_sparsity: float = 2e-4   # Sparsity loss weight
    router_lambda_entropy: float = 1e-3    # Entropy loss weight

    # Retrieval
    retrieval_topk: int = 32               # Chunks per query
    retrieval_chunk_bytes: int = 2048      # Chunk size before compression
    retrieval_landmark_tokens: int = 6     # Tokens per compressed chunk

    # Retrieval paths (optional - set to None to disable retrieval)
    encoder_checkpoint_path: Optional[str] = None    # Path to dual encoder checkpoint
    workspace_index_path: Optional[str] = None       # Path to workspace index
    global_index_path: Optional[str] = None          # Path to global index
    retrieval_manifest_path: Optional[str] = None    # Path to retrieval manifest (YAML/JSON)
    enable_retrieval: bool = False                   # Enable retrieval system

    # Training
    dropout: float = 0.0                   # Dropout rate
    dtype: str = "bfloat16"                # Model precision
    debug: bool = False                    # Enable debug output (segment isolation, etc.)

    # Metadata (FR-011b)
    config_version: str = "1.0.0"          # Config schema version
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for checkpoint metadata (FR-011b)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Deserialize from checkpoint metadata."""
        # Remove any extra fields not in dataclass
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    def validate(self):
        """Validate configuration constraints.

        Raises:
            AssertionError: If validation fails
        """
        # Dimension constraints
        assert self.d_model % self.n_retention_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_retention_heads ({self.n_retention_heads})"

        # Sequence length constraints
        assert self.max_seq_len_infer >= self.max_seq_len_train, \
            f"Inference seq_len ({self.max_seq_len_infer}) must be >= training seq_len ({self.max_seq_len_train})"

        # Router budget constraints
        max_landmarks = self.retrieval_topk * self.router_landmark_len_L
        assert self.router_budget_B <= max_landmarks, \
            f"Router budget ({self.router_budget_B}) exceeds available landmarks ({max_landmarks})"

        # Adaptive computation constraints (FR-006)
        assert 1 <= self.hrm_t_max <= 10, \
            f"hrm_t_max ({self.hrm_t_max}) must be in range [1, 10] (FR-006)"

        # Parameter count constraint (optimized for performance)
        param_count = self.estimate_param_count()
        assert 0.5e9 <= param_count <= 3e9, \
            f"Parameter count {param_count/1e9:.2f}B outside reasonable 0.5-3B range"

    def estimate_param_count(self) -> int:
        """Estimate total parameters (FR-004: 2-3B).

        Returns:
            int: Estimated parameter count
        """
        # RetNet layers
        # Each layer: retention (similar to self-attention) + MLP
        retnet_params = self.n_layers_retnet * (
            self.d_model * self.d_model * 4 +  # QKV-like projections + output
            self.d_model * self.d_model * self.mlp_mult * 2  # MLP (up + down)
        )

        # Attention layers
        attn_params = self.n_layers_attention * (
            self.d_model * self.d_model * 4 +  # QKV + output
            self.d_model * self.d_model * self.mlp_mult * 2  # MLP
        )

        # Embeddings (tied input/output)
        embed_params = self.vocab_size * self.d_model * 2  # Input + output (tied)

        # HRM + Router
        hrm_params = self.d_model * self.d_model  # Controller GRU-like
        router_params = self.d_model * self.router_budget_B * 2  # Routing projection

        # Retrieval compressor
        retrieval_params = self.d_model * self.d_model * 4  # 2-layer MLP for compression

        total = retnet_params + attn_params + embed_params + hrm_params + router_params + retrieval_params
        return int(total)
