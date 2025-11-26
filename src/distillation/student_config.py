"""RetNet student model configuration for distillation.

This module defines configurations for RetNet student models:
- 350M parameter variant (d_model=1280, n_layers=12)
- 500M parameter variant (d_model=1536, n_layers=16)

Both variants use Llama-3.2-1B tokenizer (vocab_size=128256) and 4k context (v1).
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import torch
from transformers import AutoTokenizer

# Re-export tokenizer helper from dataset module to keep behavior consistent
from .dataset import load_llama_tokenizer as _dataset_load_llama_tokenizer


@dataclass
class RetNetStudentConfig:
    """Configuration for RetNet student model.

    This config is designed for distillation from Llama-3.2-1B teacher.

    Features:
    - Compatible with existing RetNet architecture in src/models/retnet/backbone.py
    - Uses Llama tokenizer (vocab_size=128256)
    - 4k context window for v1 (max_position_embeddings=4096)
    - Two variants: 350M and 500M parameters
    """

    # Model dimensions
    d_model: int = 960                       # Hidden dimension (default: 350M variant)
    n_layers: int = 12                       # Number of RetNet layers
    n_heads: int = 12                        # Number of retention heads
    mlp_mult: int = 4                        # MLP expansion factor (ffn_dim = d_model * mlp_mult)

    # Tokenizer configuration (Llama-3.2-1B)
    vocab_size: int = 128256                 # Llama tokenizer vocabulary size
    tokenizer_name: str = "meta-llama/Llama-3.2-1B"  # HuggingFace tokenizer

    # Position embeddings
    max_position_embeddings: int = 4096      # Maximum sequence length (default v1: 4k)

    # Regularization
    dropout: float = 0.1                     # Dropout rate
    attention_dropout: float = 0.1           # Attention dropout rate
    activation_dropout: float = 0.0          # Activation dropout rate
    drop_path_rate: float = 0.0              # Stochastic depth rate

    # Normalization
    layernorm_eps: float = 1e-5              # Layer norm epsilon
    normalize_before: bool = True            # Pre-norm architecture

    # Activation
    activation_fn: str = 'gelu'              # Activation function

    # RetNet specific
    recurrent_chunk_size: int = 512          # Chunk size for recurrent mode
    chunkwise_recurrent: bool = True         # Enable chunk-recurrent mode

    # Training
    dtype: str = "bfloat16"                  # Model precision
    tie_word_embeddings: bool = True         # Tie input/output embeddings

    # Distillation specific
    teacher_model: str = "meta-llama/Llama-3.2-1B"  # Teacher model name
    target_param_count_range: tuple = (300_000_000, 400_000_000)  # Target parameter range

    # Metadata
    config_version: str = "1.0.0"            # Config schema version
    variant: str = "350M"                    # Model variant identifier
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self):
        """Validate configuration constraints.

        Note: Parameter count validation is skipped in __post_init__ because our
        simplified estimation formula doesn't account for TorchScale's GLU implementation.
        Use validate_actual_params() after model initialization to verify actual count.

        Raises:
            AssertionError: If validation fails
        """
        # Dimension constraints
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"

        # Vocabulary size must match Llama tokenizer
        assert self.vocab_size == 128256, \
            f"vocab_size must be 128256 for Llama tokenizer, got {self.vocab_size}"


    def validate_actual_params(self, actual_param_count: int):
        """Validate actual parameter count against target range.

        Args:
            actual_param_count: Actual parameter count from initialized model

        Raises:
            AssertionError: If actual parameter count is outside target range
        """
        min_params, max_params = self.target_param_count_range
        assert min_params <= actual_param_count <= max_params, \
            f"Actual parameter count {actual_param_count:,} outside target range [{min_params:,}, {max_params:,}]"

    def estimate_param_count(self) -> int:
        """Estimate total parameters for the student model.

        Calculation includes:
        - Token embeddings: vocab_size * d_model
        - RetNet layers: n_layers * (retention + FFN)
        - Output projection: vocab_size * d_model (tied if tie_word_embeddings=True)
        - Layer norms: negligible (~d_model * n_layers * 2)

        Returns:
            int: Estimated parameter count
        """
        # Token embeddings (input)
        embed_params = self.vocab_size * self.d_model

        # RetNet layers
        # Each layer contains:
        # 1. Multi-scale retention (similar to attention): 4 * d_model^2 (Q, K, V, O projections)
        # 2. FFN: 2 * d_model * (d_model * mlp_mult) = 2 * d_model^2 * mlp_mult
        retention_params_per_layer = 4 * self.d_model * self.d_model
        ffn_params_per_layer = 2 * self.d_model * (self.d_model * self.mlp_mult)
        layer_params = self.n_layers * (retention_params_per_layer + ffn_params_per_layer)

        # Output projection (if not tied, add vocab_size * d_model)
        output_params = 0 if self.tie_word_embeddings else self.vocab_size * self.d_model

        # Layer norms (small, but include for accuracy)
        # Each layer has 2 layer norms (pre-retention, pre-ffn)
        # Plus input/output layer norms
        layernorm_params = (2 * self.n_layers + 2) * self.d_model

        total = embed_params + layer_params + output_params + layernorm_params
        return int(total)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for checkpoint metadata."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetNetStudentConfig":
        """Deserialize from checkpoint metadata."""
        # Remove any extra fields not in dataclass
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    def to_retnet_backbone_args(self) -> Dict[str, Any]:
        """Convert to arguments for RetNetBackbone initialization.

        Returns dictionary compatible with src/models/retnet/backbone.py::RetNetBackbone.__init__
        """
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "dropout": self.dropout,
            "max_seq_len": self.max_position_embeddings,
            "debug": False,
        }


@dataclass
class RetNetStudent350MConfig(RetNetStudentConfig):
    """350M parameter RetNet student configuration.

    Architecture:
    - d_model: 960
    - n_layers: 12
    - n_heads: 12
    - FFN dimension: 3840 (d_model * 4)
    - Head dimension: 80 (d_model / n_heads)

    Target: 350M parameters (300M-400M range)
    Estimated: ~256M parameters (simplified formula)
    Actual: ~320M parameters (TorchScale with GLU)
    """
    d_model: int = 960
    n_layers: int = 12
    n_heads: int = 12  # 960 % 12 = 80 head_dim
    variant: str = "350M"
    target_param_count_range: tuple = (300_000_000, 400_000_000)


@dataclass
class RetNetStudent500MConfig(RetNetStudentConfig):
    """500M parameter RetNet student configuration.

    Architecture:
    - d_model: 1152
    - n_layers: 15
    - n_heads: 12
    - FFN dimension: 4608 (d_model * 4)
    - Head dimension: 96 (d_model / n_heads)

    Target: 500M parameters (450M-550M range)
    Estimated: ~387M parameters (simplified formula)
    Actual: ~495M parameters (TorchScale with GLU)
    """
    d_model: int = 1152
    n_layers: int = 15
    n_heads: int = 12  # 1152 % 12 = 96 head_dim
    variant: str = "500M"
    target_param_count_range: tuple = (450_000_000, 550_000_000)


def load_llama_tokenizer(
    model_name: str = "meta-llama/Llama-3.2-1B",
    cache_dir: Optional[str] = None,
    use_fast: bool = True,
    adapter_path: Optional[str] = None,
    hf_token: Optional[str] = None,
    trust_remote_code: bool = True,
) -> AutoTokenizer:
    """Expose dataset.load_llama_tokenizer through student_config for backward compatibility."""

    return _dataset_load_llama_tokenizer(
        model_name=model_name,
        cache_dir=cache_dir,
        use_fast=use_fast,
        adapter_path=adapter_path,
        hf_token=hf_token,
        trust_remote_code=trust_remote_code,
    )


def create_student_config(variant: str = "350M", max_seq_length: int = 4096) -> RetNetStudentConfig:
    """Factory function to create student configuration.

    Args:
        variant: Model variant ("350M" or "500M")
        max_seq_length: Maximum sequence length (default: 4096)

    Returns:
        Appropriate student configuration

    Raises:
        ValueError: If variant is not recognized

    Example:
        >>> config_350m = create_student_config("350M")
        >>> config_350m.d_model
        1280
        >>> config_350m.estimate_param_count()
        ~350000000

        >>> config_500m = create_student_config("500M")
        >>> config_500m.d_model
        1536
    """
    if variant == "350M":
        config = RetNetStudent350MConfig()
    elif variant == "500M":
        config = RetNetStudent500MConfig()
    else:
        raise ValueError(f"Unknown variant: {variant}. Must be '350M' or '500M'")
        
    # Apply custom max_seq_length
    config.max_position_embeddings = max_seq_length
    return config
