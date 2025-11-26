"""
Configuration dataclasses for model, training, and curriculum per plan.md.

Defines all hyperparameters and settings for the training system.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


@dataclass
class ModelConfig:
    """
    Model architecture configuration.

    Per plan.md: 12-28 layer decoder-only transformer, 50-150M params
    """

    # Architecture
    d_model: int = 512  # Hidden dimension
    n_layers: int = 12  # Number of transformer layers
    n_heads: int = 8  # Number of attention heads
    ffn_mult: int = 4  # FFN dimension multiplier (ffn_dim = d_model * ffn_mult)
    vocab_size: int = 49180  # StarCoder2 base (49152) + special tokens (~28)
    max_seq_len: int = 4096  # Maximum sequence length

    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1

    # Position encoding
    use_rope: bool = True  # Use RoPE (Rotary Position Embedding)

    def __post_init__(self):
        """Validate configuration."""
        assert self.d_model > 0, "d_model must be positive"
        assert self.n_layers > 0, "n_layers must be positive"
        assert self.n_heads > 0 and self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )
        assert 0 <= self.dropout <= 1, "dropout must be in [0, 1]"

    @property
    def ffn_dim(self) -> int:
        """Compute feedforward dimension."""
        return self.d_model * self.ffn_mult

    @property
    def head_dim(self) -> int:
        """Compute dimension per attention head."""
        return self.d_model // self.n_heads

    def estimate_params(self) -> int:
        """
        Estimate total model parameters.

        Rough estimate: (12 * d_model^2 * ffn_mult + vocab * d_model) * n_layers

        Returns:
            Estimated parameter count
        """
        # Attention: 4 * d_model^2 (Q, K, V, O projections)
        attention_params = 4 * self.d_model ** 2

        # FFN: 2 * d_model * ffn_dim
        ffn_params = 2 * self.d_model * self.ffn_dim

        # LayerNorm: 2 * d_model (per layer, 2 norms)
        norm_params = 2 * 2 * self.d_model

        # Total per layer
        per_layer_params = attention_params + ffn_params + norm_params

        # Embedding: vocab * d_model
        embedding_params = self.vocab_size * self.d_model

        # LM head: d_model * vocab (often tied with embedding)
        lm_head_params = 0  # Assuming tied embeddings

        total_params = self.n_layers * per_layer_params + embedding_params + lm_head_params

        return total_params


@dataclass
class TrainingConfig:
    """
    Training hyperparameters per plan.md and research.md.
    """

    # Optimizer
    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.01
    eps: float = 1e-8

    # Learning rate schedule
    warmup_steps: int = 1000
    total_steps: int = 200000
    lr_schedule: str = "cosine"  # "cosine" or "linear"

    # Batch size and gradient accumulation
    tokens_per_batch: int = 524288  # Target tokens per batch
    grad_accum_steps: int = 8  # Auto-calculated if None
    max_seq_len: int = 2048  # Context window during training

    # Mixed precision
    use_amp: bool = True  # Automatic Mixed Precision
    amp_dtype: str = "bf16"  # "bf16" or "fp16"

    # Gradient clipping
    grad_clip: float = 1.0

    # Checkpointing
    checkpoint_interval: int = 2000  # Save every N steps
    checkpoint_dir: str = "checkpoints"

    # Logging
    log_interval: int = 100  # Log metrics every N steps
    eval_interval: int = 2000  # Evaluate every N steps

    # Reproducibility
    seed: int = 42

    def __post_init__(self):
        """Validate training configuration."""
        assert self.learning_rate > 0, "learning_rate must be positive"
        assert 0 <= self.weight_decay <= 1, "weight_decay must be in [0, 1]"
        assert self.warmup_steps >= 0, "warmup_steps must be non-negative"
        assert self.total_steps > 0, "total_steps must be positive"
        assert self.grad_clip > 0, "grad_clip must be positive"
        assert self.amp_dtype in ["bf16", "fp16"], "amp_dtype must be 'bf16' or 'fp16'"


@dataclass
class BandConfig:
    """Configuration for a single curriculum band."""

    band_id: str  # "A0", "A1", etc.
    description: str
    pack_size: int  # Number of Q/A pairs per sequence
    promotion_gates: Dict[str, float]  # {"em": 0.985, "format": 0.995}
    sample_counts: Dict[str, int] = field(
        default_factory=lambda: {"train": 10000, "val": 1000, "test": 1000}
    )

    # Special flags for FORMAT band
    teacher_forced: bool = False  # Use teacher forcing (only for FORMAT band)
    no_replay: bool = False  # Exclude from global replay and review
    expected_steps: Optional[int] = None  # Expected training steps

    def __post_init__(self):
        """Validate band configuration."""
        assert self.pack_size > 0, "pack_size must be positive"
        for metric, threshold in self.promotion_gates.items():
            assert 0 <= threshold <= 1, f"Gate {metric} threshold must be in [0, 1]"


@dataclass
class CurriculumConfig:
    """
    Curriculum progression configuration per plan.md.
    """

    # Band definitions
    bands: List[BandConfig] = field(default_factory=list)

    # Sampling ratios (current/review/preview)
    current_ratio: float = 0.7
    review_ratio: float = 0.2
    preview_ratio: float = 0.1

    # Global replay
    global_replay_ratio: float = 0.2  # 20% of batch from all prior bands

    # Promotion requirements
    consecutive_evals_required: int = 2  # Gates must be met for N consecutive evals
    regression_tolerance: float = 0.02  # Max allowed metric drop (2%)

    def __post_init__(self):
        """Validate curriculum configuration."""
        total_ratio = self.current_ratio + self.review_ratio + self.preview_ratio
        assert abs(total_ratio - 1.0) < 1e-6, (
            f"Sampling ratios must sum to 1.0, got {total_ratio}"
        )
        assert 0 <= self.global_replay_ratio <= 1, "global_replay_ratio must be in [0, 1]"
        assert self.consecutive_evals_required >= 1, "consecutive_evals_required must be >= 1"
        assert 0 <= self.regression_tolerance <= 1, "regression_tolerance must be in [0, 1]"

    def get_band(self, band_id: str) -> Optional[BandConfig]:
        """Get band configuration by ID."""
        for band in self.bands:
            if band.band_id == band_id:
                return band
        return None


def load_config_from_yaml(config_path: Path) -> Dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_config_to_yaml(config: Dict, config_path: Path):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        config_path: Path to save YAML file
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# Example usage
if __name__ == "__main__":
    # Create model config
    model_config = ModelConfig(
        d_model=512,
        n_layers=12,
        n_heads=8,
        ffn_mult=4,
        vocab_size=256,
    )

    print("=== Model Configuration ===")
    print(f"Architecture: {model_config.n_layers}L, {model_config.d_model}D, {model_config.n_heads}H")
    print(f"FFN dimension: {model_config.ffn_dim}")
    print(f"Head dimension: {model_config.head_dim}")
    print(f"Estimated params: {model_config.estimate_params() / 1e6:.1f}M")

    # Create training config
    train_config = TrainingConfig(
        learning_rate=3e-4,
        warmup_steps=1000,
        total_steps=200000,
        tokens_per_batch=524288,
    )

    print("\n=== Training Configuration ===")
    print(f"Learning rate: {train_config.learning_rate}")
    print(f"Warmup steps: {train_config.warmup_steps}")
    print(f"Total steps: {train_config.total_steps}")
    print(f"Tokens per batch: {train_config.tokens_per_batch}")

    # Create curriculum config
    curriculum_config = CurriculumConfig(
        bands=[
            BandConfig("A0", "Copy and compare", 128, {"em": 0.985}),
            BandConfig("A1", "1-digit addition", 128, {"em": 0.985}),
            BandConfig("A2", "Multi-digit addition", 64, {"em": 0.985, "carry": 0.98}),
        ],
        current_ratio=0.7,
        review_ratio=0.2,
        preview_ratio=0.1,
    )

    print("\n=== Curriculum Configuration ===")
    print(f"Number of bands: {len(curriculum_config.bands)}")
    print(f"Sampling ratios: {curriculum_config.current_ratio:.0%}/{curriculum_config.review_ratio:.0%}/{curriculum_config.preview_ratio:.0%}")
    print(f"Global replay: {curriculum_config.global_replay_ratio:.0%}")

    for band in curriculum_config.bands:
        print(f"\n  {band.band_id}: {band.description}")
        print(f"    Pack size: {band.pack_size}")
        print(f"    Gates: {band.promotion_gates}")

    print("\n✓ All configuration dataclasses defined")
