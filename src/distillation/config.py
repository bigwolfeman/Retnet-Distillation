"""
Configuration loader and management for distillation training.

Implements:
- YAML config file loading
- CLI argument override merging
- Configuration validation
- Default value provision
- Type conversion and coercion

Task: T075
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
import argparse


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_project_path(path_str: Optional[str]) -> Optional[Path]:
    """Resolve project-relative paths to absolute paths.

    Args:
        path_str: Path string (may be relative).
    Returns:
        Absolute Path or None.
    """
    if path_str is None:
        return None

    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    else:
        path = path.resolve()
    return path


def _is_subpath(candidate: Path, reference: Path) -> bool:
    """Return True if candidate is inside reference."""
    try:
        candidate.relative_to(reference)
        return True
    except ValueError:
        return False


@dataclass
class SaddleEscapeConfig:
    """Configuration for saddle point detection and escape."""

    # Master switch
    enabled: bool = True

    # Detection parameters - CONSERVATIVE defaults
    grad_norm_threshold: float = 0.55
    loss_improvement_threshold: float = 0.0005  # 0.05% - very low threshold
    min_loss_threshold: float = 100.0
    patience_steps: int = 150  # Long patience

    # Intervention parameters - GENTLE defaults
    interventions_enabled: bool = False  # Start disabled - detection only!
    cooldown_steps: int = 250  # Long cooldown between nudges

    # Logging
    log_to_wandb: bool = True
    create_wandb_alerts: bool = False  # Don't spam alerts


@dataclass
class TrainingConfig:
    """Complete training configuration with all hyperparameters.

    This combines all configuration from YAML + CLI arguments.
    """
    # Model configuration
    model_variant: str = "350M"  # 350M or 500M
    model_config_path: Optional[str] = None  # Path to model config YAML

    # Adaptive Computation Time (ACT) settings
    use_act: bool = False  # Enable ACT wrapper for adaptive computation
    act_max_steps: int = 10  # Maximum pondering steps
    act_epsilon: float = 0.01  # Halting threshold (1 - epsilon)
    act_ponder_penalty: float = 0.01  # Weight for pondering cost loss
    act_use_geometric_prior: bool = False  # Use geometric prior regularization (PonderNet style)
    act_prior_lambda: float = 0.5  # Lambda for geometric prior

    # Training hyperparameters
    max_steps: int = 60000
    physical_batch_size: int = 1
    gradient_accumulation_steps: int = 256
    max_seq_length: int = 4096

    # Optimizer settings
    optimizer_type: str = "adamw"  # "adamw" or "muon"
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    use_8bit_optimizer: bool = True

    # Muon-specific settings
    muon_momentum: float = 0.95  # Muon momentum (beta)
    muon_grad_clip: float = 1.0  # Muon gradient clipping
    muon_zero_clip_percent: float = 0.0  # Muon zero clip percent
    muon_aux_lr_scale: float = 0.25  # LR multiplier for auxiliary AdamW branch
    muon_ready_check: bool = True  # Validate environment/smoke-test before enabling Muon
    muon_clip_threshold: float = 10.0  # Max allowed spectral product for Q/K pairs (<=0 disables)
    muon_clip_alpha: float = 0.5  # Split factor for distributing clip between Q and K
    muon_clip_pairs: List[List[str]] = field(
        default_factory=lambda: [["q_proj", "k_proj"]]
    )

    # Scheduler settings
    # Warmup configuration (batch-based)
    warmup_batches: int = 200  # Linear warmup over N batches (not optimizer steps)
                                # 200 batches ≈ 1 minute
                                # Reaches full LR quickly for faster iteration
    plateau_batches: int = 0   # Hold at max LR for N batches after warmup (not optimizer steps)
                                # 0 = no plateau, immediate cosine decay after warmup
                                # For 60K step run: suggest 30K steps ≈ 1.92M batches (30000 * 64 grad_accum)
    scheduler_type: str = "cosine_warmup"
    cosine_t0: int = 10000
    cosine_tmult: int = 2
    min_lr: float = 3.0e-5

    # Mixed precision
    use_bf16: bool = True
    gradient_checkpointing: bool = False

    # Teacher settings
    teacher_mode: str = "direct"  # direct, cached, network
    teacher_device: str = "cuda"  # Device for teacher model (e.g., "cuda", "cuda:0", "cuda:1")
                                   # Use different device than student for memory isolation
                                   # Reduces memory conflicts when using DirectTeacherClient
    teacher_url: str = "http://localhost:8080"
    teacher_model: str = "meta-llama/Llama-3.2-1B-Instruct"
    teacher_adapter_path: Optional[str] = None  # Optional PEFT adapter path for teacher model
    teacher_topk: int = 512  # Increased from 128 for better probability coverage
    teacher_temperature: float = 2.0
    teacher_timeout: float = 30.0
    teacher_max_retries: int = 3
    cache_logits: bool = False  # Cache logits while training
    cache_dir: str = "data/teacher_cache/"  # Directory for cached logits

    # Distillation loss settings
    distill_alpha: float = 0.2  # Hard CE mixing coefficient

    # Reverse KL configuration
    reverse_kl: bool = False  # Use reverse KL divergence (KL(student || teacher) instead of KL(teacher || student))
    reverse_kl_warmup_steps: int = 0  # Optional warmup before flipping to reverse KL (0 = disabled)

    # Parameter schedules
    alpha_warmup_steps: int = 0  # Steps before ramping alpha (0 = no schedule, use distill_alpha directly)
    alpha_initial: float = 0.0  # Starting alpha during warmup
    alpha_final: float = 0.2  # Target alpha after warmup (overrides distill_alpha if warmup enabled)

    temperature_warmup_steps: int = 0  # Steps before ramping temperature (0 = no schedule)
    temperature_initial: float = 2.5  # Starting temperature during warmup
    temperature_final: float = 1.0  # Target temperature after warmup

    # Data configuration
    train_data_path: str = "data/train"
    val_data_path: str = "data/val"
    tokenizer_name: str = "meta-llama/Llama-3.2-1B"
    num_workers: int = 4  # DataLoader worker processes
    ram_cache_mb: int = 0  # RAM cache size in MB (0 = disabled, use lazy loading)

    # Streaming data pipeline configuration
    use_streaming_loader: bool = False  # Use memory-efficient streaming dataloader
    use_prefetch_loader: bool = True    # Wrap with GPU prefetch loader (recommended)
    streaming_prefetch_factor: int = 2  # Batches to prefetch per worker
    streaming_num_workers: int = 2      # Workers for streaming mode (lower than default)

    # Sequence packing configuration
    use_packed_sequences: bool = False  # Pack multiple short examples into full 4k sequences
    pack_max_length: int = 4000  # Target length for packing (leaves buffer for BOS/EOS)

    # Pretokenized data configuration
    use_pretokenized_data: bool = False  # Use manifest-based pretokenized dataset
    pretokenized_splits: Optional[List[str]] = None  # Filter to specific splits (None = all)

    # Async teacher configuration
    async_teacher: bool = False  # Enable async teacher prefetch for overlapped inference
    force_async_teacher: bool = False  # Override auto-disable when teacher/student on same device
    teacher_prefetch_batches: int = 1  # Pipeline depth (1 = prefetch one batch ahead)

    # Checkpointing
    output_dir: str = "runs/stage1_kd"
    checkpoint_dir: Optional[str] = None  # Defaults to {output_dir}/checkpoints
    save_interval: int = 5000
    keep_last_n: int = 3
    max_total_size_gb: float = 100.0
    resume_from: Optional[str] = None  # Path to checkpoint to resume from

    # Evaluation
    eval_interval: int = 5000
    eval_perplexity: bool = True
    eval_niah: bool = True
    eval_perplexity_samples: int = 1000
    eval_niah_samples: int = 100

    # Telemetry
    log_interval: int = 10
    log_memory_debug: bool = False  # Enable detailed memory logging (CPU overhead)
    log_sample_debug: bool = False  # Enable sample data logging (token IDs, decoded text) for first 10 steps
    enable_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_offline: bool = False

    # Resource limits
    max_vram_gb: float = 30.0

    # Misc
    seed: int = 42
    device: str = "cuda"

    # Saddle escape
    saddle_escape: SaddleEscapeConfig = field(default_factory=SaddleEscapeConfig)

    # CE Pretrain Mode
    pretrain_ce_only: bool = False  # Enable cross-entropy only pretraining (no teacher)

    def __post_init__(self):
        """Post-initialization validation and derivations."""
        # Resolve output/checkpoint/data paths relative to repo root for consistency
        output_path = _resolve_project_path(self.output_dir)
        self.output_dir = str(output_path)

        if self.checkpoint_dir is None:
            checkpoint_path = output_path / "checkpoints"
        else:
            checkpoint_path = _resolve_project_path(self.checkpoint_dir)
        self.checkpoint_dir = str(checkpoint_path)

        if _is_subpath(output_path, checkpoint_path):
            raise ValueError(
                "output_dir cannot be inside checkpoint_dir. "
                "Please set --output-dir outside the checkpoints folder."
            )

        train_data_path = _resolve_project_path(self.train_data_path)
        if train_data_path:
            self.train_data_path = str(train_data_path)

        val_data_path = _resolve_project_path(self.val_data_path)
        if val_data_path:
            self.val_data_path = str(val_data_path)

        cache_path = _resolve_project_path(self.cache_dir) if self.cache_dir else None
        if cache_path:
            self.cache_dir = str(cache_path)

        if self.model_config_path:
            model_config_path = _resolve_project_path(self.model_config_path)
            if model_config_path:
                self.model_config_path = str(model_config_path)

        if self.resume_from:
            resume_path = _resolve_project_path(self.resume_from)
            if resume_path:
                self.resume_from = str(resume_path)

        # Handle legacy warmup_steps config (migrate to warmup_batches)
        if hasattr(self, 'warmup_steps') and not hasattr(self, 'warmup_batches'):
            logger.warning(
                f"Legacy config: warmup_steps={self.warmup_steps} detected. "
                f"Converting to warmup_batches. Please update config to use warmup_batches instead."
            )
            # Convert optimizer steps to batches (approximate)
            self.warmup_batches = self.warmup_steps * self.gradient_accumulation_steps

        # Validate model variant
        valid_variants = ["350M", "500M", "titan_mac_350M", "titan_mac_500M", "titan_retention_350M"]
        if self.model_variant not in valid_variants:
            raise ValueError(f"Invalid model_variant: {self.model_variant}. Must be one of {valid_variants}")

        # Validate effective batch size
        effective_batch_size = self.physical_batch_size * self.gradient_accumulation_steps
        if effective_batch_size < 1:
            raise ValueError(f"Effective batch size must be >= 1, got {effective_batch_size}")

    @property
    def effective_batch_size(self) -> int:
        """Compute effective batch size."""
        return self.physical_batch_size * self.gradient_accumulation_steps

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """Create from dictionary."""
        # Filter to valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        # Handle nested SaddleEscapeConfig
        if 'saddle_escape' in filtered_data and isinstance(filtered_data['saddle_escape'], dict):
            filtered_data['saddle_escape'] = SaddleEscapeConfig(**filtered_data['saddle_escape'])

        return cls(**filtered_data)


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary with configuration

    Raises:
        FileNotFoundError: If config file not found
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.info(f"Loading config from: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Empty config file: {config_path}")

    # FIX #3: Check for deprecated config marker
    if config.get('__deprecated__'):
        error_msg = config.get('__error__', 'This config file is deprecated')
        raise RuntimeError(error_msg)

    # Flatten nested configuration
    flattened = _flatten_config(config)

    logger.info(f"Loaded config with {len(flattened)} keys")

    return flattened


def _flatten_config(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested configuration dictionary.

    Example:
        {"training": {"lr": 0.001}} -> {"training_lr": 0.001}

    Args:
        config: Nested configuration dictionary
        prefix: Prefix for flattened keys

    Returns:
        Flattened dictionary
    """
    flattened = {}

    for key, value in config.items():
        if prefix:
            full_key = f"{prefix}_{key}"
        else:
            full_key = key

        if isinstance(value, dict):
            # Recursively flatten nested dicts
            flattened.update(_flatten_config(value, full_key))
        else:
            flattened[full_key] = value

    return flattened


def _unflatten_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Unflatten configuration dictionary.

    Example:
        {"training_lr": 0.001} -> {"training": {"lr": 0.001}}

    Args:
        config: Flattened configuration dictionary

    Returns:
        Nested dictionary
    """
    unflattened = {}

    for key, value in config.items():
        parts = key.split('_')

        # Navigate/create nested structure
        current = unflattened
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                # If we encounter a non-dict value, replace it with a dict
                # This handles cases where a boolean/scalar was set before we knew it needed to be nested
                current[part] = {}
            current = current[part]

        # Set value
        current[parts[-1]] = value

    return unflattened


def merge_configs(
    base_config: Dict[str, Any],
    override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge two configurations, with override taking precedence.

    Args:
        base_config: Base configuration (e.g., from YAML)
        override_config: Override configuration (e.g., from CLI)

    Returns:
        Merged configuration
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if value is not None:  # Only override if value is not None
            merged[key] = value

    return merged


def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Train RetNet student via knowledge distillation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )

    # Model configuration
    parser.add_argument(
        "--model-variant",
        type=str,
        choices=["350M", "500M"],
        help="Student model variant (350M or 500M)",
    )
    parser.add_argument(
        "--model-config-path",
        type=str,
        help="Path to model-specific config YAML",
    )

    # ACT configuration
    parser.add_argument(
        "--use-act",
        action="store_true",
        default=None,
        help="Enable Adaptive Computation Time wrapper",
    )
    parser.add_argument(
        "--act-max-steps",
        type=int,
        help="Maximum pondering steps for ACT",
    )
    parser.add_argument(
        "--act-epsilon",
        type=float,
        help="Halting threshold for ACT (default: 0.01)",
    )
    parser.add_argument(
        "--act-ponder-penalty",
        type=float,
        help="Weight for pondering cost loss",
    )
    parser.add_argument(
        "--act-use-geometric-prior",
        action="store_true",
        default=None,
        help="Use geometric prior regularization (PonderNet style)",
    )
    parser.add_argument(
        "--act-prior-lambda",
        type=float,
        help="Lambda for geometric prior distribution",
    )

    # Training hyperparameters
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--physical-batch-size",
        type=int,
        help="Physical batch size per device",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        help="Maximum sequence length",
    )

    # Optimizer settings
    parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        help="Weight decay coefficient",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        help="Maximum gradient norm for clipping",
    )
    parser.add_argument(
        "--use-8bit-optimizer",
        action="store_true",
        default=None,
        help="Use 8-bit AdamW optimizer",
    )
    parser.add_argument(
        "--no-8bit-optimizer",
        action="store_false",
        dest="use_8bit_optimizer",
        help="Disable 8-bit optimizer (use standard AdamW)",
    )

    # Scheduler settings
    parser.add_argument(
        "--warmup-batches",
        type=int,
        help="Number of warmup batches (batch-based warmup, not optimizer steps)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        help="[DEPRECATED] Number of warmup steps - use --warmup-batches instead",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        help="Minimum learning rate",
    )

    # Mixed precision
    parser.add_argument(
        "--use-bf16",
        action="store_true",
        default=None,
        help="Use BF16 mixed precision",
    )
    parser.add_argument(
        "--no-bf16",
        action="store_false",
        dest="use_bf16",
        help="Disable BF16 (use FP32)",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=None,
        help="Enable gradient checkpointing",
    )

    # Teacher settings
    parser.add_argument(
        "--teacher-mode",
        type=str,
        choices=["direct", "cached", "network"],
        help="Teacher mode: direct (load in memory), cached (read from disk), network (vLLM server)",
    )
    parser.add_argument(
        "--teacher-device",
        type=str,
        help="Device for teacher model (e.g., 'cuda', 'cuda:0', 'cuda:1'). Use different device than student for memory isolation.",
    )
    parser.add_argument(
        "--teacher-url",
        type=str,
        help="vLLM teacher server URL (for network mode)",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        help="Teacher model identifier",
    )
    parser.add_argument(
        "--teacher-adapter-path",
        type=str,
        help="Path to PEFT adapter for teacher model (optional)",
    )
    parser.add_argument(
        "--teacher-topk",
        type=int,
        help="Number of top-k logits from teacher",
    )
    parser.add_argument(
        "--teacher-temperature",
        type=float,
        help="Temperature for teacher softmax",
    )
    parser.add_argument(
        "--cache-logits",
        action="store_true",
        default=None,
        help="Cache teacher logits while training",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Directory for cached logits",
    )

    # Distillation settings
    parser.add_argument(
        "--distill-alpha",
        type=float,
        help="Hard CE mixing coefficient (alpha)",
    )
    parser.add_argument(
        "--reverse-kl",
        action="store_true",
        default=None,
        help="Use reverse KL divergence (KL(student || teacher) instead of KL(teacher || student))",
    )
    parser.add_argument(
        "--reverse-kl-warmup-steps",
        type=int,
        help="Optional warmup before flipping to reverse KL (0 = disabled)",
    )
    parser.add_argument(
        "--alpha-warmup-steps",
        type=int,
        help="Steps before ramping alpha (0 = no schedule)",
    )
    parser.add_argument(
        "--alpha-initial",
        type=float,
        help="Starting alpha during warmup",
    )
    parser.add_argument(
        "--alpha-final",
        type=float,
        help="Target alpha after warmup",
    )
    parser.add_argument(
        "--temperature-warmup-steps",
        type=int,
        help="Steps before ramping temperature (0 = no schedule)",
    )
    parser.add_argument(
        "--temperature-initial",
        type=float,
        help="Starting temperature during warmup",
    )
    parser.add_argument(
        "--temperature-final",
        type=float,
        help="Target temperature after warmup",
    )

    # Data configuration
    parser.add_argument(
        "--train-data-path",
        type=str,
        help="Path to training data",
    )
    parser.add_argument(
        "--val-data-path",
        type=str,
        help="Path to validation data",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        help="HuggingFace tokenizer name",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--ram-cache-mb",
        type=int,
        metavar="MB",
        dest="ram_cache_mb",
        help="RAM cache size in MB for parquet data (0 = disabled, lazy loading)",
    )

    # Pretokenized data configuration
    parser.add_argument(
        "--use-pretokenized-data",
        action="store_true",
        default=None,
        help="Use manifest-based pretokenized dataset",
    )
    parser.add_argument(
        "--pretokenized-splits",
        nargs="+",
        type=str,
        help="Filter to specific splits from manifest (space-separated list, e.g., --pretokenized-splits openhermes numina_cot)",
    )

    # Async teacher configuration
    parser.add_argument(
        "--async-teacher",
        action="store_true",
        default=None,
        help="Enable async teacher prefetch to overlap teacher inference with student training",
    )
    parser.add_argument(
        "--no-async-teacher",
        action="store_false",
        dest="async_teacher",
        help="Disable async teacher prefetch (use synchronous mode)",
    )
    parser.add_argument(
        "--force-async-teacher",
        action="store_true",
        default=None,
        help="Force async mode even when teacher/student are on same device (overrides auto-disable)",
    )
    parser.add_argument(
        "--teacher-prefetch-batches",
        type=int,
        help="Pipeline depth for async teacher (default: 1)",
    )

    # Checkpointing
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for logs and checkpoints",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Checkpoint directory (defaults to {output_dir}/checkpoints)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        help="Steps between checkpoint saves",
    )
    parser.add_argument(
        "--keep-last-n",
        type=int,
        help="Number of recent checkpoints to keep",
    )
    parser.add_argument(
        "--resume",
        "--resume-from",
        dest="resume_from",
        type=str,
        help="Path to checkpoint to resume from (restores model/optimizer/LR state)",
    )

    # Evaluation
    parser.add_argument(
        "--eval-interval",
        type=int,
        help="Steps between evaluations",
    )
    parser.add_argument(
        "--no-eval-perplexity",
        action="store_false",
        dest="eval_perplexity",
        default=None,
        help="Disable perplexity evaluation",
    )
    parser.add_argument(
        "--no-eval-niah",
        action="store_false",
        dest="eval_niah",
        default=None,
        help="Disable NIAH evaluation",
    )

    # Telemetry
    parser.add_argument(
        "--log-interval",
        type=int,
        help="Steps between logging",
    )
    parser.add_argument(
        "--log-memory-debug",
        action="store_true",
        default=None,
        help="Enable detailed memory logging at key checkpoints (has CPU overhead)",
    )
    parser.add_argument(
        "--log-sample-debug",
        action="store_true",
        default=None,
        help="Enable sample data logging (token IDs, decoded text) for first 10 steps",
    )
    parser.add_argument(
        "--enable-wandb",
        action="store_true",
        default=None,
        help="Enable wandb logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        help="Wandb run name",
    )
    parser.add_argument(
        "--wandb-offline",
        action="store_true",
        default=None,
        help="Use wandb offline mode",
    )

    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode (validate config without training)",
    )

    # Saddle escape
    parser.add_argument(
        "--enable-saddle-interventions",
        action="store_true",
        help="Enable saddle point interventions (default: detection only)",
    )

    # CE Pretrain Mode
    parser.add_argument(
        "--pretrain-ce-only",
        action="store_true",
        default=None,
        help="Enable CE-only pretraining mode (skip teacher model)",
    )

    return parser.parse_args()


def create_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    """Create training configuration from CLI arguments and YAML file.

    Priority (highest to lowest):
    1. CLI arguments (if provided)
    2. YAML config file (if provided)
    3. Default values

    Args:
        args: Parsed CLI arguments

    Returns:
        Complete training configuration
    """
    # Start with defaults
    config_dict = TrainingConfig().to_dict()

    # Load YAML config if provided
    if args.config:
        yaml_config = load_yaml_config(args.config)
        config_dict = merge_configs(config_dict, yaml_config)
        logger.info(f"Loaded YAML config from: {args.config}")

    # Override with CLI arguments
    cli_overrides = {}
    for key, value in vars(args).items():
        # Skip special keys
        if key in ["config", "dry_run", "enable_saddle_interventions"]:
            continue

        # Only override if value is not None
        if value is not None:
            cli_overrides[key] = value

    if cli_overrides:
        config_dict = merge_configs(config_dict, cli_overrides)
        logger.info(f"Applied {len(cli_overrides)} CLI overrides")

    # Create TrainingConfig from merged dict
    config = TrainingConfig.from_dict(config_dict)

    # Handle enable_saddle_interventions CLI flag
    if hasattr(args, 'enable_saddle_interventions') and args.enable_saddle_interventions:
        config.saddle_escape.interventions_enabled = True
        logger.info("CLI override: Saddle point interventions ENABLED")

    return config


def validate_config(config: TrainingConfig, skip_data_checks: bool = False):
    """Validate training configuration.

    Args:
        config: Training configuration to validate
        skip_data_checks: Skip checking if data paths exist (for dry-run mode)

    Raises:
        ValueError: If configuration is invalid
    """
    logger.info("Validating configuration...")

    # Validate paths exist (skip in dry-run mode)
    if not skip_data_checks:
        if not Path(config.train_data_path).exists():
            raise ValueError(f"Training data not found: {config.train_data_path}")

        if not Path(config.val_data_path).exists():
            logger.warning(f"Validation data not found: {config.val_data_path}")
    else:
        logger.info("Skipping data path validation (dry-run mode)")

    # Validate resume checkpoint
    if config.resume_from and not Path(config.resume_from).exists():
        raise ValueError(f"Resume checkpoint not found: {config.resume_from}")

    # Validate teacher URL
    if not config.teacher_url:
        raise ValueError("teacher_url must be specified")

    # Validate wandb settings
    if config.enable_wandb and not config.wandb_project:
        raise ValueError("wandb_project must be specified when enable_wandb=True")

    # Validate model variant
    valid_variants = ["350M", "500M", "titan_mac_350M", "titan_mac_500M", "titan_retention_350M"]
    if config.model_variant not in valid_variants:
        raise ValueError(f"Invalid model_variant: {config.model_variant}. Must be one of {valid_variants}")

    # Validate batch size
    if config.effective_batch_size < 1:
        raise ValueError(f"Effective batch size must be >= 1")

    # Validate sequence length with direct teacher mode
    if config.teacher_mode == "direct" and config.max_seq_length > 8192:
        logger.warning("=" * 80)
        logger.warning("WARNING: High sequence length with direct teacher mode")
        logger.warning(f"  max_seq_length: {config.max_seq_length}")
        logger.warning(f"  teacher_mode: {config.teacher_mode}")
        logger.warning("")
        logger.warning("Direct teacher mode loads the teacher model in VRAM alongside the student.")
        logger.warning("Sequence lengths > 8192 can cause significant memory overhead (~8-10GB+).")
        logger.warning("")
        logger.warning("Recommendations:")
        logger.warning("  1. Reduce max_seq_length to 4096 or 8192 (saves ~4-8GB)")
        logger.warning("  2. Use teacher_mode='network' to run teacher on separate GPU/server")
        logger.warning("  3. Use teacher_mode='cached' with pre-cached logits")
        logger.warning("=" * 80)
        # Don't raise error - just warn. User might have enough VRAM.

    logger.info("Configuration validated successfully")


def print_config(config: TrainingConfig):
    """Print configuration in human-readable format.

    Args:
        config: Training configuration to print
    """
    logger.info("=" * 80)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 80)

    # Model configuration
    logger.info("MODEL:")
    logger.info(f"  Variant: {config.model_variant}")
    logger.info(f"  Model config: {config.model_config_path}")
    logger.info(f"  Max sequence length: {config.max_seq_length}")

    # Training hyperparameters
    logger.info("TRAINING:")
    logger.info(f"  Max steps: {config.max_steps:,}")
    logger.info(f"  Physical batch size: {config.physical_batch_size}")
    logger.info(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {config.effective_batch_size}")
    logger.info(f"  Learning rate: {config.learning_rate:.2e}")
    logger.info(f"  Weight decay: {config.weight_decay}")
    logger.info(f"  Max grad norm: {config.max_grad_norm}")
    logger.info(f"  Warmup batches: {config.warmup_batches}")
    logger.info(f"  Use BF16: {config.use_bf16}")
    logger.info(f"  Use 8-bit optimizer: {config.use_8bit_optimizer}")
    logger.info(f"  Gradient checkpointing: {config.gradient_checkpointing}")

    # Teacher configuration
    logger.info("TEACHER:")
    logger.info(f"  Mode: {config.teacher_mode}")
    logger.info(f"  Device: {config.teacher_device}")
    logger.info(f"  URL: {config.teacher_url}")
    logger.info(f"  Model: {config.teacher_model}")
    logger.info(f"  Adapter path: {config.teacher_adapter_path or 'None'}")
    logger.info(f"  Top-k: {config.teacher_topk}")
    logger.info(f"  Temperature: {config.teacher_temperature}")

    # Data configuration
    logger.info("DATA:")
    logger.info(f"  Train data: {config.train_data_path}")
    logger.info(f"  Val data: {config.val_data_path}")
    logger.info(f"  Tokenizer: {config.tokenizer_name}")
    logger.info(f"  Num workers: {config.num_workers}")

    # Checkpointing
    logger.info("CHECKPOINTING:")
    logger.info(f"  Output dir: {config.output_dir}")
    logger.info(f"  Checkpoint dir: {config.checkpoint_dir}")
    logger.info(f"  Save interval: {config.save_interval}")
    logger.info(f"  Keep last N: {config.keep_last_n}")
    logger.info(f"  Resume from: {config.resume_from or 'None'}")

    # Evaluation
    logger.info("EVALUATION:")
    logger.info(f"  Eval interval: {config.eval_interval}")
    logger.info(f"  Eval perplexity: {config.eval_perplexity}")
    logger.info(f"  Eval NIAH: {config.eval_niah}")

    # Telemetry
    logger.info("TELEMETRY:")
    logger.info(f"  Log interval: {config.log_interval}")
    logger.info(f"  Wandb enabled: {config.enable_wandb}")
    if config.enable_wandb:
        logger.info(f"  Wandb project: {config.wandb_project}")
        logger.info(f"  Wandb run: {config.wandb_run_name}")

    # Misc
    logger.info("MISC:")
    logger.info(f"  Seed: {config.seed}")
    logger.info(f"  Device: {config.device}")

    logger.info("=" * 80)


def save_config(config: TrainingConfig, output_path: Union[str, Path]):
    """Save configuration to YAML file.

    Args:
        config: Training configuration to save
        output_path: Path to save YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to nested dict for readability
    config_dict = config.to_dict()
    nested_dict = _unflatten_config(config_dict)

    with open(output_path, 'w') as f:
        yaml.dump(nested_dict, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved configuration to: {output_path}")

