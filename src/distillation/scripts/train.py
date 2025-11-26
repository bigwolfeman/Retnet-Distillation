#!/usr/bin/env python3
"""
Main training script for knowledge distillation.

This script ties together all components from Phases 1-4:
- Student model (RetNet 350M/500M)
- VLLMTeacherClient (real vLLM server)
- Data pipeline
- Optimizer and scheduler
- Telemetry logger
- Evaluation runner
- Checkpoint manager

Tasks: T071-T074
"""

import os
import sys
import json
import logging
import signal
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# FIX #2: Configure CUDA allocator to reduce fragmentation
# - expandable_segments: Allows memory segments to grow dynamically
# - max_split_size_mb: Limits block splitting to reduce fragmentation (trades allocation speed for memory efficiency)
# This prevents the 4-25GB memory fragmentation observed during long training runs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Optional wandb import for enhanced logging
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    logging.warning("wandb not installed. Enhanced wandb logging will be disabled.")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from distillation.config import (
    parse_cli_args,
    create_config_from_args,
    validate_config,
    print_config,
    save_config,
    TrainingConfig,
)
from distillation.student_config import create_student_config, RetNetStudentConfig
from distillation.vllm_teacher_client import VLLMTeacherClient
from distillation.cached_teacher_client import CachedTeacherClient
from distillation.direct_teacher_client import DirectTeacherClient
from distillation.caching_wrapper import CachingTeacherWrapper
from distillation.dataset import SimpleDataLoader, load_llama_tokenizer, create_streaming_dataloaders
from distillation.packed_dataset import PackedDataLoader
from distillation.prefetch_dataloader import PrefetchDataLoader
from distillation.optimizer import create_optimizer, create_scheduler
from distillation.telemetry import TelemetryLogger, OutputSink
from distillation.evaluation.runner import EvaluationRunner
from distillation.evaluation.perplexity import PerplexityConfig
from distillation.evaluation.niah import NIAHConfig
from distillation.checkpoint import CheckpointManager
from distillation.trainer import DistillationTrainer, TrainingConfig as LegacyTrainingConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def _resolve_checkpoint_path(path: Path) -> Path:
    """Resolve a user-supplied checkpoint path.

    Accepts either a direct checkpoint file or a directory containing checkpoints.
    Returns the actual checkpoint file path.
    """
    if path.is_file():
        return path

    if path.is_dir():
        latest_txt = path / "latest_checkpoint.txt"
        if latest_txt.exists():
            try:
                text = latest_txt.read_text().strip()
                if text:
                    candidate = Path(text)
                    if not candidate.is_absolute():
                        candidate = (path / candidate).resolve()
                    if candidate.exists():
                        return candidate
                    logger.warning(f"latest_checkpoint.txt points to missing file: {candidate}")
            except Exception as e:
                logger.warning(f"Failed to read {latest_txt}: {e}")

        latest_link = path / "checkpoint_latest.pt"
        if latest_link.exists():
            return latest_link.resolve() if latest_link.is_symlink() else latest_link

        checkpoints = sorted(path.glob("checkpoint_*.pt"))
        if checkpoints:
            return checkpoints[-1]

        raise FileNotFoundError(f"No checkpoint files found in directory: {path}")

    raise FileNotFoundError(f"Checkpoint path not found: {path}")


class TrainingState:
    """Training state tracker for crash recovery.

    Tracks:
    - global_step
    - epoch
    - best_val_loss
    - best_checkpoint_path
    - training_start_time
    - last_eval_step
    - last_save_step
    """

    def __init__(self, state_file: Path):
        """Initialize training state tracker.

        Args:
            state_file: Path to training_state.json
        """
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """Load training state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                logger.info(f"Loaded training state from: {self.state_file}")
                return state
            except Exception as e:
                logger.warning(f"Failed to load training state: {e}")

        # Default state
        return {
            'global_step': 0,
            'epoch': 0,
            'best_val_loss': float('inf'),
            'best_checkpoint_path': None,
            'training_start_time': time.time(),
            'last_eval_step': 0,
            'last_save_step': 0,
        }

    def save(self):
        """Save training state to disk."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
            logger.debug(f"Saved training state to: {self.state_file}")
        except Exception as e:
            logger.error(f"Failed to save training state: {e}")

    def update(self, **kwargs):
        """Update training state.

        Args:
            **kwargs: Key-value pairs to update
        """
        self.state.update(kwargs)
        self.save()

    def get(self, key: str, default: Any = None) -> Any:
        """Get state value.

        Args:
            key: State key
            default: Default value if key not found

        Returns:
            State value
        """
        return self.state.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Get state value by key."""
        return self.state[key]

    def __setitem__(self, key: str, value: Any):
        """Set state value and save."""
        self.state[key] = value
        self.save()


def setup_signal_handlers(trainer: DistillationTrainer, checkpoint_manager: CheckpointManager):
    """Setup signal handlers for graceful shutdown.

    Args:
        trainer: Trainer instance
        checkpoint_manager: Checkpoint manager instance
    """
    def signal_handler(signum, frame):
        """Handle shutdown signals."""
        logger.info(f"\nReceived signal {signum}, saving checkpoint and shutting down...")

        try:
            # Save checkpoint
            state_dict = trainer.get_state_dict()
            checkpoint_path = checkpoint_manager.save_checkpoint(
                state_dict,
                step=trainer.global_step,
                is_best=False,
            )
            logger.info(f"Saved checkpoint to: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

        logger.info("Shutdown complete")
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Registered signal handlers (SIGINT, SIGTERM)")


def load_student_model(
    student_config: RetNetStudentConfig,
    device: torch.device,
    use_bf16: bool = True,
    gradient_checkpointing: bool = True,  # FIX #4: Wire gradient_checkpointing from config
) -> nn.Module:
    """Load student model (RetNet or TitanMAC).

    Args:
        student_config: Student model configuration
        device: Device to load model to
        use_bf16: Use BF16 precision
        gradient_checkpointing: Enable gradient checkpointing for memory savings

    Returns:
        Loaded student model
    """
    logger.info("Loading student model...")
    logger.info(f"  Variant: {student_config.variant}")
    logger.info(f"  d_model: {student_config.d_model}")
    logger.info(f"  n_layers: {student_config.n_layers}")
    logger.info(f"  n_heads: {student_config.n_heads}")

    # Check if this is a TitanMAC variant
    if student_config.variant.startswith("titan_mac"):
        logger.info("Loading TitanMAC model...")
        from src.models.titans.titan_init import create_titan_mac_model

        model = create_titan_mac_model(
            variant=student_config.variant,
            vocab_size=student_config.vocab_size,
            device=device,
            use_bf16=use_bf16,
        )

        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"TitanMAC model loaded: {param_count:,} parameters")

        return model

    # Otherwise, load RetNet model
    # Import RetNet model
    try:
        from models.retnet.backbone import RetNetBackbone, RetNetOutputHead
    except ImportError:
        # Try alternative import path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from models.retnet.backbone import RetNetBackbone, RetNetOutputHead

    # Create model
    model_kwargs = student_config.to_retnet_backbone_args()
    # FIX #4: Pass gradient_checkpointing parameter
    model_kwargs['checkpoint_activations'] = gradient_checkpointing
    model = RetNetBackbone(**model_kwargs)
    logger.info(f"  Gradient checkpointing: {gradient_checkpointing}")

    # Add output head (lm_head) for vocab projection
    # FIXED: Use add_module() to properly register as PyTorch submodule
    # Create WITHOUT weight tying initially - we'll tie after dtype conversion
    model.add_module('lm_head', RetNetOutputHead(
        d_model=student_config.d_model,
        vocab_size=student_config.vocab_size,
        tie_weights=False,  # Will tie manually after BF16 conversion
        embedding_layer=None,
    ))

    # Move to device BEFORE tying weights so both tensors live on same device
    # Keep parameters in FP32 for optimizer stability; bf16 is handled via autocast
    model = model.to(device)
    if use_bf16:
        logger.info("BF16 autocast enabled for forward/backward (parameters stay FP32)")

    # NOW tie weights after both are in BF16
    # This saves ~147M parameters (29% of model size!)
    model.lm_head.proj.weight = model.embed.weight
    logger.info("Weight tying enabled: lm_head.proj shares weights with embed (saves 147M params)")

    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Student model loaded: {param_count:,} parameters")

    # ============================================================================
    # DIAGNOSTIC: Verify retention decay parameters are learnable
    # ============================================================================
    print("\n" + "="*80)
    print("RETENTION DECAY PARAMETER AUDIT")
    print("="*80)

    decay_params = [(n, p.shape, p.requires_grad, p.numel()) for n, p in model.named_parameters()
                    if 'decay' in n.lower()]
    decay_buffers = [(n, b.shape) for n, b in model.named_buffers()
                     if 'decay' in n.lower()]

    num_decay_tensors = len(decay_params)
    num_decay_elements = sum(numel for _, _, _, numel in decay_params)

    print(f"\n✓ Learnable decay parameter tensors: {num_decay_tensors}")
    print(f"✓ Total decay parameter elements: {num_decay_elements}")
    if decay_params:
        for name, shape, grad, numel in decay_params[:5]:
            print(f"    ✓ {name}: {shape} = {numel} elements (requires_grad={grad})")
        if len(decay_params) > 5:
            print(f"    ... and {len(decay_params) - 5} more")
    else:
        print("    ⚠️  WARNING: NO LEARNABLE DECAY PARAMETERS!")

    print(f"\n✓ Fixed decay buffers: {len(decay_buffers)}")
    if decay_buffers:
        print("    ⚠️  PROBLEM: Decay is still a BUFFER (not learnable)!")
        for name, shape in decay_buffers[:3]:
            print(f"    {name}: {shape}")

    expected_elements = 12  # 1 shared RetNetRelPos instance × 12 heads (not per-layer!)
    print(f"\n✓ Expected: {expected_elements} learnable decay elements")
    print(f"✓ Actual:   {num_decay_elements} learnable decay elements")
    print(f"\n  Note: RetNet uses a SHARED retnet_rel_pos module (not one per layer)")
    print(f"  So we expect 12 elements (1 tensor × 12 heads), not 192 (16 layers × 12 heads)")

    if num_decay_elements == expected_elements and len(decay_buffers) == 0:
        print("\n✅ SUCCESS: Retention decay parameters are learnable!")
    elif num_decay_elements == 0 and len(decay_buffers) > 0:
        print("\n❌ FAILURE: Decay is still a fixed buffer. TorchScale patch did not apply.")
    elif num_decay_elements > 0 and len(decay_buffers) > 0:
        print("\n⚠️  WARNING: Both parameters AND buffers exist. Check for duplicate 'decay' attributes.")
    else:
        print(f"\n⚠️  WARNING: Found {num_decay_elements} elements, expected {expected_elements}")

    print("="*80 + "\n")
    # ============================================================================

    # Validate parameter count
    try:
        student_config.validate_actual_params(param_count)
        logger.info(f"Parameter count within target range: {student_config.target_param_count_range}")
    except AssertionError as e:
        logger.warning(f"Parameter count validation failed: {e}")

    return model


def create_data_loaders(
    config: TrainingConfig,
    tokenizer: AutoTokenizer,
) -> tuple[DataLoader, Optional[DataLoader]]:
    """Create training and validation data loaders.

    Args:
        config: Training configuration
        tokenizer: Tokenizer

    Returns:
        Tuple of (train_loader, val_loader)
    """
    logger.info("Creating data loaders...")
    logger.info(f"  Packed sequences: {config.use_packed_sequences}")
    logger.info(f"  Streaming mode: {config.use_streaming_loader}")
    logger.info(f"  Prefetch wrapper: {config.use_prefetch_loader}")

    # Detect data format (pretokenized manifest > parquet > JSONL)
    train_path = Path(config.train_data_path)

    # Check for manifest-based pretokenized data (highest priority)
    is_pretokenized_manifest = False
    if train_path.is_dir() and (train_path / "manifest.json").exists():
        is_pretokenized_manifest = True
    elif config.use_pretokenized_data:
        # User explicitly requested pretokenized data but manifest not found
        raise ValueError(
            f"Pretokenized data requested (--use-pretokenized-data) but "
            f"manifest.json not found in {train_path}. "
            f"Ensure the directory contains a manifest.json file."
        )

    # Check for parquet data (fallback)
    is_parquet = False
    if not is_pretokenized_manifest:
        if train_path.is_dir():
            # Check if directory contains parquet files
            parquet_files = list(train_path.glob("*.parquet"))
            is_parquet = len(parquet_files) > 0
        elif str(train_path).endswith('.parquet'):
            is_parquet = True

    # Load datasets with appropriate loader
    if is_pretokenized_manifest:
        logger.info("Detected manifest-based pretokenized format - using PretokenizedShardDataset")
        logger.info(f"  Manifest path: {train_path / 'manifest.json'}")
        if config.pretokenized_splits:
            logger.info(f"  Split filter: {config.pretokenized_splits}")
        else:
            logger.info(f"  Split filter: None (using all splits)")

        if config.use_packed_sequences:
            logger.warning("Packed sequences requested but pretokenized format detected - packing not supported")
            logger.warning("Falling back to standard PretokenizedShardDataset")

        from src.distillation.dataset import PretokenizedShardDataset

        train_dataset = PretokenizedShardDataset(
            data_path=config.train_data_path,
            max_length=config.max_seq_length,
            splits=config.pretokenized_splits,
            tokenizer_pad_token_id=tokenizer.pad_token_id,
        )
        logger.info(f"Training dataset loaded: {len(train_dataset):,} sequences")
        logger.info(f"  Using pad_token_id: {tokenizer.pad_token_id} (from tokenizer)")

        # Validation dataset (optional)
        val_dataset = None
        val_path = Path(config.val_data_path)
        if val_path.exists() and (val_path / "manifest.json").exists():
            val_dataset = PretokenizedShardDataset(
                data_path=config.val_data_path,
                max_length=config.max_seq_length,
                splits=config.pretokenized_splits,
                tokenizer_pad_token_id=tokenizer.pad_token_id,
            )
            logger.info(f"Validation dataset loaded: {len(val_dataset):,} sequences")
        else:
            logger.warning(f"Validation data not found or no manifest: {config.val_data_path}")

    elif is_parquet:
        logger.info("Detected parquet format - using ParquetDataLoader")
        logger.info(f"  RAM cache: {config.ram_cache_mb} MB ({'enabled' if config.ram_cache_mb > 0 else 'disabled'})")

        if config.use_packed_sequences:
            logger.warning("Packed sequences requested but parquet format detected - packing not supported for parquet")
            logger.warning("Falling back to standard ParquetDataLoader")

        from src.distillation.dataset import ParquetDataLoader

        train_dataset = ParquetDataLoader(
            data_path=config.train_data_path,
            ram_cache_mb=config.ram_cache_mb,
            max_length=config.max_seq_length  # Truncate to prevent OOM
        )
        logger.info(f"Training dataset loaded: {len(train_dataset)} examples")

        val_dataset = None
        if Path(config.val_data_path).exists():
            val_dataset = ParquetDataLoader(
                data_path=config.val_data_path,
                ram_cache_mb=config.ram_cache_mb,
                max_length=config.max_seq_length  # Truncate to prevent OOM
            )
            logger.info(f"Validation dataset loaded: {len(val_dataset)} examples")
        else:
            logger.warning(f"Validation data not found: {config.val_data_path}")
    else:
        # JSONL/text format
        if config.use_packed_sequences:
            logger.info("Detected JSONL/text format - using PackedDataLoader")
            logger.info(f"  Pack target: {config.pack_max_length} tokens")
            train_dataset = PackedDataLoader(
                data_path=config.train_data_path,
                max_length=config.max_seq_length,
                pack_max_length=config.pack_max_length,
                tokenizer=tokenizer,
            )
            logger.info(f"Training dataset loaded: {len(train_dataset)} packed sequences")

            # Show packing stats
            stats = train_dataset.get_packing_stats()
            logger.info(f"  Packing efficiency: {stats['packing_efficiency']:.1f}%")
            logger.info(f"  Avg docs per pack: {stats['avg_docs_per_pack']:.2f}")
            logger.info(f"  Avg tokens per pack: {stats['avg_tokens_per_pack']:.1f}")

            val_dataset = None
            if Path(config.val_data_path).exists():
                val_dataset = PackedDataLoader(
                    data_path=config.val_data_path,
                    max_length=config.max_seq_length,
                    pack_max_length=config.pack_max_length,
                    tokenizer=tokenizer,
                )
                logger.info(f"Validation dataset loaded: {len(val_dataset)} packed sequences")
            else:
                logger.warning(f"Validation data not found: {config.val_data_path}")
        else:
            logger.info("Detected JSONL/text format - using SimpleDataLoader")
            train_dataset = SimpleDataLoader(
                data_path=config.train_data_path,
                max_length=config.max_seq_length,
                tokenizer=tokenizer,
            )
            logger.info(f"Training dataset loaded: {len(train_dataset)} examples")

            val_dataset = None
            if Path(config.val_data_path).exists():
                val_dataset = SimpleDataLoader(
                    data_path=config.val_data_path,
                    max_length=config.max_seq_length,
                    tokenizer=tokenizer,
                )
                logger.info(f"Validation dataset loaded: {len(val_dataset)} examples")
            else:
                logger.warning(f"Validation data not found: {config.val_data_path}")

    # Create dataloaders based on streaming mode
    if config.use_streaming_loader:
        logger.info("Using streaming dataloaders (memory-efficient mode)")
        logger.info(f"  Streaming workers: {config.streaming_num_workers}")
        logger.info(f"  Prefetch factor: {config.streaming_prefetch_factor}")

        train_loader, val_loader = create_streaming_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=config.physical_batch_size,
            num_workers=config.streaming_num_workers,
            pin_memory=True,
            prefetch_factor=config.streaming_prefetch_factor,
            shuffle_train=True,
            drop_last=True,
        )
    else:
        logger.info("Using standard dataloaders (default mode)")

        # FIX #3: Reduce workers for pretokenized data to minimize RAM usage
        # Pretokenized data doesn't need tokenization, so fewer workers suffice
        # With 4 workers × 8.6GB dataset + prefetch buffers → ~113GB RAM (TOO MUCH!)
        # Reduce to 1 worker for pretokenized, keep original for raw text
        effective_num_workers = config.num_workers
        if is_pretokenized_manifest:
            effective_num_workers = 1  # Single worker for pretokenized (no CPU bottleneck)
            logger.info("  FIX: Reducing workers from {} to 1 for pretokenized data (saves ~100GB RAM)".format(config.num_workers))
        logger.info(f"  Workers: {effective_num_workers}")

        # Only set prefetch_factor if num_workers > 0
        train_loader_kwargs = {
            "batch_size": config.physical_batch_size,
            "shuffle": True,
            "num_workers": effective_num_workers,
            "collate_fn": train_dataset.collate_fn,
            "pin_memory": True,
        }
        if effective_num_workers > 0:
            train_loader_kwargs["prefetch_factor"] = 2

        train_loader = DataLoader(train_dataset, **train_loader_kwargs)

        val_loader = None
        if val_dataset is not None:
            val_loader_kwargs = {
                "batch_size": config.physical_batch_size,
                "shuffle": False,
                "num_workers": effective_num_workers,
                "collate_fn": val_dataset.collate_fn,
                "pin_memory": True,
            }
            if effective_num_workers > 0:
                val_loader_kwargs["prefetch_factor"] = 2

            val_loader = DataLoader(val_dataset, **val_loader_kwargs)

    # Wrap with GPU prefetch loader if enabled
    if config.use_prefetch_loader:
        logger.info("Wrapping dataloaders with PrefetchDataLoader for GPU prefetching")
        train_loader = PrefetchDataLoader(train_loader, device=config.device)
        if val_loader is not None:
            val_loader = PrefetchDataLoader(val_loader, device=config.device)

    logger.info(f"Data loaders created successfully")
    return train_loader, val_loader


def create_teacher_client(config: TrainingConfig, tokenizer: Optional[AutoTokenizer] = None):
    """
    Factory function for teacher client based on mode.

    Supports 4 execution paths:
    1. Direct (no cache): Load teacher in memory, generate logits on-the-fly
    2. Direct + Cache: Load teacher, generate logits AND save to disk
    3. Cached Only: Read from pre-cached logits
    4. Network (vLLM): Query remote vLLM server

    Args:
        config: Training configuration
        tokenizer: Pre-loaded tokenizer (optional, used for DirectTeacherClient consistency)

    Returns:
        Initialized teacher client

    Raises:
        ValueError: If invalid teacher_mode
        RuntimeError: If teacher initialization fails
    """
    logger.info("=" * 80)
    logger.info("TEACHER CLIENT INITIALIZATION")
    logger.info("=" * 80)
    logger.info(f"Teacher mode: {config.teacher_mode}")

    if config.teacher_mode == "direct":
        # Direct mode: Load teacher model locally
        logger.info("Mode: DIRECT (local in-process inference)")
        logger.info(f"  Model: {config.teacher_model}")
        logger.info(f"  Device: {config.teacher_device}")
        logger.info(f"  Top-k: {config.teacher_topk}")
        logger.info(f"  Cache logits: {config.cache_logits}")

        # Device isolation warning
        if config.teacher_device == config.device and config.async_teacher and not config.force_async_teacher:
            logger.warning("⚠️  WARNING: Teacher and student on same device with async enabled")
            logger.warning(f"   Both models on: {config.device}")
            logger.warning("   This may cause 8-10GB memory overhead with no performance benefit")
            logger.warning("   Consider: (1) using different devices, or (2) disabling async_teacher")

        teacher = DirectTeacherClient(
            model_name=config.teacher_model,
            device=config.teacher_device,
            # Use fp32 for teacher to avoid BF16 rounding that can make top-k mass exceed 1.0
            torch_dtype=torch.float32,
            topk=config.teacher_topk,
            use_flash_attention=True,  # Enable Flash Attention 2 for KV cache optimization (~2GB savings)
            adapter_path=config.teacher_adapter_path,  # Optional PEFT adapter
            tokenizer=tokenizer,  # FIX: Pass shared tokenizer to ensure consistency
        )

        # Health check
        if not teacher.health_check():
            raise RuntimeError("DirectTeacherClient health check failed")

        logger.info(f"DirectTeacherClient initialized on {config.teacher_device} and health check passed")

        # Wrap with caching if requested
        if config.cache_logits:
            logger.info(f"Wrapping with CachingTeacherWrapper: {config.cache_dir}")
            teacher = CachingTeacherWrapper(
                teacher_client=teacher,
                cache_dir=config.cache_dir,
                shard_size=1000,
                auto_flush=True,
            )
            logger.info("Caching enabled - logits will be saved while training")

    elif config.teacher_mode == "cached":
        # Cached mode: Read from pre-cached logits
        logger.info("Mode: CACHED (read from disk)")
        logger.info(f"  Cache dir: {config.cache_dir}")

        teacher = CachedTeacherClient(
            cache_dir=config.cache_dir,
            fallback_url=None,  # No fallback - pure cached mode
            max_loaded_shards=10,
        )

        logger.info("CachedTeacherClient initialized")
        logger.info(f"  Cached sequences: {len(teacher.cache_index)}")

    elif config.teacher_mode == "network":
        # Network mode: Query remote vLLM server
        logger.info("Mode: NETWORK (vLLM server)")
        logger.info(f"  URL: {config.teacher_url}")
        logger.info(f"  Model: {config.teacher_model}")
        logger.info(f"  Cache logits: {config.cache_logits}")

        teacher = VLLMTeacherClient(
            base_url=config.teacher_url,
            model=config.teacher_model,
            timeout=config.teacher_timeout,
            max_retries=config.teacher_max_retries,
        )

        # Health check
        if not teacher.health_check():
            raise RuntimeError(
                f"Teacher server not reachable at {config.teacher_url}. "
                "Please ensure vLLM server is running."
            )

        logger.info("VLLMTeacherClient initialized and health check passed")

        # Get model info
        try:
            model_info = teacher.get_model_info()
            logger.info(f"Teacher model info: {model_info}")
        except Exception as e:
            logger.warning(f"Failed to get teacher model info: {e}")

        # Wrap with caching if requested
        if config.cache_logits:
            logger.info(f"Wrapping with CachingTeacherWrapper: {config.cache_dir}")
            teacher = CachingTeacherWrapper(
                teacher_client=teacher,
                cache_dir=config.cache_dir,
                shard_size=1000,
                auto_flush=True,
            )
            logger.info("Caching enabled - logits will be saved while training")

    else:
        raise ValueError(
            f"Invalid teacher_mode: {config.teacher_mode}. "
            "Must be 'direct', 'cached', or 'network'"
        )

    logger.info("=" * 80)

    # FIX #1: Auto-disable async for same-device teachers (memory optimization)
    # Async mode causes 8-10GB memory overhead when teacher and student run on same GPU
    if config.async_teacher:
        # Check if teacher is on same device as student (training device)
        teacher_device = getattr(teacher, 'device', None)
        training_device = config.device
        force_async = getattr(config, 'force_async_teacher', False)

        # Normalize device strings for comparison (cuda == cuda:0)
        def normalize_device(dev):
            if dev is None:
                return None
            dev_str = str(dev)
            # "cuda" is equivalent to "cuda:0"
            if dev_str == "cuda":
                return "cuda:0"
            return dev_str

        teacher_dev_normalized = normalize_device(teacher_device)
        training_dev_normalized = normalize_device(training_device)
        devices_match = teacher_dev_normalized == training_dev_normalized

        logger.info(f"Async teacher check:")
        logger.info(f"  Teacher device: {teacher_device} → {teacher_dev_normalized}")
        logger.info(f"  Training device: {training_device} → {training_dev_normalized}")
        logger.info(f"  Devices match: {devices_match}")
        logger.info(f"  Force async: {force_async}")

        # Auto-disable if same device (unless force_async_teacher is set)
        if devices_match and not force_async:
            logger.warning("=" * 80)
            logger.warning("🚨 ASYNC TEACHER AUTO-DISABLED 🚨")
            logger.warning(f"Teacher and student are both on {training_dev_normalized}.")
            logger.warning("Async mode causes 8-10GB memory overhead with no performance benefit")
            logger.warning("when both models run on the same GPU.")
            logger.warning("")
            logger.warning("To override: Set 'force_async_teacher: true' in config")
            logger.warning("To fix properly: Use different GPUs (teacher_device: cuda:1)")
            logger.warning("=" * 80)
            config.async_teacher = False
            logger.info(f"config.async_teacher set to: {config.async_teacher}")

    # Wrap with AsyncTeacherClient if enabled (AFTER caching wrapper and auto-disable check)
    # logger.info(f"DEBUG: Final config.async_teacher = {config.async_teacher}")
    if config.async_teacher:
        from distillation.async_teacher import AsyncTeacherClient
        logger.info("Wrapping teacher client with AsyncTeacherClient for async prefetch")
        teacher = AsyncTeacherClient(
            teacher_client=teacher,
            max_queue_depth=4,
            max_workers=1,
        )
        logger.info("Async teacher prefetch enabled")

    return teacher


def create_trainer(
    model: nn.Module,
    config: TrainingConfig,
    student_config: RetNetStudentConfig,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    tokenizer: AutoTokenizer,
    output_dir: Path,
    teacher_client: Optional[Any] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> Tuple[DistillationTrainer, int]:
    """Create distillation trainer with all components.

    Args:
        model: Student model
        config: Training configuration
        student_config: Student model configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        tokenizer: Tokenizer
        output_dir: Output directory
        teacher_client: Pre-initialized teacher client (optional)
        optimizer: Pre-initialized optimizer with custom param groups (optional)
        scheduler: Pre-initialized LR scheduler (optional)

    Returns:
        Tuple of (initialized trainer, warmup_steps_calculated)
    """
    logger.info("Creating distillation trainer...")

    # Convert TrainingConfig to legacy TrainingConfig for trainer
    # PHASE 1B FIX: Convert warmup_batches to warmup_steps by dividing by gradient_accumulation_steps
    # This ensures proper warmup duration in optimizer steps instead of batches
    warmup_steps_calculated = max(1, config.warmup_batches // config.gradient_accumulation_steps)
    logger.info(f"PHASE 1B: Converting warmup_batches={config.warmup_batches} to warmup_steps={warmup_steps_calculated} "
                f"(gradient_accumulation_steps={config.gradient_accumulation_steps})")

    legacy_config = LegacyTrainingConfig(
        physical_batch_size=config.physical_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_grad_norm=config.max_grad_norm,
        optimizer_type=config.optimizer_type,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=warmup_steps_calculated,  # PHASE 1B: Now uses properly converted warmup steps
        max_steps=config.max_steps,
        use_bf16=config.use_bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        log_interval=config.log_interval,
        eval_interval=config.eval_interval,
        save_interval=config.save_interval,
        teacher_endpoint=f"{config.teacher_url}/v1/completions",
        teacher_topk=config.teacher_topk,
        teacher_temperature=config.teacher_temperature,
        distill_alpha=config.distill_alpha,
        log_memory_debug=config.log_memory_debug,  # FIX #1: Pass memory debug flag
        muon_aux_lr_scale=config.muon_aux_lr_scale,
        muon_ready_check=config.muon_ready_check,
        muon_clip_threshold=config.muon_clip_threshold,
        muon_clip_alpha=config.muon_clip_alpha,
        muon_clip_pairs=config.muon_clip_pairs,
    )

    # Create trainer with optimizer and scheduler
    trainer = DistillationTrainer(
        model=model,
        config=legacy_config,
        student_config=student_config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        checkpoint_dir=output_dir / "checkpoints",
        tokenizer=tokenizer,
        enable_full_eval=config.eval_perplexity or config.eval_niah,
        teacher_client=teacher_client,
        optimizer=optimizer,  # Pass shared optimizer with special param groups
        scheduler=scheduler,  # Pass LR scheduler for warm restarts
        pretrain_ce_only=config.pretrain_ce_only,  # Pass CE pretrain flag explicitly
    )

    logger.info("Trainer created successfully")

    return trainer, warmup_steps_calculated


def main():
    """Main training entry point."""
    # Parse CLI arguments
    args = parse_cli_args()

    # Create configuration
    config = create_config_from_args(args)

    # Enable TF32 for faster matmul operations on Ampere+ GPUs (RTX 30xx/40xx/50xx)
    # This gives ~2× speedup on FP32 operations with minimal accuracy impact
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logging.info("✓ TF32 enabled for matmul and cudnn operations")
        # FIX #2: Log CUDA allocator configuration
        alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "default")
        logging.info(f"🔧 CUDA allocator configured: {alloc_conf}")

    # Setup output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup file logging
    log_file = output_dir / "train.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 80)
    logger.info("KNOWLEDGE DISTILLATION TRAINING")
    logger.info("=" * 80)

    # Print configuration
    print_config(config)

    # Dry run mode
    if args.dry_run:
        logger.info("Dry run mode: validating configuration and exiting")
        validate_config(config, skip_data_checks=True)
        logger.info("✅ Configuration is valid (dry-run mode: data paths not checked)")
        return

    # Validate configuration
    validate_config(config)

    # Save configuration
    config_file = output_dir / "config.yaml"
    save_config(config, config_file)

    # Initialize wandb if enabled (separate from telemetry for custom batching)
    wandb_initialized = False
    if config.enable_wandb:
        print("=" * 80)
        print("WANDB INITIALIZATION")
        print("=" * 80)
        if not HAS_WANDB:
            logger.warning("Wandb requested but not installed. Wandb logging will be disabled.")
            print("WARNING: Wandb not installed - logging disabled")
        elif not config.wandb_project:
            logger.error("Wandb enabled but wandb_project not specified. Wandb logging will be disabled.")
            print("ERROR: Wandb project not specified - logging disabled")
        else:
            try:
                print(f"Initializing wandb...")
                print(f"  Project: {config.wandb_project}")
                print(f"  Run name: {config.wandb_run_name}")
                print(f"  Mode: {'offline' if config.wandb_offline else 'online'}")
                wandb.init(
                    project=config.wandb_project,
                    name=config.wandb_run_name,
                    config=config.to_dict(),
                    mode="offline" if config.wandb_offline else "online",
                )
                wandb_initialized = True
                logger.info(f"Wandb initialized: project={config.wandb_project}, name={config.wandb_run_name}")
                logger.info(f"  Mode: {'offline' if config.wandb_offline else 'online'}")
                logger.info(f"  Metrics will be collected every 100 batches and committed every 10 optimizer steps")
                print("SUCCESS: Wandb initialized successfully!")
                print("  Metrics will be collected every 100 batches")
                print("  Metrics will be committed every 10 optimizer steps (2560 batches with gradient_accumulation=256)")
            except Exception as e:
                logger.error(f"Failed to initialize wandb: {e}")
                logger.warning("Continuing without wandb logging")
                print(f"ERROR: Failed to initialize wandb: {e}")
        print("=" * 80)
    else:
        print("=" * 80)
        print("WANDB DISABLED")
        print("=" * 80)
        print("Wandb logging is disabled in config (enable_wandb=False)")
        print("To enable: Set enable_wandb=true in config file or pass --enable-wandb flag")
        print("=" * 80)

    # Set random seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    logger.info(f"Random seed set to: {config.seed}")

    # Setup device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = load_llama_tokenizer(
        model_name=config.tokenizer_name,
        use_fast=True,
        adapter_path=config.teacher_adapter_path,
    )

    # Create student config
    # Handle TitanMAC variants (which don't use RetNetStudentConfig)
    if config.model_variant.startswith("titan_mac"):
        student_config = None  # Titan models handle their own config
    else:
        student_config = create_student_config(
            variant=config.model_variant,
            max_seq_length=config.max_seq_length  # Pass configured seq length
        )

    # Load student model (RetNet)
    model = load_student_model(
        student_config=student_config,
        device=device,
        use_bf16=config.use_bf16,
        gradient_checkpointing=config.gradient_checkpointing,  # FIX #4
    )

    # Initialize teacher client (skip in CE pretrain mode - T010, T011, T014)
    if config.pretrain_ce_only:
        logger.info("=" * 80)
        logger.info("CE PRETRAIN MODE ENABLED")
        logger.info("=" * 80)
        logger.info("Training with pure cross-entropy loss (no teacher model)")

        # T010: Warn about ignored teacher settings
        teacher_settings = ['teacher_model', 'teacher_url', 'teacher_topk',
                            'teacher_temperature', 'teacher_adapter_path']
        present_settings = [s for s in teacher_settings
                           if getattr(config, s, None) and s != 'teacher_url']  # teacher_url always has default
        if present_settings:
            logger.warning(f"⚠️  CE pretrain mode: ignoring teacher settings: {present_settings}")

        # T011: Skip teacher initialization
        teacher_client = None
        logger.info("Teacher client: None (CE pretrain mode)")
    else:
        teacher_client = create_teacher_client(config)

    # Create data loaders
    train_loader, val_loader = create_data_loaders(config, tokenizer)

    # Create optimizer with proper parameter groups (boosted LR for retention decay params)
    logger.info("Creating optimizer...")
    optimizer = create_optimizer(
        model=model,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        use_8bit=config.use_8bit_optimizer,
        decay_param_lr_multiplier=1.0,  # REDUCED from 10.0 to prevent instability with learnable decay
        optimizer_type=config.optimizer_type,
        muon_momentum=config.muon_momentum,
        muon_grad_clip=config.muon_grad_clip,
        muon_zero_clip_percent=config.muon_zero_clip_percent,
        muon_aux_lr_scale=config.muon_aux_lr_scale,
        muon_ready_check=config.muon_ready_check,
    )

    # Create scheduler with warmup, plateau, and cosine annealing
    logger.info("Creating learning rate scheduler...")
    # PHASE 1B FIX: Convert warmup_batches to warmup_steps by dividing by gradient_accumulation_steps
    warmup_steps_calculated = max(1, config.warmup_batches // config.gradient_accumulation_steps)
    plateau_steps_calculated = max(0, config.plateau_batches // config.gradient_accumulation_steps)
    logger.info(f"Scheduler warmup: {warmup_steps_calculated} steps (from {config.warmup_batches} batches / {config.gradient_accumulation_steps} accumulation)")
    logger.info(f"Scheduler plateau: {plateau_steps_calculated} steps (from {config.plateau_batches} batches / {config.gradient_accumulation_steps} accumulation)")

    scheduler = create_scheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps_calculated,  # PHASE 1B: Properly converted warmup steps
        plateau_steps=plateau_steps_calculated,  # Plateau phase: hold at max LR
        T_0=config.cosine_t0,
        T_mult=config.cosine_tmult,
        eta_min=config.min_lr,
    )

    # Initialize saddle escape system
    saddle_manager = None
    if config.saddle_escape.enabled:
        from distillation.saddle_escape import SaddleEscapeManager

        saddle_manager = SaddleEscapeManager(config.saddle_escape)
        logger.info("✅ Saddle escape system enabled")
        if not config.saddle_escape.interventions_enabled:
            logger.info("   Mode: DETECTION ONLY (interventions disabled)")
        else:
            logger.info("   Mode: DETECTION + INTERVENTIONS")

    # Create telemetry logger (without wandb sink - we handle wandb separately for batched commits)
    logger.info("Creating telemetry logger...")
    telemetry_sinks = [OutputSink.FILE, OutputSink.CONSOLE]
    # Note: We don't add OutputSink.WANDB here because we handle wandb separately
    # with custom batching (collect every 100 batches, commit every 400 optimizer steps)

    telemetry = TelemetryLogger(
        log_dir=output_dir / "logs",
        log_interval=config.log_interval,
        sinks=telemetry_sinks,
        enable_wandb=False,  # Disable wandb in telemetry (we handle it separately)
        wandb_project=None,
        wandb_run_name=None,
        wandb_config=None,
        wandb_offline=False,
    )

    # Create checkpoint manager
    logger.info("Creating checkpoint manager...")
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=Path(config.checkpoint_dir),
        keep_last_n=config.keep_last_n,
        save_best=True,
        max_total_size_gb=config.max_total_size_gb,
    )

    # Initialize training state
    state_file = output_dir / "training_state.json"
    training_state = TrainingState(state_file)

    # Create trainer with optimizer and scheduler
    # PHASE 1B: Now returns warmup_steps_calculated for use in scheduler and logging
    trainer, warmup_steps_for_trainer = create_trainer(
        model=model,
        config=config,
        student_config=student_config,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        output_dir=output_dir,
        teacher_client=teacher_client,
        optimizer=optimizer,  # Pass optimizer with 50x LR for retention decay params
        scheduler=scheduler,  # Pass scheduler for warm restarts
    )

    # Auto-resume from latest checkpoint if available
    resume_path = None
    if config.resume_from:
        try:
            resume_path = _resolve_checkpoint_path(Path(config.resume_from))
            logger.info(f"Resuming from specified checkpoint: {resume_path}")
        except Exception as e:
            logger.error(f"Unable to resolve --resume path '{config.resume_from}': {e}")
            raise
    else:
        logger.info("No --resume flag specified; starting fresh training run.")

    # Load checkpoint if resume path was determined
    if resume_path:
        try:
            state_dict = checkpoint_manager.load_checkpoint(
                checkpoint_path=resume_path,
                device=device,
            )
            trainer.load_state_dict(state_dict)

            # CE pretrain mode transition logging (T005)
            checkpoint_config = state_dict.get('config', {})
            checkpoint_ce_mode = checkpoint_config.get('pretrain_ce_only', False)
            if checkpoint_ce_mode and not config.pretrain_ce_only:
                logger.info("🔄 Resuming from CE pretrain checkpoint in KD mode")
            elif not checkpoint_ce_mode and config.pretrain_ce_only:
                logger.info("🔄 Resuming from KD checkpoint in CE pretrain mode")

            # Tokenizer consistency warning (T006)
            checkpoint_tokenizer = checkpoint_config.get('tokenizer_name')
            if checkpoint_tokenizer and checkpoint_tokenizer != config.tokenizer_name:
                logger.warning(
                    f"⚠️ Tokenizer mismatch: checkpoint={checkpoint_tokenizer}, "
                    f"config={config.tokenizer_name}"
                )

            # Update training state
            training_state.update(
                global_step=trainer.global_step,
                epoch=trainer.epoch,
                best_val_loss=trainer.best_val_loss,
                last_save_step=trainer.global_step,  # Prevent immediate duplicate save on resume
            )

            logger.info(f"✅ Resumed from step {trainer.global_step}, epoch {trainer.epoch}")
        except Exception as e:
            logger.error(f"Failed to resume from checkpoint: {e}")
            raise
    else:
        logger.info("No checkpoint found - starting fresh training")

    # Setup signal handlers for graceful shutdown
    setup_signal_handlers(trainer, checkpoint_manager)

    # Training loop
    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)

    training_start_time = time.time()
    accumulation_step = 0  # Counter for gradient accumulation
    total_batches = 0  # Total batch counter for progress tracking

    # Wandb metrics buffer for batched commits
    # Strategy: Collect metrics every 100 batches, commit to wandb every 400 optimizer steps
    wandb_metrics_buffer = {}
    warmup_completed_logged = False  # Track if we've logged warmup completion

    # Track diversity losses across gradient accumulation steps
    diversity_losses_buffer = []  # Accumulate diversity losses across batches

    try:
        # Main training loop
        while trainer.global_step < config.max_steps:
            # Run one epoch
            for batch in train_loader:
                # Training step
                step_start = time.time()
                metrics = trainer._train_step(batch, accumulation_step)
                step_time = time.time() - step_start

                # Accumulate diversity loss if present
                if 'diversity_loss' in metrics:
                    diversity_losses_buffer.append(metrics['diversity_loss'])

                # Increment counters
                accumulation_step += 1
                total_batches += 1

                # Show batch progress every 50 batches
                if total_batches % 50 == 0:
                    # Convert loss to scalar for display (if it's a tensor)
                    batch_loss = metrics['loss'].item() if torch.is_tensor(metrics['loss']) else metrics['loss']
                    logger.info(
                        f"Batch {total_batches} | Accumulation: {accumulation_step}/{config.gradient_accumulation_steps} | "
                        f"Loss: {batch_loss:.4f}"
                    )

                # Optimizer step (after gradient accumulation)
                if accumulation_step >= config.gradient_accumulation_steps:
                    grad_norm = trainer._optimizer_step()

                    # Increment global step counter (counts optimizer steps, not batches)
                    trainer.global_step += 1

                    # Reset accumulation counter
                    accumulation_step = 0

                    # Compute average diversity loss across accumulation steps (BEFORE clearing buffer)
                    avg_diversity_loss = None
                    if diversity_losses_buffer:
                        # Convert tensors to scalars and average
                        div_losses_scalar = [d.item() if torch.is_tensor(d) else d for d in diversity_losses_buffer]
                        avg_diversity_loss = sum(div_losses_scalar) / len(div_losses_scalar)
                        # Clear buffer for next accumulation window
                        diversity_losses_buffer.clear()

                    # Get actual learning rates from trainer's optimizer parameter groups
                    # Bug fix: Previously used a dummy scheduler that was never connected to the real optimizer
                    lr_per_group = {}
                    for i, param_group in enumerate(trainer.optimizer.param_groups):
                        group_name = param_group.get('name', f'group_{i}')
                        lr_per_group[group_name] = param_group['lr']

                    # Base LR for logging (use first param group as reference)
                    current_lr = trainer.optimizer.param_groups[0]['lr']

                    # Check if warmup just completed
                    if trainer.global_step == warmup_steps_for_trainer:
                        logger.info("=" * 80)
                        logger.info("====== WARMUP FINISHED ======")
                        logger.info(f"Warmup completed at batch {trainer.total_batches}")
                        logger.info(f"Learning rate now at full value: {current_lr:.2e}")
                        logger.info(f"  LR breakdown by param group: {lr_per_group}")
                        logger.info("=" * 80)

                        # Log warmup completion to wandb (once)
                        if wandb_initialized and not warmup_completed_logged:
                            wandb_metrics_buffer.update({
                                "warmup/completed_at_batch": trainer.total_batches,
                                "warmup/completed_at_step": trainer.global_step,
                            })
                            warmup_completed_logged = True
                            logger.info(f"Logged warmup completion to wandb buffer")

                    # Convert tensor metrics to scalars for logging
                    # FIX: metrics['loss'] is a tensor from _train_step, need .item() for JSON serialization
                    loss_scalar = metrics['loss'].item() if torch.is_tensor(metrics['loss']) else metrics['loss']
                    grad_norm_scalar = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm

                    # Saddle escape check
                    saddle_intervention = None
                    if saddle_manager:
                        saddle_intervention = saddle_manager.check_and_intervene(
                            grad_norm=grad_norm_scalar,
                            loss=loss_scalar,
                            step=trainer.global_step,
                            optimizer=trainer.optimizer,
                            scheduler=trainer.scheduler,
                        )

                    # Always log optimizer steps
                    log_msg = (
                        f"[Optimizer-step {trainer.global_step}] Loss: {loss_scalar:.4f} | "
                        f"LR: {current_lr:.2e} | Grad Norm: {grad_norm_scalar:.4f}"
                    )
                    if avg_diversity_loss is not None:
                        log_msg += f" | Div loss: {avg_diversity_loss:.6f}"
                    logger.info(log_msg)

                    # Add optimizer-step-level metrics to wandb buffer
                    if wandb_initialized:
                        # logger.info(f"DEBUG: Adding optimizer metrics at step {trainer.global_step}")

                        # Calculate GPU memory (Bug #4 fix: Track both allocated and peak)
                        gpu_mem_allocated_gb = 0.0
                        gpu_mem_peak_gb = 0.0
                        if torch.cuda.is_available():
                            gpu_mem_allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                            gpu_mem_peak_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)

                        # Calculate throughput metrics
                        tokens_per_sec = (config.max_seq_length * config.physical_batch_size) / step_time if step_time > 0 else 0.0
                        batches_per_sec = 1.0 / step_time if step_time > 0 else 0.0

                        # Base metrics (convert all tensors to scalars)
                        wandb_metrics_buffer.update({
                            "train/loss": loss_scalar,
                            "train/learning_rate": current_lr,  # Base LR for backwards compatibility
                            "train/grad_norm": grad_norm_scalar,
                            "train/optimizer_step": trainer.global_step,
                            "train/batch_count": total_batches,
                            "train/epoch": trainer.epoch,
                            # Performance metrics (now logged every optimizer step for consistent visualization)
                            "performance/step_time": step_time,
                            "performance/tokens_per_sec": tokens_per_sec,
                            "performance/batches_per_sec": batches_per_sec,
                            "performance/gpu_memory_allocated_gb": gpu_mem_allocated_gb,
                            "performance/gpu_memory_peak_gb": gpu_mem_peak_gb,
                        })

                        # Per-group learning rates (NEW: Fix for Bug #1)
                        for group_name, lr_value in lr_per_group.items():
                            wandb_metrics_buffer[f"train/lr_{group_name}"] = lr_value

                        # Add diversity loss if present (diversity regularization)
                        if avg_diversity_loss is not None:
                            wandb_metrics_buffer["train/diversity_loss"] = avg_diversity_loss

                        # Add saddle escape metrics
                        if saddle_manager and config.saddle_escape.log_to_wandb:
                            detection = saddle_intervention.get('detection', {}) if saddle_intervention else {}

                            # Add saddle metrics to buffer
                            wandb_metrics_buffer.update({
                                'saddle/is_stuck': int(detection.get('is_stuck', False)),
                                'saddle/stuck_counter': detection.get('stuck_counter', 0),
                                'saddle/loss_improvement': detection.get('loss_improvement', 0.0),
                                'saddle/grad_norm_below_threshold': int(detection.get('grad_norm_low', False)),
                                'saddle/intervention_active': int(saddle_manager.current_intervention is not None),
                                'saddle/total_interventions': saddle_manager.total_interventions,
                            })

                            # Log intervention events
                            if saddle_intervention and saddle_intervention.get('action') != 'none':
                                wandb_metrics_buffer.update({
                                    'saddle/intervention_type': saddle_intervention['action'],
                                    'saddle/intervention_number': saddle_intervention.get('intervention_number', 0),
                                    'saddle/lr_boost_factor': saddle_intervention.get('factor', 0),
                                })

                                logger.warning(
                                    f"📊 Wandb: Logged saddle intervention #{saddle_intervention['intervention_number']}"
                                )

                        # logger.info(f"DEBUG: Buffer after optimizer metrics: size={len(wandb_metrics_buffer)}, keys={list(wandb_metrics_buffer.keys())}")

                    # Commit to wandb every optimizer step
                    if wandb_initialized and trainer.global_step > 0:
                        # logger.info(f"DEBUG: Commit condition met! Step={trainer.global_step}, buffer_size={len(wandb_metrics_buffer)}")
                        if len(wandb_metrics_buffer) > 0:
                            try:
                                # logger.info(f"DEBUG: About to commit these metrics to wandb: {list(wandb_metrics_buffer.keys())}")
                                wandb.log(wandb_metrics_buffer, step=trainer.global_step)
                                # logger.info(f"DEBUG: Commit successful! Committed {len(wandb_metrics_buffer)} metrics to wandb at step {trainer.global_step}")
                                logger.info(f"Committed {len(wandb_metrics_buffer)} metrics to wandb at step {trainer.global_step}")
                                wandb_metrics_buffer.clear()
                            except Exception as e:
                                logger.error(f"Failed to log to wandb: {e}")
                                # logger.error(f"DEBUG: Exception details: {type(e).__name__}: {str(e)}")
                        else:
                            # logger.warning(f"DEBUG: Commit condition met but buffer is empty at step {trainer.global_step}")
                            pass

                    # Detailed telemetry logging at intervals
                    if telemetry.should_log(trainer.global_step):
                        # Build telemetry kwargs with optional diversity loss
                        # FIX: Use scalar values (already converted above) for JSON serialization
                        telemetry_kwargs = {
                            'step': trainer.global_step,
                            'epoch': trainer.epoch,
                            'loss': loss_scalar,  # Already converted to scalar above
                            'learning_rate': current_lr,  # Use actual LR from trainer's optimizer
                            'grad_norm': grad_norm_scalar,  # Already converted to scalar above
                            'num_tokens': config.max_seq_length * config.physical_batch_size,
                            'batch_size': config.physical_batch_size,
                            'step_time': step_time,
                        }
                        # Add diversity loss if present
                        if avg_diversity_loss is not None:
                            telemetry_kwargs['diversity_loss'] = avg_diversity_loss

                        telemetry.log_step(**telemetry_kwargs)

                    # Update training state
                    training_state.update(
                        global_step=trainer.global_step,
                        epoch=trainer.epoch,
                    )

                # Evaluation
                if trainer.global_step % config.eval_interval == 0 and trainer.global_step > 0:
                    # Prevent duplicate evaluations on the same step (observed repeated evals)
                    if training_state.get('last_eval_step', 0) == trainer.global_step:
                        logger.info(f"Skipping duplicate evaluation at step {trainer.global_step}")
                        continue

                    logger.info("=" * 80)
                    logger.info(f"Running evaluation at step {trainer.global_step}")
                    logger.info("=" * 80)

                    # Bug #6 fix: Log memory BEFORE evaluation
                    if torch.cuda.is_available():
                        mem_before = torch.cuda.memory_allocated() / (1024 ** 3)
                        mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                        logger.info(f"Memory BEFORE eval: {mem_before:.2f} GB allocated, {mem_reserved:.2f} GB reserved")

                    # Configure evaluation
                    perplexity_config = None
                    niah_config = None

                    if config.eval_perplexity:
                        perplexity_config = PerplexityConfig(
                            max_samples=config.eval_perplexity_samples
                        )

                    # T027: Skip NIAH in CE pretrain mode (requires teacher logits)
                    if config.eval_niah and not config.pretrain_ce_only:
                        niah_config = NIAHConfig(
                            context_length=config.max_seq_length,
                            num_samples=config.eval_niah_samples,
                        )
                    elif config.eval_niah and config.pretrain_ce_only:
                        if trainer.global_step == config.eval_interval:  # Log only on first eval
                            logger.info("ℹ️  NIAH evaluation skipped in CE pretrain mode (requires teacher)")

                    # Run evaluation (T026, T027: handle CE pretrain mode)
                    eval_results = trainer.evaluation_runner.run_all(
                        val_dataloader=val_loader,
                        perplexity_config=perplexity_config,
                        niah_config=niah_config,
                        run_perplexity=config.eval_perplexity,
                        run_niah=config.eval_niah and not config.pretrain_ce_only,  # T027: Skip NIAH in CE mode
                        output_dir=output_dir / "eval",
                        step=trainer.global_step,
                    )

                    # Bug #6 fix: Log memory AFTER evaluation
                    if torch.cuda.is_available():
                        mem_after = torch.cuda.memory_allocated() / (1024 ** 3)
                        mem_reserved_after = torch.cuda.memory_reserved() / (1024 ** 3)
                        mem_free = (mem_reserved_after - mem_after)
                        logger.info(f"Memory AFTER eval: {mem_after:.2f} GB allocated, {mem_reserved_after:.2f} GB reserved, {mem_free:.2f} GB free")

                        # Check if we're at risk of OOM
                        if mem_free < 3.0:  # Less than 3GB free
                            logger.warning(f"WARNING: Low memory after eval! Only {mem_free:.2f} GB free. Training may OOM on next step.")

                    # Update best validation loss
                    if 'perplexity' in eval_results and 'loss' in eval_results['perplexity']:
                        val_loss = eval_results['perplexity']['loss']
                        is_best = val_loss < training_state.get('best_val_loss', float('inf'))

                        if is_best:
                            training_state.update(best_val_loss=val_loss)
                            logger.info(f"New best validation loss: {val_loss:.4f}")
                    else:
                        is_best = False

                    # Log evaluation to telemetry
                    if 'perplexity' in eval_results:
                        # Sanitize eval_results to convert tensors to scalars before logging
                        # This prevents JSON serialization errors in telemetry
                        from distillation.telemetry import _sanitize_for_json
                        eval_results_sanitized = _sanitize_for_json(eval_results)

                        telemetry.log_evaluation(
                            step=trainer.global_step,
                            epoch=trainer.epoch,
                            eval_loss=eval_results_sanitized['perplexity'].get('loss', float('inf')),
                            eval_metrics=eval_results_sanitized,
                        )

                        # Add validation metrics to wandb buffer
                        if wandb_initialized:
                            val_loss = eval_results['perplexity'].get('loss', float('inf'))
                            val_perplexity = eval_results['perplexity'].get('perplexity', float('inf'))
                            wandb_metrics_buffer.update({
                                "val/loss": val_loss,
                                "val/perplexity": val_perplexity,
                            })
                            logger.info(f"Added validation metrics to wandb buffer (loss={val_loss:.4f}, ppl={val_perplexity:.2f})")

                    # Remember that we already evaluated at this step
                    training_state.update(last_eval_step=trainer.global_step)

                # Checkpointing
                if trainer.global_step % config.save_interval == 0 and trainer.global_step > 0:
                    # Prevent repeated saves at the same step when resuming
                    if training_state.get('last_save_step', 0) == trainer.global_step:
                        logger.info(f"Skipping duplicate checkpoint at step {trainer.global_step}")
                        continue

                    logger.info(f"Saving checkpoint at step {trainer.global_step}")

                    # Determine if best checkpoint
                    is_best = False
                    if 'best_val_loss' in training_state.state:
                        is_best = (trainer.best_val_loss <= training_state.get('best_val_loss', float('inf')))

                    # Save checkpoint
                    try:
                        state_dict = trainer.get_state_dict()
                        checkpoint_path = checkpoint_manager.save_checkpoint(
                            state_dict,
                            step=trainer.global_step,
                            is_best=is_best,
                        )

                        if is_best:
                            training_state.update(best_checkpoint_path=str(checkpoint_path))

                        logger.info(f"Checkpoint saved: {checkpoint_path}")
                    except Exception as e:
                        logger.error(f"Failed to save checkpoint: {e}")
                    else:
                        training_state.update(last_save_step=trainer.global_step)

                # Check if training is complete
                if trainer.global_step >= config.max_steps:
                    break

            trainer.epoch += 1

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    finally:
        # Finalize telemetry
        telemetry.finalize()

        # Finish wandb run
        if wandb_initialized:
            try:
                # Commit any remaining metrics in buffer
                if len(wandb_metrics_buffer) > 0:
                    wandb.log(wandb_metrics_buffer, step=trainer.global_step)
                    logger.info(f"Committed final {len(wandb_metrics_buffer)} metrics to wandb")

                wandb.finish()
                logger.info("Wandb run finished successfully")
            except Exception as e:
                logger.error(f"Failed to finish wandb: {e}")

        # Close teacher client (handles both sync and async clients)
        if teacher_client is not None:
            if config.async_teacher and hasattr(teacher_client, 'close'):
                logger.info("Shutting down async teacher client...")
                teacher_client.close()
            elif hasattr(teacher_client, 'close'):
                teacher_client.close()

    # PROFILING: Log profiling results if enabled
    if hasattr(trainer, '_log_profiling_results'):
        trainer._log_profiling_results()

    # Training complete
    training_time = time.time() - training_start_time
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"  Total steps: {trainer.global_step}")
    logger.info(f"  Total epochs: {trainer.epoch}")
    logger.info(f"  Training time: {training_time / 3600:.2f} hours")
    logger.info(f"  Best validation loss: {training_state.get('best_val_loss', 'N/A')}")
    logger.info(f"  Best checkpoint: {training_state.get('best_checkpoint_path', 'N/A')}")
    logger.info("=" * 80)

    # Print telemetry summary
    summary = telemetry.get_summary_stats()
    logger.info("Telemetry summary:")
    logger.info(f"  Total runtime: {summary['total_runtime_sec'] / 3600:.2f} hours")
    logger.info(f"  Avg tokens/sec: {summary['moving_averages']['tokens_per_sec_avg']:.1f}")
    logger.info(f"  Final VRAM: {summary['current_memory']['gpu_memory_allocated_gb']:.2f} GB")


if __name__ == "__main__":
    main()

