#!/usr/bin/env python3
"""
Teacher model finetuning script using PEFT LoRA.

This script finetunes meta-llama/Llama-3.2-1B-Instruct on the pretokenized
distillation corpus using parameter-efficient fine-tuning (LoRA).

Features:
- LoRA injection on all attention and MLP layers
- BF16 mixed precision training
- Gradient checkpointing for memory efficiency
- SFTTrainer for supervised finetuning
- Validation perplexity tracking with early stopping
- Resume capability from checkpoint
- Optional model merging for deployment

Target VRAM: ≤16 GB

Usage:
    # Train with default config
    python scripts/finetune_teacher.py --config configs/teacher_ft.yaml

    # Train and save merged model
    python scripts/finetune_teacher.py --config configs/teacher_ft.yaml --save-merged

    # Resume from checkpoint
    python scripts/finetune_teacher.py --config configs/teacher_ft.yaml --resume-from runs/teacher_ft/checkpoint-1000

    # Use RAM disk for faster I/O (copies 8.7GB to /dev/shm)
    python scripts/finetune_teacher.py --config configs/teacher_ft.yaml --ram-disk

    # Use RAM disk and clean up after training
    python scripts/finetune_teacher.py --config configs/teacher_ft.yaml --ram-disk --ram-disk-cleanup
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import time
import shutil
import subprocess
import random

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
)
from tqdm.auto import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.distillation.dataset import PretokenizedShardDataset


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Finetune teacher model with LoRA")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--save-merged",
        action="store_true",
        help="Save merged model (base + LoRA) for deployment",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from checkpoint directory",
    )
    parser.add_argument(
        "--ram-disk",
        action="store_true",
        help="Copy dataset to RAM disk (/dev/shm) for faster I/O",
    )
    parser.add_argument(
        "--ram-disk-cleanup",
        action="store_true",
        help="Remove data from RAM disk after training (requires --ram-disk)",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from: {config_path}")

    # Normalize config keys to handle both naming conventions
    # Map teacher_model -> base_model or model_name -> base_model
    if 'teacher_model' in config and 'base_model' not in config:
        config['base_model'] = config['teacher_model']
    if 'model_name' in config and 'base_model' not in config:
        config['base_model'] = config['model_name']
    if 'base_model' in config and 'model_name' not in config:
        config['model_name'] = config['base_model']

    # Flatten nested lora_config if present
    if 'lora_config' in config and isinstance(config['lora_config'], dict):
        lora_cfg = config['lora_config']
        config['lora_rank'] = lora_cfg.get('r', lora_cfg.get('lora_rank', 64))
        config['lora_alpha'] = lora_cfg.get('lora_alpha', 128)
        config['lora_dropout'] = lora_cfg.get('lora_dropout', 0.05)
        config['target_modules'] = lora_cfg.get('target_modules', None)

    # Set defaults for missing keys
    config.setdefault('use_bf16', config.get('bf16', True))
    config.setdefault('num_workers', config.get('dataloader_num_workers', 0))
    config.setdefault('lora_rank', 64)
    config.setdefault('lora_alpha', 128)
    config.setdefault('lora_dropout', 0.05)
    config.setdefault('tokenizer_name', config.get('tokenizer') or config.get('base_model'))
    config.setdefault('allow_tokenizer_mismatch', False)

    return config


def setup_model_and_tokenizer(
    model_name: str,
    tokenizer_name: Optional[str] = None,
    use_bf16: bool = True,
    gradient_checkpointing: bool = True,
    device_map: str = "cuda",
    use_flash_attention_2: bool = False,
) -> tuple:
    """Load base model and tokenizer.

    Args:
        model_name: HuggingFace model name
        use_bf16: Use BF16 precision
        gradient_checkpointing: Enable gradient checkpointing
        device_map: Device map for model loading
        use_flash_attention_2: Use Flash Attention 2 (requires flash-attn package)

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info("Loading tokenizer...")
    tokenizer_source = tokenizer_name or model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        use_fast=True,
        trust_remote_code=True,
    )

    # Set pad token if not set or if it's the same as eos_token (Option A fix)
    # This ensures we can mask padding without masking valid EOS tokens
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        # Add new pad token
        num_added = tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        if num_added > 0:
            logger.info(f"Added new pad_token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
        else:
            logger.warning("Could not add pad token despite trying.")

    logger.info(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
    logger.info(f"  bos_token: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")
    logger.info(f"  eos_token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
    logger.info(f"  pad_token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")

    logger.info("Loading base model...")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Device map: {device_map}")
    logger.info(f"  BF16: {use_bf16}")
    logger.info(f"  Gradient checkpointing: {gradient_checkpointing}")
    logger.info(f"  Flash Attention 2: {use_flash_attention_2}")

    # Load model in BF16 if requested
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float32

    # Prepare model loading kwargs
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "trust_remote_code": True,
    }

    # Add Flash Attention 2 if requested
    if use_flash_attention_2:
        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Flash Attention 2 will be used")
        except ImportError:
            logger.warning(
                "Flash Attention 2 requested but flash-attn not installed. "
                "Install with: pip install flash-attn --no-build-isolation"
            )
            logger.warning("Falling back to standard attention")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )

    # Resize embeddings if we added a token
    if len(tokenizer) > model.config.vocab_size:
        logger.info(f"Resizing embeddings from {model.config.vocab_size} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    # Enable gradient checkpointing for memory efficiency
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Base model loaded: {total_params:,} total params, {trainable_params:,} trainable")

    return model, tokenizer


def setup_lora(
    model: nn.Module,
    lora_rank: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None,
) -> nn.Module:
    """Apply LoRA to model.

    Args:
        model: Base model
        lora_rank: LoRA rank (r)
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: LoRA dropout rate
        target_modules: Target modules for LoRA (None = all attention + MLP)

    Returns:
        PEFT model with LoRA
    """
    logger.info("Configuring LoRA...")
    logger.info(f"  Rank: {lora_rank}")
    logger.info(f"  Alpha: {lora_alpha}")
    logger.info(f"  Dropout: {lora_dropout}")

    # Default target modules for Llama models (attention + MLP)
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    logger.info(f"  Target modules: {target_modules}")

    # Create LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        inference_mode=False,
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Count parameters after LoRA
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = 100 * trainable_params / total_params

    logger.info("LoRA applied successfully")
    logger.info(f"  Total params: {total_params:,}")
    logger.info(f"  Trainable params: {trainable_params:,} ({trainable_pct:.2f}%)")

    # Print trainable parameters
    model.print_trainable_parameters()

    return model


def compute_perplexity(
    model: nn.Module,
    eval_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: int = 1000,
) -> float:
    """Compute perplexity on validation set.

    Args:
        model: Model to evaluate
        eval_dataloader: Validation dataloader
        device: Device to run evaluation on
        max_samples: Maximum number of samples to evaluate

    Returns:
        Perplexity value
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    logger.info(f"Computing perplexity (max {max_samples} samples)...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            if batch_idx * batch['input_ids'].size(0) >= max_samples:
                break

            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            # Accumulate loss
            # Only count non-padding tokens
            num_tokens = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens
            num_batches += 1

    # Compute perplexity
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    logger.info(f"Perplexity: {perplexity:.2f} (avg loss: {avg_loss:.4f}, {num_batches} batches)")

    model.train()
    return perplexity


def check_ram_disk_space(required_gb: float = 9.0) -> tuple[bool, float]:
    """Check if /dev/shm has enough space.

    Args:
        required_gb: Required space in GB

    Returns:
        Tuple of (has_space, available_gb)
    """
    try:
        result = subprocess.run(
            ['df', '--block-size=1G', '/dev/shm'],
            capture_output=True,
            text=True,
            check=True
        )
        # Parse output: Filesystem  1G-blocks  Used  Available  Use%  Mounted on
        lines = result.stdout.strip().split('\n')
        if len(lines) >= 2:
            parts = lines[1].split()
            available_gb = float(parts[3])
            return available_gb >= required_gb, available_gb
    except (subprocess.CalledProcessError, IndexError, ValueError) as e:
        logger.warning(f"Failed to check /dev/shm space: {e}")
        return False, 0.0

    return False, 0.0


def get_directory_size(path: Path) -> int:
    """Get total size of directory in bytes.

    Args:
        path: Directory path

    Returns:
        Total size in bytes
    """
    total = 0
    try:
        for entry in path.rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
    except Exception as e:
        logger.warning(f"Error calculating directory size: {e}")
    return total


def copy_with_progress(src: Path, dst: Path, desc: str = "Copying"):
    """Copy directory with progress bar.

    Args:
        src: Source directory
        dst: Destination directory
        desc: Description for progress bar
    """
    # Create destination directory
    dst.mkdir(parents=True, exist_ok=True)

    # Get list of all files to copy
    all_files = []
    for entry in src.rglob('*'):
        if entry.is_file():
            all_files.append(entry)

    total_size = sum(f.stat().st_size for f in all_files)

    # Copy with progress
    with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
        for src_file in all_files:
            # Compute relative path and destination
            rel_path = src_file.relative_to(src)
            dst_file = dst / rel_path

            # Create parent directory if needed
            dst_file.parent.mkdir(parents=True, exist_ok=True)

            # Copy file
            shutil.copy2(src_file, dst_file)
            pbar.update(src_file.stat().st_size)


def setup_ram_disk(config: Dict[str, Any], cleanup: bool = False) -> tuple[Dict[str, Any], Optional[Path]]:
    """Setup RAM disk for training data.

    Copies dataset to /dev/shm for faster I/O. Checks for available space,
    creates directory structure, and updates config paths.

    Args:
        config: Configuration dictionary
        cleanup: If True, remove data from RAM disk after training

    Returns:
        Tuple of (updated_config, ram_disk_path)

    Raises:
        RuntimeError: If /dev/shm has insufficient space
        FileNotFoundError: If source data path doesn't exist
    """
    # Get source data path
    data_path = config.get('data_path') or config.get('train_data_path')
    if not data_path:
        raise ValueError("No data_path or train_data_path found in config")

    src_path = Path(data_path)
    if not src_path.exists():
        raise FileNotFoundError(f"Source data path not found: {src_path}")

    # Define RAM disk path
    ram_disk_base = Path("/dev/shm/distillation_data")
    ram_disk_path = ram_disk_base / src_path.name

    logger.info("=" * 80)
    logger.info("RAM DISK SETUP")
    logger.info("=" * 80)
    logger.info(f"Source: {src_path}")
    logger.info(f"Destination: {ram_disk_path}")

    # Check if data already exists in RAM disk
    if ram_disk_path.exists() and (ram_disk_path / "manifest.json").exists():
        logger.info("Dataset already exists in RAM disk - skipping copy")

        # Log RAM disk usage
        try:
            result = subprocess.run(
                ['df', '-h', '/dev/shm'],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("Current /dev/shm usage:")
            for line in result.stdout.strip().split('\n'):
                logger.info(f"  {line}")
        except subprocess.CalledProcessError:
            pass
    else:
        # Calculate required space
        src_size_bytes = get_directory_size(src_path)
        src_size_gb = src_size_bytes / (1024**3)
        required_gb = src_size_gb + 1.0  # Add 1GB buffer

        logger.info(f"Dataset size: {src_size_gb:.2f} GB")
        logger.info(f"Required space: {required_gb:.2f} GB (includes buffer)")

        # Check available space
        has_space, available_gb = check_ram_disk_space(required_gb)
        if not has_space:
            raise RuntimeError(
                f"Insufficient space in /dev/shm. "
                f"Required: {required_gb:.2f} GB, Available: {available_gb:.2f} GB. "
                f"Consider increasing /dev/shm size or disabling --ram-disk flag."
            )

        logger.info(f"Available space in /dev/shm: {available_gb:.2f} GB")
        logger.info(f"Copying dataset to RAM disk ({src_size_gb:.2f} GB)...")

        # Copy data with progress
        try:
            copy_with_progress(
                src_path,
                ram_disk_path,
                desc=f"Copying to RAM disk ({src_size_gb:.1f}GB)"
            )
            logger.info("Dataset copy complete")
        except Exception as e:
            # Clean up partial copy on failure
            if ram_disk_path.exists():
                logger.error("Copy failed - cleaning up partial data...")
                shutil.rmtree(ram_disk_path, ignore_errors=True)
            raise RuntimeError(f"Failed to copy dataset to RAM disk: {e}") from e

        # Set permissions
        try:
            os.chmod(ram_disk_path, 0o755)
            for entry in ram_disk_path.rglob('*'):
                if entry.is_file():
                    os.chmod(entry, 0o644)
                elif entry.is_dir():
                    os.chmod(entry, 0o755)
        except Exception as e:
            logger.warning(f"Failed to set permissions: {e}")

        # Log final RAM disk usage
        try:
            result = subprocess.run(
                ['df', '-h', '/dev/shm'],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("Final /dev/shm usage:")
            for line in result.stdout.strip().split('\n'):
                logger.info(f"  {line}")
        except subprocess.CalledProcessError:
            pass

    # Update config to point to RAM disk
    updated_config = config.copy()
    if 'data_path' in updated_config:
        updated_config['data_path'] = str(ram_disk_path)
    if 'train_data_path' in updated_config:
        updated_config['train_data_path'] = str(ram_disk_path)
    if 'val_data_path' in updated_config:
        # Update validation path if it points to same location
        if updated_config['val_data_path'] == str(src_path):
            updated_config['val_data_path'] = str(ram_disk_path)

    logger.info(f"Dataset ready in RAM at {ram_disk_path}")
    logger.info("Note: Data will persist in RAM until system reboot")
    if cleanup:
        logger.info("Data will be removed from RAM disk after training completes")
    logger.info("=" * 80)

    return updated_config, ram_disk_path if cleanup else None


def cleanup_ram_disk(ram_disk_path: Path):
    """Remove dataset from RAM disk.

    Args:
        ram_disk_path: Path to RAM disk data directory
    """
    if ram_disk_path and ram_disk_path.exists():
        logger.info("=" * 80)
        logger.info("RAM DISK CLEANUP")
        logger.info("=" * 80)
        logger.info(f"Removing data from RAM disk: {ram_disk_path}")

        try:
            shutil.rmtree(ram_disk_path)
            logger.info("RAM disk cleanup complete")

            # Log updated usage
            result = subprocess.run(
                ['df', '-h', '/dev/shm'],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("Updated /dev/shm usage:")
            for line in result.stdout.strip().split('\n'):
                logger.info(f"  {line}")
        except Exception as e:
            logger.error(f"Failed to clean up RAM disk: {e}")

        logger.info("=" * 80)


class CustomTrainer(Trainer):
    """Custom Trainer with validation perplexity tracking and early stopping.

    Features:
    - Computes perplexity after each epoch
    - Early stopping based on perplexity improvement
    - Logs best checkpoint path
    """

    def __init__(
        self,
        *args,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
        val_samples: int = 1000,
        early_stop_threshold: float = 0.10,
        early_stop_patience: int = 3,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """Initialize custom trainer.

        Args:
            val_dataloader: Validation dataloader
            val_samples: Number of validation samples
            early_stop_threshold: Stop if ppl improvement < this threshold
            early_stop_patience: Stop after this many epochs without improvement
            max_tokens: Stop after this many tokens (None = no limit)
        """
        super().__init__(*args, **kwargs)
        self.val_dataloader = val_dataloader
        self.val_samples = val_samples
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_patience = early_stop_patience
        self.max_tokens = max_tokens

        # Tracking
        self.best_perplexity = float('inf')
        self.best_checkpoint = None
        self.epochs_without_improvement = 0
        self.total_tokens_seen = 0

    def evaluate(
        self,
        eval_dataset = None,
        ignore_keys = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """Override evaluate to compute perplexity."""
        # Call parent evaluation
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # Compute perplexity if val_dataloader provided
        if self.val_dataloader is not None:
            perplexity = compute_perplexity(
                model=self.model,
                eval_dataloader=self.val_dataloader,
                device=self.args.device,
                max_samples=self.val_samples,
            )

            metrics[f"{metric_key_prefix}_perplexity"] = perplexity

            # Check for improvement
            ppl_improvement = (self.best_perplexity - perplexity) / self.best_perplexity

            if perplexity < self.best_perplexity:
                logger.info(f"New best perplexity: {perplexity:.2f} (prev: {self.best_perplexity:.2f}, improvement: {ppl_improvement:.1%})")
                self.best_perplexity = perplexity
                self.best_checkpoint = self.state.best_model_checkpoint
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                logger.info(f"No improvement: {perplexity:.2f} vs best {self.best_perplexity:.2f} ({self.epochs_without_improvement}/{self.early_stop_patience} patience)")

            # Early stopping check
            if self.epochs_without_improvement >= self.early_stop_patience:
                if ppl_improvement < self.early_stop_threshold:
                    logger.warning(f"Early stopping: {self.epochs_without_improvement} epochs without {self.early_stop_threshold:.0%} improvement")
                    self.control.should_training_stop = True

        return metrics

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training step to track token count."""
        # Track tokens
        if "input_ids" in inputs:
            # Use processing_class (tokenizer) with fallback for compatibility
            tokenizer = getattr(self, 'processing_class', None) or getattr(self, 'tokenizer', None)
            if tokenizer is not None:
                batch_tokens = (inputs["input_ids"] != tokenizer.pad_token_id).sum().item()
                self.total_tokens_seen += batch_tokens

                # Check token limit
                if self.max_tokens is not None and self.total_tokens_seen >= self.max_tokens:
                    logger.warning(f"Reached max tokens: {self.total_tokens_seen:,} >= {self.max_tokens:,}")
                    self.control.should_training_stop = True

        # Call parent with correct signature based on transformers version
        if num_items_in_batch is not None:
            return super().training_step(model, inputs, num_items_in_batch)
        else:
            return super().training_step(model, inputs)


def main():
    """Main training entry point."""
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Setup RAM disk if requested
    ram_disk_cleanup_path = None
    if args.ram_disk:
        try:
            config, ram_disk_cleanup_path = setup_ram_disk(
                config,
                cleanup=args.ram_disk_cleanup
            )
        except Exception as e:
            logger.error(f"Failed to setup RAM disk: {e}")
            logger.error("Exiting. Remove --ram-disk flag to use disk-based loading.")
            sys.exit(1)

    # Override output dir if specified
    if args.output_dir:
        config['output_dir'] = args.output_dir

    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup file logging
    log_file = output_dir / "finetune.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 80)
    logger.info("TEACHER FINETUNING WITH LORA")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Config: {args.config}")

    # Print key config
    logger.info("Configuration:")
    logger.info(f"  Base model: {config['base_model']}")
    logger.info(f"  Max seq length: {config['max_seq_length']}")
    logger.info(f"  LoRA rank: {config['lora_rank']}")
    logger.info(f"  LoRA alpha: {config['lora_alpha']}")
    logger.info(f"  Learning rate: {config['learning_rate']}")
    logger.info(f"  Batch size: {config.get('per_device_train_batch_size', 1)}")
    logger.info(f"  Gradient accumulation: {config.get('gradient_accumulation_steps', 8)}")
    logger.info(f"  Max steps: {config.get('max_steps', 'auto')}")
    logger.info(f"  Max tokens: {config.get('max_tokens', 'unlimited')}")
    logger.info(f"  Epochs: {config.get('num_train_epochs', 'auto')}")

    # Enable TF32 for faster training
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 enabled for matmul and cudnn")

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        model_name=config['base_model'],
        tokenizer_name=config.get('tokenizer_name'),
        use_bf16=config.get('use_bf16', True),
        gradient_checkpointing=config.get('gradient_checkpointing', True),
        use_flash_attention_2=config.get('use_flash_attention_2', False),
    )

    # Apply LoRA
    model = setup_lora(
        model=model,
        lora_rank=config['lora_rank'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config.get('lora_dropout', 0.05),
    )

    # Load dataset
    logger.info("Loading dataset...")
    # Support both data_path and train_data_path
    data_path = config.get('data_path') or config.get('train_data_path')
    logger.info(f"  Data path: {data_path}")
    logger.info(f"  Max length: {config['max_seq_length']}")

    train_dataset = PretokenizedShardDataset(
        data_path=data_path,
        max_length=config['max_seq_length'],
        splits=config.get('splits') or config.get('pretokenized_splits'),
        tokenizer_pad_token_id=tokenizer.pad_token_id,
        tokenizer_eos_token_id=tokenizer.eos_token_id,
    )

    manifest_tokenizer = getattr(train_dataset, 'manifest', {}).get('tokenizer')
    if manifest_tokenizer and manifest_tokenizer != config['tokenizer_name']:
        msg = (
            "Pretokenized data was produced with tokenizer '%s' but the finetune "
            "script is loading '%s'. Retokenize the data or set allow_tokenizer_mismatch=true "
            "(not recommended)."
        ) % (manifest_tokenizer, config['tokenizer_name'])
        if config.get('allow_tokenizer_mismatch'):
            logger.warning(msg)
        else:
            raise ValueError(msg)

    logger.info(f"Training dataset loaded: {len(train_dataset):,} sequences")

    # Create validation dataset (use small subset for efficiency)
    val_dataset = None
    val_dataloader = None
    if config.get('validation_split', 0.0) > 0:
        # Simple validation split (take last N samples)
        val_size = int(len(train_dataset) * config['validation_split'])
        train_size = len(train_dataset) - val_size

        logger.info(f"Splitting dataset: {train_size:,} train, {val_size:,} val")

        # Create validation dataset (we'll just use a subset for perplexity)
        val_dataset = torch.utils.data.Subset(train_dataset, range(train_size, len(train_dataset)))
        train_dataset = torch.utils.data.Subset(train_dataset, range(train_size))

        # Create validation dataloader
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config['per_device_eval_batch_size'],
            shuffle=False,
            collate_fn=train_dataset.dataset.collate_fn if hasattr(train_dataset, 'dataset') else train_dataset.collate_fn,
        )

        logger.info(f"Validation dataloader created: {len(val_dataloader)} batches")

    # Training arguments
    # Support both warmup_steps and warmup_ratio
    warmup_steps = config.get('warmup_steps', 0)
    if warmup_steps == 0 and 'warmup_ratio' in config:
        warmup_ratio = config['warmup_ratio']
    else:
        warmup_ratio = 0.0

    # Configure W&B reporting
    if config.get('enable_wandb', False):
        report_to = config.get('report_to', ['wandb'])
        # Set environment variables before TrainingArguments
        os.environ['WANDB_PROJECT'] = config.get('wandb_project', 'teacher-finetuning')
        if config.get('wandb_run_name'):
            os.environ['WANDB_RUN_NAME'] = config['wandb_run_name']
        logger.info("W&B logging ENABLED")
        logger.info(f"  Project: {config.get('wandb_project', 'teacher-finetuning')}")
        logger.info(f"  Run name: {config.get('wandb_run_name', 'llama-1b-domain-adapt')}")
    else:
        report_to = config.get('report_to', ['none'])
        logger.info("W&B logging disabled")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.get('per_device_train_batch_size', 1),
        per_device_eval_batch_size=config.get('per_device_eval_batch_size', 1),
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 8),
        learning_rate=config['learning_rate'],
        lr_scheduler_type=config.get('lr_scheduler_type', 'cosine'),
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        max_steps=config.get('max_steps', -1),
        num_train_epochs=config.get('num_train_epochs', 1),
        bf16=config.get('use_bf16', config.get('bf16', True)),
        fp16=config.get('fp16', False),
        logging_steps=config.get('logging_steps', config.get('logging_strategy', 50)),
        logging_first_step=config.get('logging_first_step', True),
        save_steps=config.get('save_steps', 500),
        save_total_limit=config.get('save_total_limit', 3),
        save_strategy=config.get('save_strategy', 'steps'),
        eval_strategy=config.get('eval_strategy', config.get('evaluation_strategy', 'steps' if val_dataset else 'no')),
        eval_steps=config.get('eval_steps', 500) if val_dataset else None,
        load_best_model_at_end=config.get('load_best_model_at_end', True if val_dataset else False),
        metric_for_best_model=config.get('metric_for_best_model', 'eval_perplexity' if val_dataset else None),
        greater_is_better=config.get('greater_is_better', False),
        report_to=report_to,
        gradient_checkpointing=config.get('gradient_checkpointing', True),
        max_grad_norm=config.get('max_grad_norm', 1.0),
        weight_decay=config.get('weight_decay', 0.01),
        seed=config.get('seed', config.get('data_seed', 42)),
        data_seed=config.get('data_seed', config.get('seed', 42)),
        dataloader_num_workers=config.get('num_workers', config.get('dataloader_num_workers', 0)),
        dataloader_pin_memory=config.get('dataloader_pin_memory', True),
        dataloader_persistent_workers=config.get('persistent_workers', False) if config.get('num_workers', config.get('dataloader_num_workers', 0)) > 0 else False,
        dataloader_prefetch_factor=config.get('prefetch_factor', 2) if config.get('num_workers', config.get('dataloader_num_workers', 0)) > 0 else None,
        remove_unused_columns=config.get('remove_unused_columns', False),
        ddp_find_unused_parameters=False,
        optim=config.get('optim', 'adamw_torch'),
        save_safetensors=config.get('save_safetensors', True),
        save_only_model=config.get('save_only_model', False),
    )

    logger.info("Training arguments configured")
    logger.info(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps: {training_args.max_steps if training_args.max_steps > 0 else 'auto'}")

    # Setup W&B if enabled
    if 'wandb' in training_args.report_to:
        try:
            import wandb
            os.environ['WANDB_PROJECT'] = config.get('wandb_project', 'teacher-finetuning')
            if config.get('wandb_run_name'):
                os.environ['WANDB_RUN_NAME'] = config['wandb_run_name']
            logger.info(f"W&B tracking enabled: project={config.get('wandb_project')}")
        except ImportError:
            logger.warning("W&B requested but not installed. Continuing without W&B.")

    # Create trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        val_dataloader=val_dataloader,
        val_samples=config.get('val_samples', config.get('eval_perplexity_samples', 1000)),
        early_stop_threshold=config.get('early_stop_threshold', config.get('early_stopping_threshold', 0.001)),
        early_stop_patience=config.get('early_stop_patience', config.get('early_stopping_patience', 5)),
        max_tokens=config.get('max_tokens', None),
    )

    logger.info("Trainer created")

    # Resume from checkpoint if specified
    resume_from_checkpoint = args.resume_from
    if resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")

    # Start training
    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)

    training_start_time = time.time()

    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")

        # Save checkpoint on interruption
        interrupted_checkpoint_path = output_dir / f"checkpoint-interrupted-step-{trainer.state.global_step}"
        logger.info(f"Saving checkpoint to: {interrupted_checkpoint_path}")

        try:
            # Save model, optimizer, scheduler, and trainer state
            trainer.save_model(str(interrupted_checkpoint_path))

            # Save trainer state (includes step count, best metrics, etc.)
            trainer.state.save_to_json(str(interrupted_checkpoint_path / "trainer_state.json"))

            # Save optimizer and scheduler if they exist
            if trainer.optimizer is not None:
                torch.save(
                    trainer.optimizer.state_dict(),
                    str(interrupted_checkpoint_path / "optimizer.pt")
                )
            if trainer.lr_scheduler is not None:
                torch.save(
                    trainer.lr_scheduler.state_dict(),
                    str(interrupted_checkpoint_path / "scheduler.pt")
                )

            # Save RNG states for reproducibility
            torch.save(
                {
                    'python': random.getstate(),
                    'numpy': np.random.get_state(),
                    'cpu': torch.random.get_rng_state(),
                    'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
                str(interrupted_checkpoint_path / "rng_state.pth")
            )

            logger.info(f"Checkpoint saved successfully at step {trainer.state.global_step}")
            logger.info(f"Resume with: --resume-from {interrupted_checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        # Clean up RAM disk on error if requested
        if ram_disk_cleanup_path:
            cleanup_ram_disk(ram_disk_cleanup_path)
        raise
    finally:
        # Clean up RAM disk if requested
        if ram_disk_cleanup_path:
            cleanup_ram_disk(ram_disk_cleanup_path)

    training_time = time.time() - training_start_time

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Training time: {training_time / 3600:.2f} hours")
    logger.info(f"Total tokens: {trainer.total_tokens_seen:,}")
    if hasattr(trainer, 'best_perplexity'):
        logger.info(f"Best perplexity: {trainer.best_perplexity:.2f}")
        logger.info(f"Best checkpoint: {trainer.best_checkpoint}")

    # Save final LoRA adapter
    final_adapter_path = output_dir / "final_adapter"
    logger.info(f"Saving final LoRA adapter to: {final_adapter_path}")
    model.save_pretrained(final_adapter_path)
    tokenizer.save_pretrained(final_adapter_path)

    # Save merged model if requested (via CLI or config)
    should_save_merged = args.save_merged or config.get('save_merged_model', False)
    if should_save_merged:
        logger.info("Merging LoRA adapter with base model...")

        # Merge and unload
        merged_model = model.merge_and_unload()

        # Get merged output path
        merged_output_path = config.get('merged_output_path', output_dir / "merged_model")
        merged_output_path = Path(merged_output_path)

        logger.info(f"Saving merged model to: {merged_output_path}")
        merged_model.save_pretrained(str(merged_output_path))
        tokenizer.save_pretrained(str(merged_output_path))

        logger.info("Merged model saved successfully")
        logger.info(f"  Path: {merged_output_path}")
        logger.info(f"  Use with: AutoModelForCausalLM.from_pretrained('{merged_output_path}')")

    # Print summary
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"LoRA adapter: {final_adapter_path}")
    if args.save_merged:
        logger.info(f"Merged model: {output_dir / 'merged_model'}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
