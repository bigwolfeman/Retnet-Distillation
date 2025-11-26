"""
Training loop for knowledge distillation.

Implements gradient accumulation, mixed precision, checkpointing,
and periodic evaluation.

Tasks: T037-T040
"""

import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass, field

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logging.warning("psutil not installed. System memory monitoring will be disabled.")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
try:
    # PyTorch 2.0+ unified API
    from torch.amp import autocast, GradScaler
except ImportError:
    # PyTorch 1.x fallback
    from torch.cuda.amp import autocast, GradScaler

from .losses import SparseKLLoss
from .vllm_teacher_client import VLLMTeacherClient
from .student_config import RetNetStudentConfig
from .evaluation.runner import EvaluationRunner
from .evaluation.perplexity import PerplexityConfig
from .evaluation.niah import NIAHConfig


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for distillation training.

    Attributes:
        physical_batch_size: Actual batch size per forward pass (default: 1 for 32GB VRAM)
        gradient_accumulation_steps: Number of steps to accumulate gradients (default: 256)
        max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
        learning_rate: Learning rate for optimizer (default: 1e-4)
        weight_decay: Weight decay for optimizer (default: 0.01)
        warmup_steps: Number of warmup steps (default: 1000)
        max_steps: Maximum training steps (default: 100000)
        use_bf16: Use BF16 mixed precision (default: True)
        gradient_checkpointing: Use gradient checkpointing (default: False)
        log_interval: Steps between logging (default: 10)
        log_memory_debug: Enable detailed memory logging at key checkpoints (default: False)
        eval_interval: Steps between evaluation (default: 1000)
        save_interval: Steps between checkpoints (default: 5000)
        teacher_endpoint: Teacher server endpoint URL
        teacher_topk: Number of top-k logits from teacher (default: 512)
        teacher_temperature: Temperature for teacher softmax (default: 2.0)
        distill_alpha: Mixing coefficient for hard/soft targets (default: 0.2)
    """

    # Batch and gradient settings
    physical_batch_size: int = 1
    gradient_accumulation_steps: int = 256
    max_grad_norm: float = 1.0

    # Optimizer settings
    optimizer_type: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    use_8bit_optimizer: bool = False  # Use 8-bit AdamW from bitsandbytes
    muon_aux_lr_scale: float = 0.25
    muon_ready_check: bool = True
    muon_clip_threshold: float = 10.0
    muon_clip_alpha: float = 0.5
    muon_clip_pairs: List[List[str]] = field(default_factory=lambda: [["q_proj", "k_proj"]])

    # Mixed precision
    use_bf16: bool = True
    gradient_checkpointing: bool = False

    # Logging and checkpointing
    log_interval: int = 10
    log_memory_debug: bool = False  # Enable detailed memory logging at key checkpoints
    eval_interval: int = 1000
    save_interval: int = 5000

    # Teacher settings
    teacher_endpoint: str = "http://localhost:8000/v1/completions"
    teacher_model: str = "meta-llama/Llama-3.2-1B"
    teacher_topk: int = 512  # Increased from 128 for better probability coverage
    teacher_temperature: float = 2.0

    # Distillation settings
    distill_alpha: float = 0.2
    alpha_warmup_steps: int = 0  # Steps to ramp alpha (0 = no schedule)
    alpha_initial: float = 0.0  # Starting alpha during warmup
    alpha_final: float = 0.2  # Target alpha after warmup

    # Temperature scheduling
    temperature_warmup_steps: int = 0  # Steps to ramp temperature (0 = no schedule)
    temperature_initial: float = 2.5  # Starting temperature during warmup
    temperature_final: float = 1.0  # Target temperature after warmup

    # Reverse KL settings
    reverse_kl: bool = False  # Use reverse KL divergence
    reverse_kl_warmup_steps: int = 0  # Optional warmup before flipping

    @property
    def effective_batch_size(self) -> int:
        """Compute effective batch size (physical * accumulation)."""
        return self.physical_batch_size * self.gradient_accumulation_steps


class DistillationTrainer:
    """Training loop for knowledge distillation from teacher to student.

    Implements:
    - Basic training loop with teacher logit fetching (T037)
    - Gradient accumulation for large effective batch sizes (T038)
    - Gradient clipping for training stability (T039)
    - BF16 mixed precision for efficiency (T040)

    Features:
    - Fetches teacher logits via HTTP client (teacher_client.py)
    - Computes sparse-KL distillation loss (losses.py)
    - Handles padding mask (labels=-100)
    - Tracks training metrics and timing
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        student_config: RetNetStudentConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        checkpoint_dir: Optional[Path] = None,
        tokenizer: Optional[Any] = None,
        enable_full_eval: bool = False,
        teacher_client: Optional[Any] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        pretrain_ce_only: bool = False,
    ):
        """Initialize distillation trainer.

        Args:
            model: Student model (RetNet)
            config: Training configuration
            student_config: Student model configuration
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            checkpoint_dir: Directory for saving checkpoints (optional)
            tokenizer: Tokenizer for evaluation (optional, required for NIAH)
            enable_full_eval: Enable full evaluation suite (perplexity + NIAH) (T059)
            teacher_client: Pre-initialized teacher client (DirectTeacherClient, VLLMTeacherClient, etc.)
                           If None, creates VLLMTeacherClient from config for backward compatibility
            optimizer: Pre-initialized optimizer (optional, for shared optimizer with special param groups)
            scheduler: Pre-initialized LR scheduler (optional, for warm restarts)
            pretrain_ce_only: Enable CE-only pretraining mode (no teacher, pure cross-entropy loss)
        """
        self.model = model
        self.config = config
        self.student_config = student_config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.tokenizer = tokenizer
        self.enable_full_eval = enable_full_eval

        # Training state
        self.global_step = 0
        self.total_batches = 0  # DEPRECATED: No longer used for warmup (Phase 1b fix). Kept for backward compat.
        self.epoch = 0
        self.best_val_loss = float('inf')

        # PROFILING: Timing accumulators
        self.enable_profiling = False  # TEMPORARY: Set to False to disable
        self.profiling_data = {
            'teacher_times': [],
            'student_fwd_times': [],
            'loss_comp_times': [],
            'backward_times': [],
            'optimizer_times': [],
            'total_step_times': [],
        }

        # Move model to GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        if self.config.use_bf16:
            logger.info("Keeping weights in FP32; BF16 autocast will be used for compute")

        # FIX: Ensure weight tying is preserved after model dtype conversion
        # BF16 conversion can break weight tying if not careful
        if hasattr(self.model, 'lm_head') and hasattr(self.model, 'embed'):
            # Re-tie weights to ensure they share the same tensor
            self.model.lm_head.proj.weight = self.model.embed.weight
            logger.info("✓ Weight tying re-applied after BF16 conversion")

        # Initialize optimizer (use provided optimizer or create default)
        if optimizer is not None:
            self.optimizer = optimizer
            logger.info("Using provided optimizer with custom parameter groups")
        else:
            self.optimizer = self._create_optimizer()
            logger.info("Created default optimizer (backward compatibility mode)")

        # Initialize scheduler (use provided scheduler or None)
        self.scheduler = scheduler
        if self.scheduler is not None:
            logger.info("Using provided LR scheduler (CosineAnnealingWarmRestartsWithWarmup)")
        else:
            logger.info("No LR scheduler provided - using manual warmup schedule")

        # Muon clip configuration (QK-clip) for stabilizing logits
        self._muon_clip_config = self._build_muon_clip_config()
        self._muon_clip_logged = False

        # VALIDATION: Verify weight tying and optimizer setup
        all_opt_param_ids = {id(p) for group in self.optimizer.param_groups for p in group['params']}
        if hasattr(self.model, 'embed'):
            embed_in_opt = id(self.model.embed.weight) in all_opt_param_ids
            logger.info(f"✓ Optimizer contains embed.weight: {embed_in_opt}")
            if not embed_in_opt:
                logger.error("CRITICAL: embed.weight missing from optimizer!")
                logger.error(f"  Total params in optimizer: {len(all_opt_param_ids)}")
                logger.error(f"  embed.weight id: {id(self.model.embed.weight)}")
                raise RuntimeError("embed.weight not in optimizer param groups!")

            # CHECK: Verify requires_grad
            logger.info(f"✓ embed.weight.requires_grad: {self.model.embed.weight.requires_grad}")
            if not self.model.embed.weight.requires_grad:
                logger.error("CRITICAL: embed.weight.requires_grad is False!")
                raise RuntimeError("embed.weight is frozen (requires_grad=False)!")

        if hasattr(self.model, 'lm_head') and hasattr(self.model, 'embed'):
            weights_tied = self.model.embed.weight is self.model.lm_head.proj.weight
            logger.info(f"✓ Weight tying intact: {weights_tied}")
            if not weights_tied:
                logger.error("CRITICAL: Weight tying broken!")
                logger.error(f"  embed.weight id: {id(self.model.embed.weight)}")
                logger.error(f"  lm_head.proj.weight id: {id(self.model.lm_head.proj.weight)}")
                raise RuntimeError("Weight tying is broken - embed and lm_head are separate tensors!")

            logger.info(f"✓ lm_head.proj.weight.requires_grad: {self.model.lm_head.proj.weight.requires_grad}")

        # CHECK: Model training mode
        logger.info(f"✓ Model in training mode: {self.model.training}")

        # VALIDATION: Verify persistent tokens are trainable
        for module in self.model.modules():
            if hasattr(module, 'persistent_tokens'):
                if not module.persistent_tokens.requires_grad:
                    raise RuntimeError(
                        f"Persistent tokens frozen during training! "
                        f"requires_grad={module.persistent_tokens.requires_grad}"
                    )
                logger.info(f"✓ Persistent tokens trainable: requires_grad=True")

        # Initialize loss function
        # Determine initial values for schedules
        initial_alpha = (self.config.alpha_initial
                        if self.config.alpha_warmup_steps > 0
                        else self.config.distill_alpha)
        initial_temperature = (self.config.temperature_initial
                              if self.config.temperature_warmup_steps > 0
                              else self.config.teacher_temperature)
        initial_reverse_kl = (False
                             if self.config.reverse_kl_warmup_steps > 0
                             else self.config.reverse_kl)

        self.loss_fn = SparseKLLoss(
            temperature=initial_temperature,
            alpha=initial_alpha,
            reverse_kl=initial_reverse_kl,
        )

        # Track if we've logged parameter changes
        self._logged_alpha_schedule_start = False
        self._logged_alpha_schedule_complete = False
        self._logged_temperature_schedule_start = False
        self._logged_temperature_schedule_complete = False
        self._logged_reverse_kl_flip = False

        # CE Pretrain Mode flag (T013) - skips teacher and uses pure CE loss
        # Use explicit parameter (not config) to avoid legacy config propagation issues
        self.pretrain_ce_only = pretrain_ce_only
        if self.pretrain_ce_only:
            logger.info("✓ CE Pretrain Mode: Using pure cross-entropy loss (no teacher)")

        # Initialize teacher client
        if teacher_client is not None:
            # Use provided teacher client (DirectTeacherClient, CachedTeacherClient, etc.)
            self.teacher_client = teacher_client
            logger.info("Using provided teacher client")
        elif self.pretrain_ce_only:
            # T013: In CE pretrain mode, teacher_client is None
            self.teacher_client = None
            logger.info("Teacher client: None (CE pretrain mode)")
        else:
            # Backward compatibility: Create VLLMTeacherClient from config
            # Extract base URL and model name from endpoint
            # Expected format: "http://host:port/v1/completions"
            base_url = self.config.teacher_endpoint.replace("/v1/completions", "")
            # Model name should be provided in config, fallback to meta-llama/Llama-3.2-1B
            model_name = getattr(self.config, 'teacher_model', 'meta-llama/Llama-3.2-1B')

            self.teacher_client = VLLMTeacherClient(
                base_url=base_url,
                model=model_name,
                timeout=30.0,
                max_retries=3,
            )
            logger.info("Created VLLMTeacherClient from config (backward compatibility)")

        # Initialize BF16 gradient scaler (for mixed precision)
        # Note: BF16 typically doesn't need gradient scaling, but we include it
        # for compatibility with FP16 workflows
        try:
            # PyTorch 2.0+ unified API
            if self.config.use_bf16:
                self.scaler = GradScaler('cuda', enabled=False)  # BF16 doesn't need scaling
            else:
                self.scaler = GradScaler('cuda', enabled=True)
        except TypeError:
            # PyTorch 1.x fallback
            if self.config.use_bf16:
                self.scaler = GradScaler(enabled=False)
            else:
                self.scaler = GradScaler(enabled=True)

        # Training metrics
        self.metrics: Dict[str, List[float]] = {
            'loss': [],
            'kl_loss': [],
            'ce_loss': [],
            'grad_norm': [],
            'learning_rate': [],
            'steps_per_sec': [],
        }

        # Initialize evaluation runner (T059)
        self.evaluation_runner = None
        if self.enable_full_eval and self.tokenizer is not None:
            self.evaluation_runner = EvaluationRunner(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
            )
            logger.info("Initialized EvaluationRunner for periodic evaluation")

        # Async teacher prefetch support
        self.async_teacher_enabled = hasattr(teacher_client, 'submit') and hasattr(teacher_client, 'get')
        self._pending_teacher_future = None

        if not self.async_teacher_enabled:
            logger.info("Async teacher prefetch: DISABLED (teacher client has no async methods)")
        else:
            logger.info("Async teacher prefetch: ENABLED (AsyncTeacherClient wrapper detected)")

        logger.info(f"Initialized DistillationTrainer:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Physical batch size: {self.config.physical_batch_size}")
        logger.info(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {self.config.effective_batch_size}")
        logger.info(f"  Max grad norm: {self.config.max_grad_norm}")
        logger.info(f"  Learning rate: {self.config.learning_rate}")
        logger.info(f"  Use BF16: {self.config.use_bf16}")
        logger.info(f"  Teacher endpoint: {self.config.teacher_endpoint}")
        logger.info(f"  Full evaluation: {self.enable_full_eval}")
        logger.info(f"  Async teacher prefetch: {self.async_teacher_enabled}")
        logger.info(f"LOSS CONFIGURATION:")
        logger.info(f"  Initial alpha: {self.loss_fn.alpha:.3f}")
        logger.info(f"  Initial temperature: {self.loss_fn.temperature:.3f}")
        logger.info(f"  Initial reverse_kl: {self.loss_fn.reverse_kl}")
        if self.config.alpha_warmup_steps > 0:
            logger.info(f"  Alpha schedule: {self.config.alpha_initial:.3f} -> {self.config.alpha_final:.3f} over {self.config.alpha_warmup_steps} steps")
        if self.config.temperature_warmup_steps > 0:
            logger.info(f"  Temperature schedule: {self.config.temperature_initial:.3f} -> {self.config.temperature_final:.3f} over {self.config.temperature_warmup_steps} steps")
        if self.config.reverse_kl_warmup_steps > 0:
            logger.info(f"  Reverse KL warmup: flip to {self.config.reverse_kl} at step {self.config.reverse_kl_warmup_steps}")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay.

        Returns:
            AdamW optimizer
        """
        # Separate parameters: no weight decay for biases and layer norms
        no_decay = ['bias', 'layer_norm', 'layernorm', 'norm']

        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters()
                          if not any(nd in n.lower() for nd in no_decay)],
                'weight_decay': self.config.weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                          if any(nd in n.lower() for nd in no_decay)],
                'weight_decay': 0.0,
            }
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        return optimizer

    def _get_learning_rate(self) -> float:
        """Compute learning rate with linear warmup over optimizer steps.

        PHASE 1B FIX: Changed warmup to use optimizer steps instead of batches.

        Previously: warmup_batches=300 with gradient_accumulation_steps=256 meant
        only ~1 optimizer step of warmup (300/256 ≈ 1.17 steps).

        Now: warmup_steps=300 means 300 actual optimizer steps, giving proper warmup.

        Note: If scheduler is provided, this method is not used. The scheduler
        handles LR updates automatically via scheduler.step().

        Returns:
            Current learning rate
        """
        # If scheduler is provided, it handles LR - return current LR from optimizer
        if self.scheduler is not None:
            # Return base LR from first param group (scheduler manages it)
            return self.optimizer.param_groups[0]['lr']

        # PHASE 1B FIX: Use global_step (optimizer steps) instead of total_batches
        # Config.warmup_steps is the number of optimizer steps for warmup (default: 1000)
        warmup_steps = self.config.warmup_steps

        if self.global_step < warmup_steps:
            # Linear warmup over optimizer steps
            warmup_factor = self.global_step / warmup_steps if warmup_steps > 0 else 1.0
            return self.config.learning_rate * warmup_factor
        else:
            # Constant learning rate after warmup
            return self.config.learning_rate

    def _update_learning_rate(self):
        """Update optimizer learning rate.

        Note: If scheduler is provided, this is a no-op since the scheduler
        handles LR updates via scheduler.step() in _optimizer_step().
        """
        # If scheduler is provided, it handles LR updates - skip manual update
        if self.scheduler is not None:
            return

        # Manual LR update for backward compatibility (no scheduler)
        lr = self._get_learning_rate()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _update_loss_parameters(self, step: int):
        """Update loss function parameters according to schedules.

        Implements linear ramps for alpha and temperature over warmup steps.
        Implements reverse KL warmup (start with forward KL, flip after warmup).

        Args:
            step: Current global training step (optimizer steps, not batches)
        """
        updated = False

        # Alpha schedule (linear ramp)
        if self.config.alpha_warmup_steps > 0:
            if step == 0 and not self._logged_alpha_schedule_start:
                logger.info(
                    f"Alpha schedule: ramping from {self.config.alpha_initial:.3f} to "
                    f"{self.config.alpha_final:.3f} over {self.config.alpha_warmup_steps} steps"
                )
                self._logged_alpha_schedule_start = True

            if step < self.config.alpha_warmup_steps:
                # Linear ramp
                progress = step / self.config.alpha_warmup_steps
                current_alpha = (self.config.alpha_initial +
                               progress * (self.config.alpha_final - self.config.alpha_initial))
                self.loss_fn.alpha = current_alpha
                updated = True
            elif step == self.config.alpha_warmup_steps and not self._logged_alpha_schedule_complete:
                # Reached final value
                self.loss_fn.alpha = self.config.alpha_final
                logger.info(
                    f"Alpha schedule complete at step {step}: "
                    f"alpha = {self.config.alpha_final:.3f}"
                )
                self._logged_alpha_schedule_complete = True
                updated = True

        # Temperature schedule (linear ramp)
        if self.config.temperature_warmup_steps > 0:
            if step == 0 and not self._logged_temperature_schedule_start:
                logger.info(
                    f"Temperature schedule: ramping from {self.config.temperature_initial:.3f} to "
                    f"{self.config.temperature_final:.3f} over {self.config.temperature_warmup_steps} steps"
                )
                self._logged_temperature_schedule_start = True

            if step < self.config.temperature_warmup_steps:
                # Linear ramp
                progress = step / self.config.temperature_warmup_steps
                current_temperature = (self.config.temperature_initial +
                                     progress * (self.config.temperature_final - self.config.temperature_initial))
                self.loss_fn.temperature = current_temperature
                updated = True
            elif step == self.config.temperature_warmup_steps and not self._logged_temperature_schedule_complete:
                # Reached final value
                self.loss_fn.temperature = self.config.temperature_final
                logger.info(
                    f"Temperature schedule complete at step {step}: "
                    f"temperature = {self.config.temperature_final:.3f}"
                )
                self._logged_temperature_schedule_complete = True
                updated = True

        # Reverse KL warmup (flip from forward to reverse after warmup)
        if self.config.reverse_kl_warmup_steps > 0:
            if step >= self.config.reverse_kl_warmup_steps and not self._logged_reverse_kl_flip:
                # Flip to reverse KL
                self.loss_fn.reverse_kl = self.config.reverse_kl
                logger.info(
                    f"Reverse KL warmup complete at step {step}: "
                    f"flipped to reverse_kl = {self.config.reverse_kl}"
                )
                self._logged_reverse_kl_flip = True
                updated = True

        # Periodic logging (every 100 steps) when schedules are active
        if updated and step % 100 == 0 and step > 0:
            logger.info(
                f"Step {step}: Loss params - "
                f"alpha={self.loss_fn.alpha:.3f}, "
                f"temperature={self.loss_fn.temperature:.3f}, "
                f"reverse_kl={self.loss_fn.reverse_kl}"
            )

    def _fetch_teacher_logits(
        self,
        input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fetch teacher log-probabilities via HTTP client.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]

        Returns:
            Tuple of:
                - teacher_topk_indices: [batch_size, seq_len, k]
                - teacher_topk_logprobs: [batch_size, seq_len, k]
                - teacher_other_logprob: [batch_size, seq_len, 1]
        """
        # Log input shape for diagnostics (first time only)
        if not hasattr(self, '_logged_teacher_input_shape'):
            logger.info(f"[TEACHER] Processing input_ids with shape: {input_ids.shape} (batch_size={input_ids.shape[0]}, seq_len={input_ids.shape[1]})")
            self._logged_teacher_input_shape = True

        # Use GPU-native tensor API if available (DirectTeacherClient)
        # This eliminates CPU/GPU synchronization overhead (64x per optimizer step!)
        if hasattr(self.teacher_client, 'get_top_k_logits_tensors'):
            # GPU-NATIVE PATH: No CPU syncs, everything stays on GPU
            if not hasattr(self, '_logged_tensor_api_mode'):
                logger.critical("=" * 80)
                logger.critical("🟢 USING TENSOR API PATH (GPU-native, no CPU/GPU syncs)")
                logger.critical("=" * 80)
                self._logged_tensor_api_mode = True

            indices, logprobs_float, other_logprob = self.teacher_client.get_top_k_logits_tensors(
                input_ids=input_ids,
                topk=self.config.teacher_topk,
            )
            # No conversion needed - teacher client already returns log-probabilities
            return indices, logprobs_float, other_logprob

        # FALLBACK PATH: List-based API for VLLMTeacherClient (network-based)
        # NOTE: This .cpu().tolist() is necessary for network API
        # Called once per batch (every gradient_accumulation_steps), not every accumulation step
        if not hasattr(self, '_logged_tensor_api_mode'):
            logger.critical("=" * 80)
            logger.critical("🔴 USING LIST API PATH (fallback with CPU/GPU syncs)")
            logger.critical("=" * 80)
            self._logged_tensor_api_mode = True

        input_ids_list = input_ids.cpu().tolist()

        # Query teacher using VLLMTeacherClient
        # NOTE: Temperature=1.0 gets raw logits from teacher
        # Temperature scaling is applied in loss function (SparseKLLoss)
        results = self.teacher_client.get_prompt_logprobs(
            input_ids=input_ids_list,
            topk=self.config.teacher_topk,
            temperature=1.0,  # Always use 1.0 to get raw logits
        )

        # Parse response - VLLMTeacherClient returns list of dicts
        # Each dict has: indices, logprobs (both as lists)
        batch_size = len(results)

        # Build tensors from results
        all_indices = []
        all_logprobs = []

        for result in results:
            # indices is List[List[int]], logprobs is List[List[float]]
            indices = result['indices']
            logprobs = result['logprobs']

            # Convert to tensors and pad if needed
            seq_indices = []
            seq_logprobs = []

            for pos_indices, pos_logprobs in zip(indices, logprobs):
                # Handle empty positions (like BOS)
                if not pos_indices:
                    # Use padding values with -inf for log-probabilities
                    seq_indices.append([0] * self.config.teacher_topk)
                    seq_logprobs.append([-float('inf')] * self.config.teacher_topk)
                else:
                    # Pad to topk if needed
                    while len(pos_indices) < self.config.teacher_topk:
                        pos_indices.append(0)
                        pos_logprobs.append(-float('inf'))
                    seq_indices.append(pos_indices[:self.config.teacher_topk])
                    seq_logprobs.append(pos_logprobs[:self.config.teacher_topk])

            all_indices.append(seq_indices)
            all_logprobs.append(seq_logprobs)

        # Convert to tensors
        indices = torch.tensor(all_indices, dtype=torch.long, device=self.device)
        logprobs_float = torch.tensor(all_logprobs, dtype=torch.float32, device=self.device)

        # Compute other_logprob from top-k logprobs
        # FIX: Calculate proper other_logprob instead of hardcoded placeholder
        # Convert logprobs to probabilities and sum them
        probs = torch.exp(logprobs_float)  # [batch_size, seq_len, k]
        total_prob = probs.sum(dim=-1, keepdim=True)  # [batch_size, seq_len, 1]

        # Other logprob is log(1 - total_prob), clamped to avoid log(0)
        other_prob = torch.clamp(1.0 - total_prob, min=1e-8)
        other_logprob = torch.log(other_prob)  # [batch_size, seq_len, 1]

        return indices, logprobs_float, other_logprob

    def _log_memory_checkpoint(self, checkpoint_name: str):
        """Log detailed memory usage at a checkpoint (FIX #5).

        This method logs GPU memory usage to help debug memory issues and verify
        that memory optimizations are working correctly.

        Controlled by config.log_memory_debug flag or MEMORY_DEBUG env var.

        Args:
            checkpoint_name: Name of the checkpoint (e.g., "after_teacher_fetch")
        """
        # Check if memory debugging is enabled
        if not (self.config.log_memory_debug or os.getenv("MEMORY_DEBUG")):
            return

        # Log memory summary
        logger.info("=" * 80)
        logger.info(f"MEMORY CHECKPOINT: {checkpoint_name} (step {self.global_step})")
        logger.info(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        logger.info(f"  Peak:      {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        logger.info(f"  Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")

        # Full summary only with config flag (not just env var)
        if self.config.log_memory_debug:
            logger.info("\n" + torch.cuda.memory_summary())

        logger.info("=" * 80)

    def _train_step(
        self,
        batch: Dict[str, torch.Tensor],
        accumulation_step: int,
        teacher_logits: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        next_batch: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """Execute a single training step.

        Args:
            batch: Batch of data with keys: input_ids, attention_mask, labels
            accumulation_step: Current accumulation step (0 to accumulation_steps-1)
            teacher_logits: Pre-fetched teacher logits (async mode) or None (sync mode)
            next_batch: Next batch for async prefetch (optional)

        Returns:
            Dictionary of metrics for this step
        """
        # PROFILING: Start total step timer
        if self.enable_profiling:
            torch.cuda.synchronize()
            step_start = time.time()

        # Increment total batch counter (for warmup tracking)
        self.total_batches += 1

        # Move batch to device (non-blocking for async transfer)
        input_ids = batch['input_ids'].to(self.device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
        labels = batch.get('labels', batch['input_ids']).to(self.device, non_blocking=True)

        # DEBUG: Verify attention mask (reduced frequency to avoid CPU sync overhead)
        # FIX: Changed from every 100 steps to every 1000 steps, removed .tolist() calls
        if self.global_step == 0 or self.global_step % 1000 == 0:
            if attention_mask is not None:
                mask_sum = attention_mask.sum().item()
                mask_total = attention_mask.numel()
                mask_pct = (mask_sum / mask_total) * 100
                logger.info(f"DEBUG Step {self.global_step}: Attention mask {mask_sum}/{mask_total} = {mask_pct:.2f}% valid")
                logger.info(f"  Input shape: {input_ids.shape}, Mask shape: {attention_mask.shape}, Labels shape: {labels.shape}")
                # Removed .tolist() calls to avoid CPU sync
            else:
                logger.info(f"DEBUG Step {self.global_step}: attention_mask is None!")

        # FIX #5: Memory checkpoint at step start
        self._log_memory_checkpoint("step_start")

        # Fetch teacher logits (skip in CE pretrain mode - T012)
        teacher_topk_indices = None
        teacher_topk_logprobs = None
        teacher_other_logprob = None

        if not self.pretrain_ce_only:
            # PROFILING: Time teacher inference
            if self.enable_profiling:
                torch.cuda.synchronize()
                teacher_start = time.time()

            with torch.no_grad():
                if teacher_logits is not None:
                    # Async mode: use pre-fetched log-probabilities
                    teacher_topk_indices, teacher_topk_logprobs, teacher_other_logprob = teacher_logits
                else:
                    # Sync mode: fetch on demand
                    teacher_topk_indices, teacher_topk_logprobs, teacher_other_logprob = \
                        self._fetch_teacher_logits(input_ids)

        # FIX #5: Memory checkpoint after teacher fetch
        self._log_memory_checkpoint("after_teacher_fetch")

        # Prefetch next batch asynchronously if enabled
        if self.async_teacher_enabled and next_batch is not None:
            try:
                # FIX #2: Use GPU-native tensor API when available to avoid CPU/GPU sync
                # Check if teacher client supports submit_tensors() for GPU-only async prefetch
                if hasattr(self.teacher_client, 'submit_tensors'):
                    # FIX #2: Memory profiling for tensor API path
                    mem_before = torch.cuda.max_memory_allocated()

                    # GPU-native path: submit tensors directly (no .cpu().tolist() conversion)
                    next_input_ids = next_batch['input_ids']  # Keep on GPU
                    self.teacher_client.submit_tensors(
                        input_ids=next_input_ids,
                        topk=self.config.teacher_topk,
                    )
                    self._pending_teacher_future = True

                    # FIX #2: Log memory impact of tensor API
                    mem_after = torch.cuda.max_memory_allocated()
                    mem_delta = (mem_after - mem_before) / 1e9

                    # Log first successful tensor-based prefetch
                    if not hasattr(self, '_async_tensor_api_logged'):
                        logger.info(
                            "FIX #2: Async teacher prefetch using GPU-native tensor API "
                            "(eliminates 64+ CPU/GPU syncs per step)"
                        )
                        logger.info(f"FIX #2: Memory impact of tensor API async submit: {mem_delta:.3f} GB")
                        self._async_tensor_api_logged = True
                else:
                    # Fallback: use list-based API (e.g., for VLLMTeacherClient)
                    next_input_ids = next_batch['input_ids'].cpu().tolist()
                    self.teacher_client.submit(
                        input_ids=next_input_ids,
                        topk=self.config.teacher_topk,
                        temperature=1.0,
                    )
                    self._pending_teacher_future = True
                    # Log first successful list-based prefetch
                    if not hasattr(self, '_async_first_submit_logged'):
                        logger.info(
                            "Async teacher prefetch: using list-based API "
                            "(tensor API not available in wrapped client)"
                        )
                        self._async_first_submit_logged = True
            except Exception as e:
                logger.warning(f"Async teacher prefetch failed, will fall back to sync: {e}")
                self._pending_teacher_future = False

        if self.enable_profiling and not self.pretrain_ce_only:
            torch.cuda.synchronize()
            teacher_time = time.time() - teacher_start

        # DIAGNOSTIC: Check for constant data/teacher outputs (bug investigation)
        # FIX: Removed .cpu().tolist() calls to avoid CPU sync overhead
        if self.global_step < 5:
            logger.info(f"[DIAG Step {self.global_step}] Model training mode: {self.model.training}")
            if not self.pretrain_ce_only and teacher_topk_logprobs is not None:
                logger.info(f"[DIAG Step {self.global_step}] Teacher topk_logprobs shape: {teacher_topk_logprobs.shape}")
            elif self.pretrain_ce_only:
                logger.info(f"[DIAG Step {self.global_step}] CE Pretrain Mode: No teacher logits")
            # Removed hash and .tolist() calls to avoid CPU sync

        # Forward pass with mixed precision
        # Handle PyTorch version compatibility
        try:
            # PyTorch 2.0+ unified API
            autocast_ctx = autocast('cuda', dtype=torch.bfloat16, enabled=self.config.use_bf16)
        except TypeError:
            # PyTorch 1.x fallback
            autocast_ctx = autocast(enabled=self.config.use_bf16)

        # PROFILING: Time student forward pass
        if self.enable_profiling:
            torch.cuda.synchronize()
            student_fwd_start = time.time()

        with autocast_ctx:
            # Get student logits and auxiliary losses
            # Assuming model returns logits directly (or handle model-specific API)
            aux_losses = {}
            if hasattr(self.model, 'forward_train'):
                # RetNetBackbone: returns hidden states
                hidden_states = self.model.forward_train(input_ids)
                # Need output head to get logits (assuming model has lm_head)
                if hasattr(self.model, 'lm_head'):
                    student_logits = self.model.lm_head(hidden_states)
                else:
                    raise ValueError("Model must have 'lm_head' or return logits directly")
            else:
                # Standard forward pass
                outputs = self.model(input_ids, step=self.global_step, return_aux_loss=True)

                # Handle different return formats
                if isinstance(outputs, tuple):
                    if len(outputs) == 3:
                        # (loss, logits, aux_losses) - TitanMAC format
                        _, student_logits, aux_losses = outputs
                    elif len(outputs) == 2:
                        # (loss, logits) - standard format
                        _, student_logits = outputs
                    else:
                        student_logits = outputs[0]
                elif hasattr(outputs, 'logits'):
                    student_logits = outputs.logits
                else:
                    student_logits = outputs

        if self.enable_profiling:
            torch.cuda.synchronize()
            student_fwd_time = time.time() - student_fwd_start

        # Compute loss in full precision to avoid BF16 underflow/NaNs in KL log-subtract
        with autocast("cuda", enabled=False):

            # DIAGNOSTIC: Check student logits variation
            # FIX: Removed .tolist() and .item() calls to avoid CPU sync
            if self.global_step < 5:
                logger.info(f"[DIAG Step {self.global_step}] Student logits shape: {student_logits.shape}, dtype: {student_logits.dtype}")
                # Removed .tolist() and .item() calls to avoid CPU sync

            # PROFILING: Time loss computation
            if self.enable_profiling:
                torch.cuda.synchronize()
                loss_start = time.time()

            # Compute loss (T012: CE pretrain mode uses pure cross-entropy)
            if self.pretrain_ce_only:
                # CE pretrain mode: pure cross-entropy loss on labels
                vocab_size = student_logits.size(-1)
                loss = F.cross_entropy(
                    student_logits.float().view(-1, vocab_size),
                    labels.view(-1),
                    reduction='mean',
                )
            else:
                # Normal KD mode: sparse KL distillation loss
                loss = self.loss_fn(
                    student_logits=student_logits.float(),
                    teacher_topk_indices=teacher_topk_indices,
                    teacher_topk_logprobs=teacher_topk_logprobs,
                    teacher_other_logprob=teacher_other_logprob,
                    hard_targets=labels,
                    attention_mask=attention_mask,  # FIX: Pass mask to avoid gradient noise from padding
                )

            # MEMORY FIX: Free student logits immediately after loss computation
            # student_logits is ~1GB and accumulates across gradient_accumulation_steps
            # Deleting here saves ~4GB with gradient_accumulation_steps=4
            del student_logits
            # REMOVED: caused 64x/step memory fragmentation
            # torch.cuda.empty_cache()

            # Add diversity regularization loss if present
            diversity_loss = None
            if aux_losses and 'diversity_loss' in aux_losses:
                diversity_loss = aux_losses['diversity_loss']
                # Add to main loss (already scaled by diversity_penalty_weight in retention_block)
                loss = loss + diversity_loss

            # Scale loss by accumulation steps (for gradient averaging)
            loss = loss / self.config.gradient_accumulation_steps

            if self.enable_profiling:
                torch.cuda.synchronize()
                loss_time = time.time() - loss_start

        # PROFILING: Time backward pass
        if self.enable_profiling:
            torch.cuda.synchronize()
            backward_start = time.time()

        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()

        # FIX #5: Memory checkpoint after backward
        self._log_memory_checkpoint("after_backward")

        if self.enable_profiling:
            torch.cuda.synchronize()
            backward_time = time.time() - backward_start

        # GRADIENT FLOW CHECK: Verify gradients reach embedding layer
        # FIX: Removed .item() calls to avoid CPU sync
        if self.global_step < 3 and hasattr(self.model, 'embed'):
            if self.model.embed.weight.grad is not None:
                logger.info(f"[GRAD-CHECK Step {self.global_step}] embed.weight.grad exists - shape: {self.model.embed.weight.grad.shape}")
                # Removed .item() calls to avoid CPU sync
            else:
                logger.error(f"[GRAD-CHECK Step {self.global_step}] CRITICAL: embed.weight.grad is None! Gradients not flowing to embedding layer!")

        # PROFILING: Record step timing breakdown (only on optimizer steps)
        if self.enable_profiling and accumulation_step == self.config.gradient_accumulation_steps - 1:
            torch.cuda.synchronize()
            total_step_time = time.time() - step_start
            self.profiling_data['teacher_times'].append(teacher_time)
            self.profiling_data['student_fwd_times'].append(student_fwd_time)
            self.profiling_data['loss_comp_times'].append(loss_time)
            self.profiling_data['backward_times'].append(backward_time)
            self.profiling_data['total_step_times'].append(total_step_time)

        # Collect metrics
        # FIX: Keep loss as tensor to avoid CPU sync every batch (was 31.3% overhead)
        # Only convert to scalar when actually logging
        metrics = {
            'loss': loss.detach() * self.config.gradient_accumulation_steps,  # Unscale for logging, keep as tensor
        }

        # Add diversity loss to metrics if present
        if diversity_loss is not None:
            metrics['diversity_loss'] = diversity_loss.detach() * self.config.gradient_accumulation_steps

        return metrics

    def _optimizer_step(self) -> torch.Tensor:
        """Execute optimizer step with gradient clipping.

        Returns:
            Gradient norm before clipping (as tensor to avoid CPU sync)
        """
        # PROFILING: Time optimizer step
        if self.enable_profiling:
            torch.cuda.synchronize()
            optimizer_start = time.time()

        # WEIGHT UPDATE CHECK: Save embed weights before optimizer step
        # FIX: Removed .item() to avoid CPU sync, will check tensor directly
        if self.global_step < 3 and hasattr(self.model, 'embed'):
            embed_before = self.model.embed.weight[0, :10].clone().detach()

        # Unscale gradients (for BF16, this is a no-op)
        self.scaler.unscale_(self.optimizer)

        # Guardrail: skip optimizer step if any grad is non-finite (prevents NaN poisoning)
        grad_is_finite = True
        for p in self.model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                grad_is_finite = False
                break
        if not grad_is_finite:
            logger.error(f"[Step {self.global_step}] Non-finite gradients detected, skipping optimizer step")
            self.optimizer.zero_grad(set_to_none=True)
            return torch.tensor(float('nan'), device=self.device)

        # QUICK DIAGNOSTIC: Log per-group norms before clipping
        if self.global_step % 10 == 0:
            logger.info(f"[Step {self.global_step}] BEFORE clipping:")
            for idx, group in enumerate(self.optimizer.param_groups):
                name = group.get('name', f'group_{idx}')
                params = [p for p in group['params'] if p.grad is not None]
                if params:
                    norm = torch.nn.utils.clip_grad_norm_(params, float('inf'), error_if_nonfinite=False)
                    logger.info(f"  {name}: {norm.item():.2f}")

        # PER-GROUP GRADIENT CLIPPING
        # CRITICAL FIX: Clip each parameter group independently to prevent
        # LayerNorm's large gradients (~16) from crushing main params (~1.7)
        #
        # Before (global): LN norm ~16 forces 30× scale-down on all params
        # After (per-group): LN clipped separately, main params keep healthy gradients
        group_norms = []
        for idx, group in enumerate(self.optimizer.param_groups):
            name = group.get('name', f'group_{idx}')
            params = [p for p in group['params'] if p.grad is not None]

            if not params:
                continue

            # Set per-group thresholds
            if 'layernorm' in name.lower() or 'norm' in name.lower():
                # LayerNorm gets its own threshold (allow larger norms)
                clip_threshold = 1.0  # Let LN have larger gradients
            elif 'retention_decay' in name.lower():
                # Retention decay (if learnable in future)
                clip_threshold = self.config.max_grad_norm
            else:
                # Main parameters (Q/K/V, FFN, embeddings)
                clip_threshold = self.config.max_grad_norm  # 0.5

            # Clip this group
            norm = torch.nn.utils.clip_grad_norm_(params, clip_threshold, error_if_nonfinite=False)
            group_norms.append(norm)

        # Compute overall norm for logging (max of group norms)
        grad_norm = max(group_norms) if group_norms else torch.tensor(0.0, device=self.device)

        # QUICK DIAGNOSTIC: Log per-group norms after clipping
        if self.global_step % 10 == 0:
            logger.info(f"[Step {self.global_step}] AFTER per-group clipping:")
            for idx, group in enumerate(self.optimizer.param_groups):
                name = group.get('name', f'group_{idx}')
                params = [p for p in group['params'] if p.grad is not None]
                if params:
                    norm = torch.nn.utils.clip_grad_norm_(params, float('inf'), error_if_nonfinite=False)
                    logger.info(f"  {name}: {norm.item():.2f}")

        # Optimizer step with gradient scaling
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # FIX #5: Memory checkpoint after optimizer step
        self._log_memory_checkpoint("after_optimizer_step")

        # Step scheduler if provided - CRITICAL: Must call for warm restarts - do not remove
        if self.scheduler is not None:
            self.scheduler.step()

        # WEIGHT UPDATE CHECK: Verify embed weights changed
        # FIX: Removed .item() calls, check tensor directly
        if self.global_step < 3 and hasattr(self.model, 'embed'):
            embed_after = self.model.embed.weight[0, :10]
            max_change = (embed_after - embed_before).abs().max()
            logger.info(f"[WEIGHT-CHECK Step {self.global_step}] embed.weight changed (max delta on GPU)")
            if max_change < 1e-8:
                logger.error(f"[WEIGHT-CHECK Step {self.global_step}] CRITICAL: Embedding weights DID NOT UPDATE!")

        # Zero gradients
        self.optimizer.zero_grad(set_to_none=True)

        # PROFILING: Record optimizer time
        if self.enable_profiling:
            torch.cuda.synchronize()
            optimizer_time = time.time() - optimizer_start
            self.profiling_data['optimizer_times'].append(optimizer_time)

        # Apply spectral clipping for Muon (QK-Clip) if configured
        self._apply_muon_qk_clip()

        # FIX: Return grad_norm as tensor to avoid CPU sync every optimizer step
        # Caller will convert to scalar only when logging
        return grad_norm

    def _build_muon_clip_config(self):
        """Prepare MuonClip settings from config."""
        optimizer_type = getattr(self.config, 'optimizer_type', 'adamw')
        threshold = getattr(self.config, 'muon_clip_threshold', 0.0)
        if optimizer_type != "muon" or threshold <= 0:
            return None

        raw_pairs = getattr(self.config, 'muon_clip_pairs', [])
        normalized_pairs = []
        for pair in raw_pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                logger.warning(f"Skipping invalid muon_clip pair definition: {pair}")
                continue
            q_tag, k_tag = pair
            normalized_pairs.append({
                'q': q_tag,
                'k': k_tag,
                'q_lower': q_tag.lower(),
                'k_lower': k_tag.lower(),
            })

        if not normalized_pairs:
            logger.warning("Muon clip enabled but no valid Q/K pairs configured; disabling MuonClip.")
            return None

        alpha = getattr(self.config, 'muon_clip_alpha', 0.5)
        alpha = min(max(alpha, 0.0), 1.0)
        return {
            'threshold': float(threshold),
            'alpha': float(alpha),
            'pairs': normalized_pairs,
        }

    def _apply_muon_qk_clip(self):
        """Clamp Q/K projection matrices to keep attention logits bounded."""
        cfg = getattr(self, '_muon_clip_config', None)
        if not cfg:
            return

        pair_maps = [dict() for _ in cfg['pairs']]
        # Build mapping of shared prefixes -> tensors
        for name, param in self.model.named_parameters():
            if not param.requires_grad or param.ndim < 2:
                continue

            lowered = name.lower()
            for idx, pair in enumerate(cfg['pairs']):
                if pair['q_lower'] in lowered:
                    key = lowered.replace(pair['q_lower'], "")
                    pair_maps[idx].setdefault(key, {})['q'] = param
                    break
                if pair['k_lower'] in lowered:
                    key = lowered.replace(pair['k_lower'], "")
                    pair_maps[idx].setdefault(key, {})['k'] = param
                    break

        total_clipped = 0
        threshold = cfg['threshold']
        alpha = cfg['alpha']
        eps = 1e-12

        for slots in pair_maps:
            for tensors in slots.values():
                q_param = tensors.get('q')
                k_param = tensors.get('k')
                if q_param is None or k_param is None:
                    continue

                q_norm = torch.linalg.norm(q_param).item()
                k_norm = torch.linalg.norm(k_param).item()
                score = q_norm * k_norm
                if score <= threshold or not math.isfinite(score):
                    continue

                ratio = max(threshold / (score + eps), 0.0)
                if ratio >= 1.0:
                    continue

                scale_q = ratio ** alpha
                scale_k = ratio ** (1.0 - alpha)
                q_param.data.mul_(scale_q)
                k_param.data.mul_(scale_k)
                total_clipped += 1

        if total_clipped:
            if not self._muon_clip_logged:
                logger.info(
                    f"MuonClip active: threshold={threshold}, alpha={alpha} "
                    f"(clipped {total_clipped} Q/K pairs)"
                )
                self._muon_clip_logged = True
            else:
                logger.debug(f"MuonClip clipped {total_clipped} Q/K pairs this step")

    def _log_profiling_results(self):
        """Log detailed profiling breakdown."""
        if not self.enable_profiling or len(self.profiling_data['total_step_times']) == 0:
            return

        import statistics
        import json

        # Calculate averages
        n = len(self.profiling_data['total_step_times'])
        avg_teacher = statistics.mean(self.profiling_data['teacher_times'])
        avg_student_fwd = statistics.mean(self.profiling_data['student_fwd_times'])
        avg_loss = statistics.mean(self.profiling_data['loss_comp_times'])
        avg_backward = statistics.mean(self.profiling_data['backward_times'])
        avg_optimizer = statistics.mean(self.profiling_data['optimizer_times'])
        avg_total = statistics.mean(self.profiling_data['total_step_times'])

        # Calculate percentages
        teacher_pct = (avg_teacher / avg_total) * 100
        student_fwd_pct = (avg_student_fwd / avg_total) * 100
        loss_pct = (avg_loss / avg_total) * 100
        backward_pct = (avg_backward / avg_total) * 100
        optimizer_pct = (avg_optimizer / avg_total) * 100
        accounted = teacher_pct + student_fwd_pct + loss_pct + backward_pct + optimizer_pct
        other_pct = 100 - accounted

        # Log detailed breakdown
        logger.info("=" * 80)
        logger.info(f"PROFILING BREAKDOWN (averaged over {n} steps)")
        logger.info("=" * 80)
        logger.info(f"Teacher inference:   {avg_teacher:.4f}s ({teacher_pct:5.1f}%)")
        logger.info(f"Student forward:     {avg_student_fwd:.4f}s ({student_fwd_pct:5.1f}%)")
        logger.info(f"Loss computation:    {avg_loss:.4f}s ({loss_pct:5.1f}%)")
        logger.info(f"Backward pass:       {avg_backward:.4f}s ({backward_pct:5.1f}%)")
        logger.info(f"Optimizer step:      {avg_optimizer:.4f}s ({optimizer_pct:5.1f}%)")
        logger.info(f"Other/overhead:      {avg_total - avg_teacher - avg_student_fwd - avg_loss - avg_backward - avg_optimizer:.4f}s ({other_pct:5.1f}%)")
        logger.info("-" * 80)
        logger.info(f"TOTAL per step:      {avg_total:.4f}s (100.0%)")
        logger.info("=" * 80)

        # Determine bottleneck
        if teacher_pct > 40:
            logger.info("CONCLUSION: Teacher is the bottleneck (>40% of step time)")
            logger.info("RECOMMENDATION: Pursue offline distillation or async teacher inference")
        elif student_fwd_pct + backward_pct > 60:
            logger.info("CONCLUSION: Student architecture is the bottleneck (>60% of step time)")
            logger.info("RECOMMENDATION: Profile student in isolation and optimize architecture")
        else:
            logger.info("CONCLUSION: Mixed bottleneck - both teacher and student contribute")
            logger.info("RECOMMENDATION: Optimize both distillation setup and student architecture")

        # Save to JSON
        results = {
            'num_steps': n,
            'breakdown': {
                'teacher_inference': {'time_s': avg_teacher, 'percentage': teacher_pct},
                'student_forward': {'time_s': avg_student_fwd, 'percentage': student_fwd_pct},
                'loss_computation': {'time_s': avg_loss, 'percentage': loss_pct},
                'backward_pass': {'time_s': avg_backward, 'percentage': backward_pct},
                'optimizer_step': {'time_s': avg_optimizer, 'percentage': optimizer_pct},
                'other_overhead': {'time_s': avg_total - avg_teacher - avg_student_fwd - avg_loss - avg_backward - avg_optimizer, 'percentage': other_pct},
            },
            'total_per_step': {'time_s': avg_total},
            'raw_data': self.profiling_data,
        }

        # Save results
        output_path = Path("TESTING/results/profiling_breakdown.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Profiling results saved to: {output_path}")
        logger.info("=" * 80)

    def train(self) -> Dict[str, Any]:
        """Main training loop.

        Returns:
            Training statistics
        """
        logger.info("Starting training...")
        logger.info(f"  Max steps: {self.config.max_steps}")
        logger.info(f"  Effective batch size: {self.config.effective_batch_size}")

        self.model.train()

        # Training state
        accumulation_step = 0
        epoch_losses = []
        epoch_diversity_losses = []  # Track diversity losses
        step_start_time = time.time()

        # Training loop with async prefetch support
        while self.global_step < self.config.max_steps:
            # Convert dataloader to iterator for batch peeking
            dataloader_iter = iter(self.train_dataloader)

            # Cold start: fetch first batch
            try:
                current_batch = next(dataloader_iter)
            except StopIteration:
                logger.warning("Empty dataloader, ending epoch")
                self.epoch += 1
                continue

            # Track if we have a pending async request
            has_pending = False

            while True:
                # Peek next batch for prefetch
                try:
                    next_batch = next(dataloader_iter)
                except StopIteration:
                    next_batch = None

                # Update learning rate
                self._update_learning_rate()

                # Get teacher logits (from previous async request or sync fetch)
                teacher_logits = None
                if self.async_teacher_enabled and has_pending:
                    try:
                        # Retrieve async result
                        teacher_logits = self.teacher_client.get()
                        if not hasattr(self, '_async_first_fetch_logged'):
                            logger.info("Async teacher prefetch: successfully retrieved first prefetched batch")
                            self._async_first_fetch_logged = True
                    except Exception as e:
                        logger.warning(f"Async teacher get() failed, falling back to sync: {e}")
                        teacher_logits = None
                        self.async_teacher_enabled = False  # Disable async for remainder of training

                # Execute training step (will prefetch next batch if available)
                metrics = self._train_step(
                    current_batch,
                    accumulation_step,
                    teacher_logits=teacher_logits,
                    next_batch=next_batch,
                )

                # FIX: Store loss as tensor (no CPU sync)
                epoch_losses.append(metrics['loss'])
                # Track diversity loss if present
                if 'diversity_loss' in metrics:
                    epoch_diversity_losses.append(metrics['diversity_loss'])

                accumulation_step += 1

                # Optimizer step after accumulation (T038)
                if accumulation_step >= self.config.gradient_accumulation_steps:
                    # DIAGNOSTIC: Check param before step
                    # FIX: Removed .item() to avoid CPU sync
                    if self.global_step < 3:
                        param_before = next(iter(self.model.parameters())).clone().detach()

                    # Clip gradients and step optimizer
                    grad_norm = self._optimizer_step()

                    # DIAGNOSTIC: Check param after step
                    # FIX: Removed .item() call
                    if self.global_step < 3:
                        param_after = next(iter(self.model.parameters()))
                        param_change = (param_after - param_before).abs().max()
                        logger.info(f"[DIAG Step {self.global_step}] Param changed (checking on GPU)")

                    # Update global step
                    self.global_step += 1
                    accumulation_step = 0

                    # Update loss parameters according to schedules
                    self._update_loss_parameters(self.global_step)

                    # Compute timing
                    step_end_time = time.time()
                    steps_per_sec = 1.0 / (step_end_time - step_start_time)
                    step_start_time = step_end_time

                    # Log metrics
                    if self.global_step % 64 == 0:
                        # FIX: Only now convert tensors to scalars (batched CPU sync)
                        # Convert recent losses from tensors to scalars
                        recent_losses = epoch_losses[-64:]
                        recent_losses_scalar = [l.item() if torch.is_tensor(l) else l for l in recent_losses]
                        avg_loss = sum(recent_losses_scalar) / len(recent_losses_scalar)
                        grad_norm_scalar = grad_norm.item()  # Convert grad_norm tensor to scalar
                        lr = self._get_learning_rate()

                        # Compute average diversity loss if present
                        avg_diversity_loss = None
                        if epoch_diversity_losses:
                            recent_div_losses = epoch_diversity_losses[-64:]
                            if recent_div_losses:
                                recent_div_losses_scalar = [l.item() if torch.is_tensor(l) else l for l in recent_div_losses]
                                avg_diversity_loss = sum(recent_div_losses_scalar) / len(recent_div_losses_scalar)

                        # Build log message
                        log_msg = (
                            f"Step {self.global_step}/{self.config.max_steps} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"Grad norm: {grad_norm_scalar:.4f} | "
                            f"LR: {lr:.2e} | "
                            f"Steps/sec: {steps_per_sec:.2f}"
                        )
                        if avg_diversity_loss is not None:
                            log_msg += f" | Div loss: {avg_diversity_loss:.6f}"

                        logger.info(log_msg)

                        # CRITICAL FIX: Monitor system memory to detect leaks
                        if HAS_PSUTIL:
                            sys_mem = psutil.virtual_memory()
                            sys_mem_used_gb = sys_mem.used / (1024 ** 3)
                            sys_mem_percent = sys_mem.percent

                            swap = psutil.swap_memory()
                            swap_used_gb = swap.used / (1024 ** 3)

                            # Log system memory status
                            gpu_mem_allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                            logger.info(f"Memory: GPU {gpu_mem_allocated_gb:.1f}GB | RAM {sys_mem_used_gb:.1f}GB ({sys_mem_percent:.1f}%) | Swap {swap_used_gb:.1f}GB")

                            # CRITICAL: Alert if swap is being used (indicates memory leak!)
                            if swap_used_gb > 1.0:
                                logger.warning(f"⚠️  SWAP IN USE: {swap_used_gb:.1f}GB - MEMORY LEAK DETECTED!")
                                logger.warning(f"⚠️  System RAM: {sys_mem_used_gb:.1f}GB used of {sys_mem.total / (1024 ** 3):.1f}GB total")

                        # Record metrics
                        self.metrics['loss'].append(avg_loss)
                        self.metrics['grad_norm'].append(grad_norm_scalar)
                        self.metrics['learning_rate'].append(lr)
                        self.metrics['steps_per_sec'].append(steps_per_sec)

                        # Add diversity loss to metrics dict if tracking
                        if avg_diversity_loss is not None:
                            if 'diversity_loss' not in self.metrics:
                                self.metrics['diversity_loss'] = []
                            self.metrics['diversity_loss'].append(avg_diversity_loss)

                    # CRITICAL FIX: Clean up parquet file cache to prevent memory leak
                    if self.global_step % 1000 == 0:
                        # Check if dataset has cleanup method (PretokenizedShardDataset)
                        if hasattr(self.train_dataset, 'cleanup_file_cache'):
                            self.train_dataset.cleanup_file_cache()
                            logger.info(f"Step {self.global_step}: Cleaned up parquet file cache to prevent memory leak")

                    # Evaluation (T059)
                    if self.global_step % self.config.eval_interval == 0:
                        # Run full evaluation suite if enabled
                        if self.enable_full_eval and self.evaluation_runner is not None:
                            logger.info("=" * 80)
                            logger.info(f"Running full evaluation at step {self.global_step}")
                            logger.info("=" * 80)

                            # Configure evaluation
                            perplexity_config = PerplexityConfig(max_samples=1000)
                            niah_config = NIAHConfig(
                                context_length=4096,
                                num_samples=20,  # Quick NIAH test
                            )

                            # Run evaluation
                            eval_results = self.evaluation_runner.run_all(
                                val_dataloader=self.val_dataloader,
                                perplexity_config=perplexity_config,
                                niah_config=niah_config,
                                output_dir=self.checkpoint_dir / "eval" if self.checkpoint_dir else None,
                                step=self.global_step,
                            )

                            # Update best validation loss
                            if 'perplexity' in eval_results and 'loss' in eval_results['perplexity']:
                                val_loss = eval_results['perplexity']['loss']
                                if val_loss < self.best_val_loss:
                                    self.best_val_loss = val_loss
                                    logger.info(f"New best validation loss: {val_loss:.4f}")

                        # Fallback: simple validation if full eval disabled
                        elif self.val_dataloader:
                            val_loss = self.evaluate()
                            logger.info(f"Validation loss: {val_loss:.4f}")

                            # Save best model
                            if val_loss < self.best_val_loss:
                                self.best_val_loss = val_loss
                                logger.info(f"New best validation loss: {val_loss:.4f}")

                    # Checkpointing (handled externally via checkpoint.py)
                    # This is just a hook for external checkpoint saving
                    if self.checkpoint_dir and self.global_step % self.config.save_interval == 0:
                        logger.info(f"Checkpoint hook: step {self.global_step}")

                    # Check if training is complete
                    if self.global_step >= self.config.max_steps:
                        break

                # Advance to next batch
                current_batch = next_batch
                has_pending = (next_batch is not None and self.async_teacher_enabled)

                # Check if we've reached end of epoch
                if current_batch is None:
                    break

            self.epoch += 1

        logger.info("Training complete!")
        logger.info(f"  Total steps: {self.global_step}")
        logger.info(f"  Total epochs: {self.epoch}")

        # PROFILING: Log final profiling results
        self._log_profiling_results()

        # Convert final loss to scalar if it's a tensor
        final_loss = epoch_losses[-1] if epoch_losses else None
        if final_loss is not None and torch.is_tensor(final_loss):
            final_loss = final_loss.item()

        return {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'final_loss': final_loss,
            'best_val_loss': self.best_val_loss,
        }

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate model on validation set.

        Returns:
            Average validation loss
        """
        if not self.val_dataloader:
            logger.warning("No validation dataloader provided, skipping evaluation")
            return float('inf')

        logger.info("Running validation...")
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in self.val_dataloader:
            # Move batch to device (non-blocking for async transfer)
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device, non_blocking=True)
            labels = batch.get('labels', batch['input_ids']).to(self.device, non_blocking=True)

            # Fetch teacher log-probabilities
            teacher_topk_indices, teacher_topk_logprobs, teacher_other_logprob = \
                self._fetch_teacher_logits(input_ids)

            # Forward pass
            # Handle PyTorch version compatibility
            try:
                autocast_ctx = autocast('cuda', dtype=torch.bfloat16, enabled=self.config.use_bf16)
            except TypeError:
                autocast_ctx = autocast(enabled=self.config.use_bf16)

            with autocast_ctx:
                if hasattr(self.model, 'forward_train'):
                    hidden_states = self.model.forward_train(input_ids)
                    if hasattr(self.model, 'lm_head'):
                        student_logits = self.model.lm_head(hidden_states)
                    else:
                        raise ValueError("Model must have 'lm_head' or return logits directly")
                else:
                    outputs = self.model(input_ids, step=self.global_step)
                    if hasattr(outputs, 'logits'):
                        student_logits = outputs.logits
                    else:
                        student_logits = outputs

                # Compute loss
                loss = self.loss_fn(
                    student_logits=student_logits,
                    teacher_topk_indices=teacher_topk_indices,
                    teacher_topk_logprobs=teacher_topk_logprobs,
                    teacher_other_logprob=teacher_other_logprob,
                    hard_targets=labels,
                    attention_mask=attention_mask,  # FIX: Pass mask for validation too
                )

                # MEMORY FIX: Free student logits after loss computation (evaluation mode)
                del student_logits
                # REMOVED: caused memory fragmentation in eval loop
                # torch.cuda.empty_cache()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

        self.model.train()
        return avg_loss

    def get_state_dict(self) -> Dict[str, Any]:
        """Get trainer state for checkpointing.

        Returns:
            State dictionary containing all training state
        """
        state_dict = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'metrics': self.metrics,
            'config': {
                'training_config': vars(self.config),
                # Bug #3 fix: Add defensive None check for Titan models (student_config is None)
                'student_config': self.student_config.to_dict() if self.student_config else None,
                # CE pretrain mode metadata for provenance tracking
                'pretrain_ce_only': getattr(self.config, 'pretrain_ce_only', False),
                'tokenizer_name': getattr(self.tokenizer, 'name_or_path', None) if self.tokenizer else None,
            },
            'rng_states': {
                'python': None,  # Will be set by checkpoint.py
                'numpy': None,   # Will be set by checkpoint.py
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            }
        }

        # Save scheduler state if scheduler is provided
        if self.scheduler is not None:
            state_dict['scheduler_state_dict'] = self.scheduler.state_dict()

        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load trainer state from checkpoint.

        Args:
            state_dict: State dictionary from checkpoint
        """
        self.global_step = state_dict['global_step']
        self.epoch = state_dict['epoch']
        self.best_val_loss = state_dict['best_val_loss']
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.scaler.load_state_dict(state_dict['scaler_state_dict'])
        self.metrics = state_dict.get('metrics', self.metrics)

        # Restore scheduler state if present
        if self.scheduler is not None and 'scheduler_state_dict' in state_dict:
            self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
            logger.info("Loaded scheduler state from checkpoint")
        elif self.scheduler is not None:
            logger.warning("Scheduler is active but no scheduler state found in checkpoint")

        # Restore RNG states
        # Note: RNG restoration is primarily handled by CheckpointManager._restore_rng_states()
        # This is a fallback for direct state_dict loading
        rng_states = state_dict.get('rng_states', {})

        try:
            if rng_states.get('torch') is not None:
                torch_rng = rng_states['torch']
                # FIX: Ensure RNG state is on CPU (same fix as in checkpoint.py)
                if isinstance(torch_rng, torch.Tensor):
                    if torch_rng.device.type != 'cpu':
                        torch_rng = torch_rng.cpu()
                    if torch_rng.dtype != torch.uint8:
                        logger.warning(f"Unexpected torch RNG state dtype: {torch_rng.dtype}, converting to uint8")
                        torch_rng = torch_rng.to(torch.uint8)
                    torch.set_rng_state(torch_rng)
                    logger.debug("Restored PyTorch RNG state")

            if rng_states.get('cuda') is not None and torch.cuda.is_available():
                cuda_rng = rng_states['cuda']
                # FIX: Ensure CUDA RNG state is on CPU (same fix as in checkpoint.py)
                if isinstance(cuda_rng, torch.Tensor):
                    if cuda_rng.device.type != 'cpu':
                        cuda_rng = cuda_rng.cpu()
                    if cuda_rng.dtype != torch.uint8:
                        logger.warning(f"Unexpected CUDA RNG state dtype: {cuda_rng.dtype}, converting to uint8")
                        cuda_rng = cuda_rng.to(torch.uint8)
                    torch.cuda.set_rng_state(cuda_rng)
                    logger.debug("Restored CUDA RNG state")
        except Exception as e:
            logger.warning(f"Failed to restore RNG states: {e}")
            logger.warning("Training will continue without RNG state restoration")

        logger.info(f"Loaded checkpoint: step={self.global_step}, epoch={self.epoch}")

