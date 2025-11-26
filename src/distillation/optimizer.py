"""
Optimizer and learning rate scheduler for distillation training.

Implements 8-bit AdamW optimizer with cosine annealing and linear warmup
for memory-efficient training on 32GB VRAM.

Tasks: T044-T046
"""

import logging
import math
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from packaging import version

try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False
    logging.warning(
        "bitsandbytes not available. 8-bit optimizer will fall back to standard AdamW. "
        "Install with: pip install bitsandbytes>=0.41.0"
    )

try:
    from torch.optim import Muon
    HAS_MUON = True
except ImportError:
    HAS_MUON = False
    logging.warning(
        "Muon not available. Requires PyTorch 2.9+ or install with: pip install muon-optimizer"
    )


logger = logging.getLogger(__name__)


_MIN_TORCH_VERSION_FOR_MUON = version.parse("2.1.0")
_MUON_SMOKE_TEST_PASSED = False


def _ensure_muon_ready(run_smoke_test: bool = True):
    """Validate Muon availability and run a minimal smoke test."""
    global _MUON_SMOKE_TEST_PASSED

    if not HAS_MUON:
        raise RuntimeError(
            "Muon optimizer requested but not installed. Install with: pip install muon-optimizer"
        )

    torch_version = version.parse(torch.__version__.split("+")[0])
    if torch_version < _MIN_TORCH_VERSION_FOR_MUON:
        raise RuntimeError(
            f"Muon requires torch>={_MIN_TORCH_VERSION_FOR_MUON}, found {torch.__version__}"
        )

    if not run_smoke_test or _MUON_SMOKE_TEST_PASSED:
        return

    try:
        test_param = torch.nn.Parameter(torch.randn(4, 4))
        opt = Muon(
            [{"params": [test_param]}],
            lr=1e-3,
            weight_decay=0.0,
            momentum=0.9,
            nesterov=True,
            ns_steps=1,
        )
        loss = (test_param ** 2).sum()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        _MUON_SMOKE_TEST_PASSED = True
        logger.info("Muon smoke test passed")
    except Exception as exc:
        raise RuntimeError("Muon smoke test failed. Check installation.") from exc


def create_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    use_8bit: bool = True,
    no_decay_bias_and_norm: bool = True,
    decay_param_lr_multiplier: float = 10.0,  # Boost retention decay LR without saturating
    optimizer_type: str = "adamw",  # "adamw" or "muon"
    muon_momentum: float = 0.95,
    muon_grad_clip: float = 1.0,
    muon_zero_clip_percent: float = 0.0,
    muon_aux_lr_scale: float = 0.25,
    muon_ready_check: bool = True,
) -> tuple:
    """Create optimizer with support for AdamW (8-bit or standard) or Muon.

    Uses bitsandbytes AdamW8bit for ~75% memory reduction in optimizer states,
    critical for fitting training in 32GB VRAM budget.

    Muon is an alternative optimizer that can avoid parameter fuzzing.

    Args:
        model: Model to optimize
        lr: Learning rate (default: 1e-4)
        betas: Adam beta coefficients (default: (0.9, 0.999))
        eps: Adam epsilon for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0.01)
        use_8bit: Use 8-bit optimizer (default: True, requires bitsandbytes)
        no_decay_bias_and_norm: Exclude biases and norms from weight decay (default: True)
        optimizer_type: Optimizer type - "adamw" or "muon" (default: "adamw")
        muon_momentum: Muon momentum/beta (default: 0.95)
        muon_grad_clip: Muon gradient clipping (default: 1.0)
        muon_zero_clip_percent: Muon zero clip percent (default: 0.0)

    Returns:
        Optimizer instance (AdamW8bit, AdamW, or Muon)

    Example:
        >>> model = RetNetBackbone(config)
        >>> optimizer = create_optimizer(model, lr=1e-4, use_8bit=True)
        >>> # Memory savings: ~75% reduction vs FP32 optimizer
        >>> # For 1B params: ~8GB optimizer state (8-bit) vs ~32GB (FP32)

    Note:
        8-bit optimizer requires bitsandbytes>=0.41.0
        Memory calculation:
        - FP32 AdamW: 8 bytes/param * 4 states (params, gradients, m, v) = 32 bytes/param
        - 8-bit AdamW: 2 bytes/param * 4 states = 8 bytes/param
        - Savings: 24 bytes/param = 75% reduction
    """
    # Parameter groups: separate norm/decay/no-decay/retention-decay (TIER 3)
    if no_decay_bias_and_norm:
        # Separate parameters into 4 groups for differential learning rates
        norm_params = []
        retention_decay_params = []  # TIER 3: NEW GROUP
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # LayerNorm weights get higher LR and zero weight decay
            if 'norm' in name.lower() and 'weight' in name:
                norm_params.append(param)
            # Retention decay parameters get higher LR to compensate for tiny grads
            # Match both 'raw_log_decay' (custom impl) and 'retention.decay' (TorchScale)
            elif 'raw_log_decay' in name or ('decay' in name and 'retention' in name.lower()):
                retention_decay_params.append(param)
                logger.info(f"  [Retention Decay] Found param: {name} ({param.numel()} params)")
            # Biases and other norm parameters (bias) get zero weight decay
            elif 'bias' in name.lower() or 'layer_norm' in name.lower() or 'layernorm' in name.lower():
                no_decay_params.append(param)
            # Everything else gets standard weight decay
            else:
                decay_params.append(param)

        optimizer_grouped_parameters = [
            {
                'params': norm_params,
                'lr': lr * 10,  # 10x higher LR for LayerNorm (respects config base LR)
                'weight_decay': 0.0,
                'name': 'layer_norm',
            },
            {
                'params': retention_decay_params,
                'lr': lr * decay_param_lr_multiplier,
                'weight_decay': 0.0,  # Critical: no weight decay for bounded params
                'name': 'retention_decay',
            },
            {
                'params': decay_params,
                'lr': lr,  # Explicit base LR from config
                'weight_decay': weight_decay,
                'name': 'standard',
            },
            {
                'params': no_decay_params,
                'lr': lr,  # Explicit base LR from config
                'weight_decay': 0.0,
                'name': 'no_decay',
            }
        ]

        # Count parameters in each group
        num_norm = sum(p.numel() for p in norm_params)
        num_retention_decay = sum(p.numel() for p in retention_decay_params)  # TIER 3
        num_decay = sum(p.numel() for p in decay_params)
        num_no_decay = sum(p.numel() for p in no_decay_params)

        logger.info(f"Optimizer parameter groups:")
        logger.info(f"  LayerNorm (10x LR): {num_norm:,} params")
        logger.info(f"  Retention decay ({decay_param_lr_multiplier:.0f}x LR): {num_retention_decay:,} params")
        logger.info(f"  With weight decay: {num_decay:,} params")
        logger.info(f"  Without weight decay (bias): {num_no_decay:,} params")
    else:
        # Single group with uniform weight decay
        optimizer_grouped_parameters = [
            {
                'params': [p for p in model.parameters() if p.requires_grad],
                'weight_decay': weight_decay,
            }
        ]

        num_params = sum(p.numel() for p in optimizer_grouped_parameters[0]['params'])
        logger.info(f"Optimizer: {num_params:,} trainable params")

    # Create optimizer (Muon, 8-bit AdamW, or standard AdamW)
    if optimizer_type == "muon":
        if muon_ready_check:
            _ensure_muon_ready(run_smoke_test=True)

        # Muon only supports ≥2D parameters (weight matrices)
        # Filter parameter groups to only include 2D+ parameters for Muon
        muon_groups = []
        adamw_groups = []

        for group in optimizer_grouped_parameters:
            muon_params = [p for p in group['params'] if p.ndim >= 2]
            adamw_params = [p for p in group['params'] if p.ndim < 2]

            if muon_params:
                muon_group = group.copy()
                muon_group['params'] = muon_params
                muon_groups.append(muon_group)

            if adamw_params:
                adamw_group = group.copy()
                adamw_group['params'] = adamw_params
                adamw_group['lr'] = group.get('lr', lr) * muon_aux_lr_scale
                adamw_groups.append(adamw_group)

        # Count parameters
        num_muon = sum(p.numel() for g in muon_groups for p in g['params'])
        num_adamw = sum(p.numel() for g in adamw_groups for p in g['params'])

        # Create Muon for 2D+ parameters
        optimizer = Muon(
            muon_groups,
            lr=lr,
            weight_decay=weight_decay,
            momentum=muon_momentum,
            nesterov=True,
            ns_steps=5,
        )
        logger.info(f"Created Muon optimizer")
        logger.info(f"  Base LR: {lr:.2e}, momentum: {muon_momentum}, weight_decay: {weight_decay}")
        logger.info(f"  Newton-Schulz steps: 5, nesterov: True")
        logger.info(f"  Muon params (≥2D): {num_muon:,}")
        logger.info(f"  AdamW params (<2D): {num_adamw:,}")

        # Create AdamW for <2D parameters (biases, norms, etc.)
        if adamw_groups:
            if use_8bit and HAS_BITSANDBYTES:
                adamw_optimizer = bnb.optim.AdamW8bit(
                    adamw_groups,
                    lr=lr,
                    betas=betas,
                    eps=eps,
                )
                logger.info("Created 8-bit AdamW optimizer for <2D params")
            else:
                adamw_optimizer = torch.optim.AdamW(
                    adamw_groups,
                    lr=lr,
                    betas=betas,
                    eps=eps,
                )
                logger.info("Created standard AdamW optimizer for <2D params")

            # Return hybrid optimizer (Muon + AdamW)
            # We'll need to handle this in the training loop
            from distillation.hybrid_optimizer import HybridOptimizer
            optimizer = HybridOptimizer([optimizer, adamw_optimizer])
            logger.info(f"Using hybrid Muon+AdamW optimizer")

        # Log per-group learning rates for verification
        logger.info(f"  Parameter group learning rates:")
        if no_decay_bias_and_norm:
            logger.info(f"    LayerNorm (10x): {lr * 10:.2e}")
            logger.info(f"    Retention decay ({decay_param_lr_multiplier:.0f}x): {lr * decay_param_lr_multiplier:.2e}")
            logger.info(f"    Decay (1x): {lr:.2e}")
            logger.info(f"    No-decay (1x): {lr:.2e}")

    elif use_8bit and HAS_BITSANDBYTES:
        optimizer = bnb.optim.AdamW8bit(
            optimizer_grouped_parameters,
            lr=lr,
            betas=betas,
            eps=eps,
        )
        logger.info(f"Created 8-bit AdamW optimizer (bitsandbytes)")
        logger.info(f"  Memory savings: ~75% vs FP32 AdamW")
        logger.info(f"  Base LR: {lr:.2e}, betas: {betas}, weight_decay: {weight_decay}")
        # Log per-group learning rates for verification
        logger.info(f"  Parameter group learning rates:")
        if no_decay_bias_and_norm:
            logger.info(f"    LayerNorm (10x): {lr * 10:.2e}")
            logger.info(f"    Retention decay ({decay_param_lr_multiplier:.0f}x): {lr * decay_param_lr_multiplier:.2e}")
            logger.info(f"    Decay (1x): {lr:.2e}")
            logger.info(f"    No-decay (1x): {lr:.2e}")
    else:
        if use_8bit and not HAS_BITSANDBYTES:
            logger.warning(
                "8-bit optimizer requested but bitsandbytes not available. "
                "Falling back to standard AdamW."
            )

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=lr,
            betas=betas,
            eps=eps,
        )
        logger.info(f"Created standard AdamW optimizer")
        logger.info(f"  Base LR: {lr:.2e}, betas: {betas}, weight_decay: {weight_decay}")
        # Log per-group learning rates for verification
        logger.info(f"  Parameter group learning rates:")
        if no_decay_bias_and_norm:
            logger.info(f"    LayerNorm (10x): {lr * 10:.2e}")
            logger.info(f"    Retention decay ({decay_param_lr_multiplier:.0f}x): {lr * decay_param_lr_multiplier:.2e}")
            logger.info(f"    Decay (1x): {lr:.2e}")
            logger.info(f"    No-decay (1x): {lr:.2e}")

    # TIER 3 FIX: Return metadata for potential per-group clipping
    # For now, metadata is minimal (just for backward compatibility)
    metadata = {'lr_multiplier': decay_param_lr_multiplier}
    return optimizer


class CosineAnnealingWarmRestartsWithWarmup(_LRScheduler):
    """Cosine annealing learning rate scheduler with warm restarts, linear warmup, and optional plateau.

    Combines:
    1. Linear warmup: LR increases from 0 to base_lr over warmup_steps
    2. Plateau (optional): LR held constant at base_lr for plateau_steps
    3. Cosine annealing with warm restarts: LR decays following cosine curve,
       then restarts at base_lr with progressively longer cycles

    Schedule (with plateau):
    - Warmup: steps 0-9 (linear 0 -> base_lr)
    - Plateau: steps 10-30009 (30k steps at base_lr)
    - Cycle 1: steps 30010-40009 (10k steps, cosine decay)
    - Cycle 2: steps 40010-60009 (20k steps, cosine decay)
    - Cycle 3: steps 60010-100009 (40k steps, cosine decay)
    - ...

    Schedule (without plateau, plateau_steps=0):
    - Warmup: steps 0-999 (linear 0 -> base_lr)
    - Cycle 1: steps 1000-10999 (10k steps, cosine decay)
    - Cycle 2: steps 11000-30999 (20k steps, cosine decay)
    - Cycle 3: steps 31000-70999 (40k steps, cosine decay)
    - ...

    Args:
        optimizer: Optimizer to schedule
        warmup_steps: Number of linear warmup steps (default: 1000)
        plateau_steps: Number of steps to hold at max LR after warmup (default: 0)
        T_0: Steps in first cosine cycle after warmup+plateau (default: 10000)
        T_mult: Multiplier for cycle length after each restart (default: 2)
        eta_min: Minimum learning rate (default: 1e-6, ~10% of typical lr=1e-4)
        last_epoch: Index of last epoch (default: -1)

    Example:
        >>> optimizer = create_optimizer(model, lr=1e-4)
        >>> scheduler = CosineAnnealingWarmRestartsWithWarmup(
        ...     optimizer,
        ...     warmup_steps=1000,
        ...     plateau_steps=30000,
        ...     T_0=10000,
        ...     T_mult=2,
        ...     eta_min=1e-6,
        ... )
        >>> for epoch in range(num_epochs):
        ...     for batch in dataloader:
        ...         loss = train_step(batch)
        ...         loss.backward()
        ...         optimizer.step()
        ...         scheduler.step()  # Call after each optimizer step
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 1000,
        plateau_steps: int = 0,
        T_0: int = 10000,
        T_mult: int = 2,
        eta_min: float = 1e-6,
        last_epoch: int = -1,
    ):
        """Initialize scheduler.

        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of linear warmup steps
            plateau_steps: Number of steps to hold at max LR after warmup (default: 0)
            T_0: Steps in first cosine cycle after warmup+plateau
            T_mult: Multiplier for cycle length after restart
            eta_min: Minimum learning rate
            last_epoch: Index of last epoch (-1 = start from beginning)
        """
        self.warmup_steps = warmup_steps
        self.plateau_steps = plateau_steps
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min

        # Base learning rates (stored in optimizer)
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        super().__init__(optimizer, last_epoch)

        logger.info(f"Created cosine annealing scheduler with warm restarts:")
        logger.info(f"  Warmup steps: {warmup_steps}")
        logger.info(f"  Plateau steps: {plateau_steps}")
        logger.info(f"  T_0 (first cycle): {T_0}")
        logger.info(f"  T_mult (cycle multiplier): {T_mult}")
        logger.info(f"  Min LR: {eta_min:.2e}")
        if plateau_steps > 0:
            logger.info(f"  Schedule: warmup ({warmup_steps}) -> plateau ({plateau_steps}) -> decay")
        else:
            logger.info(f"  Schedule: warmup ({warmup_steps}) -> decay")
        logger.info(f"  Cycle schedule: {T_0} -> {T_0*T_mult} -> {T_0*T_mult**2} -> ...")

    def get_lr(self) -> List[float]:
        """Compute learning rate for current step.

        Returns:
            List of learning rates (one per parameter group)
        """
        # Current step (starts at 0 after first .step() call)
        # Note: last_epoch is -1 initially, becomes 0 after first step()
        step = max(0, self.last_epoch)

        # Phase 1: Linear warmup
        if step < self.warmup_steps:
            # Linear warmup from 0 to base_lr
            # Handle step 0 specially to avoid division by zero
            if step == 0:
                warmup_factor = 0.0
            else:
                warmup_factor = step / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        # Phase 2: Plateau (constant LR at max_lr)
        if step < self.warmup_steps + self.plateau_steps:
            # Hold at base_lr for plateau_steps
            return self.base_lrs

        # Phase 3: Cosine annealing with warm restarts
        # Adjust step to account for warmup and plateau
        step_after_warmup = step - self.warmup_steps - self.plateau_steps

        # Determine which cycle we're in and position within cycle
        # We need to recompute this each time to handle state loads correctly
        current_cycle = 0
        cycle_start_step = 0
        cycle_length = self.T_0

        while step_after_warmup >= cycle_start_step + cycle_length:
            cycle_start_step += cycle_length
            current_cycle += 1
            cycle_length = self.T_0 * (self.T_mult ** current_cycle)

        # Position within current cycle
        step_in_cycle = step_after_warmup - cycle_start_step

        # Cosine annealing within current cycle
        # lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * step_in_cycle / cycle_length))
        cos_factor = 0.5 * (1 + math.cos(math.pi * step_in_cycle / cycle_length))

        lrs = [
            self.eta_min + (base_lr - self.eta_min) * cos_factor
            for base_lr in self.base_lrs
        ]

        return lrs

    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state for checkpointing.

        Returns:
            State dictionary with all scheduler state
        """
        state = {
            'warmup_steps': self.warmup_steps,
            'plateau_steps': self.plateau_steps,
            'T_0': self.T_0,
            'T_mult': self.T_mult,
            'eta_min': self.eta_min,
            'base_lrs': self.base_lrs,
            'last_epoch': self.last_epoch,
        }
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load scheduler state from checkpoint.

        Args:
            state_dict: State dictionary from checkpoint
        """
        self.warmup_steps = state_dict['warmup_steps']
        self.plateau_steps = state_dict.get('plateau_steps', 0)  # Backward compatible
        self.T_0 = state_dict['T_0']
        self.T_mult = state_dict['T_mult']
        self.eta_min = state_dict['eta_min']
        self.base_lrs = state_dict['base_lrs']
        self.last_epoch = state_dict['last_epoch']


def create_scheduler(
    optimizer: Optimizer,
    warmup_steps: int = 1000,
    plateau_steps: int = 0,
    T_0: int = 10000,
    T_mult: int = 2,
    eta_min: float = 1e-6,
) -> CosineAnnealingWarmRestartsWithWarmup:
    """Create learning rate scheduler with warmup, optional plateau, and cosine annealing.

    Args:
        optimizer: Optimizer to schedule
        warmup_steps: Number of linear warmup steps (default: 1000)
        plateau_steps: Number of steps to hold at max LR after warmup (default: 0)
        T_0: Steps in first cosine cycle (default: 10000)
        T_mult: Multiplier for cycle length after restart (default: 2)
        eta_min: Minimum learning rate (default: 1e-6)

    Returns:
        Learning rate scheduler

    Example:
        >>> optimizer = create_optimizer(model, lr=1e-4)
        >>> scheduler = create_scheduler(optimizer, warmup_steps=1000, plateau_steps=30000, T_0=10000)
        >>> # Training loop
        >>> for step in range(max_steps):
        ...     loss = train_step(batch)
        ...     loss.backward()
        ...     optimizer.step()
        ...     scheduler.step()  # Update LR after each step
    """
    scheduler = CosineAnnealingWarmRestartsWithWarmup(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        plateau_steps=plateau_steps,
        T_0=T_0,
        T_mult=T_mult,
        eta_min=eta_min,
    )

    return scheduler


def get_optimizer_memory_footprint(optimizer: Optimizer) -> Dict[str, float]:
    """Estimate optimizer memory footprint.

    Args:
        optimizer: Optimizer instance

    Returns:
        Dictionary with memory estimates in GB:
            - 'state_memory_gb': Memory for optimizer state (m, v, etc.)
            - 'param_memory_gb': Memory for parameters
            - 'total_memory_gb': Total optimizer memory

    Example:
        >>> optimizer = create_optimizer(model, use_8bit=True)
        >>> memory = get_optimizer_memory_footprint(optimizer)
        >>> print(f"Optimizer memory: {memory['total_memory_gb']:.2f} GB")
    """
    # Count parameters
    total_params = sum(
        p.numel()
        for group in optimizer.param_groups
        for p in group['params']
    )

    # Estimate state memory based on optimizer type
    if isinstance(optimizer, torch.optim.AdamW):
        # Standard AdamW: 2 states (m, v) * 4 bytes (FP32) = 8 bytes/param
        state_bytes_per_param = 8
        optimizer_type = "AdamW (FP32)"
    elif HAS_BITSANDBYTES and isinstance(optimizer, bnb.optim.AdamW8bit):
        # 8-bit AdamW: 2 states (m, v) * 1 byte (INT8) = 2 bytes/param
        state_bytes_per_param = 2
        optimizer_type = "AdamW8bit"
    else:
        # Unknown optimizer, assume FP32
        state_bytes_per_param = 8
        optimizer_type = "Unknown"

    # Compute memory
    state_memory_gb = (total_params * state_bytes_per_param) / (1024 ** 3)

    # Parameter memory (FP32 = 4 bytes, BF16 = 2 bytes)
    # Assume FP32 for gradient accumulation
    param_memory_gb = (total_params * 4) / (1024 ** 3)

    total_memory_gb = state_memory_gb + param_memory_gb

    return {
        'optimizer_type': optimizer_type,
        'total_params': total_params,
        'state_memory_gb': state_memory_gb,
        'param_memory_gb': param_memory_gb,
        'total_memory_gb': total_memory_gb,
        'state_bytes_per_param': state_bytes_per_param,
    }
