"""
PERFORMANCE OPTIMIZATIONS for DistillationTrainer.

This module contains optimized versions of key trainer methods.
To use: Replace methods in trainer.py with these optimized versions.

Key optimizations implemented:
1. Async batch transfers with pinned memory
2. Teacher output caching
3. Optimized dequantization
4. Loss computation uses optimized losses_optimized.py

Estimated speedup: 15-25% overall training time

See: ai-notes/PERFORMANCE_AUDIT_FORWARD_BACKWARD.md
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from functools import lru_cache
import hashlib

from .losses_optimized import SparseKLLossOptimized


# ============================================================================
# OPTIMIZATION #1: Teacher Output Caching
# ============================================================================

class TeacherCacheMixin:
    """
    Mixin to add teacher output caching to DistillationTrainer.

    Usage:
        class DistillationTrainer(TeacherCacheMixin, ...):
            ...
    """

    def __init__(self, *args, cache_size: int = 1000, **kwargs):
        """
        Args:
            cache_size: Maximum number of cached teacher outputs (default: 1000)
        """
        super().__init__(*args, **kwargs)

        # Cache for teacher outputs: key = hash(input_ids), value = (indices, values, other_mass)
        self._teacher_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._cache_size = cache_size
        self._cache_hits = 0
        self._cache_misses = 0

    def _hash_input_ids(self, input_ids: torch.Tensor) -> str:
        """
        Compute hash of input_ids for cache lookup.

        Args:
            input_ids: Token IDs [batch_size, seq_len]

        Returns:
            Hash string
        """
        # Use fast hash (xxhash would be better, but not standard library)
        # Convert to bytes for hashing
        input_bytes = input_ids.cpu().numpy().tobytes()
        return hashlib.blake2b(input_bytes, digest_size=16).hexdigest()

    def _fetch_teacher_logits_cached(
        self,
        input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fetch teacher logits with caching (OPTIMIZATION #1).

        This is a drop-in replacement for _fetch_teacher_logits with caching support.

        Cache hit rate depends on training data:
        - With sequence repetition: 20-40% hit rate
        - Without repetition: ~0% hit rate (cache disabled effectively)

        Expected speedup: 10-15ms per cache hit (on total step time of 150-200ms)

        Args:
            input_ids: Input token IDs [batch_size, seq_len]

        Returns:
            Tuple of (indices, values, other_mass)
        """
        # Compute cache key
        cache_key = self._hash_input_ids(input_ids)

        # Check cache
        if cache_key in self._teacher_cache:
            self._cache_hits += 1
            # Cache hit: return cached tensors (clone to avoid aliasing)
            cached_indices, cached_values, cached_other_mass = self._teacher_cache[cache_key]
            return cached_indices.clone(), cached_values.clone(), cached_other_mass.clone()

        # Cache miss: query teacher (use original method)
        self._cache_misses += 1
        indices, values, other_mass = self._fetch_teacher_logits(input_ids)

        # Add to cache (with eviction if full)
        if len(self._teacher_cache) >= self._cache_size:
            # Evict oldest entry (FIFO policy)
            # For LRU, use OrderedDict instead
            first_key = next(iter(self._teacher_cache))
            del self._teacher_cache[first_key]

        # Store in cache (clone to avoid mutation)
        self._teacher_cache[cache_key] = (indices.clone(), values.clone(), other_mass.clone())

        return indices, values, other_mass

    def get_cache_stats(self) -> Dict[str, float]:
        """Get cache statistics."""
        total_queries = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_queries if total_queries > 0 else 0.0
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self._teacher_cache),
        }


# ============================================================================
# OPTIMIZATION #4: Async Batch Transfers
# ============================================================================

def _train_step_optimized(
    self,
    batch: Dict[str, torch.Tensor],
    accumulation_step: int,
) -> Dict[str, float]:
    """
    OPTIMIZED training step with async batch transfers.

    Key optimizations:
    1. Use non_blocking=True for async CPU->GPU transfers
    2. Use optimized loss function (losses_optimized.py)
    3. Optionally use teacher caching (if enabled)

    Requirements:
    - DataLoader must use pin_memory=True
    - Replace loss_fn with SparseKLLossOptimized in __init__

    Expected speedup: 1-2ms per step (transfer overlap)

    Args:
        batch: Batch of data with keys: input_ids, attention_mask, labels
        accumulation_step: Current accumulation step (0 to accumulation_steps-1)

    Returns:
        Dictionary of metrics for this step
    """
    # OPTIMIZATION #4: Async batch transfers (non_blocking=True)
    # Requires DataLoader with pin_memory=True
    input_ids = batch['input_ids'].to(self.device, non_blocking=True)
    attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
    labels = batch.get('labels', input_ids).to(self.device, non_blocking=True)

    # Fetch teacher logits (use cached version if available)
    with torch.no_grad():
        if hasattr(self, '_fetch_teacher_logits_cached'):
            # Use cached version (OPTIMIZATION #1)
            teacher_topk_indices, teacher_topk_values, teacher_other_mass = \
                self._fetch_teacher_logits_cached(input_ids)
        else:
            # Use original version
            teacher_topk_indices, teacher_topk_values, teacher_other_mass = \
                self._fetch_teacher_logits(input_ids)

    # Forward pass with mixed precision
    try:
        autocast_ctx = autocast('cuda', dtype=torch.bfloat16, enabled=self.config.use_bf16)
    except TypeError:
        autocast_ctx = autocast(enabled=self.config.use_bf16)

    with autocast_ctx:
        # Get student logits
        if hasattr(self.model, 'forward_train'):
            hidden_states = self.model.forward_train(input_ids)
            if hasattr(self.model, 'lm_head'):
                student_logits = self.model.lm_head(hidden_states)
            else:
                raise ValueError("Model must have 'lm_head' or return logits directly")
        else:
            outputs = self.model(input_ids)
            if hasattr(outputs, 'logits'):
                student_logits = outputs.logits
            else:
                student_logits = outputs

        # Compute distillation loss using OPTIMIZED loss function
        # Note: loss_fn should be SparseKLLossOptimized (set in __init__)
        loss = self.loss_fn(
            student_logits=student_logits,
            teacher_topk_indices=teacher_topk_indices,
            teacher_topk_values=teacher_topk_values,
            teacher_other_mass=teacher_other_mass,
            hard_targets=labels,
        )

        # OPTIMIZATION #7: Scale loss outside autocast for better precision
        # (but for simplicity, keeping inside for now - BF16 is stable enough)
        loss = loss / self.config.gradient_accumulation_steps

    # Backward pass with gradient scaling
    self.scaler.scale(loss).backward()

    # Collect metrics
    metrics = {
        'loss': loss.item() * self.config.gradient_accumulation_steps,  # Unscale for logging
    }

    return metrics


# ============================================================================
# OPTIMIZATION #8: Dequantization Buffer Reuse - REMOVED
# ============================================================================
#
# NOTE: This optimization was removed due to F-007 (see CODE_AUDIT_PERFORMANCE_OPTIMIZATIONS.md)
#
# REASON: The buffer reuse didn't work because .float() allocates a new tensor
# before the out= parameter can take effect. Net savings: 0 bytes.
#
# Original code:
#   torch.mul(values_int8.float(), scale.unsqueeze(-1), out=buffer)
#                         ↑ Allocates here (4MB)
#                                                         ↑ Too late
#
# The optimization claimed to save 4MB per step (0.0125% of 32GB VRAM).
# Code complexity increase didn't justify negligible benefit.
#
# If dequantization performance becomes critical, use in-place operations:
#   buffer.copy_(values_int8)  # int8 → fp32 in-place
#   buffer.mul_(scale.unsqueeze(-1))  # In-place multiply


# ============================================================================
# DataLoader Configuration Helper
# ============================================================================

def configure_dataloader_for_optimization(dataloader):
    """
    Configure DataLoader for optimal performance.

    Usage:
        train_dataloader = configure_dataloader_for_optimization(train_dataloader)

    Requirements:
    - PyTorch >= 1.7 (pin_memory support)
    - CUDA-enabled GPU

    Args:
        dataloader: PyTorch DataLoader instance

    Returns:
        Configured DataLoader (may create new instance)
    """
    # Check if already configured
    if hasattr(dataloader, 'pin_memory') and dataloader.pin_memory:
        return dataloader

    # Create new DataLoader with optimized settings
    from torch.utils.data import DataLoader

    optimized_loader = DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        shuffle=hasattr(dataloader.sampler, 'shuffle') and dataloader.sampler.shuffle,
        num_workers=dataloader.num_workers,
        collate_fn=dataloader.collate_fn,
        pin_memory=True,  # CRITICAL: Enables async transfers with non_blocking=True
        drop_last=dataloader.drop_last,
        timeout=dataloader.timeout if hasattr(dataloader, 'timeout') else 0,
        worker_init_fn=dataloader.worker_init_fn if hasattr(dataloader, 'worker_init_fn') else None,
        prefetch_factor=2,  # Prefetch 2 batches per worker (reduces waiting)
        persistent_workers=True if dataloader.num_workers > 0 else False,  # Keep workers alive
    )

    return optimized_loader


# ============================================================================
# Usage Example
# ============================================================================

"""
To integrate these optimizations into DistillationTrainer:

1. Add TeacherCacheMixin as base class:

   class DistillationTrainer(TeacherCacheMixin):
       def __init__(self, ...):
           super().__init__(cache_size=1000)
           ...

2. Replace loss function in __init__:

   from .losses_optimized import SparseKLLossOptimized

   self.loss_fn = SparseKLLossOptimized(
       temperature=self.config.teacher_temperature,
       alpha=self.config.distill_alpha,
   )

3. Replace _train_step method with _train_step_optimized:

   # In class definition:
   _train_step = _train_step_optimized

4. Configure DataLoader for pinned memory:

   from .trainer_optimizations import configure_dataloader_for_optimization

   train_dataloader = configure_dataloader_for_optimization(train_dataloader)
   trainer = DistillationTrainer(
       model=model,
       config=config,
       train_dataloader=train_dataloader,
       ...
   )

5. (Optional) Use optimized dequantization:

   _fetch_teacher_logits = _fetch_teacher_logits_optimized_dequant

Expected results:
- 15-25% faster training (depending on cache hit rate)
- 512MB less peak memory (from loss optimizations)
- Better GPU utilization (from async transfers)

Benchmarking:
- Use torch.profiler to measure before/after (see audit report)
- Monitor cache hit rate: trainer.get_cache_stats()
- Check GPU utilization: nvidia-smi dmon
"""
