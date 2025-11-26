"""
Saddle point detection and escape for training optimization.

Detects when training is stuck in flat regions with:
- Low gradient norms
- Minimal loss improvement
- High absolute loss

Then applies gentle interventions to help escape.
"""

from collections import deque
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SaddlePointDetector:
    """Detects when training is stuck in a saddle point or flat region."""

    def __init__(
        self,
        grad_norm_threshold: float = 0.55,
        loss_improvement_threshold: float = 0.0005,  # 0.05% - very conservative
        min_loss_threshold: float = 100.0,
        patience_steps: int = 150,  # Long patience - gentle approach
    ):
        """
        Initialize saddle point detector.

        Args:
            grad_norm_threshold: Trigger when grad_norm < this
            loss_improvement_threshold: Min improvement rate to be "healthy"
            min_loss_threshold: Only intervene if loss > this
            patience_steps: Steps to wait before declaring "stuck"
        """
        self.grad_norm_threshold = grad_norm_threshold
        self.loss_improvement_threshold = loss_improvement_threshold
        self.min_loss_threshold = min_loss_threshold
        self.patience_steps = patience_steps

        self.loss_history = deque(maxlen=patience_steps)
        self.grad_norm_history = deque(maxlen=patience_steps)
        self.stuck_counter = 0

    def update(self, grad_norm: float, loss: float, step: int) -> Dict:
        """
        Update detector state and check for saddle point.

        Args:
            grad_norm: Current gradient norm
            loss: Current loss value
            step: Current training step

        Returns:
            dict with detection results
        """
        self.loss_history.append(loss)
        self.grad_norm_history.append(grad_norm)

        # Need enough history
        if len(self.loss_history) < self.patience_steps:
            return {
                'is_stuck': False,
                'reason': 'insufficient_history',
                'stuck_counter': 0,
                'grad_norm': grad_norm,
                'loss': loss,
            }

        # Check conditions
        grad_norm_low = grad_norm < self.grad_norm_threshold
        loss_high = loss > self.min_loss_threshold

        # Calculate loss improvement over window
        loss_start = self.loss_history[0]
        loss_end = self.loss_history[-1]
        loss_improvement = (loss_start - loss_end) / loss_start if loss_start > 0 else 0
        loss_not_improving = loss_improvement < self.loss_improvement_threshold

        # Update stuck counter
        if grad_norm_low and loss_high and loss_not_improving:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        is_stuck = self.stuck_counter >= self.patience_steps

        return {
            'is_stuck': is_stuck,
            'grad_norm_low': grad_norm_low,
            'loss_high': loss_high,
            'loss_not_improving': loss_not_improving,
            'loss_improvement': loss_improvement,
            'stuck_counter': self.stuck_counter,
            'grad_norm': grad_norm,
            'loss': loss,
        }

    def reset(self):
        """Reset detector state."""
        self.stuck_counter = 0
        self.loss_history.clear()
        self.grad_norm_history.clear()

    def get_loss_improvement(self) -> float:
        """Get current loss improvement rate."""
        if len(self.loss_history) < 2:
            return 0.0

        loss_start = self.loss_history[0]
        loss_end = self.loss_history[-1]
        return (loss_start - loss_end) / loss_start if loss_start > 0 else 0.0


class SaddleEscapeManager:
    """Manages saddle point detection and gentle interventions."""

    def __init__(self, config):
        """
        Initialize saddle escape manager.

        Args:
            config: Configuration object with saddle_escape settings
        """
        self.config = config
        self.detector = SaddlePointDetector(
            grad_norm_threshold=config.grad_norm_threshold,
            loss_improvement_threshold=config.loss_improvement_threshold,
            min_loss_threshold=config.min_loss_threshold,
            patience_steps=config.patience_steps,
        )

        self.intervention_history = []
        self.current_intervention = None
        self.steps_since_intervention = 0
        self.total_interventions = 0
        self.successful_escapes = 0

        # Intervention enabled/disabled
        self.interventions_enabled = config.interventions_enabled

        logger.info("SaddleEscapeManager initialized:")
        logger.info(f"  Grad norm threshold: {config.grad_norm_threshold}")
        logger.info(f"  Loss improvement threshold: {config.loss_improvement_threshold}")
        logger.info(f"  Patience steps: {config.patience_steps}")
        logger.info(f"  Interventions enabled: {self.interventions_enabled}")

    def check_and_intervene(
        self,
        grad_norm: float,
        loss: float,
        step: int,
        optimizer,
        scheduler=None,
    ) -> Dict:
        """
        Check for saddle point and optionally intervene.

        Args:
            grad_norm: Current gradient norm
            loss: Current loss
            step: Current training step
            optimizer: Optimizer instance
            scheduler: LR scheduler (optional)

        Returns:
            dict with intervention details
        """
        # Update detector
        detection = self.detector.update(grad_norm, loss, step)

        # Track ongoing intervention
        if self.current_intervention:
            self.steps_since_intervention += 1

            # Check if intervention complete
            if self.steps_since_intervention >= self.current_intervention['duration']:
                self._end_intervention(optimizer, scheduler)

                # Check escape success
                if self._check_escape_success():
                    self.successful_escapes += 1
                    self.detector.reset()
                    logger.info("✅ Successfully escaped saddle point")

        # Check cooldown
        if self._in_cooldown(step):
            return {
                'action': 'none',
                'reason': 'cooldown',
                'detection': detection,
            }

        # Apply intervention if stuck and interventions enabled
        if detection['is_stuck'] and not self.current_intervention and self.interventions_enabled:
            return self._apply_intervention(step, optimizer, scheduler, detection)

        return {
            'action': 'none',
            'detection': detection,
        }

    def _apply_intervention(self, step, optimizer, scheduler, detection):
        """Apply gentle LR boost intervention."""
        num_interventions = len(self.intervention_history)

        logger.warning(
            f"🚨 SADDLE POINT DETECTED at step {step}: "
            f"grad_norm={detection['grad_norm']:.4f}, "
            f"loss={detection['loss']:.2f}, "
            f"improvement={detection['loss_improvement']:.4f}"
        )

        # Gentle escalation: 1.5x → 2.0x → 2.5x → 3.0x
        if num_interventions == 0:
            factor = 1.5  # Very gentle first nudge
            duration = 50
        elif num_interventions == 1:
            factor = 2.0  # Moderate nudge
            duration = 50
        elif num_interventions == 2:
            factor = 2.5  # Stronger nudge
            duration = 75
        else:
            factor = 3.0  # Maximum nudge
            duration = 100

        result = self._lr_boost(step, optimizer, factor, duration)
        self.total_interventions += 1
        result['intervention_number'] = self.total_interventions
        result['detection'] = detection

        return result

    def _lr_boost(self, step, optimizer, factor, duration):
        """Apply temporary LR boost."""
        # Save original LRs
        original_lrs = [group['lr'] for group in optimizer.param_groups]

        # Apply boost
        for i, group in enumerate(optimizer.param_groups):
            group['lr'] = original_lrs[i] * factor

        intervention = {
            'type': 'lr_boost',
            'start_step': step,
            'duration': duration,
            'factor': factor,
            'original_lrs': original_lrs,
        }

        self.current_intervention = intervention
        self.steps_since_intervention = 0

        logger.warning(
            f"🚀 GENTLE NUDGE: LR boost {factor:.1f}x for {duration} steps "
            f"(LR: {original_lrs[0]:.6f} → {original_lrs[0]*factor:.6f})"
        )

        return {
            'action': 'lr_boost',
            'factor': factor,
            'duration': duration,
            'original_lr': original_lrs[0],
            'boosted_lr': original_lrs[0] * factor,
        }

    def _end_intervention(self, optimizer, scheduler):
        """End intervention and restore LR."""
        if self.current_intervention and self.current_intervention['type'] == 'lr_boost':
            # Restore original LRs
            for i, group in enumerate(optimizer.param_groups):
                group['lr'] = self.current_intervention['original_lrs'][i]

            logger.info(
                f"✅ NUDGE ENDED: Restored LR to {self.current_intervention['original_lrs'][0]:.6f}"
            )

        # Record history
        if self.current_intervention:
            intervention_record = self.current_intervention.copy()
            intervention_record['end_step'] = intervention_record['start_step'] + intervention_record['duration']
            self.intervention_history.append(intervention_record)

        self.current_intervention = None
        self.steps_since_intervention = 0

    def _in_cooldown(self, step):
        """Check if in cooldown period."""
        if not self.intervention_history:
            return False

        last_intervention_end = self.intervention_history[-1]['end_step']
        cooldown_steps = self.config.cooldown_steps

        return (step - last_intervention_end) < cooldown_steps

    def _check_escape_success(self):
        """Check if successfully escaped."""
        if len(self.detector.loss_history) < 20:
            return False

        recent_losses = list(self.detector.loss_history)[-20:]
        improvement = (recent_losses[0] - recent_losses[-1]) / recent_losses[0]

        # Success if >1% improvement in 20 steps
        return improvement > 0.01

    def get_stats(self):
        """Get intervention statistics."""
        return {
            'total_interventions': self.total_interventions,
            'successful_escapes': self.successful_escapes,
            'success_rate': self.successful_escapes / self.total_interventions if self.total_interventions > 0 else 0,
            'current_intervention_active': self.current_intervention is not None,
        }
