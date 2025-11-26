"""
Real-time monitoring of log_decay during training.

This monitor tracks retention heads to detect boundary violations that cause
gradient death and eventual NaN failures.

Usage:
    from src.distillation.monitors.log_decay_monitor import LogDecayMonitor

    # After creating model
    monitor = LogDecayMonitor(model, log_every_n_steps=10)

    # In training loop, after loss computation
    monitor.log_step(step=trainer.global_step, loss=loss.item(), grad_norm=grad_norm)

Author: Claude (Monitoring Wizard)
Date: 2025-11-08
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class LogDecayMonitor:
    """Monitor log_decay parameters during training to detect boundary issues."""

    def __init__(self, model: nn.Module, log_every_n_steps: int = 10):
        """Initialize log_decay monitor.

        Args:
            model: Neural network model with log_decay parameters
            log_every_n_steps: Logging frequency (default: every 10 steps)
        """
        self.model = model
        self.log_every_n_steps = log_every_n_steps
        self.step = 0
        self.history: List[Dict] = []

        # Bounds (same as in retention forward pass)
        self.alpha_min = 1e-5
        self.alpha_max = 0.9999
        self.boundary_threshold = 1e-3

        # Count total log_decay parameters
        self.log_decay_params = []
        for name, param in model.named_parameters():
            if 'log_decay' in name:
                self.log_decay_params.append((name, param))

        if not self.log_decay_params:
            logger.warning("No log_decay parameters found in model! Monitor will be inactive.")
        else:
            logger.info(f"LogDecayMonitor initialized: tracking {len(self.log_decay_params)} log_decay parameters")

    def should_log(self) -> bool:
        """Check if we should log at current step."""
        return self.step % self.log_every_n_steps == 0

    def check_boundaries(self) -> Dict:
        """Check if any log_decay parameters are at boundaries.

        Returns:
            Dictionary with violation counts and statistics
        """
        violations = {
            'min': 0,
            'max': 0,
            'total_heads': 0,
            'alpha_min': float('inf'),
            'alpha_max': float('-inf'),
            'alpha_mean': 0.0,
            'log_mean': 0.0,
        }

        all_alphas = []
        all_logs = []

        for name, param in self.log_decay_params:
            # Apply same transformation as in forward pass
            with torch.no_grad():
                alpha = torch.exp(param.data).clamp(min=self.alpha_min, max=self.alpha_max)

            # Check boundaries
            at_lower = (alpha - self.alpha_min).abs() < self.boundary_threshold
            at_upper = (self.alpha_max - alpha).abs() < self.boundary_threshold

            violations['min'] += at_lower.sum().item()
            violations['max'] += at_upper.sum().item()
            violations['total_heads'] += alpha.numel()

            # Collect for global stats
            all_alphas.extend(alpha.cpu().flatten().tolist())
            all_logs.extend(param.data.cpu().flatten().tolist())

        # Compute global statistics
        if all_alphas:
            violations['alpha_min'] = min(all_alphas)
            violations['alpha_max'] = max(all_alphas)
            violations['alpha_mean'] = sum(all_alphas) / len(all_alphas)
            violations['log_mean'] = sum(all_logs) / len(all_logs)

        return violations

    def log_step(self, step: int, loss: float, grad_norm: float):
        """Log current state of log_decay parameters.

        Args:
            step: Current training step
            loss: Current loss value
            grad_norm: Gradient norm (pre-clipping)
        """
        self.step = step

        if not self.should_log():
            return

        violations = self.check_boundaries()

        # Collect statistics
        stats = {
            'step': step,
            'loss': loss,
            'grad_norm': grad_norm,
            'boundary_violations_min': violations['min'],
            'boundary_violations_max': violations['max'],
            'total_heads': violations['total_heads'],
            'alpha_min': violations['alpha_min'],
            'alpha_max': violations['alpha_max'],
            'alpha_mean': violations['alpha_mean'],
            'log_mean': violations['log_mean'],
        }

        self.history.append(stats)

        # Print warning if violations detected
        total_violations = violations['min'] + violations['max']
        if total_violations > 0:
            pct = 100 * total_violations / violations['total_heads']
            logger.warning(
                f"[Step {step}] {total_violations}/{violations['total_heads']} heads at boundaries ({pct:.1f}%)"
            )
            logger.warning(
                f"  Lower bound: {violations['min']}, Upper bound: {violations['max']}"
            )
            logger.warning(
                f"  Alpha range: [{violations['alpha_min']:.6f}, {violations['alpha_max']:.6f}], mean: {violations['alpha_mean']:.6f}"
            )

            # Critical threshold: if >25% of heads are stuck, warn loudly
            if pct > 25:
                logger.error("=" * 80)
                logger.error("CRITICAL: >25% of heads stuck at boundaries!")
                logger.error("This will cause gradient death and eventual NaN failure")
                logger.error("Recommended action: Stop training and implement sigmoid parameterization")
                logger.error("=" * 80)

    def get_statistics(self) -> Optional[Dict]:
        """Get latest statistics.

        Returns:
            Latest statistics dict, or None if no data
        """
        if not self.history:
            return None
        return self.history[-1]

    def get_history(self) -> List[Dict]:
        """Get full history of statistics.

        Returns:
            List of statistics dicts
        """
        return self.history

    def reset(self):
        """Reset history."""
        self.history = []
        self.step = 0

    def save_history(self, path: str):
        """Save history to JSON file.

        Args:
            path: Path to save JSON file
        """
        import json
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

        logger.info(f"Saved log_decay history to: {path}")

    def plot_history(self, save_path: Optional[str] = None):
        """Plot log_decay statistics over time.

        Args:
            save_path: Path to save plot (if None, displays plot)
        """
        if not self.history:
            logger.warning("No history to plot!")
            return

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not installed, cannot plot")
            return

        # Extract data
        steps = [s['step'] for s in self.history]
        alpha_mins = [s['alpha_min'] for s in self.history]
        alpha_maxs = [s['alpha_max'] for s in self.history]
        alpha_means = [s['alpha_mean'] for s in self.history]
        violations_min = [s['boundary_violations_min'] for s in self.history]
        violations_max = [s['boundary_violations_max'] for s in self.history]
        total_heads = self.history[0]['total_heads']

        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Plot 1: Alpha values
        axes[0].plot(steps, alpha_mins, label='Alpha Min', color='blue', alpha=0.7)
        axes[0].plot(steps, alpha_maxs, label='Alpha Max', color='red', alpha=0.7)
        axes[0].plot(steps, alpha_means, label='Alpha Mean', color='green', alpha=0.7)
        axes[0].axhline(y=self.alpha_min, color='blue', linestyle='--', alpha=0.3, label=f'Lower Bound ({self.alpha_min})')
        axes[0].axhline(y=self.alpha_max, color='red', linestyle='--', alpha=0.3, label=f'Upper Bound ({self.alpha_max})')
        axes[0].set_xlabel('Training Step')
        axes[0].set_ylabel('Alpha Value')
        axes[0].set_title('Retention Decay Rates (Alpha) Over Training')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Boundary violations
        axes[1].plot(steps, violations_min, label='At Lower Bound', color='blue', marker='o', markersize=3)
        axes[1].plot(steps, violations_max, label='At Upper Bound', color='red', marker='o', markersize=3)
        axes[1].fill_between(steps, 0, [v1 + v2 for v1, v2 in zip(violations_min, violations_max)],
                             alpha=0.3, color='orange', label='Total Violations')
        axes[1].set_xlabel('Training Step')
        axes[1].set_ylabel(f'Number of Heads (out of {total_heads})')
        axes[1].set_title('Boundary Violations Over Training')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            from pathlib import Path
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150)
            logger.info(f"Saved plot to: {save_path}")
        else:
            plt.show()

        plt.close()
