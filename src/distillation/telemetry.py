"""
Telemetry and logging for training runs.

Implements structured logging of metrics (loss, throughput, VRAM, etc.)
with optional wandb integration.

Tasks: T048-T051
"""

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Deque
from enum import Enum

import torch

# Optional wandb import
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    logging.warning(
        "wandb not available. Wandb logging will be disabled. "
        "Install with: pip install wandb"
    )


logger = logging.getLogger(__name__)


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert tensors to scalars/lists in nested dicts/lists.

    This function ensures all tensor objects are converted to JSON-serializable
    types (scalars or lists) before writing to files or logging.

    Args:
        obj: Object to sanitize (can be tensor, dict, list, or primitive)

    Returns:
        JSON-serializable version of the object

    Example:
        >>> tensor_dict = {'loss': torch.tensor(0.5), 'metrics': {'acc': torch.tensor(0.95)}}
        >>> sanitized = _sanitize_for_json(tensor_dict)
        >>> # {'loss': 0.5, 'metrics': {'acc': 0.95}}
    """
    if torch.is_tensor(obj):
        # Convert tensor to scalar (for single element) or list (for multi-element)
        if obj.numel() == 1:
            return obj.item()
        else:
            return obj.tolist()
    elif isinstance(obj, dict):
        # Recursively sanitize dictionary values
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # Recursively sanitize list/tuple elements
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        # Primitives are already JSON-serializable
        return obj
    else:
        # For other types, try to convert to string
        try:
            return str(obj)
        except Exception:
            return "<non-serializable>"


class OutputSink(Enum):
    """Output sink types for telemetry."""
    FILE = "file"
    CONSOLE = "console"
    WANDB = "wandb"


@dataclass
class TrainingMetrics:
    """Structured training metrics for a single step.

    Core metrics:
        step: Global training step
        epoch: Training epoch
        timestamp: Unix timestamp

    Loss metrics:
        loss: Total loss
        distillation_loss: Soft KL divergence loss (optional)
        hard_ce_loss: Hard cross-entropy loss (optional)
        soft_kl_loss: Soft KL loss component (optional)

    Optimizer metrics:
        learning_rate: Current learning rate
        grad_norm: Gradient norm (before clipping)

    Resource metrics:
        gpu_memory_allocated_gb: Allocated GPU memory in GB
        gpu_memory_reserved_gb: Reserved GPU memory in GB
        vram_utilization: GPU memory utilization percentage

    Throughput metrics:
        tokens_per_sec: Tokens processed per second
        samples_per_sec: Samples processed per second
        steps_per_sec: Training steps per second

    Moving averages (optional):
        tokens_per_sec_avg: Moving average of tokens/sec
        samples_per_sec_avg: Moving average of samples/sec
        steps_per_sec_avg: Moving average of steps/sec
    """
    # Core metrics
    step: int
    epoch: int
    timestamp: float

    # Loss metrics
    loss: float
    distillation_loss: Optional[float] = None
    hard_ce_loss: Optional[float] = None
    soft_kl_loss: Optional[float] = None

    # Optimizer metrics
    learning_rate: float = 0.0
    grad_norm: float = 0.0

    # Resource metrics
    gpu_memory_allocated_gb: float = 0.0
    gpu_memory_reserved_gb: float = 0.0
    vram_utilization: float = 0.0

    # Throughput metrics
    tokens_per_sec: float = 0.0
    samples_per_sec: float = 0.0
    steps_per_sec: float = 0.0

    # Moving averages (optional)
    tokens_per_sec_avg: Optional[float] = None
    samples_per_sec_avg: Optional[float] = None
    steps_per_sec_avg: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class TelemetryLogger:
    """Telemetry logger for training monitoring.

    Features:
    - JSON log output every N steps
    - Multiple output sinks (file, console, optional wandb)
    - Comprehensive metrics tracking
    - Throughput measurement with moving averages
    - GPU memory tracking
    - Configurable log interval
    - Minimal overhead

    Example:
        >>> telemetry = TelemetryLogger(
        ...     log_dir="./logs",
        ...     log_interval=10,
        ...     sinks=[OutputSink.FILE, OutputSink.CONSOLE],
        ...     enable_wandb=False,
        ... )
        >>> telemetry.log_step(
        ...     step=100,
        ...     epoch=1,
        ...     loss=1.5,
        ...     learning_rate=1e-4,
        ...     grad_norm=0.5,
        ...     num_tokens=4096,
        ...     batch_size=1,
        ...     step_time=0.5,
        ... )
    """

    def __init__(
        self,
        log_dir: Optional[Union[str, Path]] = None,
        log_interval: int = 10,
        sinks: Optional[List[OutputSink]] = None,
        enable_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        wandb_offline: bool = False,
        moving_avg_window: int = 100,
    ):
        """Initialize telemetry logger.

        Args:
            log_dir: Directory for log files (required if FILE sink enabled)
            log_interval: Steps between logging (default: 10)
            sinks: List of output sinks (default: [FILE, CONSOLE])
            enable_wandb: Enable wandb logging (default: False)
            wandb_project: Wandb project name (required if wandb enabled)
            wandb_run_name: Wandb run name (optional)
            wandb_config: Wandb configuration dict (optional)
            wandb_offline: Use wandb offline mode (default: False)
            moving_avg_window: Window size for moving averages (default: 100)
        """
        self.log_interval = log_interval
        self.sinks = sinks or [OutputSink.FILE, OutputSink.CONSOLE]
        self.enable_wandb = enable_wandb and HAS_WANDB
        self.moving_avg_window = moving_avg_window

        # Validate wandb settings
        if self.enable_wandb:
            if not HAS_WANDB:
                logger.warning("Wandb requested but not installed. Disabling wandb logging.")
                self.enable_wandb = False
            elif not wandb_project:
                raise ValueError("wandb_project must be specified when enable_wandb=True")

        # Setup file logging
        self.log_dir = None
        self.log_file = None
        if OutputSink.FILE in self.sinks:
            if not log_dir:
                raise ValueError("log_dir must be specified when FILE sink is enabled")
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = self.log_dir / "training_metrics.jsonl"
            logger.info(f"Telemetry logging to: {self.log_file}")

        # Setup wandb
        self.wandb_run = None
        if self.enable_wandb:
            try:
                if wandb_offline:
                    wandb.init(
                        project=wandb_project,
                        name=wandb_run_name,
                        config=wandb_config,
                        mode="offline",
                    )
                else:
                    wandb.init(
                        project=wandb_project,
                        name=wandb_run_name,
                        config=wandb_config,
                    )
                self.wandb_run = wandb.run
                logger.info(f"Wandb initialized: project={wandb_project}, run={wandb_run_name}")
            except Exception as e:
                logger.error(f"Failed to initialize wandb: {e}")
                self.enable_wandb = False

        # Moving averages
        self.tokens_per_sec_history: Deque[float] = deque(maxlen=moving_avg_window)
        self.samples_per_sec_history: Deque[float] = deque(maxlen=moving_avg_window)
        self.steps_per_sec_history: Deque[float] = deque(maxlen=moving_avg_window)

        # State tracking
        self.last_log_step = -1
        self.start_time = time.time()

        logger.info(f"TelemetryLogger initialized:")
        logger.info(f"  Log interval: {log_interval}")
        logger.info(f"  Sinks: {[s.value for s in self.sinks]}")
        logger.info(f"  Wandb enabled: {self.enable_wandb}")
        logger.info(f"  Moving avg window: {moving_avg_window}")

    def should_log(self, step: int) -> bool:
        """Check if we should log at this step.

        Args:
            step: Current training step

        Returns:
            True if we should log at this step
        """
        return step % self.log_interval == 0 or step == 0

    def _get_gpu_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics.

        Returns:
            Dictionary with memory stats in GB
        """
        if not torch.cuda.is_available():
            return {
                'gpu_memory_allocated_gb': 0.0,
                'gpu_memory_reserved_gb': 0.0,
                'vram_utilization': 0.0,
            }

        allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)

        # Get GPU properties for total memory
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)

        # Utilization as percentage of total memory
        utilization = (allocated / total_memory) * 100 if total_memory > 0 else 0.0

        return {
            'gpu_memory_allocated_gb': allocated,
            'gpu_memory_reserved_gb': reserved,
            'vram_utilization': utilization,
        }

    def _compute_moving_averages(self) -> Dict[str, float]:
        """Compute moving averages of throughput metrics.

        Returns:
            Dictionary with moving averages
        """
        return {
            'tokens_per_sec_avg': sum(self.tokens_per_sec_history) / len(self.tokens_per_sec_history) if self.tokens_per_sec_history else 0.0,
            'samples_per_sec_avg': sum(self.samples_per_sec_history) / len(self.samples_per_sec_history) if self.samples_per_sec_history else 0.0,
            'steps_per_sec_avg': sum(self.steps_per_sec_history) / len(self.steps_per_sec_history) if self.steps_per_sec_history else 0.0,
        }

    def log_step(
        self,
        step: int,
        epoch: int,
        loss: float,
        learning_rate: float = 0.0,
        grad_norm: float = 0.0,
        num_tokens: Optional[int] = None,
        batch_size: Optional[int] = None,
        step_time: Optional[float] = None,
        distillation_loss: Optional[float] = None,
        hard_ce_loss: Optional[float] = None,
        soft_kl_loss: Optional[float] = None,
        **extra_metrics: Any,
    ) -> Optional[TrainingMetrics]:
        """Log training metrics for a step.

        Args:
            step: Global training step
            epoch: Training epoch
            loss: Total loss
            learning_rate: Current learning rate
            grad_norm: Gradient norm (before clipping)
            num_tokens: Number of tokens processed (for throughput)
            batch_size: Batch size (for throughput)
            step_time: Time taken for this step in seconds (for throughput)
            distillation_loss: Distillation loss component (optional)
            hard_ce_loss: Hard cross-entropy loss component (optional)
            soft_kl_loss: Soft KL loss component (optional)
            **extra_metrics: Additional metrics to log

        Returns:
            TrainingMetrics object if logged, None otherwise
        """
        # Check if we should log at this step
        if not self.should_log(step):
            # Still update moving averages for throughput
            if step_time and step_time > 0:
                if num_tokens:
                    self.tokens_per_sec_history.append(num_tokens / step_time)
                if batch_size:
                    self.samples_per_sec_history.append(batch_size / step_time)
                self.steps_per_sec_history.append(1.0 / step_time)
            return None

        # Get GPU memory stats
        memory_stats = self._get_gpu_memory_stats()

        # Compute throughput metrics
        tokens_per_sec = 0.0
        samples_per_sec = 0.0
        steps_per_sec = 0.0

        if step_time and step_time > 0:
            if num_tokens:
                tokens_per_sec = num_tokens / step_time
                self.tokens_per_sec_history.append(tokens_per_sec)
            if batch_size:
                samples_per_sec = batch_size / step_time
                self.samples_per_sec_history.append(samples_per_sec)
            steps_per_sec = 1.0 / step_time
            self.steps_per_sec_history.append(steps_per_sec)

        # Compute moving averages
        moving_avgs = self._compute_moving_averages()

        # Create metrics object
        metrics = TrainingMetrics(
            step=step,
            epoch=epoch,
            timestamp=time.time(),
            loss=loss,
            distillation_loss=distillation_loss,
            hard_ce_loss=hard_ce_loss,
            soft_kl_loss=soft_kl_loss,
            learning_rate=learning_rate,
            grad_norm=grad_norm,
            tokens_per_sec=tokens_per_sec,
            samples_per_sec=samples_per_sec,
            steps_per_sec=steps_per_sec,
            tokens_per_sec_avg=moving_avgs['tokens_per_sec_avg'],
            samples_per_sec_avg=moving_avgs['samples_per_sec_avg'],
            steps_per_sec_avg=moving_avgs['steps_per_sec_avg'],
            **memory_stats,
        )

        # Log to sinks
        if OutputSink.FILE in self.sinks:
            self._log_to_file(metrics)

        if OutputSink.CONSOLE in self.sinks:
            self._log_to_console(metrics, extra_metrics)

        if OutputSink.WANDB in self.sinks and self.enable_wandb:
            self._log_to_wandb(metrics, extra_metrics)

        self.last_log_step = step
        return metrics

    def _log_to_file(self, metrics: TrainingMetrics):
        """Log metrics to file (JSONL format).

        Args:
            metrics: Training metrics to log
        """
        if not self.log_file:
            return

        try:
            # Sanitize metrics to convert any tensors to scalars/lists
            metrics_dict = metrics.to_dict()
            sanitized_dict = _sanitize_for_json(metrics_dict)
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(sanitized_dict) + '\n')
        except Exception as e:
            logger.error(f"Failed to write to log file: {e}")

    def _log_to_console(self, metrics: TrainingMetrics, extra_metrics: Dict[str, Any]):
        """Log metrics to console (human-readable format).

        Args:
            metrics: Training metrics to log
            extra_metrics: Additional metrics to log
        """
        # Format key metrics for display
        log_msg = (
            f"Step {metrics.step:6d} | Epoch {metrics.epoch:3d} | "
            f"Loss: {metrics.loss:7.4f} | "
            f"LR: {metrics.learning_rate:.2e} | "
            f"Grad: {metrics.grad_norm:6.3f} | "
            f"VRAM: {metrics.gpu_memory_allocated_gb:5.2f}GB ({metrics.vram_utilization:5.1f}%) | "
            f"Tokens/s: {metrics.tokens_per_sec:7.1f} (avg: {metrics.tokens_per_sec_avg:7.1f})"
        )

        # Add extra metrics if provided
        if extra_metrics:
            extra_str = " | ".join(f"{k}: {v}" for k, v in extra_metrics.items())
            log_msg += f" | {extra_str}"

        logger.info(log_msg)

    def _log_to_wandb(self, metrics: TrainingMetrics, extra_metrics: Dict[str, Any]):
        """Log metrics to wandb.

        Args:
            metrics: Training metrics to log
            extra_metrics: Additional metrics to log
        """
        if not self.enable_wandb or not self.wandb_run:
            return

        try:
            # Prepare metrics dict for wandb
            wandb_metrics = metrics.to_dict()
            wandb_metrics.update(extra_metrics)

            # Log to wandb
            wandb.log(wandb_metrics, step=metrics.step)
        except Exception as e:
            logger.error(f"Failed to log to wandb: {e}")

    def log_evaluation(
        self,
        step: int,
        epoch: int,
        eval_loss: float,
        eval_metrics: Optional[Dict[str, Any]] = None,
    ):
        """Log evaluation metrics.

        Args:
            step: Global training step
            epoch: Training epoch
            eval_loss: Evaluation loss
            eval_metrics: Additional evaluation metrics
        """
        metrics_dict = {
            'step': step,
            'epoch': epoch,
            'eval_loss': eval_loss,
            'timestamp': time.time(),
        }

        if eval_metrics:
            metrics_dict.update({f'eval_{k}': v for k, v in eval_metrics.items()})

        # Log to sinks
        if OutputSink.FILE in self.sinks and self.log_file:
            try:
                # Sanitize metrics_dict to convert tensors to scalars/lists
                sanitized_dict = _sanitize_for_json(metrics_dict)
                with open(self.log_file, 'a') as f:
                    f.write(json.dumps(sanitized_dict) + '\n')
            except Exception as e:
                logger.error(f"Failed to write evaluation to log file: {e}")

        if OutputSink.CONSOLE in self.sinks:
            logger.info(f"Evaluation | Step {step:6d} | Epoch {epoch:3d} | Loss: {eval_loss:7.4f}")

        if OutputSink.WANDB in self.sinks and self.enable_wandb:
            try:
                wandb.log(metrics_dict, step=step)
            except Exception as e:
                logger.error(f"Failed to log evaluation to wandb: {e}")

    def finalize(self):
        """Finalize telemetry logging (close files, finish wandb, etc.)."""
        if self.enable_wandb and self.wandb_run:
            try:
                wandb.finish()
                logger.info("Wandb run finished")
            except Exception as e:
                logger.error(f"Failed to finish wandb: {e}")

        logger.info(f"Telemetry logging finalized. Total runtime: {time.time() - self.start_time:.2f}s")

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of telemetry session.

        Returns:
            Dictionary with summary stats
        """
        moving_avgs = self._compute_moving_averages()
        memory_stats = self._get_gpu_memory_stats()

        return {
            'total_runtime_sec': time.time() - self.start_time,
            'last_log_step': self.last_log_step,
            'moving_averages': moving_avgs,
            'current_memory': memory_stats,
            'log_file': str(self.log_file) if self.log_file else None,
            'wandb_enabled': self.enable_wandb,
        }
