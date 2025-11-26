"""
Checkpoint management for distillation training.

Implements atomic checkpoint save/load with corruption prevention
and resume functionality.

Tasks: T041-T042
"""

import logging
import os
import random
from pathlib import Path
from typing import Dict, Any, Optional, List
import tempfile
import shutil

import torch
import numpy as np


logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoint save/load with atomic writes and corruption prevention.

    Features:
    - Atomic save with os.replace (prevents corruption) (T041)
    - Load/resume from checkpoint (T042)
    - RNG state preservation for reproducibility
    - Automatic checkpoint rotation (keep last N checkpoints)
    - Corruption detection and recovery

    Checkpoint format:
    - checkpoint_{step}.pt: Main checkpoint file
    - checkpoint_latest.pt: Symlink/copy to latest checkpoint
    - checkpoint_best.pt: Best validation checkpoint
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        keep_last_n: int = 3,
        save_best: bool = True,
        max_total_size_gb: float = 100.0,
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for saving checkpoints
            keep_last_n: Number of recent checkpoints to keep (default: 3 for T061)
            save_best: Whether to save best validation checkpoint (default: True)
            max_total_size_gb: Maximum total checkpoint size in GB (default: 100.0 for T062)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_last_n = keep_last_n
        self.save_best = save_best
        self.max_total_size_gb = max_total_size_gb

        # Create checkpoint directory
        # Handle migration from old single-file checkpoints to new directory format
        if self.checkpoint_dir.exists() and not self.checkpoint_dir.is_dir():
            # Old checkpoint file exists - rename it to preserve it
            backup_path = self.checkpoint_dir.parent / f"{self.checkpoint_dir.name}.backup"
            logger.warning(f"Found old checkpoint file format at {self.checkpoint_dir}")
            logger.warning(f"Renaming to {backup_path} to preserve it")
            self.checkpoint_dir.rename(backup_path)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Path to latest checkpoint tracker (T063)
        self.latest_checkpoint_file = self.checkpoint_dir / "latest_checkpoint.txt"

        logger.info(f"Initialized CheckpointManager:")
        logger.info(f"  Checkpoint dir: {self.checkpoint_dir}")
        logger.info(f"  Keep last N: {self.keep_last_n}")
        logger.info(f"  Save best: {self.save_best}")
        logger.info(f"  Max total size: {self.max_total_size_gb:.1f} GB")

    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        step: int,
        is_best: bool = False,
    ) -> Path:
        """Save checkpoint with atomic write to prevent corruption.

        Uses atomic save pattern:
        1. Write to temporary file
        2. Sync to disk
        3. Atomically rename to final path (os.replace)

        This prevents corruption if process is killed during save.

        Args:
            state_dict: Full training state (model, optimizer, RNG, etc.)
            step: Global training step
            is_best: Whether this is the best validation checkpoint

        Returns:
            Path to saved checkpoint

        Example:
            >>> manager = CheckpointManager(Path("checkpoints"))
            >>> state = trainer.get_state_dict()
            >>> path = manager.save_checkpoint(state, step=1000)
        """
        # Add RNG states to state_dict
        state_dict = self._add_rng_states(state_dict)

        # Checkpoint filename
        checkpoint_name = f"checkpoint_{step}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        logger.info(f"Saving checkpoint: {checkpoint_path}")

        # Atomic save: write to temp file then replace
        # This prevents corruption if process is killed during save
        with tempfile.NamedTemporaryFile(
            mode='wb',
            dir=self.checkpoint_dir,
            delete=False,
            prefix=f".tmp_checkpoint_{step}_",
            suffix=".pt"
        ) as tmp_file:
            temp_path = Path(tmp_file.name)

            try:
                # Save to temporary file
                torch.save(state_dict, tmp_file)

                # Ensure data is written to disk
                tmp_file.flush()
                os.fsync(tmp_file.fileno())

                # Atomically replace existing file (if any)
                # os.replace is atomic on POSIX systems
                os.replace(temp_path, checkpoint_path)

                logger.info(f"Checkpoint saved: {checkpoint_path} ({self._format_size(checkpoint_path)})")

            except Exception as e:
                # Clean up temp file on error
                if temp_path.exists():
                    temp_path.unlink()
                raise RuntimeError(f"Failed to save checkpoint: {e}") from e

        # Update latest symlink/copy
        self._update_latest_checkpoint(checkpoint_path)

        # Save best checkpoint if requested
        if is_best and self.save_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            shutil.copy2(checkpoint_path, best_path)
            logger.info(f"Saved best checkpoint: {best_path}")

        # Update latest_checkpoint.txt (T063)
        self._update_latest_checkpoint_file(checkpoint_path)

        # Rotate old checkpoints (T061)
        self._rotate_checkpoints()

        # Prune checkpoints by size (T062)
        self._prune_checkpoints_by_size()

        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Load checkpoint from disk.

        Args:
            checkpoint_path: Path to checkpoint file (None = load latest)
            device: Device to load tensors to (None = use saved device)

        Returns:
            Loaded state dictionary

        Raises:
            FileNotFoundError: If checkpoint not found
            RuntimeError: If checkpoint is corrupted

        Example:
            >>> manager = CheckpointManager(Path("checkpoints"))
            >>> state = manager.load_checkpoint()  # Load latest
            >>> trainer.load_state_dict(state)
        """
        # If no path specified, load latest
        if checkpoint_path is None:
            checkpoint_path = self._get_latest_checkpoint()
            if checkpoint_path is None:
                raise FileNotFoundError(
                    f"No checkpoints found in {self.checkpoint_dir}"
                )

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint: {checkpoint_path}")

        try:
            # Load checkpoint
            # Use weights_only=False for compatibility with RNG states (numpy arrays)
            if device is not None:
                state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
            else:
                state_dict = torch.load(checkpoint_path, weights_only=False)

            # Restore RNG states
            self._restore_rng_states(state_dict)

            logger.info(f"Checkpoint loaded successfully:")
            logger.info(f"  Step: {state_dict.get('global_step', 'unknown')}")
            logger.info(f"  Epoch: {state_dict.get('epoch', 'unknown')}")

            return state_dict

        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint {checkpoint_path}: {e}"
            ) from e

    def resume_from_checkpoint(
        self,
        checkpoint_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ) -> Optional[Dict[str, Any]]:
        """Resume training from checkpoint if available.

        This is a convenience method that handles missing checkpoints gracefully
        (returns None instead of raising error).

        Args:
            checkpoint_path: Path to checkpoint file (None = load latest)
            device: Device to load tensors to

        Returns:
            Loaded state dictionary, or None if no checkpoint found

        Example:
            >>> manager = CheckpointManager(Path("checkpoints"))
            >>> state = manager.resume_from_checkpoint()
            >>> if state:
            >>>     trainer.load_state_dict(state)
            >>>     print(f"Resumed from step {state['global_step']}")
            >>> else:
            >>>     print("Starting training from scratch")
        """
        try:
            return self.load_checkpoint(checkpoint_path, device)
        except FileNotFoundError:
            logger.info("No checkpoint found, starting from scratch")
            return None

    def _add_rng_states(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Add RNG states to state dictionary for reproducibility.

        Args:
            state_dict: State dictionary to augment

        Returns:
            State dictionary with RNG states
        """
        # Create copy to avoid modifying input
        state_dict = state_dict.copy()

        # Add/update RNG states
        if 'rng_states' not in state_dict:
            state_dict['rng_states'] = {}

        state_dict['rng_states']['python'] = random.getstate()
        state_dict['rng_states']['numpy'] = np.random.get_state()
        state_dict['rng_states']['torch'] = torch.get_rng_state()
        if torch.cuda.is_available():
            state_dict['rng_states']['cuda'] = torch.cuda.get_rng_state()

        return state_dict

    def _restore_rng_states(self, state_dict: Dict[str, Any]):
        """Restore RNG states from checkpoint.

        Args:
            state_dict: State dictionary with RNG states
        """
        if 'rng_states' not in state_dict:
            logger.warning("No RNG states found in checkpoint")
            return

        rng_states = state_dict['rng_states']

        try:
            # Restore Python RNG state
            if rng_states.get('python') is not None:
                random.setstate(rng_states['python'])
                logger.debug("Restored Python RNG state")

            # Restore NumPy RNG state
            if rng_states.get('numpy') is not None:
                np.random.set_state(rng_states['numpy'])
                logger.debug("Restored NumPy RNG state")

            # Restore PyTorch RNG state
            if rng_states.get('torch') is not None:
                torch_rng = rng_states['torch']

                # FIX: Ensure RNG state is on CPU and is a ByteTensor
                # When checkpoint is loaded with map_location='cuda', tensors are moved to CUDA
                # but torch.set_rng_state() requires a CPU ByteTensor
                if isinstance(torch_rng, torch.Tensor):
                    if torch_rng.device.type != 'cpu':
                        torch_rng = torch_rng.cpu()
                    # Ensure it's a ByteTensor (uint8)
                    if torch_rng.dtype != torch.uint8:
                        logger.warning(f"Unexpected torch RNG state dtype: {torch_rng.dtype}, converting to uint8")
                        torch_rng = torch_rng.to(torch.uint8)
                    torch.set_rng_state(torch_rng)
                    logger.debug("Restored PyTorch RNG state")
                else:
                    logger.warning(f"Unexpected torch RNG state type: {type(torch_rng)}, skipping")

            # Restore CUDA RNG state
            if rng_states.get('cuda') is not None and torch.cuda.is_available():
                cuda_rng = rng_states['cuda']

                # FIX: Ensure CUDA RNG state is on CPU and is a ByteTensor
                # torch.cuda.set_rng_state() also requires CPU ByteTensor
                if isinstance(cuda_rng, torch.Tensor):
                    if cuda_rng.device.type != 'cpu':
                        cuda_rng = cuda_rng.cpu()
                    # Ensure it's a ByteTensor (uint8)
                    if cuda_rng.dtype != torch.uint8:
                        logger.warning(f"Unexpected CUDA RNG state dtype: {cuda_rng.dtype}, converting to uint8")
                        cuda_rng = cuda_rng.to(torch.uint8)
                    torch.cuda.set_rng_state(cuda_rng)
                    logger.debug("Restored CUDA RNG state")
                else:
                    logger.warning(f"Unexpected CUDA RNG state type: {type(cuda_rng)}, skipping")

            logger.info("✅ RNG states restored successfully")

        except Exception as e:
            logger.warning(f"⚠️  Failed to restore RNG states: {e}")
            logger.warning("   Training will continue without RNG state restoration")
            logger.warning("   This may affect reproducibility but won't affect model training")
            # Don't raise - just warn and continue

    def _update_latest_checkpoint(self, checkpoint_path: Path):
        """Update latest checkpoint symlink/copy.

        Args:
            checkpoint_path: Path to latest checkpoint
        """
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"

        try:
            # Try creating symlink (not supported on all filesystems)
            if latest_path.exists() or latest_path.is_symlink():
                latest_path.unlink()
            latest_path.symlink_to(checkpoint_path.name)
            logger.debug(f"Updated latest checkpoint symlink: {latest_path}")

        except (OSError, NotImplementedError):
            # Fallback to copy if symlinks not supported
            shutil.copy2(checkpoint_path, latest_path)
            logger.debug(f"Updated latest checkpoint copy: {latest_path}")

    def _get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint.

        Implements automatic resumption (T063):
        - First check latest_checkpoint.txt
        - Fallback to checkpoint_latest.pt
        - Fallback to highest step number

        Returns:
            Path to latest checkpoint, or None if no checkpoints found
        """
        # First: check latest_checkpoint.txt (T063)
        if self.latest_checkpoint_file.exists():
            try:
                with open(self.latest_checkpoint_file, 'r') as f:
                    path_str = f.read().strip()
                    if path_str:
                        latest_path = Path(path_str)
                        if latest_path.exists():
                            logger.debug(f"Found checkpoint from latest_checkpoint.txt: {latest_path.name}")
                            return latest_path
                        else:
                            logger.warning(f"Checkpoint in latest_checkpoint.txt not found: {latest_path}")
            except Exception as e:
                logger.warning(f"Failed to read latest_checkpoint.txt: {e}")

        # Second: check for explicit latest checkpoint
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        if latest_path.exists():
            # Resolve symlink if needed
            if latest_path.is_symlink():
                return latest_path.resolve()
            else:
                return latest_path

        # Fallback: find checkpoint with highest step number
        checkpoints = self._list_checkpoints()
        if checkpoints:
            return checkpoints[-1]  # Last checkpoint (highest step)

        return None

    def _list_checkpoints(self) -> List[Path]:
        """List all checkpoint files sorted by step number.

        Returns:
            List of checkpoint paths sorted by step number
        """
        # Find all checkpoint files
        checkpoint_pattern = "checkpoint_*.pt"
        checkpoints = []

        for path in self.checkpoint_dir.glob(checkpoint_pattern):
            # Skip special checkpoints
            if path.name in ["checkpoint_latest.pt", "checkpoint_best.pt"]:
                continue

            # Extract step number
            try:
                step_str = path.stem.split('_')[-1]
                step = int(step_str)
                checkpoints.append((step, path))
            except (ValueError, IndexError):
                logger.warning(f"Skipping invalid checkpoint name: {path.name}")
                continue

        # Sort by step number
        checkpoints.sort(key=lambda x: x[0])

        return [path for _, path in checkpoints]

    def _rotate_checkpoints(self):
        """Remove old checkpoints, keeping only last N + best.

        Implements retention policy (T061):
        - Keep last N checkpoints (default: 3)
        - Always keep best checkpoint
        """
        if self.keep_last_n <= 0:
            return  # Keep all checkpoints

        checkpoints = self._list_checkpoints()

        # Remove oldest checkpoints if we exceed limit
        # Note: Always preserve checkpoint_best.pt (not in _list_checkpoints)
        if len(checkpoints) > self.keep_last_n:
            num_to_remove = len(checkpoints) - self.keep_last_n
            to_remove = checkpoints[:num_to_remove]

            for path in to_remove:
                logger.info(f"Removing old checkpoint (rotation): {path.name}")
                path.unlink()

    def _update_latest_checkpoint_file(self, checkpoint_path: Path):
        """Update latest_checkpoint.txt with path to latest checkpoint.

        Implements automatic resumption (T063):
        - Write absolute path to latest checkpoint
        - Used for crash recovery

        Args:
            checkpoint_path: Path to latest checkpoint
        """
        try:
            with open(self.latest_checkpoint_file, 'w') as f:
                f.write(str(checkpoint_path.absolute()))
            logger.debug(f"Updated latest_checkpoint.txt: {checkpoint_path.name}")
        except Exception as e:
            logger.warning(f"Failed to update latest_checkpoint.txt: {e}")

    def _get_total_checkpoint_size(self) -> float:
        """Get total size of all checkpoints in GB.

        Returns:
            Total size in GB
        """
        total_size = 0
        for path in self.checkpoint_dir.glob("checkpoint_*.pt"):
            if path.is_file():
                total_size += path.stat().st_size

        return total_size / (1024**3)  # Convert bytes to GB

    def _prune_checkpoints_by_size(self):
        """Prune checkpoints to maintain total size limit.

        Implements checkpoint pruning (T062):
        - Auto-delete oldest checkpoints to maintain ≤100GB total
        - Always preserve checkpoint_best.pt
        - Preserve at least 1 checkpoint even if over limit
        """
        if self.max_total_size_gb <= 0:
            return  # No size limit

        total_size = self._get_total_checkpoint_size()

        if total_size <= self.max_total_size_gb:
            return  # Within limit

        logger.warning(f"Total checkpoint size ({total_size:.1f} GB) exceeds limit ({self.max_total_size_gb:.1f} GB)")
        logger.info("Pruning old checkpoints...")

        checkpoints = self._list_checkpoints()

        # Preserve at least 1 checkpoint
        if len(checkpoints) <= 1:
            logger.warning("Only 1 checkpoint remaining, cannot prune further")
            return

        # Remove oldest checkpoints until we're under the limit
        for path in checkpoints[:-1]:  # Keep at least the last checkpoint
            # Remove checkpoint
            size_mb = path.stat().st_size / (1024**2)
            logger.info(f"Removing checkpoint (size limit): {path.name} ({size_mb:.1f} MB)")
            path.unlink()

            # Check if we're under the limit now
            total_size = self._get_total_checkpoint_size()
            if total_size <= self.max_total_size_gb:
                logger.info(f"Total checkpoint size now: {total_size:.1f} GB")
                break

    def _format_size(self, path: Path) -> str:
        """Format file size in human-readable format.

        Args:
            path: Path to file

        Returns:
            Formatted size string (e.g., "1.5 GB")
        """
        size = path.stat().st_size
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} PB"

    def get_checkpoint_info(self, checkpoint_path: Optional[Path] = None) -> Dict[str, Any]:
        """Get information about a checkpoint without loading full state.

        Args:
            checkpoint_path: Path to checkpoint (None = latest)

        Returns:
            Dictionary with checkpoint metadata

        Example:
            >>> manager = CheckpointManager(Path("checkpoints"))
            >>> info = manager.get_checkpoint_info()
            >>> print(f"Step: {info['global_step']}, Loss: {info['best_val_loss']}")
        """
        if checkpoint_path is None:
            checkpoint_path = self._get_latest_checkpoint()
            if checkpoint_path is None:
                raise FileNotFoundError("No checkpoints found")

        checkpoint_path = Path(checkpoint_path)

        # Load checkpoint
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Extract metadata
        info = {
            'checkpoint_path': str(checkpoint_path),
            'file_size': self._format_size(checkpoint_path),
            'global_step': state.get('global_step', None),
            'epoch': state.get('epoch', None),
            'best_val_loss': state.get('best_val_loss', None),
            'config': state.get('config', {}),
        }

        return info


def save_checkpoint(
    trainer,
    checkpoint_dir: Path,
    step: int,
    is_best: bool = False,
    keep_last_n: int = 5,
) -> Path:
    """Convenience function to save checkpoint from trainer.

    Args:
        trainer: Trainer instance with get_state_dict() method
        checkpoint_dir: Directory for checkpoints
        step: Global training step
        is_best: Whether this is the best validation checkpoint
        keep_last_n: Number of recent checkpoints to keep

    Returns:
        Path to saved checkpoint

    Example:
        >>> from distillation.checkpoint import save_checkpoint
        >>> path = save_checkpoint(trainer, Path("checkpoints"), step=1000)
    """
    manager = CheckpointManager(checkpoint_dir, keep_last_n=keep_last_n)
    state_dict = trainer.get_state_dict()
    return manager.save_checkpoint(state_dict, step, is_best)


def load_checkpoint(
    checkpoint_dir: Path,
    checkpoint_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Convenience function to load checkpoint.

    Args:
        checkpoint_dir: Directory containing checkpoints
        checkpoint_path: Specific checkpoint to load (None = latest)
        device: Device to load tensors to

    Returns:
        Loaded state dictionary

    Example:
        >>> from distillation.checkpoint import load_checkpoint
        >>> state = load_checkpoint(Path("checkpoints"))
        >>> trainer.load_state_dict(state)
    """
    manager = CheckpointManager(checkpoint_dir)
    return manager.load_checkpoint(checkpoint_path, device)


def resume_training(
    trainer,
    checkpoint_dir: Path,
    device: Optional[torch.device] = None,
) -> bool:
    """Convenience function to resume training from latest checkpoint.

    Args:
        trainer: Trainer instance with load_state_dict() method
        checkpoint_dir: Directory containing checkpoints
        device: Device to load tensors to

    Returns:
        True if resumed from checkpoint, False if starting from scratch

    Example:
        >>> from distillation.checkpoint import resume_training
        >>> resumed = resume_training(trainer, Path("checkpoints"))
        >>> if resumed:
        >>>     print(f"Resumed from step {trainer.global_step}")
        >>> else:
        >>>     print("Starting from scratch")
    """
    manager = CheckpointManager(checkpoint_dir)
    state = manager.resume_from_checkpoint(device=device)

    if state is not None:
        trainer.load_state_dict(state)
        return True
    else:
        return False
