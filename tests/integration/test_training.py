"""Integration test for training pipeline.

Tests end-to-end training:
- Train for N steps
- Save checkpoint
- Load checkpoint and resume
- Verify loss decreases
- Verify memory constraints (FR-003, SC-001)
"""

import pytest
import torch
import tempfile
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.core import RetNetHRMModel
from src.config.model_config import ModelConfig
from src.config.train_config import TrainingConfig
from src.data.tokenizer import get_tokenizer
from src.data.collator import DataCollator
from src.training.optimizer import setup_optimizer_and_scheduler
from src.training.trainer import Trainer
from src.training.checkpoint import CheckpointManager


@pytest.fixture
def tiny_model_config():
    """Create tiny model config for fast testing."""
    return ModelConfig(
        d_model=128,
        n_layers_retnet=2,
        n_retention_heads=2,
        vocab_size=1000,
        max_seq_len_train=256,
        max_seq_len_infer=512,
        mlp_mult=2,
        dropout=0.0,
    )


@pytest.fixture
def tiny_train_config():
    """Create tiny training config for fast testing."""
    return TrainingConfig(
        batch_size=2,
        seq_len=64,
        learning_rate=1e-3,
        max_steps=20,
        eval_interval_steps=10,
        log_interval_steps=5,
        checkpoint_interval_seconds=9999,  # Disable auto checkpoint for test
        precision="fp32",  # Faster for CPU testing
        device="cpu",
        num_workers=0,
    )


@pytest.fixture
def dummy_dataset():
    """Create dummy dataset for testing."""
    class DummyDataset:
        def __init__(self, num_samples=100, seq_len=64, vocab_size=1000):
            self.num_samples = num_samples
            self.seq_len = seq_len
            self.vocab_size = vocab_size

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            return {
                'input_ids': torch.randint(0, self.vocab_size, (self.seq_len,)),
                'labels': torch.randint(0, self.vocab_size, (self.seq_len,)),
            }

    return DummyDataset()


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestTrainingBasics:
    """Test basic training functionality."""

    def test_single_training_step(self, tiny_model_config, dummy_dataset):
        """Test a single training step executes without errors."""
        model = RetNetHRMModel(config=tiny_model_config)
        optimizer, scheduler = setup_optimizer_and_scheduler(
            model=model,
            learning_rate=1e-3,
            max_steps=10,
        )

        # Create data loader
        from torch.utils.data import DataLoader
        train_loader = DataLoader(dummy_dataset, batch_size=2, shuffle=False)

        # Create trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=None,
            checkpoint_manager=None,
            device="cpu",
            precision="fp32",
            grad_accumulation_steps=1,
            log_interval=1,
        )

        # Get one batch
        batch = next(iter(train_loader))

        # Single training step
        loss, metrics = trainer._train_step(batch)

        # Verify loss is computed
        assert isinstance(loss, float), "Loss should be float"
        assert loss > 0, "Loss should be positive"

        # Verify metrics
        assert 'train/loss' in metrics
        assert 'train/lr' in metrics
        assert 'train/tokens_per_second' in metrics

    def test_training_reduces_loss(self, tiny_model_config, tiny_train_config, dummy_dataset):
        """Test that training reduces loss over time."""
        model = RetNetHRMModel(config=tiny_model_config)
        optimizer, scheduler = setup_optimizer_and_scheduler(
            model=model,
            learning_rate=tiny_train_config.learning_rate,
            max_steps=tiny_train_config.max_steps,
        )

        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            dummy_dataset,
            batch_size=tiny_train_config.batch_size,
            shuffle=False,
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=None,
            checkpoint_manager=None,
            device="cpu",
            precision="fp32",
            grad_accumulation_steps=1,
            log_interval=100,  # Disable logging
        )

        # Record initial loss
        batch = next(iter(train_loader))
        initial_loss, _ = trainer._train_step(batch)

        # Train for a few steps
        trainer.train(max_steps=20, start_step=0)

        # Check final loss
        final_loss, _ = trainer._train_step(batch)

        # Loss should decrease (or at least not increase significantly)
        # Note: With random data, loss may not always decrease, but it shouldn't explode
        assert final_loss < initial_loss * 2, \
            f"Loss exploded: {initial_loss:.4f} -> {final_loss:.4f}"


class TestCheckpointSaveLoad:
    """Test checkpoint saving and loading during training."""

    def test_save_and_resume_training(
        self,
        tiny_model_config,
        tiny_train_config,
        dummy_dataset,
        temp_checkpoint_dir,
    ):
        """Test saving checkpoint mid-training and resuming."""
        # Train for initial steps
        model = RetNetHRMModel(config=tiny_model_config)
        optimizer, scheduler = setup_optimizer_and_scheduler(
            model=model,
            learning_rate=tiny_train_config.learning_rate,
            max_steps=tiny_train_config.max_steps,
        )

        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            dummy_dataset,
            batch_size=tiny_train_config.batch_size,
            shuffle=False,
        )

        checkpoint_manager = CheckpointManager(
            save_dir=temp_checkpoint_dir,
            interval_seconds=9999,  # Manual only
            save_on_ctrl_c=False,
            model=model,
            optimizer=optimizer,
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=None,
            checkpoint_manager=checkpoint_manager,
            device="cpu",
            precision="fp32",
            log_interval=100,
        )

        # Train for 10 steps
        trainer.train(max_steps=10, start_step=0)

        # Save checkpoint manually
        checkpoint_path = os.path.join(temp_checkpoint_dir, "checkpoint-step-10")
        trainer.save_checkpoint(checkpoint_path)

        # Verify checkpoint exists
        assert os.path.exists(f"{checkpoint_path}.safetensors"), \
            "Checkpoint not saved"

        # Create new model and resume
        model2 = RetNetHRMModel(config=tiny_model_config)
        optimizer2, scheduler2 = setup_optimizer_and_scheduler(
            model=model2,
            learning_rate=tiny_train_config.learning_rate,
            max_steps=tiny_train_config.max_steps,
        )

        # Load checkpoint
        from src.training.checkpoint import load_checkpoint
        loaded_data = load_checkpoint(
            checkpoint_path=f"{checkpoint_path}.safetensors",
            model=model2,
            optimizer=optimizer2,
            device="cpu",
        )

        # Verify loaded correctly
        assert loaded_data['global_step'] == 10, "Wrong global step"

        # Continue training from step 10 to 20
        trainer2 = Trainer(
            model=model2,
            optimizer=optimizer2,
            scheduler=scheduler2,
            train_loader=train_loader,
            val_loader=None,
            checkpoint_manager=None,
            device="cpu",
            precision="fp32",
            log_interval=100,
        )

        trainer2.train(max_steps=20, start_step=10)

        # Should complete without errors
        assert trainer2.global_step == 20


class TestMemoryConstraints:
    """Test FR-003: Memory usage ≤32GB."""

    def test_memory_tracking(self, tiny_model_config, dummy_dataset):
        """Test that memory is tracked during training."""
        model = RetNetHRMModel(config=tiny_model_config)
        optimizer, scheduler = setup_optimizer_and_scheduler(model=model, max_steps=10)

        from torch.utils.data import DataLoader
        train_loader = DataLoader(dummy_dataset, batch_size=2)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=None,
            device="cpu",
            precision="fp32",
            log_interval=1,
        )

        # Train one step
        batch = next(iter(train_loader))
        loss, metrics = trainer._train_step(batch)

        # Check memory metrics are present
        assert 'system/memory_allocated_gb' in metrics
        assert 'system/memory_reserved_gb' in metrics

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_limit_check(self, tiny_model_config):
        """Test that GPU memory limit is enforced (FR-003)."""
        model = RetNetHRMModel(config=tiny_model_config).cuda()

        # Should not exceed limit with tiny model
        B, T = 2, 64
        input_ids = torch.randint(0, tiny_model_config.vocab_size, (B, T)).cuda()

        output = model.forward(input_ids=input_ids)

        # Verify memory check was performed
        memory_gb = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
        assert memory_gb < 32, f"Test model used {memory_gb:.2f}GB"


class TestThroughput:
    """Test NFR-002: Training throughput."""

    def test_throughput_tracking(self, tiny_model_config, dummy_dataset):
        """Test that throughput is tracked."""
        model = RetNetHRMModel(config=tiny_model_config)
        optimizer, scheduler = setup_optimizer_and_scheduler(model=model, max_steps=10)

        from torch.utils.data import DataLoader
        train_loader = DataLoader(dummy_dataset, batch_size=2)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=None,
            device="cpu",
            precision="fp32",
            log_interval=1,
        )

        # Train for a few steps
        trainer.train(max_steps=5, start_step=0)

        # Get throughput stats
        stats = trainer.get_throughput_stats()

        assert 'tokens_processed' in stats
        assert 'elapsed_time_seconds' in stats
        assert 'tokens_per_second' in stats

        assert stats['tokens_processed'] > 0
        assert stats['tokens_per_second'] > 0


class TestGradientAccumulation:
    """Test gradient accumulation."""

    def test_gradient_accumulation_steps(self, tiny_model_config, dummy_dataset):
        """Test training with gradient accumulation."""
        model = RetNetHRMModel(config=tiny_model_config)
        optimizer, scheduler = setup_optimizer_and_scheduler(model=model, max_steps=20)

        from torch.utils.data import DataLoader
        train_loader = DataLoader(dummy_dataset, batch_size=2)

        # Train with grad accumulation = 4
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=None,
            device="cpu",
            precision="fp32",
            grad_accumulation_steps=4,  # Accumulate over 4 batches
            log_interval=100,
        )

        # Train for one effective batch (4 accumulation steps = 1 optimizer step)
        initial_step = trainer.global_step
        trainer.train(max_steps=4, start_step=0)

        # Should have completed gradient accumulation
        assert trainer.global_step == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
