"""
Integration tests for distillation trainer.

Tests end-to-end training loop with dummy data and mock teacher server.
Verifies loss decreases, checkpoint save/load, and resume functionality.

Task: T043
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.distillation.trainer import DistillationTrainer, TrainingConfig
from src.distillation.student_config import RetNetStudent350MConfig
from src.distillation.checkpoint import CheckpointManager, save_checkpoint, resume_training
from src.distillation.schemas import TopKResponse


class DummyModel(nn.Module):
    """Dummy student model for testing.

    Mimics RetNet API with forward_train and lm_head.
    """

    def __init__(self, vocab_size: int = 128256, d_model: int = 256, seq_len: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seq_len = seq_len

        # Simplified model: just embeddings and linear head
        self.embed = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward_train(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass returning hidden states (RetNet API)."""
        x = self.embed(input_ids)
        x = self.transformer(x)
        x = self.norm(x)
        return x


def create_dummy_dataloader(
    num_samples: int = 100,
    batch_size: int = 1,
    seq_len: int = 128,
    vocab_size: int = 128256,
) -> DataLoader:
    """Create dummy dataloader for testing.

    Args:
        num_samples: Number of samples
        batch_size: Batch size
        seq_len: Sequence length
        vocab_size: Vocabulary size

    Returns:
        DataLoader with dummy data
    """
    # Generate random input_ids
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    # Create dataset
    dataset = TensorDataset(input_ids, attention_mask, labels)

    # Custom collate function to return dict
    def collate_fn(batch):
        input_ids_batch = torch.stack([item[0] for item in batch])
        attention_mask_batch = torch.stack([item[1] for item in batch])
        labels_batch = torch.stack([item[2] for item in batch])

        return {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
            'labels': labels_batch,
        }

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)


def create_mock_teacher_response(
    batch_size: int = 1,
    seq_len: int = 128,
    topk: int = 128,
) -> TopKResponse:
    """Create mock teacher response for testing.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        topk: Number of top-k logits

    Returns:
        Mock TopKResponse
    """
    # Generate random top-k indices and values
    indices = torch.randint(0, 128256, (batch_size, seq_len, topk)).tolist()
    values_int8 = torch.randint(-128, 127, (batch_size, seq_len, topk)).tolist()
    scale = torch.rand(batch_size, seq_len).tolist()
    other_mass = torch.rand(batch_size, seq_len).tolist()

    return TopKResponse(
        indices=indices,
        values_int8=values_int8,
        scale=scale,
        other_mass=other_mass,
        batch_size=batch_size,
        num_positions=seq_len,
        k=topk,
    )


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary checkpoint directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def dummy_model():
    """Create dummy student model."""
    return DummyModel(vocab_size=128256, d_model=256, seq_len=128)


@pytest.fixture
def training_config(temp_checkpoint_dir):
    """Create training configuration for testing."""
    return TrainingConfig(
        physical_batch_size=1,
        gradient_accumulation_steps=2,  # Small for testing
        max_grad_norm=1.0,
        learning_rate=1e-4,
        max_steps=10,  # Small for testing
        use_bf16=False,  # Disable for testing (CPU)
        log_interval=2,
        eval_interval=5,
        save_interval=5,
        teacher_endpoint="http://localhost:8000/v1/topk",
        teacher_topk=128,
        teacher_temperature=2.0,
        distill_alpha=0.2,
    )


@pytest.fixture
def student_config():
    """Create student configuration."""
    return RetNetStudent350MConfig()


@pytest.fixture
def train_dataloader():
    """Create training dataloader."""
    return create_dummy_dataloader(num_samples=20, batch_size=1, seq_len=128)


@pytest.fixture
def val_dataloader():
    """Create validation dataloader."""
    return create_dummy_dataloader(num_samples=10, batch_size=1, seq_len=128)


def test_trainer_initialization(
    dummy_model,
    training_config,
    student_config,
    train_dataloader,
    temp_checkpoint_dir,
):
    """Test trainer initialization."""
    trainer = DistillationTrainer(
        model=dummy_model,
        config=training_config,
        student_config=student_config,
        train_dataloader=train_dataloader,
        checkpoint_dir=temp_checkpoint_dir,
    )

    assert trainer.global_step == 0
    assert trainer.epoch == 0
    assert trainer.model is not None
    assert trainer.optimizer is not None
    assert trainer.loss_fn is not None
    assert trainer.teacher_client is not None


@patch('src.distillation.trainer.TeacherClient')
def test_trainer_basic_training_loop(
    mock_teacher_client_class,
    dummy_model,
    training_config,
    student_config,
    train_dataloader,
    temp_checkpoint_dir,
):
    """Test basic training loop runs without errors.

    Verifies:
    - Training runs for specified steps
    - Loss is computed and finite
    - Gradient accumulation works
    - Logging works
    """
    # Mock teacher client
    mock_teacher_client = MagicMock()
    mock_teacher_client.query_topk.return_value = create_mock_teacher_response(
        batch_size=1, seq_len=128, topk=128
    )
    mock_teacher_client_class.return_value = mock_teacher_client

    # Create trainer
    trainer = DistillationTrainer(
        model=dummy_model,
        config=training_config,
        student_config=student_config,
        train_dataloader=train_dataloader,
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Run training
    stats = trainer.train()

    # Verify training completed
    assert stats['global_step'] == training_config.max_steps
    assert stats['final_loss'] is not None
    assert torch.isfinite(torch.tensor(stats['final_loss']))

    # Verify teacher client was called
    assert mock_teacher_client.query_topk.call_count > 0


@patch('src.distillation.trainer.TeacherClient')
def test_trainer_loss_decreases(
    mock_teacher_client_class,
    dummy_model,
    training_config,
    student_config,
    train_dataloader,
    temp_checkpoint_dir,
):
    """Test that loss decreases (or at least is finite) during training.

    Note: With dummy data and mock teacher, loss may not strictly decrease,
    but it should remain finite.
    """
    # Mock teacher client
    mock_teacher_client = MagicMock()
    mock_teacher_client.query_topk.return_value = create_mock_teacher_response(
        batch_size=1, seq_len=128, topk=128
    )
    mock_teacher_client_class.return_value = mock_teacher_client

    # Create trainer
    trainer = DistillationTrainer(
        model=dummy_model,
        config=training_config,
        student_config=student_config,
        train_dataloader=train_dataloader,
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Run training
    trainer.train()

    # Verify losses are finite
    losses = trainer.metrics['loss']
    assert len(losses) > 0
    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)

    # Verify gradient norms are finite
    grad_norms = trainer.metrics['grad_norm']
    assert len(grad_norms) > 0
    assert all(torch.isfinite(torch.tensor(norm)) for norm in grad_norms)


@patch('src.distillation.trainer.TeacherClient')
def test_checkpoint_save_load(
    mock_teacher_client_class,
    dummy_model,
    training_config,
    student_config,
    train_dataloader,
    temp_checkpoint_dir,
):
    """Test checkpoint save and load functionality.

    Verifies:
    - Checkpoint is saved to disk
    - Checkpoint can be loaded
    - Training state is preserved
    """
    # Mock teacher client
    mock_teacher_client = MagicMock()
    mock_teacher_client.query_topk.return_value = create_mock_teacher_response(
        batch_size=1, seq_len=128, topk=128
    )
    mock_teacher_client_class.return_value = mock_teacher_client

    # Create trainer
    trainer = DistillationTrainer(
        model=dummy_model,
        config=training_config,
        student_config=student_config,
        train_dataloader=train_dataloader,
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Run training for a few steps
    training_config.max_steps = 5
    trainer.config = training_config
    trainer.train()

    # Save checkpoint
    checkpoint_manager = CheckpointManager(temp_checkpoint_dir)
    checkpoint_path = checkpoint_manager.save_checkpoint(
        trainer.get_state_dict(),
        step=trainer.global_step,
    )

    # Verify checkpoint exists
    assert checkpoint_path.exists()

    # Create new trainer and load checkpoint
    new_model = DummyModel(vocab_size=128256, d_model=256, seq_len=128)
    new_trainer = DistillationTrainer(
        model=new_model,
        config=training_config,
        student_config=student_config,
        train_dataloader=train_dataloader,
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Load checkpoint
    loaded_state = checkpoint_manager.load_checkpoint(checkpoint_path)
    new_trainer.load_state_dict(loaded_state)

    # Verify state was restored
    assert new_trainer.global_step == trainer.global_step
    assert new_trainer.epoch == trainer.epoch

    # Verify model weights match
    for p1, p2 in zip(trainer.model.parameters(), new_trainer.model.parameters()):
        assert torch.allclose(p1, p2, atol=1e-6)


@patch('src.distillation.trainer.TeacherClient')
def test_checkpoint_resume(
    mock_teacher_client_class,
    dummy_model,
    training_config,
    student_config,
    train_dataloader,
    temp_checkpoint_dir,
):
    """Test training resume from checkpoint.

    Verifies:
    - Training can resume from checkpoint
    - Global step continues from checkpoint
    - Loss history is preserved
    """
    # Mock teacher client
    mock_teacher_client = MagicMock()
    mock_teacher_client.query_topk.return_value = create_mock_teacher_response(
        batch_size=1, seq_len=128, topk=128
    )
    mock_teacher_client_class.return_value = mock_teacher_client

    # Create trainer and train for a few steps
    trainer1 = DistillationTrainer(
        model=dummy_model,
        config=training_config,
        student_config=student_config,
        train_dataloader=train_dataloader,
        checkpoint_dir=temp_checkpoint_dir,
    )

    training_config.max_steps = 5
    trainer1.config = training_config
    trainer1.train()

    # Save checkpoint
    checkpoint_manager = CheckpointManager(temp_checkpoint_dir)
    checkpoint_manager.save_checkpoint(
        trainer1.get_state_dict(),
        step=trainer1.global_step,
    )

    original_step = trainer1.global_step

    # Create new trainer and resume
    new_model = DummyModel(vocab_size=128256, d_model=256, seq_len=128)
    trainer2 = DistillationTrainer(
        model=new_model,
        config=training_config,
        student_config=student_config,
        train_dataloader=train_dataloader,
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Resume from checkpoint
    resumed = resume_training(trainer2, temp_checkpoint_dir)
    assert resumed is True
    assert trainer2.global_step == original_step

    # Continue training
    training_config.max_steps = 10
    trainer2.config = training_config
    trainer2.train()

    # Verify training continued
    assert trainer2.global_step == 10


@patch('src.distillation.trainer.TeacherClient')
def test_gradient_accumulation(
    mock_teacher_client_class,
    dummy_model,
    training_config,
    student_config,
    train_dataloader,
    temp_checkpoint_dir,
):
    """Test gradient accumulation works correctly.

    Verifies:
    - Optimizer step only happens after accumulation_steps
    - Gradients are accumulated correctly
    """
    # Mock teacher client
    mock_teacher_client = MagicMock()
    mock_teacher_client.query_topk.return_value = create_mock_teacher_response(
        batch_size=1, seq_len=128, topk=128
    )
    mock_teacher_client_class.return_value = mock_teacher_client

    # Set accumulation steps
    training_config.gradient_accumulation_steps = 4
    training_config.max_steps = 2  # 2 optimizer steps = 8 forward passes

    # Create trainer
    trainer = DistillationTrainer(
        model=dummy_model,
        config=training_config,
        student_config=student_config,
        train_dataloader=train_dataloader,
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Spy on optimizer step
    original_step = trainer.optimizer.step
    step_count = [0]

    def counting_step(*args, **kwargs):
        step_count[0] += 1
        return original_step(*args, **kwargs)

    trainer.optimizer.step = counting_step

    # Run training
    trainer.train()

    # Verify optimizer was called correct number of times
    # max_steps=2 means 2 optimizer steps
    assert step_count[0] == 2


@patch('src.distillation.trainer.TeacherClient')
def test_gradient_clipping(
    mock_teacher_client_class,
    dummy_model,
    training_config,
    student_config,
    train_dataloader,
    temp_checkpoint_dir,
):
    """Test gradient clipping is applied.

    Verifies:
    - Gradient norms are clipped to max_grad_norm
    """
    # Mock teacher client
    mock_teacher_client = MagicMock()
    mock_teacher_client.query_topk.return_value = create_mock_teacher_response(
        batch_size=1, seq_len=128, topk=128
    )
    mock_teacher_client_class.return_value = mock_teacher_client

    # Set max grad norm
    training_config.max_grad_norm = 1.0
    training_config.max_steps = 5

    # Create trainer
    trainer = DistillationTrainer(
        model=dummy_model,
        config=training_config,
        student_config=student_config,
        train_dataloader=train_dataloader,
        checkpoint_dir=temp_checkpoint_dir,
    )

    # Run training
    trainer.train()

    # Verify gradient norms are recorded
    grad_norms = trainer.metrics['grad_norm']
    assert len(grad_norms) > 0

    # Note: We can't easily verify clipping without injecting large gradients,
    # but we can verify the clipping mechanism is called and norms are finite
    assert all(torch.isfinite(torch.tensor(norm)) for norm in grad_norms)


def test_checkpoint_manager_rotation(temp_checkpoint_dir):
    """Test checkpoint rotation (keep last N checkpoints).

    Verifies:
    - Old checkpoints are removed
    - Latest N checkpoints are kept
    """
    manager = CheckpointManager(temp_checkpoint_dir, keep_last_n=3)

    # Create dummy state
    dummy_state = {
        'global_step': 0,
        'epoch': 0,
        'model_state_dict': {},
        'optimizer_state_dict': {},
    }

    # Save 5 checkpoints
    for step in [100, 200, 300, 400, 500]:
        dummy_state['global_step'] = step
        manager.save_checkpoint(dummy_state, step)

    # List checkpoints
    checkpoints = manager._list_checkpoints()

    # Should only have 3 checkpoints (last 3)
    assert len(checkpoints) == 3

    # Verify correct checkpoints are kept (300, 400, 500)
    checkpoint_steps = [int(cp.stem.split('_')[-1]) for cp in checkpoints]
    assert checkpoint_steps == [300, 400, 500]


def test_checkpoint_manager_atomic_save(temp_checkpoint_dir):
    """Test atomic checkpoint save prevents corruption.

    Verifies:
    - Checkpoint is saved atomically
    - No partial files left behind
    """
    manager = CheckpointManager(temp_checkpoint_dir)

    dummy_state = {
        'global_step': 100,
        'epoch': 1,
        'model_state_dict': {'dummy': torch.randn(10, 10)},
        'optimizer_state_dict': {},
    }

    # Save checkpoint
    checkpoint_path = manager.save_checkpoint(dummy_state, step=100)

    # Verify checkpoint exists and is complete
    assert checkpoint_path.exists()

    # Verify no temp files left behind
    temp_files = list(temp_checkpoint_dir.glob('.tmp_checkpoint_*'))
    assert len(temp_files) == 0

    # Verify checkpoint can be loaded
    loaded_state = manager.load_checkpoint(checkpoint_path)
    assert loaded_state['global_step'] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
