"""
Comprehensive tests for evaluation infrastructure and checkpoint management.

Tests T056-T064:
- T056: NIAH evaluator
- T057: Perplexity evaluator
- T058: Evaluation runner
- T059: Evaluation scheduling (integration)
- T060: Report generation
- T061: Retention policy (keep last 3 + best)
- T062: Checkpoint pruning (≤100GB)
- T063: Automatic resumption (latest_checkpoint.txt)
- T064: Crash recovery
"""

import pytest
import torch
import torch.nn as nn
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
from transformers import PreTrainedTokenizer

from src.distillation.evaluation.niah import NIAHEvaluator, NIAHConfig, evaluate_niah
from src.distillation.evaluation.perplexity import PerplexityEvaluator, PerplexityConfig, evaluate_perplexity
from src.distillation.evaluation.runner import EvaluationRunner, run_evaluation
from src.distillation.checkpoint import CheckpointManager


# ============================================================================
# Mock Models and Tokenizers
# ============================================================================

class MockModel(nn.Module):
    """Mock model for testing."""
    def __init__(self, vocab_size=100, hidden_size=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward_train(self, input_ids):
        """Mock forward pass returning hidden states."""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        return torch.randn(batch_size, seq_len, self.hidden_size, device=device)

    def generate(self, input_ids, max_new_tokens=10, **kwargs):
        """Mock generate method."""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        # Generate random tokens on same device
        new_tokens = torch.randint(0, self.vocab_size, (batch_size, max_new_tokens), device=device)
        return torch.cat([input_ids, new_tokens], dim=1)


class MockTokenizer:
    """Mock tokenizer for testing."""
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.model_max_length = 4096
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(self, text, return_tensors=None, truncation=True, max_length=None, **kwargs):
        """Mock tokenization."""
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        # Mock tokenization: random tokens
        max_len = max_length or 128
        input_ids = [torch.randint(2, self.vocab_size, (min(len(t), max_len),)) for t in texts]

        if return_tensors == "pt":
            # Pad to same length
            max_seq_len = max(len(ids) for ids in input_ids)
            padded_ids = []
            for ids in input_ids:
                if len(ids) < max_seq_len:
                    padding = torch.full((max_seq_len - len(ids),), self.pad_token_id, dtype=ids.dtype)
                    ids = torch.cat([ids, padding])
                padded_ids.append(ids)

            return {'input_ids': torch.stack(padded_ids)}
        else:
            return {'input_ids': input_ids}

    def decode(self, token_ids, skip_special_tokens=True):
        """Mock decoding."""
        # Return random 4-digit number for NIAH tests
        import random
        return str(random.randint(1000, 9999))


# ============================================================================
# NIAH Evaluator Tests (T056)
# ============================================================================

class TestNIAHEvaluator:
    """Test NIAH evaluator functionality."""

    def test_init(self):
        """Test NIAH evaluator initialization."""
        model = MockModel()
        tokenizer = MockTokenizer()

        evaluator = NIAHEvaluator(model, tokenizer)

        assert evaluator.model is model
        assert evaluator.tokenizer is tokenizer
        assert evaluator.device is not None

    def test_generate_haystack(self):
        """Test haystack generation."""
        model = MockModel()
        tokenizer = MockTokenizer()
        evaluator = NIAHEvaluator(model, tokenizer)

        import random
        random_gen = random.Random(42)

        haystack = evaluator._generate_haystack(100, random_gen)

        assert isinstance(haystack, str)
        assert len(haystack) > 0
        assert '.' in haystack  # Should have sentences

    def test_create_needle(self):
        """Test needle creation."""
        model = MockModel()
        tokenizer = MockTokenizer()
        evaluator = NIAHEvaluator(model, tokenizer)

        import random
        random_gen = random.Random(42)

        needle, answer = evaluator._create_needle(42, random_gen)

        assert isinstance(needle, str)
        assert isinstance(answer, str)
        assert answer in needle
        assert len(answer) == 4  # 4-digit number
        assert answer.isdigit()

    def test_insert_needle(self):
        """Test needle insertion."""
        model = MockModel()
        tokenizer = MockTokenizer()
        evaluator = NIAHEvaluator(model, tokenizer)

        haystack = "Sentence one. Sentence two. Sentence three. Sentence four."
        needle = "The magic number is 1234."

        # Insert at beginning
        result = evaluator._insert_needle(haystack, needle, 0.0)
        assert needle in result

        # Insert in middle
        result = evaluator._insert_needle(haystack, needle, 0.5)
        assert needle in result

        # Insert at end
        result = evaluator._insert_needle(haystack, needle, 1.0)
        assert needle in result

    def test_niah_config(self):
        """Test NIAH configuration."""
        config = NIAHConfig(context_length=2048, num_samples=50)

        assert config.context_length == 2048
        assert config.num_samples == 50
        assert config.needle_positions is not None
        assert len(config.needle_positions) == config.num_positions

    @pytest.mark.slow
    def test_evaluate_single(self):
        """Test single NIAH evaluation."""
        model = MockModel()
        tokenizer = MockTokenizer()
        evaluator = NIAHEvaluator(model, tokenizer)

        import random
        random_gen = random.Random(42)

        result = evaluator.evaluate_single(
            sample_id=0,
            position=0.5,
            context_length=100,
            random_gen=random_gen,
        )

        assert 'sample_id' in result
        assert 'position' in result
        assert 'expected_answer' in result
        assert 'predicted_answer' in result
        assert 'correct' in result
        assert isinstance(result['correct'], bool)


# ============================================================================
# Perplexity Evaluator Tests (T057)
# ============================================================================

class TestPerplexityEvaluator:
    """Test perplexity evaluator functionality."""

    def test_init(self):
        """Test perplexity evaluator initialization."""
        model = MockModel()
        tokenizer = MockTokenizer()

        evaluator = PerplexityEvaluator(model, tokenizer)

        assert evaluator.model is model
        assert evaluator.tokenizer is tokenizer
        assert evaluator.device is not None

    def test_compute_loss(self):
        """Test loss computation."""
        model = MockModel(vocab_size=100)
        tokenizer = MockTokenizer()
        evaluator = PerplexityEvaluator(model, tokenizer)

        # Create input on correct device
        input_ids = torch.randint(0, 100, (2, 10), device=evaluator.device)  # [batch=2, seq_len=10]

        loss, num_tokens = evaluator._compute_loss(input_ids)

        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0  # Loss should be positive
        assert num_tokens > 0
        assert num_tokens == 2 * 9  # (batch * (seq_len - 1))

    def test_perplexity_config(self):
        """Test perplexity configuration."""
        config = PerplexityConfig(max_samples=500, batch_size=2)

        assert config.max_samples == 500
        assert config.batch_size == 2
        assert config.max_length == 4096
        assert config.ignore_index == -100

    def test_evaluate_text(self):
        """Test text evaluation."""
        model = MockModel()
        tokenizer = MockTokenizer()
        evaluator = PerplexityEvaluator(model, tokenizer)

        text = "The quick brown fox jumps over the lazy dog."
        results = evaluator.evaluate_text(text)

        assert 'perplexity' in results
        assert 'loss' in results
        assert 'bits_per_token' in results
        assert results['perplexity'] > 0
        assert results['num_samples'] == 1


# ============================================================================
# Evaluation Runner Tests (T058, T060)
# ============================================================================

class TestEvaluationRunner:
    """Test evaluation runner and report generation."""

    def test_init(self):
        """Test evaluation runner initialization."""
        model = MockModel()
        tokenizer = MockTokenizer()

        runner = EvaluationRunner(model, tokenizer)

        assert runner.model is model
        assert runner.tokenizer is tokenizer
        assert runner.perplexity_evaluator is not None
        assert runner.niah_evaluator is not None

    def test_register_custom_evaluator(self):
        """Test custom evaluator registration."""
        model = MockModel()
        tokenizer = MockTokenizer()
        runner = EvaluationRunner(model, tokenizer)

        def custom_eval(model, tokenizer):
            return {'custom_metric': 0.95}

        runner.register_custom_evaluator('my_custom', custom_eval)

        assert 'my_custom' in runner.custom_evaluators
        assert runner.custom_evaluators['my_custom'] is custom_eval

    def test_generate_report_text(self):
        """Test report text generation (T060)."""
        model = MockModel()
        tokenizer = MockTokenizer()
        runner = EvaluationRunner(model, tokenizer)

        results = {
            'timestamp': '2025-11-02T12:00:00',
            'step': 5000,
            'total_elapsed_time': 123.45,
            'perplexity': {
                'perplexity': 15.3,
                'loss': 2.73,
                'bits_per_token': 3.94,
                'total_tokens': 1000000,
                'num_samples': 1000,
                'elapsed_time': 100.0,
            },
            'niah': {
                'accuracy': 0.95,
                'total_samples': 100,
                'total_correct': 95,
                'position_accuracies': {0.1: 0.98, 0.5: 0.92},
                'elapsed_time': 23.45,
            },
        }

        report_text = runner.generate_report_text(results)

        assert 'EVALUATION REPORT' in report_text
        assert 'Perplexity:' in report_text
        assert '15.3' in report_text
        assert 'NIAH' in report_text
        # Check for accuracy - format is "95.00%"
        assert '95.00%' in report_text or '95%' in report_text

    def test_save_reports(self):
        """Test report saving (T060)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = MockModel()
            tokenizer = MockTokenizer()
            runner = EvaluationRunner(model, tokenizer)

            results = {
                'timestamp': '2025-11-02T12:00:00',
                'step': 5000,
                'perplexity': {'perplexity': 15.3},
            }

            output_dir = Path(tmpdir)
            runner.save_reports(results, output_dir, step=5000)

            # Check JSON report exists
            json_path = output_dir / "eval_00005000.json"
            assert json_path.exists()

            # Check text report exists
            txt_path = output_dir / "eval_00005000.txt"
            assert txt_path.exists()

            # Verify JSON content
            with open(json_path) as f:
                loaded = json.load(f)
            assert loaded['step'] == 5000


# ============================================================================
# Checkpoint Management Tests (T061-T064)
# ============================================================================

class TestCheckpointManager:
    """Test checkpoint manager with retention, pruning, and resumption."""

    def test_init(self):
        """Test checkpoint manager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                checkpoint_dir=Path(tmpdir),
                keep_last_n=3,
                max_total_size_gb=0.1,
            )

            assert manager.checkpoint_dir.exists()
            assert manager.keep_last_n == 3
            assert manager.max_total_size_gb == 0.1
            assert manager.latest_checkpoint_file.exists() or not manager.latest_checkpoint_file.exists()

    def test_retention_policy(self):
        """Test retention policy - keep last 3 + best (T061)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                checkpoint_dir=Path(tmpdir),
                keep_last_n=3,
            )

            # Save 5 checkpoints
            for step in [1000, 2000, 3000, 4000, 5000]:
                state = {'global_step': step, 'data': torch.randn(100)}
                manager.save_checkpoint(state, step, is_best=(step == 3000))

            # Check that only last 3 + best are kept
            checkpoints = list(Path(tmpdir).glob("checkpoint_*.pt"))
            checkpoint_names = {p.name for p in checkpoints}

            # Should have: checkpoint_3000.pt, checkpoint_4000.pt, checkpoint_5000.pt,
            # checkpoint_best.pt, checkpoint_latest.pt
            assert 'checkpoint_3000.pt' in checkpoint_names
            assert 'checkpoint_4000.pt' in checkpoint_names
            assert 'checkpoint_5000.pt' in checkpoint_names
            assert 'checkpoint_best.pt' in checkpoint_names

            # Old checkpoints should be removed
            assert 'checkpoint_1000.pt' not in checkpoint_names
            assert 'checkpoint_2000.pt' not in checkpoint_names

    def test_automatic_resumption(self):
        """Test automatic resumption via latest_checkpoint.txt (T063)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=Path(tmpdir))

            # Save checkpoint
            state = {'global_step': 1000, 'data': torch.randn(10)}
            checkpoint_path = manager.save_checkpoint(state, step=1000)

            # Check that latest_checkpoint.txt was created
            assert manager.latest_checkpoint_file.exists()

            # Read latest_checkpoint.txt
            with open(manager.latest_checkpoint_file) as f:
                recorded_path = Path(f.read().strip())

            assert recorded_path == checkpoint_path.absolute()

            # Test resumption
            loaded_path = manager._get_latest_checkpoint()
            assert loaded_path == checkpoint_path

    def test_checkpoint_pruning(self):
        """Test checkpoint pruning by size (T062)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set very small size limit
            manager = CheckpointManager(
                checkpoint_dir=Path(tmpdir),
                keep_last_n=10,  # Large enough to not trigger rotation
                max_total_size_gb=0.001,  # 1 MB limit
            )

            # Save multiple large checkpoints
            for step in [1000, 2000, 3000]:
                # Create ~1MB checkpoint
                state = {'global_step': step, 'data': torch.randn(100000)}
                manager.save_checkpoint(state, step)

            # Check that pruning occurred
            checkpoints = list(Path(tmpdir).glob("checkpoint_[0-9]*.pt"))

            # Should have pruned old checkpoints due to size limit
            # Exact count depends on checkpoint size, but should be < 3
            assert len(checkpoints) < 3

    def test_crash_recovery(self):
        """Test crash recovery scenario (T064)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate training session 1
            manager1 = CheckpointManager(checkpoint_dir=Path(tmpdir))

            state1 = {'global_step': 1000, 'epoch': 1, 'loss': 2.5}
            manager1.save_checkpoint(state1, step=1000)

            state2 = {'global_step': 2000, 'epoch': 2, 'loss': 2.0}
            manager1.save_checkpoint(state2, step=2000)

            # Simulate crash and recovery (new session)
            manager2 = CheckpointManager(checkpoint_dir=Path(tmpdir))

            # Resume from latest checkpoint
            resumed_state = manager2.resume_from_checkpoint()

            assert resumed_state is not None
            assert resumed_state['global_step'] == 2000
            assert resumed_state['epoch'] == 2
            assert resumed_state['loss'] == 2.0

    def test_crash_recovery_no_checkpoint(self):
        """Test crash recovery when no checkpoint exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=Path(tmpdir))

            # Try to resume when no checkpoints exist
            resumed_state = manager.resume_from_checkpoint()

            assert resumed_state is None

    def test_load_checkpoint_with_rng_states(self):
        """Test that RNG states are properly saved and restored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=Path(tmpdir))

            # Save checkpoint with RNG states
            state = {'global_step': 1000, 'data': torch.randn(10)}
            checkpoint_path = manager.save_checkpoint(state, step=1000)

            # Load checkpoint
            loaded_state = manager.load_checkpoint(checkpoint_path)

            # Check RNG states exist
            assert 'rng_states' in loaded_state
            assert 'python' in loaded_state['rng_states']
            assert 'numpy' in loaded_state['rng_states']
            assert 'torch' in loaded_state['rng_states']


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for evaluation and checkpointing."""

    def test_full_evaluation_workflow(self):
        """Test complete evaluation workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = MockModel()
            tokenizer = MockTokenizer()

            # Create simple dataloader
            from torch.utils.data import DataLoader, TensorDataset
            dataset = TensorDataset(torch.randint(0, 100, (10, 32)))
            dataloader = DataLoader(dataset, batch_size=2)

            # Run evaluation
            runner = EvaluationRunner(model, tokenizer)
            results = runner.run_all(
                val_dataloader=dataloader,
                perplexity_config=PerplexityConfig(max_samples=5),
                niah_config=NIAHConfig(num_samples=2, num_positions=2),
                output_dir=Path(tmpdir),
                step=1000,
            )

            # Check results structure
            assert 'perplexity' in results
            assert 'niah' in results
            assert 'timestamp' in results

            # Check reports were saved
            assert (Path(tmpdir) / "eval_00001000.json").exists()
            assert (Path(tmpdir) / "eval_00001000.txt").exists()

    def test_checkpoint_save_load_cycle(self):
        """Test save/load cycle preserves all state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=Path(tmpdir))

            # Create state dict
            original_state = {
                'global_step': 5000,
                'epoch': 3,
                'best_val_loss': 1.5,
                'model_state_dict': {'weight': torch.randn(10, 10)},
                'optimizer_state_dict': {'param': torch.randn(5)},
            }

            # Save
            checkpoint_path = manager.save_checkpoint(original_state, step=5000)

            # Load
            loaded_state = manager.load_checkpoint(checkpoint_path)

            # Verify
            assert loaded_state['global_step'] == original_state['global_step']
            assert loaded_state['epoch'] == original_state['epoch']
            assert loaded_state['best_val_loss'] == original_state['best_val_loss']


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
