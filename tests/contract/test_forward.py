"""Contract tests for model forward pass.

Validates model-forward.md contract:
- Shape invariants (training and inference modes)
- Memory constraints (FR-003)
- Input/output specifications
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.core import RetNetHRMModel, ModelOutput, InferenceOutput
from src.config.model_config import ModelConfig


@pytest.fixture
def small_model_config():
    """Create small model config for fast testing."""
    return ModelConfig(
        d_model=256,
        n_layers_retnet=4,
        n_retention_heads=4,
        vocab_size=1000,
        max_seq_len_train=512,
        max_seq_len_infer=1024,
        dropout=0.0,
    )


@pytest.fixture
def model(small_model_config):
    """Create model instance."""
    model = RetNetHRMModel(config=small_model_config)
    model.eval()
    return model


class TestForwardTrainingMode:
    """Test forward() in training mode (parallel)."""

    def test_forward_shape_invariants(self, model, small_model_config):
        """Test forward() returns correct shapes."""
        B, T = 2, 128
        input_ids = torch.randint(0, small_model_config.vocab_size, (B, T))
        labels = torch.randint(0, small_model_config.vocab_size, (B, T))

        # Forward pass
        output = model.forward(input_ids=input_ids, labels=labels)

        # Verify output type
        assert isinstance(output, ModelOutput), "Output must be ModelOutput"

        # Verify shapes (contract: logits = (B, T, vocab_size))
        assert output.logits.shape == (B, T, small_model_config.vocab_size), \
            f"Logits shape mismatch: {output.logits.shape} != {(B, T, small_model_config.vocab_size)}"

        # Verify loss is scalar
        assert output.loss is not None, "Loss must be computed when labels provided"
        assert output.loss.dim() == 0, "Loss must be scalar"
        assert output.loss.item() >= 0, "Loss must be non-negative"

    def test_forward_without_labels(self, model, small_model_config):
        """Test forward() without labels (no loss computation)."""
        B, T = 2, 128
        input_ids = torch.randint(0, small_model_config.vocab_size, (B, T))

        output = model.forward(input_ids=input_ids, labels=None)

        # Loss should be None when labels not provided
        assert output.loss is None, "Loss should be None without labels"
        assert output.logits.shape == (B, T, small_model_config.vocab_size)

    def test_forward_batch_sizes(self, model, small_model_config):
        """Test forward() with different batch sizes."""
        test_cases = [
            (1, 64),    # Single example
            (4, 128),   # Small batch
            (8, 32),    # Larger batch, shorter sequence
        ]

        for B, T in test_cases:
            input_ids = torch.randint(0, small_model_config.vocab_size, (B, T))
            output = model.forward(input_ids=input_ids)

            assert output.logits.shape == (B, T, small_model_config.vocab_size), \
                f"Shape mismatch for B={B}, T={T}"

    def test_forward_max_seq_len_constraint(self, model, small_model_config):
        """Test forward() respects max_seq_len_train."""
        B = 2
        T_valid = small_model_config.max_seq_len_train
        T_invalid = small_model_config.max_seq_len_train + 1

        # Valid sequence length should work
        input_ids = torch.randint(0, small_model_config.vocab_size, (B, T_valid))
        output = model.forward(input_ids=input_ids)
        assert output.logits.shape == (B, T_valid, small_model_config.vocab_size)

        # Invalid sequence length should raise assertion
        input_ids_long = torch.randint(0, small_model_config.vocab_size, (B, T_invalid))
        with pytest.raises(AssertionError, match="exceeds max_seq_len_train"):
            model.forward(input_ids=input_ids_long)

    def test_forward_dtype_consistency(self, model, small_model_config):
        """Test forward() maintains dtype consistency."""
        B, T = 2, 64
        input_ids = torch.randint(0, small_model_config.vocab_size, (B, T))

        output = model.forward(input_ids=input_ids)

        # Logits should be float (fp32 or bf16)
        assert output.logits.dtype in [torch.float32, torch.bfloat16], \
            f"Unexpected logits dtype: {output.logits.dtype}"


class TestForwardRecurrentMode:
    """Test forward_recurrent() in inference mode."""

    def test_recurrent_shape_invariants(self, model, small_model_config):
        """Test forward_recurrent() returns correct shapes."""
        B, chunk_size = 1, 1
        input_ids = torch.randint(0, small_model_config.vocab_size, (B, chunk_size))

        # Initialize state
        state = None

        # Forward pass
        output = model.forward_recurrent(input_ids=input_ids, state=state)

        # Verify output type
        assert isinstance(output, InferenceOutput), "Output must be InferenceOutput"

        # Verify shapes
        assert output.logits.shape == (B, chunk_size, small_model_config.vocab_size), \
            f"Logits shape mismatch: {output.logits.shape}"

        # Verify state returned
        assert output.state is not None, "State must be returned"

    def test_recurrent_incremental_processing(self, model, small_model_config):
        """Test recurrent mode processes incrementally with state carry."""
        B = 1
        sequence_length = 10

        # Generate sequence
        sequence = torch.randint(0, small_model_config.vocab_size, (B, sequence_length))

        # Process token by token
        state = None
        all_logits = []

        for t in range(sequence_length):
            token = sequence[:, t:t+1]  # (B, 1)
            output = model.forward_recurrent(input_ids=token, state=state)

            # Update state
            state = output.state

            # Collect logits
            all_logits.append(output.logits)

            # Verify shape
            assert output.logits.shape == (B, 1, small_model_config.vocab_size)

        # Verify we processed all tokens
        assert len(all_logits) == sequence_length

    def test_recurrent_state_persistence(self, model, small_model_config):
        """Test recurrent state is properly maintained across calls."""
        B, chunk_size = 1, 1
        input_ids = torch.randint(0, small_model_config.vocab_size, (B, chunk_size))

        # First call - initialize state
        output1 = model.forward_recurrent(input_ids=input_ids, state=None)
        state1 = output1.state

        # Second call - use previous state
        output2 = model.forward_recurrent(input_ids=input_ids, state=state1)
        state2 = output2.state

        # States should be different (modified)
        assert state1 is not state2, "State should be updated"

        # State should be a list (RetNet states per layer)
        assert isinstance(state2, list), "State should be list of layer states"
        assert len(state2) == small_model_config.n_layers_retnet, \
            f"State should have {small_model_config.n_layers_retnet} layers"

    def test_recurrent_longer_sequences(self, model, small_model_config):
        """Test recurrent mode handles sequences beyond training length."""
        B = 1
        # Process sequence longer than max_seq_len_train
        long_sequence_len = small_model_config.max_seq_len_train + 100

        state = None
        for t in range(long_sequence_len):
            token = torch.randint(0, small_model_config.vocab_size, (B, 1))
            output = model.forward_recurrent(input_ids=token, state=state)
            state = output.state

            # Should work without errors (recurrent mode doesn't have sequence length limit in same way)
            assert output.logits.shape == (B, 1, small_model_config.vocab_size)


class TestMemoryConstraints:
    """Test FR-003: Memory usage ≤32GB."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_check_mechanism(self, model, small_model_config):
        """Test that memory checking mechanism works."""
        # This test verifies the memory checking code exists and runs
        # Actual 32GB constraint testing requires real GPU

        B, T = 2, 128
        input_ids = torch.randint(0, small_model_config.vocab_size, (B, T))

        # Move to GPU
        model_gpu = model.cuda()
        input_ids_gpu = input_ids.cuda()

        # Should not raise (small model)
        output = model_gpu.forward(input_ids=input_ids_gpu)

        # Verify memory was checked (internal method exists)
        assert hasattr(model_gpu, '_check_memory_constraint')

    def test_memory_check_cpu_mode(self, model, small_model_config):
        """Test memory check works gracefully on CPU."""
        B, T = 2, 128
        input_ids = torch.randint(0, small_model_config.vocab_size, (B, T))

        # Should work without errors on CPU
        output = model.forward(input_ids=input_ids)
        assert output.logits.shape == (B, T, small_model_config.vocab_size)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_batch_raises_error(self, model, small_model_config):
        """Test that empty batch raises appropriate error."""
        # Empty batch should cause issues
        with pytest.raises((RuntimeError, AssertionError, ValueError)):
            input_ids = torch.randint(0, small_model_config.vocab_size, (0, 10))
            model.forward(input_ids=input_ids)

    def test_invalid_token_ids(self, model, small_model_config):
        """Test handling of out-of-vocabulary token IDs."""
        B, T = 2, 64
        # Token IDs outside vocabulary range
        invalid_ids = torch.randint(
            small_model_config.vocab_size,
            small_model_config.vocab_size + 100,
            (B, T)
        )

        # Should raise error (out of bounds for embedding)
        with pytest.raises((RuntimeError, IndexError)):
            model.forward(input_ids=invalid_ids)

    def test_single_token_sequence(self, model, small_model_config):
        """Test forward with single token (edge case)."""
        B, T = 1, 1
        input_ids = torch.randint(0, small_model_config.vocab_size, (B, T))

        output = model.forward(input_ids=input_ids)
        assert output.logits.shape == (B, T, small_model_config.vocab_size)


class TestModelOutputContract:
    """Test ModelOutput and InferenceOutput dataclasses."""

    def test_model_output_fields(self, model, small_model_config):
        """Test ModelOutput has all required fields."""
        B, T = 2, 64
        input_ids = torch.randint(0, small_model_config.vocab_size, (B, T))
        labels = torch.randint(0, small_model_config.vocab_size, (B, T))

        output = model.forward(input_ids=input_ids, labels=labels)

        # Required fields from model-forward.md
        assert hasattr(output, 'loss')
        assert hasattr(output, 'logits')
        assert hasattr(output, 'ponder_cost')
        assert hasattr(output, 'avg_steps')
        assert hasattr(output, 'router_stats')

        # MVP: ponder_cost, avg_steps, router_stats are None (US3, US4 not implemented)
        assert output.ponder_cost is None, "Ponder cost should be None in MVP (US3 not implemented)"
        assert output.avg_steps is None, "Avg steps should be None in MVP (US3 not implemented)"
        assert output.router_stats is None, "Router stats should be None in MVP (US4 not implemented)"

    def test_inference_output_fields(self, model, small_model_config):
        """Test InferenceOutput has all required fields."""
        B = 1
        input_ids = torch.randint(0, small_model_config.vocab_size, (B, 1))

        output = model.forward_recurrent(input_ids=input_ids, state=None)

        # Required fields from model-forward.md
        assert hasattr(output, 'logits')
        assert hasattr(output, 'state')
        assert hasattr(output, 'halted')
        assert hasattr(output, 'num_steps')

        # MVP: halted=False, num_steps=1 (US3 not implemented)
        assert output.halted == False, "Halted should be False in MVP (US3 not implemented)"
        assert output.num_steps == 1, "Num steps should be 1 in MVP (US3 not implemented)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
