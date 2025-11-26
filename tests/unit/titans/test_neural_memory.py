"""Unit tests for Titans neural memory (MAG coupling).

Tests neural memory functionality:
- MemoryMLP forward passes
- Full NeuralMemory tests are in the reference implementation tests

NOTE: SimplifiedNeuralMemory has been replaced with full NeuralMemory from titans-pytorch.
This file now only tests MemoryMLP. Full NeuralMemory integration tests are separate.

Per tasks.md T051, T053.
"""

import pytest
import torch
import torch.nn.functional as F
from src.models.titans.memory_models import MemoryMLP
from src.models.titans.neural_memory import NeuralMemState


class TestMemoryMLP:
    """Test suite for MemoryMLP."""

    @pytest.fixture
    def mlp(self):
        """Create MLP fixture."""
        return MemoryMLP(dim=32, depth=2, expansion_factor=4.0)

    def test_initialization(self, mlp):
        """Test MLP initializes correctly."""
        assert hasattr(mlp, 'weights')  # ParameterList of weight matrices

    def test_forward_2d(self, mlp):
        """Test forward with 2D input [batch, d_model]."""
        x = torch.randn(4, 32)
        y = mlp(x)

        assert y.shape == x.shape

    def test_forward_3d(self, mlp):
        """Test forward with 3D input [batch, seq_len, d_model]."""
        x = torch.randn(2, 10, 32)
        y = mlp(x)

        assert y.shape == x.shape

    def test_nonzero_output(self, mlp):
        """Test MLP produces non-zero output."""
        x = torch.randn(4, 32)
        y = mlp(x)

        assert y.abs().sum() > 0.0

    def test_gradient_flow(self, mlp):
        """Test gradients flow through MLP."""
        x = torch.randn(4, 32, requires_grad=True)
        y = mlp(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None


@pytest.mark.skip(reason="SimplifiedNeuralMemory replaced with full NeuralMemory from titans-pytorch")
class TestSimplifiedNeuralMemory:
    """Test suite for SimplifiedNeuralMemory.

    NOTE: These tests are skipped because SimplifiedNeuralMemory has been replaced
    with the full NeuralMemory implementation from titans-pytorch reference.
    """

    @pytest.fixture
    def memory(self):
        """Create memory fixture."""
        return None  # Simplified memory no longer exists

    @pytest.fixture
    def input_tensor(self):
        """Create input tensor fixture."""
        return torch.randn(2, 10, 32)

    @pytest.fixture
    def target_tensor(self):
        """Create target tensor fixture."""
        return torch.randn(2, 10, 32)

    def test_initialization(self, memory):
        """Test memory initializes correctly."""
        assert memory.d_model == 32
        assert memory.surprise_threshold == 0.5
        assert hasattr(memory, 'memory_mlp')
        assert hasattr(memory, 'to_query')
        assert hasattr(memory, 'to_gate')

    def test_forward_without_update(self, memory, input_tensor):
        """Test forward without memory update."""
        retrieved, state = memory(
            input_tensor,
            state=None,
            update_memory=False,
            target=None
        )

        assert retrieved.shape == input_tensor.shape
        assert isinstance(state, NeuralMemState)

    def test_forward_with_update(self, memory, input_tensor, target_tensor):
        """Test forward with memory update."""
        retrieved, state = memory(
            input_tensor,
            state=None,
            update_memory=True,
            target=target_tensor
        )

        assert retrieved.shape == input_tensor.shape
        assert isinstance(state, NeuralMemState)

    def test_state_initialization(self, memory, input_tensor):
        """Test state initializes correctly when None."""
        _, state = memory(input_tensor, state=None, update_memory=False)

        assert state.weights is not None
        assert len(state.weights) > 0
        assert state.total_updates == 0
        assert state.writes_used == 0

    def test_surprise_computation(self, memory):
        """Test surprise signal computation."""
        pred = torch.randn(2, 10, 32)
        target = torch.randn(2, 10, 32)

        surprise = memory._compute_surprise(pred, target)

        assert isinstance(surprise, float)
        assert surprise >= 0.0  # MSE is non-negative

    def test_surprise_zero_for_identical(self, memory):
        """Test surprise is near zero for identical pred and target."""
        x = torch.randn(2, 10, 32)

        surprise = memory._compute_surprise(x, x)

        assert surprise < 1e-6

    def test_memory_update_increments_counter(self, memory, input_tensor, target_tensor):
        """Test memory update increments counters."""
        # High surprise to trigger update
        memory.surprise_threshold = 0.0  # Always update

        _, state1 = memory(input_tensor, update_memory=False)
        assert state1.writes_used == 0

        _, state2 = memory(
            input_tensor,
            state=state1,
            update_memory=True,
            target=target_tensor
        )

        assert state2.writes_used == 1
        assert state2.total_updates == 1

    def test_memory_update_respects_budget(self, memory, input_tensor, target_tensor):
        """Test memory update respects write budget."""
        memory.surprise_threshold = 0.0  # Always want to update
        memory.max_writes_per_episode = 2

        _, state = memory(input_tensor, update_memory=False)

        # Update 3 times (should only actually update 2)
        for _ in range(3):
            _, state = memory(
                input_tensor,
                state=state,
                update_memory=True,
                target=target_tensor
            )

        # Should stop at budget
        assert state.writes_used <= 2

    def test_writes_paused_prevents_update(self, memory, input_tensor, target_tensor):
        """Test writes_paused flag prevents updates."""
        memory.surprise_threshold = 0.0  # Always want to update

        _, state = memory(input_tensor, update_memory=False)

        # Manually pause writes
        from dataclasses import replace
        state = replace(state, writes_paused=True)

        _, state2 = memory(
            input_tensor,
            state=state,
            update_memory=True,
            target=target_tensor
        )

        # Should not increment (paused)
        assert state2.writes_used == 0

    def test_ema_checkpoint_creation(self, memory, input_tensor, target_tensor):
        """Test EMA checkpoint is created after update."""
        memory.surprise_threshold = 0.0  # Always update

        _, state = memory(input_tensor, update_memory=False)
        assert state.ema_checkpoint is None

        _, state2 = memory(
            input_tensor,
            state=state,
            update_memory=True,
            target=target_tensor
        )

        assert state2.ema_checkpoint is not None
        assert len(state2.ema_checkpoint) > 0

    def test_surprise_history_tracking(self, memory, input_tensor, target_tensor):
        """Test surprise values are tracked in history."""
        memory.surprise_threshold = 0.0  # Always update

        _, state = memory(input_tensor, update_memory=False)
        assert len(state.recent_surprises) == 0

        _, state2 = memory(
            input_tensor,
            state=state,
            update_memory=True,
            target=target_tensor
        )

        assert len(state2.recent_surprises) > 0

    def test_avg_surprise_computation(self, memory, input_tensor, target_tensor):
        """Test average surprise is computed."""
        memory.surprise_threshold = 0.0

        _, state = memory(input_tensor, update_memory=False)

        # Do multiple updates
        for _ in range(5):
            _, state = memory(
                input_tensor,
                state=state,
                update_memory=True,
                target=target_tensor
            )

        assert state.avg_surprise > 0.0

    def test_drift_detection_cusum(self, memory, input_tensor):
        """Test CUSUM drift detection."""
        memory.drift_threshold = 2.0  # Low threshold for testing

        # Create state with high surprises to trigger drift
        from dataclasses import replace
        _, state = memory(input_tensor, update_memory=False)
        state = replace(state, recent_surprises=[5.0, 5.0, 5.0], avg_surprise=1.0)

        # Check drift
        state2 = memory._check_drift(state)

        # CUSUM should increase
        assert state2.cusum_stat >= state.cusum_stat

    def test_drift_triggers_pause(self, memory, input_tensor):
        """Test drift detection triggers write pause."""
        memory.drift_threshold = 1.0  # Very low threshold

        _, state = memory(input_tensor, update_memory=False)

        # Force high CUSUM to trigger drift
        from dataclasses import replace
        state = replace(state, cusum_stat=10.0)

        state2 = memory._check_drift(state)

        # Should detect drift and pause writes
        if state2.cusum_stat > memory.drift_threshold:
            assert state2.drift_detected
            assert state2.writes_paused

    def test_drift_rollback_to_ema(self, memory, input_tensor, target_tensor):
        """Test drift triggers rollback to EMA checkpoint."""
        memory.drift_threshold = 1.0
        memory.surprise_threshold = 0.0

        # Create checkpoint
        _, state = memory(
            input_tensor,
            update_memory=True,
            target=target_tensor
        )

        initial_checkpoint = state.ema_checkpoint.copy()

        # Force drift
        from dataclasses import replace
        state = replace(state, cusum_stat=10.0)

        # Check drift (should rollback)
        state2 = memory._check_drift(state)

        # CUSUM should reset if drift detected
        if state2.drift_detected:
            assert state2.cusum_stat == 0.0

    def test_resume_writes(self, memory, input_tensor):
        """Test resume_writes clears drift flags."""
        _, state = memory(input_tensor, update_memory=False)

        from dataclasses import replace
        state = replace(
            state,
            drift_detected=True,
            writes_paused=True,
            cusum_stat=5.0
        )

        state2 = memory.resume_writes(state)

        assert not state2.drift_detected
        assert not state2.writes_paused
        assert state2.cusum_stat == 0.0

    def test_gating_vector_generation(self, memory, input_tensor):
        """Test MAG gating vector is generated correctly."""
        retrieved, _ = memory(input_tensor, update_memory=False)

        gate = memory.get_gating_vector(retrieved)

        assert gate.shape == retrieved.shape
        assert (gate >= 0.0).all() and (gate <= 1.0).all()  # Sigmoid output

    def test_gating_vector_range(self, memory):
        """Test gating vector is in valid range [0, 1]."""
        # Extreme input
        x_extreme = torch.randn(2, 10, 32) * 100

        retrieved, _ = memory(x_extreme, update_memory=False)
        gate = memory.get_gating_vector(retrieved)

        assert gate.min() >= 0.0
        assert gate.max() <= 1.0

    def test_memory_weights_change_after_update(self, memory, input_tensor, target_tensor):
        """Test memory weights actually change after update."""
        memory.surprise_threshold = 0.0  # Always update

        _, state1 = memory(input_tensor, update_memory=False)
        initial_weights = {k: v.clone() for k, v in state1.weights.items()}

        _, state2 = memory(
            input_tensor,
            state=state1,
            update_memory=True,
            target=target_tensor
        )

        # Weights should have changed
        for key in initial_weights:
            assert not torch.allclose(
                initial_weights[key],
                state2.weights[key],
                atol=1e-5
            )

    def test_gradient_flow(self, memory, input_tensor):
        """Test gradients flow through memory."""
        input_tensor.requires_grad = True

        retrieved, _ = memory(input_tensor, update_memory=False)
        loss = retrieved.sum()
        loss.backward()

        assert input_tensor.grad is not None

    def test_device_placement(self):
        """Test memory works on GPU if available."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        memory = SimplifiedNeuralMemory(d_model=32, device=device)

        x = torch.randn(2, 10, 32, device=device)
        retrieved, state = memory(x, update_memory=False)

        assert retrieved.device == device

    def test_deterministic_without_update(self, memory, input_tensor):
        """Test memory is deterministic when not updating."""
        y1, _ = memory(input_tensor, update_memory=False)
        y2, _ = memory(input_tensor, update_memory=False)

        assert torch.allclose(y1, y2, atol=1e-6)

    def test_surprise_threshold_filtering(self, memory, input_tensor, target_tensor):
        """Test surprise threshold filters low-surprise updates."""
        memory.surprise_threshold = 100.0  # Very high (never update)

        _, state = memory(input_tensor, update_memory=False)

        _, state2 = memory(
            input_tensor,
            state=state,
            update_memory=True,
            target=target_tensor
        )

        # Should not update (surprise too low)
        assert state2.writes_used == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
