"""Unit tests for Adaptive Computation Time (ACT) wrapper.

Tests:
- ACT forward pass mechanics
- Halting probability computation
- Weighted output aggregation
- Pondering loss calculation
- Integration with RetNet backbone
- Edge cases (early halt, max steps reached)
"""

import pytest
import torch
import torch.nn as nn

from src.models.retnet.backbone import RetNetBackbone
from src.models.retnet.act_wrapper import (
    ACTRetNetBackbone,
    ACTLoss,
    create_act_retnet,
)


@pytest.fixture
def small_retnet():
    """Create a small RetNet for testing."""
    return RetNetBackbone(
        vocab_size=1000,
        d_model=128,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
        max_seq_len=512,
    )


@pytest.fixture
def act_retnet(small_retnet):
    """Create ACT-wrapped RetNet."""
    return ACTRetNetBackbone(
        backbone=small_retnet,
        max_steps=5,
        epsilon=0.01,
        ponder_penalty=0.01,
    )


class TestACTForwardPass:
    """Test ACT forward pass mechanics."""

    def test_forward_basic(self, act_retnet):
        """Test basic forward pass."""
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        # Forward pass
        output, act_info = act_retnet(input_ids, return_act_info=True)

        # Check output shape
        assert output.shape == (batch_size, seq_len, 128)

        # Check ACT info
        assert act_info is not None
        assert 'pondering_cost' in act_info
        assert 'halting_probs' in act_info
        assert 'remainders' in act_info
        assert 'n_steps' in act_info

        # Check shapes
        assert act_info['pondering_cost'].shape == (batch_size, seq_len)
        assert act_info['halting_probs'].shape == (batch_size, seq_len, 5)
        assert act_info['remainders'].shape == (batch_size, seq_len)
        assert act_info['n_steps'].shape == (batch_size, seq_len)

    def test_forward_without_act_info(self, act_retnet):
        """Test forward pass without returning ACT info."""
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        # Forward pass
        output, act_info = act_retnet(input_ids, return_act_info=False)

        # Check output shape
        assert output.shape == (batch_size, seq_len, 128)

        # ACT info should be None
        assert act_info is None

    def test_halting_probabilities_sum_to_one(self, act_retnet):
        """Test that halting probabilities sum to 1.0 for each position."""
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        # Forward pass
        _, act_info = act_retnet(input_ids, return_act_info=True)

        # Sum halting probs across steps
        halting_sum = act_info['halting_probs'].sum(dim=-1)

        # Should sum to 1.0 (within numerical tolerance)
        assert torch.allclose(halting_sum, torch.ones_like(halting_sum), atol=1e-5)

    def test_pondering_cost_in_valid_range(self, act_retnet):
        """Test that pondering cost is between 1 and max_steps."""
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        # Forward pass
        _, act_info = act_retnet(input_ids, return_act_info=True)

        pondering_cost = act_info['pondering_cost']

        # Should be between 1 and max_steps
        assert (pondering_cost >= 1.0).all()
        assert (pondering_cost <= act_retnet.max_steps).all()

    def test_early_halting(self):
        """Test that model can halt early if confident."""
        # Create ACT model with high epsilon (easier to halt)
        backbone = RetNetBackbone(
            vocab_size=1000,
            d_model=64,
            n_layers=1,
            n_heads=2,
            dropout=0.0,
        )
        act_model = ACTRetNetBackbone(
            backbone=backbone,
            max_steps=10,
            epsilon=0.5,  # High epsilon = easier to halt
            ponder_penalty=0.01,
        )

        input_ids = torch.randint(0, 1000, (2, 8))

        # Forward pass
        _, act_info = act_model(input_ids, return_act_info=True)

        # With high epsilon, many positions should halt in 1-2 steps
        mean_steps = act_info['pondering_cost'].mean()
        assert mean_steps < 5.0  # Should be much less than max_steps

    def test_max_steps_reached(self):
        """Test behavior when max_steps is reached."""
        # Create ACT model with very low epsilon (hard to halt)
        backbone = RetNetBackbone(
            vocab_size=1000,
            d_model=64,
            n_layers=1,
            n_heads=2,
            dropout=0.0,
        )
        act_model = ACTRetNetBackbone(
            backbone=backbone,
            max_steps=3,
            epsilon=0.0001,  # Very low epsilon = hard to halt
            ponder_penalty=0.01,
        )

        input_ids = torch.randint(0, 1000, (2, 8))

        # Forward pass
        _, act_info = act_model(input_ids, return_act_info=True)

        # Most positions should reach max_steps
        n_steps = act_info['n_steps']
        assert (n_steps <= 3).all()  # Shouldn't exceed max_steps

    def test_gradient_flow(self, act_retnet):
        """Test that gradients flow through ACT mechanism."""
        input_ids = torch.randint(0, 1000, (2, 8))

        # Forward pass
        output, act_info = act_retnet(input_ids, return_act_info=True)

        # Compute dummy loss
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Check that halting unit has gradients
        for param in act_retnet.halting_unit.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()


class TestACTLoss:
    """Test ACT loss computation."""

    def test_pondering_loss_simple(self, act_retnet):
        """Test simple pondering loss computation."""
        input_ids = torch.randint(0, 1000, (2, 8))

        # Forward pass
        _, act_info = act_retnet(input_ids, return_act_info=True)

        # Compute pondering loss
        ponder_loss = act_retnet.compute_act_loss(act_info, reduction='mean')

        # Should be a scalar
        assert ponder_loss.dim() == 0

        # Should be positive
        assert ponder_loss.item() > 0

        # Should scale with ponder_penalty
        expected_magnitude = act_retnet.ponder_penalty * act_info['pondering_cost'].mean()
        assert torch.isclose(ponder_loss, expected_magnitude, atol=1e-5)

    def test_pondering_loss_geometric_prior(self):
        """Test geometric prior pondering loss."""
        backbone = RetNetBackbone(
            vocab_size=1000,
            d_model=64,
            n_layers=1,
            n_heads=2,
            dropout=0.0,
        )
        act_model = ACTRetNetBackbone(
            backbone=backbone,
            max_steps=5,
            epsilon=0.01,
            ponder_penalty=0.01,
            use_geometric_prior=True,
            prior_lambda=0.5,
        )

        input_ids = torch.randint(0, 1000, (2, 8))

        # Forward pass
        _, act_info = act_model(input_ids, return_act_info=True)

        # Compute KL loss
        kl_loss = act_model.compute_act_loss(act_info, reduction='mean')

        # Should be a scalar
        assert kl_loss.dim() == 0

        # Should be positive (KL divergence is non-negative)
        assert kl_loss.item() >= 0

    def test_act_loss_wrapper(self, act_retnet):
        """Test ACTLoss wrapper."""
        criterion = ACTLoss(act_model=act_retnet, task_loss_weight=1.0)

        input_ids = torch.randint(0, 1000, (2, 8))

        # Forward pass
        _, act_info = act_retnet(input_ids, return_act_info=True)

        # Dummy task loss
        task_loss = torch.tensor(2.5)

        # Compute combined loss
        total_loss, loss_dict = criterion(task_loss, act_info)

        # Check total loss
        assert total_loss.item() > task_loss.item()  # Should include pondering

        # Check loss dict
        assert 'total_loss' in loss_dict
        assert 'task_loss' in loss_dict
        assert 'ponder_loss' in loss_dict
        assert 'mean_ponder_steps' in loss_dict

        # Total should equal task + ponder
        expected_total = task_loss.item() + loss_dict['ponder_loss']
        assert abs(loss_dict['total_loss'] - expected_total) < 1e-5


class TestACTRecurrentMode:
    """Test ACT in recurrent inference mode.

    NOTE: ACT in recurrent mode has limitations due to the interaction
    between pondering loops and recurrent state management. These tests
    document the current behavior and expected limitations.
    """

    @pytest.mark.xfail(reason="ACT recurrent mode needs state handling improvements")
    def test_recurrent_forward(self, act_retnet):
        """Test recurrent forward pass."""
        batch_size, chunk_size = 2, 1
        input_ids = torch.randint(0, 1000, (batch_size, chunk_size))

        # Forward pass
        output, state, diagnostics = act_retnet.forward_recurrent(input_ids)

        # Check output shape
        assert output.shape == (batch_size, chunk_size, 128)

        # Check state tuple
        assert isinstance(state, tuple)
        assert len(state) == 2

        # Check diagnostics
        assert 'halting_probs' in diagnostics
        assert 'mean_steps' in diagnostics

    @pytest.mark.xfail(reason="ACT recurrent mode needs state handling improvements")
    def test_recurrent_stateful(self, act_retnet):
        """Test stateful recurrent generation."""
        batch_size = 1
        state = None

        # Generate 5 tokens sequentially
        for _ in range(5):
            input_ids = torch.randint(0, 1000, (batch_size, 1))
            output, state, diagnostics = act_retnet.forward_recurrent(
                input_ids, state=state
            )

            # Check output
            assert output.shape == (batch_size, 1, 128)

            # State should be updated
            assert state is not None


class TestFactoryFunction:
    """Test create_act_retnet factory function."""

    def test_create_act_retnet_basic(self):
        """Test basic model creation."""
        model = create_act_retnet(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=2,
            act_max_steps=5,
        )

        assert isinstance(model, ACTRetNetBackbone)
        assert model.max_steps == 5

    def test_create_act_retnet_with_geometric_prior(self):
        """Test model creation with geometric prior."""
        model = create_act_retnet(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=2,
            act_max_steps=5,
            act_use_geometric_prior=True,
            act_prior_lambda=0.3,
        )

        assert model.use_geometric_prior
        assert model.prior_lambda == 0.3


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_input(self, act_retnet):
        """Test behavior with minimal input."""
        # Single token
        input_ids = torch.randint(0, 1000, (1, 1))

        output, act_info = act_retnet(input_ids, return_act_info=True)

        assert output.shape == (1, 1, 128)
        assert act_info['pondering_cost'].shape == (1, 1)

    def test_batch_size_one(self, act_retnet):
        """Test with batch size 1."""
        input_ids = torch.randint(0, 1000, (1, 16))

        output, act_info = act_retnet(input_ids, return_act_info=True)

        assert output.shape == (1, 16, 128)

    def test_long_sequence(self, act_retnet):
        """Test with longer sequence."""
        input_ids = torch.randint(0, 1000, (2, 128))

        output, act_info = act_retnet(input_ids, return_act_info=True)

        assert output.shape == (2, 128, 128)
        assert act_info['pondering_cost'].shape == (2, 128)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self, act_retnet):
        """Test ACT on CUDA device."""
        act_retnet = act_retnet.cuda()
        input_ids = torch.randint(0, 1000, (2, 16)).cuda()

        output, act_info = act_retnet(input_ids, return_act_info=True)

        assert output.is_cuda
        assert act_info['pondering_cost'].is_cuda


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
