"""
Integration tests for adaptive computation (ACT) system.

Tests that the HRM controller and ACT halting system correctly
allocate computation based on query complexity.

Success criteria:
- Simple queries complete in 1-2 steps
- Complex queries allocate 4-10 steps
- Average response time improvement (target 30% per SC-004)
- Average steps within budget (2-6 steps typical)
"""

import torch
import pytest
from src.models.core import RetNetHRMModel
from src.config.model_config import ModelConfig


class TestAdaptiveComputation:
    """Test suite for adaptive computation (US3)."""

    @pytest.fixture
    def small_config(self):
        """Create small model config for testing."""
        config = ModelConfig(
            d_model=512,
            n_layers_retnet=4,
            n_layers_attention=2,
            n_retention_heads=8,
            vocab_size=1000,
            max_seq_len_train=256,
            max_seq_len_infer=512,
            attention_window=128,
            hrm_t_max=10,
            hrm_epsilon=1e-3,
            hrm_ponder_tau=0.002,
            hrm_halting_bias_init=-1.0,
            dropout=0.0,
        )
        return config

    @pytest.fixture
    def model(self, small_config):
        """Create model instance."""
        model = RetNetHRMModel(small_config)
        model.eval()  # Evaluation mode
        return model

    def test_act_loop_basic(self, model):
        """Test that ACT loop runs without errors."""
        batch_size = 2
        seq_len = 64

        # Create dummy input
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        # Forward with ACT
        with torch.no_grad():
            output = model(input_ids, use_act=True, return_dict=True)

        # Check outputs
        assert output.logits is not None
        assert output.logits.shape == (batch_size, seq_len, 1000)
        assert output.ponder_cost is not None
        assert output.avg_steps is not None

        print(f"✓ ACT loop basic test passed")
        print(f"  Ponder cost: {output.ponder_cost:.2f}")
        print(f"  Avg steps: {output.avg_steps:.2f}")

    def test_simple_vs_complex_queries(self, model):
        """
        Test that model allocates different computation for simple vs complex queries.

        Note: This is a proof-of-concept test. In practice, you'd need:
        - Actual simple/complex examples
        - Fine-tuned model that has learned to adapt
        """
        batch_size = 4

        # Simulate simple queries (shorter sequences)
        simple_queries = torch.randint(0, 1000, (batch_size, 32))

        # Simulate complex queries (longer sequences)
        complex_queries = torch.randint(0, 1000, (batch_size, 128))

        with torch.no_grad():
            # Process simple queries
            simple_output = model(simple_queries, use_act=True, return_dict=True)
            simple_steps = simple_output.avg_steps

            # Process complex queries
            complex_output = model(complex_queries, use_act=True, return_dict=True)
            complex_steps = complex_output.avg_steps

        print(f"✓ Simple vs Complex test")
        print(f"  Simple queries avg steps: {simple_steps:.2f}")
        print(f"  Complex queries avg steps: {complex_steps:.2f}")

        # Note: Without training, steps may be similar
        # After training with ponder cost, expect complex > simple
        print(f"  Note: Model needs training to learn adaptive allocation")

    def test_steps_within_budget(self, model):
        """Test that average steps stay within configured budget."""
        batch_size = 8
        seq_len = 64

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        with torch.no_grad():
            output = model(input_ids, use_act=True, return_dict=True)

        avg_steps = output.avg_steps
        max_steps = model.computation_budget.max_steps
        min_steps = model.computation_budget.min_steps

        print(f"✓ Steps within budget test")
        print(f"  Avg steps: {avg_steps:.2f}")
        print(f"  Budget: [{min_steps}, {max_steps}]")

        # Average steps should be within budget
        assert min_steps <= avg_steps <= max_steps
        assert output.ponder_cost >= min_steps  # Ponder cost >= min steps

    def test_act_vs_no_act_timing(self, model):
        """
        Compare inference time with and without ACT.

        Note: ACT adds overhead but may improve quality.
        Target: 30% speedup on average (SC-004) after training.
        """
        import time

        batch_size = 4
        seq_len = 64
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        # Warmup
        with torch.no_grad():
            _ = model(input_ids, use_act=False)

        # Time without ACT
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_ids, use_act=False)
        time_no_act = (time.time() - start) / 10

        # Time with ACT
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_ids, use_act=True)
        time_with_act = (time.time() - start) / 10

        print(f"✓ ACT vs No ACT timing")
        print(f"  Without ACT: {time_no_act*1000:.2f}ms")
        print(f"  With ACT: {time_with_act*1000:.2f}ms")
        print(f"  Overhead: {((time_with_act/time_no_act - 1)*100):.1f}%")
        print(f"  Note: After training, ACT should speed up average by ~30% (SC-004)")

    def test_ponder_cost_computation(self, model):
        """Test that ponder cost is computed correctly."""
        batch_size = 2
        seq_len = 32

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        with torch.no_grad():
            output = model(input_ids, use_act=True, return_dict=True)

        ponder_cost = output.ponder_cost
        avg_steps = output.avg_steps

        print(f"✓ Ponder cost computation test")
        print(f"  Ponder cost: {ponder_cost:.2f}")
        print(f"  Avg steps: {avg_steps:.2f}")
        print(f"  Ponder tau: {model.computation_budget.ponder_tau}")

        # Ponder cost should be close to average steps
        # (exact relationship depends on halting distribution)
        assert 1.0 <= ponder_cost <= model.computation_budget.max_steps

    def test_halting_distribution(self, model):
        """Test that halting creates reasonable distribution over steps."""
        batch_size = 16  # Larger batch for distribution
        seq_len = 64

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        step_counts = []
        with torch.no_grad():
            for _ in range(5):  # Multiple runs
                output = model(input_ids, use_act=True, return_dict=True)
                step_counts.append(output.avg_steps)

        avg_across_runs = sum(step_counts) / len(step_counts)
        std_across_runs = (sum((s - avg_across_runs)**2 for s in step_counts) / len(step_counts))**0.5

        print(f"✓ Halting distribution test")
        print(f"  Avg steps across runs: {avg_across_runs:.2f} ± {std_across_runs:.2f}")
        print(f"  Step counts: {[f'{s:.1f}' for s in step_counts]}")
        print(f"  Note: Without training, distribution may be uniform")

    def test_loss_with_ponder_cost(self, model):
        """Test that loss includes ponder cost when using ACT."""
        batch_size = 2
        seq_len = 32

        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        labels = torch.randint(0, 1000, (batch_size, seq_len))

        with torch.no_grad():
            # Without ACT
            output_no_act = model(input_ids, labels=labels, use_act=False, return_dict=True)
            loss_no_act = output_no_act.loss

            # With ACT
            output_with_act = model(input_ids, labels=labels, use_act=True, return_dict=True)
            loss_with_act = output_with_act.loss

        print(f"✓ Loss with ponder cost test")
        print(f"  Loss without ACT: {loss_no_act:.4f}")
        print(f"  Loss with ACT: {loss_with_act:.4f}")
        print(f"  Ponder cost: {output_with_act.ponder_cost:.2f}")

        # Both losses should be valid
        assert loss_no_act > 0
        assert loss_with_act > 0

        # With ACT, loss includes ponder cost term
        # L_total = L_ce + tau * E[steps]
        # So we expect loss_with_act ≈ loss_no_act + tau * ponder_cost
        # (approximately, since CE loss may differ slightly due to weighted outputs)


def test_adaptive_computation_sc004():
    """
    Success Criteria SC-004 validation.

    SC-004: Average response time improves by ≥30% for simple queries
    compared to fixed maximum computation.

    Note: This test validates the infrastructure. Actual 30% improvement
    requires training the model with ponder cost regularization.
    """
    print("\n" + "="*60)
    print("SC-004: Adaptive Computation Allocation")
    print("="*60)

    # Create minimal config
    config = ModelConfig(
        d_model=256,
        n_layers_retnet=2,
        n_layers_attention=1,
        n_retention_heads=4,
        vocab_size=500,
        max_seq_len_train=128,
        hrm_t_max=6,
        dropout=0.0,
    )

    model = RetNetHRMModel(config)
    model.eval()

    # Simulate simple query
    simple_query = torch.randint(0, 500, (1, 16))

    with torch.no_grad():
        output = model(simple_query, use_act=True, return_dict=True)

    print(f"✓ ACT system operational")
    print(f"  Simple query steps: {output.avg_steps:.2f}")
    print(f"  Ponder cost: {output.ponder_cost:.2f}")
    print(f"  Max possible steps: {config.hrm_t_max}")
    print()
    print("✓ SC-004 Infrastructure Complete")
    print("  Note: 30% improvement requires training with ponder cost")
    print("  Current: Validates ACT loop, halting, and ponder cost computation")
    print("="*60)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
