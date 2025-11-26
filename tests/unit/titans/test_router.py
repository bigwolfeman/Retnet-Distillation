"""Unit tests for LinUCB router and adaptive threshold policy.

Tests LinUCBRouter contextual bandit functionality:
- Engine selection (UCB scores, ε-greedy)
- Online learning (A, b matrix updates)
- Behavior cloning pre-training
- Selection entropy monitoring
- Adaptive threshold policies

Per tasks.md T043, T044, T045.
"""

import pytest
import torch
import numpy as np
from src.models.titans.router import LinUCBRouter, AdaptiveThresholdPolicy


class TestLinUCBRouter:
    """Test suite for LinUCBRouter."""

    @pytest.fixture
    def router(self):
        """Create router fixture."""
        return LinUCBRouter(
            num_engines=3,
            context_dim=8,
            alpha=1.0,
            epsilon=0.1,
            device=torch.device('cpu')
        )

    @pytest.fixture
    def context(self):
        """Create context feature fixture."""
        return torch.randn(8)

    def test_initialization(self, router):
        """Test router initializes correctly."""
        assert router.num_engines == 3
        assert router.context_dim == 8
        assert router.alpha == 1.0
        assert router.epsilon == 0.1

        # Check A matrices initialized as λI
        for i in range(3):
            expected = router.ridge_lambda * torch.eye(8)
            assert torch.allclose(router.A[i], expected, atol=1e-6)

        # Check b vectors initialized as zeros
        assert torch.allclose(router.b, torch.zeros(3, 8, 1))

    def test_select_engine_ucb(self, router, context):
        """Test UCB-based engine selection."""
        # With epsilon=0 (pure exploitation)
        selected, scores = router.select_engine(context, epsilon=0.0, k=1)

        assert len(selected) == 1
        assert 0 <= selected[0] < 3
        assert len(scores) == 3
        assert all(isinstance(v, float) for v in scores.values())

    def test_select_multiple_engines(self, router, context):
        """Test selecting k=2 engines."""
        selected, scores = router.select_engine(context, epsilon=0.0, k=2)

        assert len(selected) == 2
        assert len(set(selected)) == 2  # No duplicates
        assert all(0 <= e < 3 for e in selected)

    def test_select_k_exceeds_limit(self, router, context):
        """Test that k>2 raises error per plan.md."""
        with pytest.raises(ValueError, match="Maximum K=2"):
            router.select_engine(context, k=3)

    def test_epsilon_greedy_exploration(self, router, context):
        """Test ε-greedy exploration."""
        # Set seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # With epsilon=1.0 (pure exploration), should be random
        selections = []
        for _ in range(100):
            selected, _ = router.select_engine(context, epsilon=1.0, k=1)
            selections.append(selected[0])

        # Should have variety (not always same engine)
        unique_selections = set(selections)
        assert len(unique_selections) > 1, "Pure exploration should select multiple engines"

    def test_exclude_engines(self, router, context):
        """Test excluding specific engines from selection."""
        selected, scores = router.select_engine(context, epsilon=0.0, k=1, exclude_engines=[0, 1])

        # Should only select engine 2
        assert selected[0] == 2

    def test_update_matrices(self, router, context):
        """Test A and b matrix updates."""
        initial_A = router.A[0].clone()
        initial_b = router.b[0].clone()

        # Update engine 0
        router.update(
            context=context,
            engine_id=0,
            reward=1.0,
            verified_correct=True,
            latency=0.5,
            cost=100.0
        )

        # A should increase (A ← A + xxᵀ)
        assert not torch.allclose(router.A[0], initial_A)
        assert (router.A[0].diag() >= initial_A.diag()).all(), "Diagonal should increase"

        # b should update (b ← b + r·x)
        assert not torch.allclose(router.b[0], initial_b)

    def test_update_tracking(self, router, context):
        """Test selection count and reward tracking."""
        assert router.selection_counts[0].item() == 0
        assert router.total_rewards[0].item() == 0.0

        # Update engine 0
        router.update(
            context=context,
            engine_id=0,
            reward=0.8,
            verified_correct=True,
            latency=0.5,
            cost=100.0
        )

        assert router.selection_counts[0].item() == 1
        assert abs(router.total_rewards[0].item() - 0.8) < 1e-5

    def test_reward_computation(self, router, context):
        """Test reward function: verified_correct - λ*latency - μ*cost."""
        # Update with None reward (should compute from components)
        router.update(
            context=context,
            engine_id=0,
            reward=None,
            verified_correct=True,  # +1
            latency=10.0,           # -0.01 * 10 = -0.1
            cost=200.0              # -0.005 * 200 = -1.0
        )

        # Expected reward: 1.0 - 0.1 - 1.0 = -0.1
        expected_reward = 1.0 - router.reward_latency_penalty * 10.0 - router.reward_cost_penalty * 200.0
        assert abs(router.total_rewards[0].item() - expected_reward) < 1e-5

    def test_ucb_exploration_bonus(self, router, context):
        """Test UCB includes exploration bonus."""
        # Fresh router has high uncertainty → high exploration bonus
        scores1 = router._compute_ucb_scores(context.unsqueeze(1))

        # After many updates, uncertainty decreases
        for _ in range(50):
            router.update(context, 0, 1.0, True, 0.1, 10.0)

        scores2 = router._compute_ucb_scores(context.unsqueeze(1))

        # Exploration bonus should decrease (scores should change)
        # This is probabilistic, but generally true
        assert not torch.allclose(scores1, scores2, atol=0.1)

    def test_pretrain_from_oracle(self, router, context):
        """Test behavior cloning pre-training."""
        initial_b = router.b[1].clone()

        # Pre-train with oracle data
        training_data = [
            (context, 1, 1.0),  # Engine 1 is oracle choice
            (context, 1, 1.0),
            (context, 1, 1.0),
        ]
        router.pretrain_from_oracle(training_data)

        # b[1] should have changed
        assert not torch.allclose(router.b[1], initial_b)

        # Engine 1 should have higher selection count
        assert router.selection_counts[1].item() == 3

    def test_get_engine_stats(self, router, context):
        """Test per-engine statistics retrieval."""
        # Update engine 0 twice
        router.update(context, 0, 1.0, True, 0.5, 100.0)
        router.update(context, 0, 0.8, True, 0.6, 120.0)

        stats = router.get_engine_stats()

        assert stats[0]['selections'] == 2
        assert abs(stats[0]['total_reward'] - 1.8) < 1e-5
        assert abs(stats[0]['avg_reward'] - 0.9) < 1e-5

    def test_selection_entropy(self, router, context):
        """Test entropy computation for collapse detection."""
        # Initial entropy should be 0 (no selections yet)
        entropy = router.get_selection_entropy()
        assert entropy == 0.0

        # Select all engines equally
        for i in range(3):
            for _ in range(10):
                router.selection_counts[i] += 1

        entropy = router.get_selection_entropy()

        # Uniform distribution entropy = log(3) ≈ 1.099 nats
        expected_entropy = np.log(3)
        assert abs(entropy - expected_entropy) < 0.01

    def test_entropy_collapse_detection(self, router, context):
        """Test entropy detects router collapse (per spec.md SC-005)."""
        # Collapse scenario: always select engine 0
        for _ in range(100):
            router.selection_counts[0] += 1

        entropy = router.get_selection_entropy()

        # Entropy should be ~0 (collapsed)
        assert entropy < 0.1, "Collapsed router should have low entropy"
        assert entropy < 1.0, "Entropy <1.0 nats indicates collapse per spec"

    def test_device_placement(self):
        """Test router works on GPU if available."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        router = LinUCBRouter(num_engines=2, context_dim=8, device=device)

        context = torch.randn(8, device=device)
        selected, scores = router.select_engine(context, k=1)

        assert len(selected) == 1
        assert router.A.device.type == device.type
        assert router.b.device.type == device.type


class TestAdaptiveThresholdPolicy:
    """Test suite for AdaptiveThresholdPolicy."""

    @pytest.fixture
    def policy(self):
        """Create policy fixture."""
        return AdaptiveThresholdPolicy(
            context_dim=8,
            device=torch.device('cpu')
        )

    @pytest.fixture
    def task_features(self):
        """Create task features fixture."""
        return torch.randn(8)

    def test_initialization(self, policy):
        """Test policy initializes with correct base thresholds."""
        assert policy.base_thresholds['math'] == 0.9
        assert policy.base_thresholds['code'] == 0.9
        assert policy.base_thresholds['text'] == 0.8
        assert policy.adjustment_range == 0.05

    def test_get_threshold_math(self, policy, task_features):
        """Test threshold retrieval for math domain."""
        tau = policy.get_threshold('math', task_features)

        # Should be base ± adjustment
        assert 0.85 <= tau <= 0.95  # 0.9 ± 0.05
        assert 0.0 <= tau <= 1.0

    def test_get_threshold_code(self, policy, task_features):
        """Test threshold retrieval for code domain."""
        tau = policy.get_threshold('code', task_features)

        assert 0.85 <= tau <= 0.95  # 0.9 ± 0.05

    def test_get_threshold_text(self, policy, task_features):
        """Test threshold retrieval for text domain."""
        tau = policy.get_threshold('text', task_features)

        assert 0.75 <= tau <= 0.85  # 0.8 ± 0.05

    def test_get_threshold_unknown_domain(self, policy, task_features):
        """Test threshold for unknown domain uses default."""
        tau = policy.get_threshold('unknown', task_features)

        # Should use default 0.85 ± 0.05
        # Allow small tolerance for floating point
        assert 0.79 <= tau <= 0.91, f"Threshold {tau} outside expected range"

    def test_threshold_clamping(self, policy):
        """Test thresholds are clamped to [0, 1]."""
        # Create extreme features
        extreme_features = torch.ones(8) * 100.0
        tau = policy.get_threshold('math', extreme_features)

        assert 0.0 <= tau <= 1.0, "Threshold must be clamped to [0, 1]"

    def test_update_from_outcome_correct_accept(self, policy, task_features):
        """Test update when accepted and correct (reward +1)."""
        initial_rewards = policy.threshold_router.total_rewards.clone()

        policy.update_from_outcome(
            domain='math',
            task_features=task_features,
            accepted=True,
            was_correct=True
        )

        # Should have positive reward
        # Action 1 (accept) should be updated
        assert policy.threshold_router.selection_counts[1].item() == 1

    def test_update_from_outcome_incorrect_reject(self, policy, task_features):
        """Test update when rejected and incorrect (reward +1)."""
        policy.update_from_outcome(
            domain='math',
            task_features=task_features,
            accepted=False,
            was_correct=False
        )

        # Action 0 (reject) should be updated
        assert policy.threshold_router.selection_counts[0].item() == 1

    def test_update_from_outcome_incorrect_accept(self, policy, task_features):
        """Test update when accepted but incorrect (reward -1)."""
        policy.update_from_outcome(
            domain='math',
            task_features=task_features,
            accepted=True,
            was_correct=False
        )

        # Should have negative reward
        assert policy.threshold_router.total_rewards[1].item() == -1.0

    def test_update_from_outcome_correct_reject(self, policy, task_features):
        """Test update when rejected but correct (reward -1)."""
        policy.update_from_outcome(
            domain='math',
            task_features=task_features,
            accepted=False,
            was_correct=True
        )

        # Should have negative reward
        assert policy.threshold_router.total_rewards[0].item() == -1.0

    def test_threshold_adaptation_over_time(self, policy, task_features):
        """Test threshold adapts with experience."""
        # Get initial threshold
        tau1 = policy.get_threshold('math', task_features)

        # Simulate many correct accepts (should learn to be lenient)
        for _ in range(20):
            policy.update_from_outcome('math', task_features, accepted=True, was_correct=True)

        # Get new threshold (may differ due to learning)
        tau2 = policy.get_threshold('math', task_features)

        # Both should be valid
        assert 0.0 <= tau1 <= 1.0
        assert 0.0 <= tau2 <= 1.0

    def test_custom_base_thresholds(self):
        """Test policy with custom base thresholds."""
        custom_thresholds = {
            'math': 0.95,
            'code': 0.85,
            'text': 0.75
        }
        policy = AdaptiveThresholdPolicy(
            base_thresholds=custom_thresholds,
            context_dim=8
        )

        task_features = torch.randn(8)
        tau = policy.get_threshold('math', task_features)

        # Should be near 0.95 ± 0.05, with small tolerance
        assert 0.89 <= tau <= 1.0, f"Threshold {tau} outside expected range"

    def test_device_placement(self):
        """Test policy works on GPU if available."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        policy = AdaptiveThresholdPolicy(context_dim=8, device=device)

        task_features = torch.randn(8, device=device)
        tau = policy.get_threshold('math', task_features)

        assert isinstance(tau, float)
        assert 0.0 <= tau <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
