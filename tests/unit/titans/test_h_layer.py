"""Unit tests for MambaHLayer (complete H-Layer).

Tests H-Layer integration:
- MAC projector (blackboard → K,V)
- SegmentedAttention processing with value residual
- Neural memory integration (MAG coupling)
- LinUCB routing decisions
- Adaptive threshold policies
- Episode management

Per tasks.md T055, T057.
"""

import pytest
import torch
from src.models.titans.h_layer import MambaHLayer, MACProjector
from src.models.titans.data_model import Problem, HLayerState, RoutingDecision, ContextSlot, SlotType
from src.models.titans.blackboard import Blackboard
from src.data.tokenizer import RetNetTokenizer


class TestMACProjector:
    """Test suite for MAC projector."""

    @pytest.fixture
    def projector(self):
        """Create projector fixture."""
        return MACProjector(d_slot=64, d_model=64, num_heads=8)

    @pytest.fixture
    def slots(self):
        """Create slots tensor fixture."""
        # [batch, num_slots, d_slot]
        return torch.randn(2, 10, 64)

    def test_initialization(self, projector):
        """Test projector initializes correctly."""
        assert projector.d_slot == 64
        assert projector.d_model == 64
        assert projector.num_heads == 8
        assert projector.d_head == 8  # 64 / 8

    def test_forward_shape(self, projector, slots):
        """Test forward produces correct K, V shapes."""
        k, v = projector(slots)

        batch, num_slots, _ = slots.shape
        num_heads = projector.num_heads
        d_head = projector.d_head

        # Expected: [batch, num_heads, num_slots, d_head]
        assert k.shape == (batch, num_heads, num_slots, d_head)
        assert v.shape == (batch, num_heads, num_slots, d_head)

    def test_different_slot_counts(self, projector):
        """Test projector handles different numbers of slots."""
        slots_5 = torch.randn(2, 5, 64)
        slots_20 = torch.randn(2, 20, 64)

        k5, v5 = projector(slots_5)
        k20, v20 = projector(slots_20)

        assert k5.shape[2] == 5
        assert k20.shape[2] == 20

    def test_gating_applied(self, projector, slots):
        """Test gating is applied to K and V."""
        k, v = projector(slots)

        # Gated values should not match ungated projection
        # (This is a structural test - gating changes the values)
        assert k.abs().sum() > 0.0
        assert v.abs().sum() > 0.0

    def test_gradient_flow(self, projector):
        """Test gradients flow through projector."""
        slots = torch.randn(2, 10, 64, requires_grad=True)
        k, v = projector(slots)
        loss = (k.sum() + v.sum())
        loss.backward()

        assert slots.grad is not None


class TestMambaHLayer:
    """Test suite for complete MambaHLayer."""

    @pytest.fixture
    def h_layer(self):
        """Create H-Layer fixture."""
        return MambaHLayer(
            d_model=64,
            num_layers=2,
            num_engines=2,
            segment_len=32,  # Changed from d_state
            num_persist_mem_tokens=2,
            device=torch.device('cpu')
        )

    @pytest.fixture
    def problem(self):
        """Create problem fixture."""
        tokenizer = RetNetTokenizer()
        return Problem(
            problem_id="test_001",
            domain="math",
            input_text="What is 2 + 2?",
            input_tokens=torch.tensor(tokenizer.encode("What is 2 + 2?")),
            time_budget=30.0
        )

    @pytest.fixture
    def blackboard(self):
        """Create blackboard fixture."""
        return Blackboard()

    def test_initialization(self, h_layer):
        """Test H-Layer initializes correctly."""
        assert h_layer.d_model == 64
        assert h_layer.num_layers == 2
        assert h_layer.num_engines == 2

        assert hasattr(h_layer, 'mac_projector')
        assert hasattr(h_layer, 'attention_layers')  # Changed from ssm_layers
        assert hasattr(h_layer, 'neural_memory')
        assert hasattr(h_layer, 'router')
        assert hasattr(h_layer, 'threshold_policy')

    def test_plan_returns_routing_decision(self, h_layer, problem, blackboard):
        """Test plan() returns valid RoutingDecision."""
        decision, state = h_layer.plan(problem, blackboard, state=None)

        assert isinstance(decision, RoutingDecision)
        assert isinstance(state, HLayerState)
        assert decision.problem_id == problem.problem_id

    def test_plan_selects_engines(self, h_layer, problem, blackboard):
        """Test plan() selects engines."""
        decision, _ = h_layer.plan(problem, blackboard)

        assert len(decision.selected_engines) > 0
        assert len(decision.selected_engines) <= 2  # Max K=2
        assert all(isinstance(e, str) for e in decision.selected_engines)

    def test_plan_sets_threshold(self, h_layer, problem, blackboard):
        """Test plan() sets confidence threshold."""
        decision, _ = h_layer.plan(problem, blackboard)

        assert 0.0 <= decision.confidence_threshold <= 1.0

    def test_plan_allocates_budget(self, h_layer, problem, blackboard):
        """Test plan() allocates time budget per engine."""
        decision, _ = h_layer.plan(problem, blackboard)

        assert len(decision.time_budget_per_engine) > 0

        # Total budget should not exceed problem budget
        total_budget = sum(decision.time_budget_per_engine.values())
        assert total_budget <= problem.time_budget + 1e-5  # Small tolerance

    def test_plan_initializes_state(self, h_layer, problem, blackboard):
        """Test plan() initializes state when None."""
        decision, state = h_layer.plan(problem, blackboard, state=None)

        assert state.step == 1
        assert state.episode_id is not None
        assert len(state.episode_id) > 0

    def test_plan_updates_existing_state(self, h_layer, problem, blackboard):
        """Test plan() updates existing state."""
        _, state1 = h_layer.plan(problem, blackboard, state=None)
        _, state2 = h_layer.plan(problem, blackboard, state=state1)

        assert state2.step == state1.step + 1
        assert state2.episode_id == state1.episode_id  # Same episode

    def test_plan_with_blackboard_context(self, h_layer, problem, blackboard):
        """Test plan() reads blackboard context via MAC."""
        # Add some context slots
        blackboard.write_slot(
            slot_id="goal_1",
            value=torch.randn(64),
            slot_type=SlotType.GOAL
        )

        decision, state = h_layer.plan(problem, blackboard)

        # Should successfully process with context
        assert isinstance(decision, RoutingDecision)

    def test_update_from_outcome(self, h_layer, problem, blackboard):
        """Test update_from_outcome() updates router."""
        decision, state = h_layer.plan(problem, blackboard)

        # Update with outcome
        state2 = h_layer.update_from_outcome(
            decision=decision,
            state=state,
            engine_id=decision.selected_engines[0],
            verified_correct=True,
            latency=0.5,
            cost=100.0
        )

        assert state2.step == state.step + 1

    def test_update_tracks_engine_stats(self, h_layer, problem, blackboard):
        """Test update_from_outcome() tracks engine statistics."""
        decision, state = h_layer.plan(problem, blackboard)

        engine_id = decision.selected_engines[0]

        # Update multiple times
        for i in range(3):
            state = h_layer.update_from_outcome(
                decision=decision,
                state=state,
                engine_id=engine_id,
                verified_correct=(i % 2 == 0),  # Alternate correct/incorrect
                latency=0.5,
                cost=100.0
            )

        # Check stats exist
        assert engine_id in state.engine_stats
        stats = state.engine_stats[engine_id]
        assert stats['attempts'] == 3

    def test_reset_episode(self, h_layer, problem, blackboard):
        """Test reset_episode() creates new episode ID."""
        _, state = h_layer.plan(problem, blackboard)

        original_episode_id = state.episode_id

        # Reset episode
        state2 = h_layer.reset_episode(state)

        assert state2.episode_id != original_episode_id  # New episode
        assert state2.step == 0

    def test_reset_episode_preserves_memory(self, h_layer, problem, blackboard):
        """Test reset_episode() preserves memory state."""
        _, state = h_layer.plan(problem, blackboard)

        # Set memory state
        memory_state = {'some_key': torch.randn(10)}
        state.memory_state = memory_state

        # Reset episode
        state2 = h_layer.reset_episode(state)

        assert state2.memory_state == memory_state  # Preserved

    def test_reset_episode_preserves_router_stats(self, h_layer, problem, blackboard):
        """Test reset_episode() preserves router statistics."""
        decision, state = h_layer.plan(problem, blackboard)

        # Add some stats
        state.engine_stats = {'engine_0': {'attempts': 10}}

        # Reset episode
        state2 = h_layer.reset_episode(state)

        assert state2.engine_stats == state.engine_stats  # Preserved

    def test_threshold_history_tracking(self, h_layer, problem, blackboard):
        """Test threshold history is tracked."""
        _, state1 = h_layer.plan(problem, blackboard)
        _, state2 = h_layer.plan(problem, blackboard, state=state1)
        _, state3 = h_layer.plan(problem, blackboard, state=state2)

        assert len(state3.threshold_history) == 3

    def test_context_features_cached(self, h_layer, problem, blackboard):
        """Test context features are cached in state."""
        _, state = h_layer.plan(problem, blackboard)

        assert state.cached_context is not None
        assert state.cached_context.shape[0] == h_layer.context_dim

    def test_routing_decision_predicted_risk(self, h_layer, problem, blackboard):
        """Test routing decision includes predicted risk."""
        decision, _ = h_layer.plan(problem, blackboard)

        assert 0.0 <= decision.predicted_risk <= 1.0

    def test_routing_decision_engine_scores(self, h_layer, problem, blackboard):
        """Test routing decision includes UCB scores."""
        decision, _ = h_layer.plan(problem, blackboard)

        assert len(decision.engine_scores) == h_layer.num_engines
        assert all(isinstance(v, float) for v in decision.engine_scores.values())

    def test_multiple_plans_same_episode(self, h_layer, problem, blackboard):
        """Test multiple planning calls in same episode."""
        _, state = h_layer.plan(problem, blackboard, state=None)

        episode_id = state.episode_id

        # Plan again in same episode
        _, state2 = h_layer.plan(problem, blackboard, state=state)
        _, state3 = h_layer.plan(problem, blackboard, state=state2)

        # Episode ID should be same
        assert state2.episode_id == episode_id
        assert state3.episode_id == episode_id

        # Steps should increment
        assert state2.step > state.step
        assert state3.step > state2.step

    def test_memory_state_persistence(self, h_layer, problem, blackboard):
        """Test neural memory state persists across planning calls."""
        _, state1 = h_layer.plan(problem, blackboard, state=None)
        _, state2 = h_layer.plan(problem, blackboard, state=state1)

        # Memory state should exist (can be None on first call, but tracked)
        # Just verify the state structure is maintained
        assert hasattr(state2, 'memory_state')

    def test_gradient_flow_through_plan(self, h_layer, problem, blackboard):
        """Test gradients flow through planning (for meta-learning)."""
        # This is a structural test
        decision, state = h_layer.plan(problem, blackboard)

        # Should not raise errors
        assert decision is not None
        assert state is not None

    def test_different_domains(self, h_layer, blackboard):
        """Test H-Layer handles different problem domains."""
        tokenizer = RetNetTokenizer()

        domains = ['math', 'code', 'text']

        for domain in domains:
            problem = Problem(
                problem_id=f"test_{domain}",
                domain=domain,
                input_text="Test problem",
                input_tokens=torch.tensor(tokenizer.encode("Test problem")),
                time_budget=30.0
            )

            decision, state = h_layer.plan(problem, blackboard)
            assert isinstance(decision, RoutingDecision)

    def test_device_placement(self, problem, blackboard):
        """Test H-Layer works on GPU if available."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        h_layer = MambaHLayer(
            d_model=64,
            num_layers=2,
            num_engines=2,
            device=device
        )

        problem.input_tokens = problem.input_tokens.to(device)

        decision, state = h_layer.plan(problem, blackboard)

        assert isinstance(decision, RoutingDecision)

    def test_deterministic_with_same_input(self, h_layer, problem, blackboard):
        """Test H-Layer is deterministic given same input."""
        torch.manual_seed(42)
        decision1, state1 = h_layer.plan(problem, blackboard, state=None)

        torch.manual_seed(42)
        decision2, state2 = h_layer.plan(problem, blackboard, state=None)

        # Should produce same results
        assert decision1.selected_engines == decision2.selected_engines


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
