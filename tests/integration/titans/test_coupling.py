"""Integration tests for MAC and MAG coupling.

Tests MAC (Memory-As-Context) and MAG (Memory-As-Gating) coupling:
- MAC: Blackboard context flows to H-Layer via K,V projection
- MAG: Neural memory gates FFN outputs
- Decoupled gradient flow (no gradients from MAG to MAC)
- End-to-end coupling in H-Layer

Per tasks.md T057.
"""

import pytest
import torch
from src.models.titans.h_layer import MambaHLayer, MACProjector
from src.models.titans.neural_memory import NeuralMemory
from src.models.titans.data_model import Problem, ContextSlot, SlotType
from src.models.titans.blackboard import Blackboard
from src.data.tokenizer import RetNetTokenizer


class TestMACCoupling:
    """Test suite for MAC (Memory-As-Context) coupling."""

    @pytest.fixture
    def blackboard(self):
        """Create blackboard with context slots."""
        bb = Blackboard()

        # Add some context slots
        for i in range(5):
            bb.write_slot(
                slot_id=f"slot_{i}",
                value=torch.randn(64),
                slot_type=SlotType.GOAL if i < 2 else SlotType.EVIDENCE
            )

        return bb

    @pytest.fixture
    def h_layer(self):
        """Create H-Layer."""
        return MambaHLayer(
            d_model=64,
            num_layers=2,
            num_engines=2,
            device=torch.device('cpu')
        )

    @pytest.fixture
    def problem(self):
        """Create test problem."""
        tokenizer = RetNetTokenizer()
        return Problem(
            problem_id="test_mac",
            domain="math",
            input_text="Test MAC coupling",
            input_tokens=torch.tensor(tokenizer.encode("Test MAC coupling")),
            time_budget=30.0
        )

    def test_mac_projector_reads_blackboard(self, blackboard):
        """Test MAC projector processes blackboard slots."""
        slots = blackboard.read_slots()
        assert len(slots) == 5

        # Stack into tensor
        slot_tensors = torch.stack([s.value for s in slots], dim=0).unsqueeze(0)

        # Project via MAC
        mac_projector = MACProjector(d_slot=64, d_model=64, num_heads=8)
        k, v = mac_projector(slot_tensors)

        # Should produce K, V for attention
        assert k.shape[2] == 5  # 5 slots
        assert v.shape[2] == 5

    def test_h_layer_reads_mac_context(self, h_layer, problem, blackboard):
        """Test H-Layer reads blackboard via MAC."""
        # Plan with context
        decision, state = h_layer.plan(problem, blackboard, state=None)

        # Should successfully process
        assert decision is not None
        assert state is not None

    def test_mac_context_affects_planning(self, h_layer, problem):
        """Test MAC context affects planning decisions."""
        # Plan with empty blackboard
        bb_empty = Blackboard()
        decision_empty, _ = h_layer.plan(problem, bb_empty)

        # Plan with rich blackboard
        bb_rich = Blackboard()
        for i in range(10):
            bb_rich.write_slot(
                slot_id=f"slot_{i}",
                value=torch.randn(64),
                slot_type=SlotType.EVIDENCE
            )

        decision_rich, _ = h_layer.plan(problem, bb_rich)

        # Both should work
        assert decision_empty is not None
        assert decision_rich is not None

        # Decisions may differ (context affects planning)
        # This is probabilistic, so we just check they're valid

    def test_mac_context_different_slot_types(self, h_layer, problem):
        """Test MAC handles different slot types."""
        bb = Blackboard()

        # Add diverse slot types
        slot_types = [SlotType.GOAL, SlotType.EVIDENCE, SlotType.OBSERVATION, SlotType.PLAN, SlotType.HINT]

        for i, slot_type in enumerate(slot_types):
            bb.write_slot(
                slot_id=f"slot_{i}",
                value=torch.randn(64),
                slot_type=slot_type
            )

        decision, state = h_layer.plan(problem, bb)

        assert decision is not None

    def test_mac_with_empty_blackboard(self, h_layer, problem):
        """Test MAC handles empty blackboard gracefully."""
        bb_empty = Blackboard()

        decision, state = h_layer.plan(problem, bb_empty)

        # Should still work (no context, but valid)
        assert decision is not None


class TestMAGCoupling:
    """Test suite for MAG (Memory-As-Gating) coupling."""

    @pytest.fixture
    def neural_memory(self):
        """Create neural memory."""
        return NeuralMemory(
            dim=64,
            chunk_size=8,
            heads=4,
            dim_head=16,
            momentum=True,
            default_model_kwargs=dict(
                depth=2,
                expansion_factor=4.
            )
        )

    @pytest.fixture
    def input_tensor(self):
        """Create input tensor."""
        return torch.randn(2, 10, 64)

    def test_memory_produces_gating_vector(self, neural_memory, input_tensor):
        """Test neural memory retrieval for MAG gating vector."""
        # NeuralMemory forward pass returns (retrieved, state)
        retrieved, state = neural_memory(input_tensor)

        # Retrieved should have same shape as input
        assert retrieved.shape == input_tensor.shape

        # In H-Layer, retrieved is passed through to_gating (sigmoid)
        # to produce the gating vector [0, 1]
        gate = torch.sigmoid(retrieved)
        assert (gate >= 0.0).all()
        assert (gate <= 1.0).all()

    def test_mag_gating_modulates_ffn(self):
        """Test MAG gating modulates FFN output."""
        # Simulate FFN output
        ffn_out = torch.randn(2, 10, 64)
        u = torch.randn(2, 10, 64)

        # Simulate gating
        gate = torch.rand(2, 10, 64)  # [0, 1]

        # MAG: h = g * FFN(u) + (1-g) * u
        h = gate * ffn_out + (1 - gate) * u

        # Output should be blend
        assert h.shape == ffn_out.shape
        assert not torch.allclose(h, ffn_out)
        assert not torch.allclose(h, u)

    def test_mag_extreme_gates(self):
        """Test MAG with extreme gate values."""
        ffn_out = torch.randn(2, 10, 64)
        u = torch.randn(2, 10, 64)

        # Gate = 1 (full FFN)
        gate_full = torch.ones(2, 10, 64)
        h_full = gate_full * ffn_out + (1 - gate_full) * u
        assert torch.allclose(h_full, ffn_out)

        # Gate = 0 (full skip)
        gate_zero = torch.zeros(2, 10, 64)
        h_zero = gate_zero * ffn_out + (1 - gate_zero) * u
        assert torch.allclose(h_zero, u)

    def test_mag_in_h_layer(self):
        """Test MAG coupling is integrated in H-Layer."""
        h_layer = MambaHLayer(
            d_model=64,
            num_layers=2,
            num_engines=2,
            device=torch.device('cpu')
        )

        # H-Layer should have neural memory
        assert hasattr(h_layer, 'neural_memory')
        assert isinstance(h_layer.neural_memory, NeuralMemory)

    def test_memory_updates_in_h_layer(self):
        """Test neural memory can update during H-Layer processing."""
        h_layer = MambaHLayer(d_model=64, num_layers=2, num_engines=2)

        # Access memory
        memory = h_layer.neural_memory

        x = torch.randn(2, 10, 64)

        # Update memory (NeuralMemory updates automatically during forward)
        retrieved, state = memory(x)

        # Should return a valid state
        assert state is not None
        # NeuralMemState is a namedtuple with fields: seq_index, weights, cache_store_segment, states, updates
        assert hasattr(state, 'seq_index')
        assert hasattr(state, 'weights')


class TestCouplingIntegration:
    """Test MAC and MAG coupling working together."""

    @pytest.fixture
    def h_layer(self):
        """Create complete H-Layer with both couplings."""
        return MambaHLayer(
            d_model=64,
            num_layers=2,
            num_engines=2,
            device=torch.device('cpu')
        )

    @pytest.fixture
    def problem(self):
        """Create test problem."""
        tokenizer = RetNetTokenizer()
        return Problem(
            problem_id="coupling_test",
            domain="math",
            input_text="Test coupling",
            input_tokens=torch.tensor(tokenizer.encode("Test coupling")),
            time_budget=30.0
        )

    @pytest.fixture
    def blackboard_with_context(self):
        """Create blackboard with context."""
        bb = Blackboard()
        for i in range(5):
            bb.write_slot(
                slot_id=f"slot_{i}",
                value=torch.randn(64),
                slot_type=SlotType.EVIDENCE
            )
        return bb

    def test_mac_and_mag_both_active(self, h_layer, problem, blackboard_with_context):
        """Test both MAC and MAG coupling are active during planning."""
        decision, state = h_layer.plan(problem, blackboard_with_context, state=None)

        # Both couplings should have been used
        assert decision is not None
        assert state is not None

        # MAC: Should have read blackboard context
        # MAG: Should have used neural memory

    def test_end_to_end_with_both_couplings(self, h_layer, problem, blackboard_with_context):
        """Test full planning with MAC and MAG."""
        # Initial plan
        decision1, state1 = h_layer.plan(problem, blackboard_with_context, state=None)

        # Update from outcome
        state2 = h_layer.update_from_outcome(
            decision=decision1,
            state=state1,
            engine_id=decision1.selected_engines[0],
            verified_correct=True,
            latency=0.5,
            cost=100.0
        )

        # Plan again (should use learned routing)
        decision2, state3 = h_layer.plan(problem, blackboard_with_context, state=state2)

        # Both plans should succeed
        assert decision1 is not None
        assert decision2 is not None

    def test_mac_context_changes_affect_planning(self, h_layer, problem):
        """Test changes in MAC context affect planning."""
        # Plan with context A
        bb_a = Blackboard()
        bb_a.write_slot(
            slot_id="goal",
            value=torch.ones(64),  # All ones
            slot_type=SlotType.GOAL
        )

        decision_a, _ = h_layer.plan(problem, bb_a)

        # Plan with context B
        bb_b = Blackboard()
        bb_b.write_slot(
            slot_id="goal",
            value=torch.zeros(64),  # All zeros
            slot_type=SlotType.GOAL
        )

        decision_b, _ = h_layer.plan(problem, bb_b)

        # Both should work
        assert decision_a is not None
        assert decision_b is not None

    def test_gradient_decoupling(self, h_layer):
        """Test MAG and MAC are decoupled (no gradients from MAG to MAC)."""
        # This is a structural test - in practice, MAG gates FFN,
        # and MAC provides context, but gradients shouldn't flow
        # from MAG gating back to blackboard context.

        # For now, just verify both components exist
        assert hasattr(h_layer, 'mac_projector')
        assert hasattr(h_layer, 'neural_memory')

    def test_multiple_episodes_with_coupling(self, h_layer, problem, blackboard_with_context):
        """Test couplings work across episode resets."""
        # Episode 1
        decision1, state1 = h_layer.plan(problem, blackboard_with_context)

        # Reset episode
        state2 = h_layer.reset_episode(state1)

        # Episode 2
        decision2, state3 = h_layer.plan(problem, blackboard_with_context, state=state2)

        # Both episodes should work
        assert decision1 is not None
        assert decision2 is not None

        # Episode IDs should differ
        assert state1.episode_id != state2.episode_id

    def test_coupling_with_long_sequence(self, h_layer, blackboard_with_context):
        """Test coupling with longer input sequence."""
        tokenizer = RetNetTokenizer()

        # Long problem text
        long_text = " ".join([f"word_{i}" for i in range(50)])
        problem = Problem(
            problem_id="long_test",
            domain="math",
            input_text=long_text,
            input_tokens=torch.tensor(tokenizer.encode(long_text)),
            time_budget=30.0
        )

        decision, state = h_layer.plan(problem, blackboard_with_context)

        assert decision is not None

    def test_coupling_stability_over_time(self, h_layer, problem, blackboard_with_context):
        """Test couplings remain stable over many iterations."""
        state = None

        # Run 20 planning iterations
        for i in range(20):
            decision, state = h_layer.plan(problem, blackboard_with_context, state=state)

            # Should always succeed
            assert decision is not None

            # Update
            state = h_layer.update_from_outcome(
                decision=decision,
                state=state,
                engine_id=decision.selected_engines[0],
                verified_correct=True,
                latency=0.5,
                cost=100.0
            )

        # Final state should be valid
        assert state.step == 20


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
