"""Contract tests for Titan-HRM interfaces (T028).

Validates that all components conform to their interface contracts:
- IBlackboard: reserve_version, write_proposal, commit_answer, read/write_slots
- ILEngine: solve, get_confidence
- IArbiter: verify_proposal, evaluate_halting

Ensures all components work together correctly per
specs/001-titan-hrm-phase/contracts/hrm-system-interface.md
"""

import pytest
import torch
from unittest.mock import Mock

from src.models.titans.blackboard import Blackboard
from src.models.titans.l_engine import SimpleLEngine
from src.models.titans.arbiter import ArbiterVerifier
from src.models.titans.verifiers import MathSymbolicVerifier, MathNumericVerifier
from src.models.titans.data_model import (
    Problem,
    SolutionProposal,
    ContextSlot,
    SlotType,
    VerificationResult,
    HaltingDecision,
    HaltingStatus,
)


# ============================================================================
# Test IBlackboard Interface Compliance
# ============================================================================

class TestBlackboardInterface:
    """Test that Blackboard implements IBlackboard interface."""

    def test_has_reserve_version_method(self):
        """Test that Blackboard has reserve_version() method."""
        bb = Blackboard()
        assert hasattr(bb, 'reserve_version')
        assert callable(bb.reserve_version)

    def test_reserve_version_returns_int(self):
        """Test that reserve_version() returns integer."""
        bb = Blackboard()
        version = bb.reserve_version()
        assert isinstance(version, int)
        assert version > 0

    def test_reserve_version_is_monotonic(self):
        """Test that reserve_version() returns monotonically increasing values."""
        bb = Blackboard()
        v1 = bb.reserve_version()
        v2 = bb.reserve_version()
        v3 = bb.reserve_version()

        assert v2 > v1
        assert v3 > v2

    def test_has_write_proposal_method(self):
        """Test that Blackboard has write_proposal() method."""
        bb = Blackboard()
        assert hasattr(bb, 'write_proposal')
        assert callable(bb.write_proposal)

    def test_write_proposal_accepts_engine_id_and_proposal(self):
        """Test write_proposal() signature."""
        bb = Blackboard()

        proposal = SolutionProposal(
            proposal_id="test_001",
            problem_id="prob_001",
            engine_id="engine_1",
            version=0,
            checksum=0,
            content=torch.tensor([1, 2, 3]),
            content_text="test",
            raw_confidence=0.5,
            calibrated_confidence=0.5,
        )

        # Should not raise
        bb.write_proposal("engine_1", proposal)

    def test_write_proposal_stores_proposal(self):
        """Test that write_proposal() stores proposal in blackboard."""
        bb = Blackboard()

        proposal = SolutionProposal(
            proposal_id="test_002",
            problem_id="prob_001",
            engine_id="engine_1",
            version=0,
            checksum=0,
            content=torch.tensor([1, 2, 3]),
            content_text="test",
            raw_confidence=0.5,
            calibrated_confidence=0.5,
        )

        bb.write_proposal("engine_1", proposal)

        # Should be retrievable
        proposals = bb.read_proposals()
        assert "engine_1" in proposals
        assert proposals["engine_1"].proposal_id == "test_002"

    def test_has_commit_answer_method(self):
        """Test that Blackboard has commit_answer() method."""
        bb = Blackboard()
        assert hasattr(bb, 'commit_answer')
        assert callable(bb.commit_answer)

    def test_commit_answer_returns_bool(self):
        """Test that commit_answer() returns boolean."""
        bb = Blackboard()

        proposal = SolutionProposal(
            proposal_id="test_003",
            problem_id="prob_001",
            engine_id="engine_1",
            version=bb.reserve_version(),
            checksum=0,
            content=torch.tensor([1, 2, 3]),
            content_text="test",
            raw_confidence=0.5,
            calibrated_confidence=0.5,
            verified=True,  # Must be verified
        )
        proposal.checksum = proposal.compute_checksum()

        result = bb.commit_answer(proposal)
        assert isinstance(result, bool)

    def test_commit_answer_requires_verification(self):
        """Test that commit_answer() requires proposal to be verified."""
        bb = Blackboard()

        unverified_proposal = SolutionProposal(
            proposal_id="test_004",
            problem_id="prob_001",
            engine_id="engine_1",
            version=bb.reserve_version(),
            checksum=0,
            content=torch.tensor([1, 2, 3]),
            content_text="test",
            raw_confidence=0.5,
            calibrated_confidence=0.5,
            verified=False,  # Not verified
        )

        result = bb.commit_answer(unverified_proposal)
        assert result is False  # Should reject unverified

    def test_has_read_slots_method(self):
        """Test that Blackboard has read_slots() method."""
        bb = Blackboard()
        assert hasattr(bb, 'read_slots')
        assert callable(bb.read_slots)

    def test_read_slots_returns_dict(self):
        """Test that read_slots() returns dictionary."""
        bb = Blackboard()
        slots = bb.read_slots()
        assert isinstance(slots, dict)

    def test_read_slots_filters_by_type(self):
        """Test that read_slots() filters by SlotType."""
        bb = Blackboard()

        # Write slots of different types
        bb.write_slot("goal_1", torch.randn(10), SlotType.GOAL)
        bb.write_slot("evidence_1", torch.randn(10), SlotType.EVIDENCE)

        # Read only GOAL slots
        goal_slots = bb.read_slots(slot_types=[SlotType.GOAL])
        assert len(goal_slots) == 1
        assert "goal_1" in goal_slots

    def test_has_write_slot_method(self):
        """Test that Blackboard has write_slot() method."""
        bb = Blackboard()
        assert hasattr(bb, 'write_slot')
        assert callable(bb.write_slot)

    def test_write_slot_accepts_correct_params(self):
        """Test write_slot() signature."""
        bb = Blackboard()

        # Should not raise
        bb.write_slot(
            slot_id="test_slot",
            value=torch.randn(10),
            slot_type=SlotType.GOAL,
        )

    def test_write_slot_stores_and_retrieves(self):
        """Test that write_slot() stores slot and read_slot() retrieves it."""
        bb = Blackboard()

        value = torch.randn(10)
        bb.write_slot("test_slot", value, SlotType.GOAL)

        slot = bb.read_slot("test_slot")
        assert slot is not None
        assert slot.slot_id == "test_slot"
        assert slot.slot_type == SlotType.GOAL
        assert torch.allclose(slot.value, value)


# ============================================================================
# Test ILEngine Interface Compliance
# ============================================================================

class TestLEngineInterface:
    """Test that SimpleLEngine implements ILEngine interface."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        from src.data.tokenizer import RetNetTokenizer
        tokenizer = Mock(spec=RetNetTokenizer)
        tokenizer.vocab_size = 50000
        tokenizer.tokenizer = Mock()
        tokenizer.tokenizer.eos_token_id = 0

        tokenizer.encode = Mock(return_value=[1, 2, 3])
        tokenizer.decode = Mock(return_value="test answer")

        return tokenizer

    @pytest.fixture
    def mock_backbone(self):
        """Create a mock RetNet backbone."""
        from src.models.retnet.backbone import RetNetBackbone
        import torch.nn as nn

        backbone = Mock(spec=RetNetBackbone)
        backbone.d_model = 512
        backbone.vocab_size = 50000
        backbone.embed = Mock(spec=nn.Embedding)
        backbone.embed.weight = torch.randn(50000, 512)

        def mock_forward_train(input_ids, segment_ids=None):
            batch_size, seq_len = input_ids.shape
            return torch.randn(batch_size, seq_len, 512)

        backbone.forward_train = Mock(side_effect=mock_forward_train)
        return backbone

    @pytest.fixture
    def l_engine(self, mock_backbone, mock_tokenizer):
        """Create a SimpleLEngine."""
        engine = SimpleLEngine(
            engine_id="test_engine",
            backbone=mock_backbone,
            tokenizer=mock_tokenizer,
            device="cpu",
        )

        # Mock output head
        def mock_output_head_forward(hidden_states):
            batch_size, seq_len, _ = hidden_states.shape
            return torch.randn(batch_size, seq_len, 50000)

        engine.output_head.forward = mock_output_head_forward
        return engine

    def test_has_solve_method(self, l_engine):
        """Test that SimpleLEngine has solve() method."""
        assert hasattr(l_engine, 'solve')
        assert callable(l_engine.solve)

    def test_solve_accepts_problem_and_blackboard(self, l_engine):
        """Test solve() signature."""
        problem = Problem(
            problem_id="test_001",
            domain="math",
            input_text="2 + 2 = ?",
            input_tokens=torch.tensor([1, 2, 3]),
        )
        blackboard = Blackboard()

        # Should not raise
        proposal = l_engine.solve(problem, blackboard)
        assert proposal is not None

    def test_solve_returns_solution_proposal(self, l_engine):
        """Test that solve() returns SolutionProposal."""
        problem = Problem(
            problem_id="test_002",
            domain="math",
            input_text="2 + 2 = ?",
            input_tokens=torch.tensor([1, 2, 3]),
        )

        proposal = l_engine.solve(problem)
        assert isinstance(proposal, SolutionProposal)

    def test_solve_proposal_has_required_fields(self, l_engine):
        """Test that solve() returns proposal with required fields."""
        problem = Problem(
            problem_id="test_003",
            domain="math",
            input_text="2 + 2 = ?",
            input_tokens=torch.tensor([1, 2, 3]),
        )

        proposal = l_engine.solve(problem)

        # Check required fields per ILEngine contract
        assert proposal.proposal_id is not None
        assert proposal.problem_id == problem.problem_id
        assert proposal.engine_id == l_engine.engine_id
        assert proposal.content is not None
        assert proposal.content_text is not None
        assert proposal.raw_confidence is not None
        assert proposal.calibrated_confidence is not None

    def test_has_get_confidence_method(self, l_engine):
        """Test that SimpleLEngine has get_confidence() method."""
        assert hasattr(l_engine, 'get_confidence')
        assert callable(l_engine.get_confidence)

    def test_get_confidence_accepts_logits(self, l_engine):
        """Test get_confidence() signature."""
        logits = torch.randn(1000)
        confidence = l_engine.get_confidence(logits)
        assert confidence is not None

    def test_get_confidence_returns_confidence_score(self, l_engine):
        """Test that get_confidence() returns ConfidenceScore (Phase 4)."""
        from src.models.titans.data_model import ConfidenceScore

        logits = torch.randn(1000)
        confidence_score = l_engine.get_confidence(logits)

        # Phase 4: now returns ConfidenceScore object
        assert isinstance(confidence_score, ConfidenceScore)
        assert 0.0 <= confidence_score.raw <= 1.0
        assert 0.0 <= confidence_score.calibrated <= 1.0

    def test_has_recalibrate_method(self, l_engine):
        """Test that SimpleLEngine has recalibrate() method."""
        assert hasattr(l_engine, 'recalibrate')
        assert callable(l_engine.recalibrate)

    def test_recalibrate_accepts_calibration_data(self, l_engine):
        """Test recalibrate() signature."""
        # Should not raise (stub in MVP)
        l_engine.recalibrate(calibration_data={'test': 'data'})


# ============================================================================
# Test IArbiter Interface Compliance
# ============================================================================

class TestArbiterInterface:
    """Test that ArbiterVerifier implements IArbiter interface."""

    @pytest.fixture
    def arbiter(self):
        """Create an ArbiterVerifier."""
        verifiers = [
            MathSymbolicVerifier(),
            MathNumericVerifier(),
        ]
        return ArbiterVerifier(
            verifiers=verifiers,
            confidence_threshold=0.8,
        )

    def test_has_verify_proposal_method(self, arbiter):
        """Test that ArbiterVerifier has verify_proposal() method."""
        assert hasattr(arbiter, 'verify_proposal')
        assert callable(arbiter.verify_proposal)

    def test_verify_proposal_accepts_proposal_and_problem(self, arbiter):
        """Test verify_proposal() signature."""
        proposal = SolutionProposal(
            proposal_id="test_001",
            problem_id="prob_001",
            engine_id="engine_1",
            version=1,
            checksum=0,
            content=torch.tensor([1, 2, 3]),
            content_text="4",
            raw_confidence=0.5,
            calibrated_confidence=0.5,
        )

        problem = Problem(
            problem_id="prob_001",
            domain="math",
            input_text="2 + 2 = ?",
            input_tokens=torch.tensor([1, 2, 3]),
            ground_truth="4",
        )

        # Should not raise
        results = arbiter.verify_proposal(proposal, problem)
        assert results is not None

    def test_verify_proposal_returns_list_of_results(self, arbiter):
        """Test that verify_proposal() returns list of VerificationResult."""
        proposal = SolutionProposal(
            proposal_id="test_002",
            problem_id="prob_001",
            engine_id="engine_1",
            version=1,
            checksum=0,
            content=torch.tensor([1, 2, 3]),
            content_text="4",
            raw_confidence=0.5,
            calibrated_confidence=0.5,
        )

        problem = Problem(
            problem_id="prob_001",
            domain="math",
            input_text="2 + 2 = ?",
            input_tokens=torch.tensor([1, 2, 3]),
            ground_truth="4",
        )

        results = arbiter.verify_proposal(proposal, problem)

        assert isinstance(results, list)
        assert len(results) > 0
        for result in results:
            assert isinstance(result, VerificationResult)

    def test_has_evaluate_halting_method(self, arbiter):
        """Test that ArbiterVerifier has evaluate_halting() method."""
        assert hasattr(arbiter, 'evaluate_halting')
        assert callable(arbiter.evaluate_halting)

    def test_evaluate_halting_accepts_correct_params(self, arbiter):
        """Test evaluate_halting() signature."""
        proposal = SolutionProposal(
            proposal_id="test_003",
            problem_id="prob_001",
            engine_id="engine_1",
            version=1,
            checksum=0,
            content=torch.tensor([1, 2, 3]),
            content_text="4",
            raw_confidence=0.9,
            calibrated_confidence=0.9,
        )

        verification_results = [
            VerificationResult(
                verifier_type=Mock(),
                proposal_id="test_003",
                passed=True,
                score=1.0,
            )
        ]

        problem = Problem(
            problem_id="prob_001",
            domain="math",
            input_text="2 + 2 = ?",
            input_tokens=torch.tensor([1, 2, 3]),
            ground_truth="4",
        )

        # Should not raise
        decision = arbiter.evaluate_halting(
            proposal=proposal,
            verification_results=verification_results,
            problem=problem,
            elapsed_time=0.5,
        )
        assert decision is not None

    def test_evaluate_halting_returns_halting_decision(self, arbiter):
        """Test that evaluate_halting() returns HaltingDecision."""
        proposal = SolutionProposal(
            proposal_id="test_004",
            problem_id="prob_001",
            engine_id="engine_1",
            version=1,
            checksum=0,
            content=torch.tensor([1, 2, 3]),
            content_text="4",
            raw_confidence=0.9,
            calibrated_confidence=0.9,
        )

        verification_results = [
            VerificationResult(
                verifier_type=Mock(),
                proposal_id="test_004",
                passed=True,
                score=1.0,
            )
        ]

        problem = Problem(
            problem_id="prob_001",
            domain="math",
            input_text="2 + 2 = ?",
            input_tokens=torch.tensor([1, 2, 3]),
            ground_truth="4",
            time_budget=30.0,
        )

        decision = arbiter.evaluate_halting(
            proposal=proposal,
            verification_results=verification_results,
            problem=problem,
            elapsed_time=0.5,
        )

        assert isinstance(decision, HaltingDecision)
        assert decision.status in [HaltingStatus.COMMIT, HaltingStatus.REFINE, HaltingStatus.TIMEOUT]

    def test_evaluate_halting_commits_on_success(self, arbiter):
        """Test that evaluate_halting() returns COMMIT when appropriate."""
        proposal = SolutionProposal(
            proposal_id="test_005",
            problem_id="prob_001",
            engine_id="engine_1",
            version=1,
            checksum=0,
            content=torch.tensor([1, 2, 3]),
            content_text="4",
            raw_confidence=0.95,  # High confidence
            calibrated_confidence=0.95,
        )

        verification_results = [
            VerificationResult(
                verifier_type=Mock(),
                proposal_id="test_005",
                passed=True,  # Verified
                score=1.0,
            )
        ]

        problem = Problem(
            problem_id="prob_001",
            domain="math",
            input_text="2 + 2 = ?",
            input_tokens=torch.tensor([1, 2, 3]),
            ground_truth="4",
            time_budget=30.0,
        )

        decision = arbiter.evaluate_halting(
            proposal=proposal,
            verification_results=verification_results,
            problem=problem,
            elapsed_time=0.5,
        )

        # High confidence + verified � COMMIT
        assert decision.status == HaltingStatus.COMMIT
        assert decision.selected_proposal == proposal.proposal_id

    def test_evaluate_halting_refines_on_low_confidence(self, arbiter):
        """Test that evaluate_halting() returns REFINE when confidence low."""
        proposal = SolutionProposal(
            proposal_id="test_006",
            problem_id="prob_001",
            engine_id="engine_1",
            version=1,
            checksum=0,
            content=torch.tensor([1, 2, 3]),
            content_text="4",
            raw_confidence=0.5,  # Low confidence
            calibrated_confidence=0.5,
        )

        verification_results = [
            VerificationResult(
                verifier_type=Mock(),
                proposal_id="test_006",
                passed=True,
                score=1.0,
            )
        ]

        problem = Problem(
            problem_id="prob_001",
            domain="math",
            input_text="2 + 2 = ?",
            input_tokens=torch.tensor([1, 2, 3]),
            ground_truth="4",
            time_budget=30.0,
        )

        decision = arbiter.evaluate_halting(
            proposal=proposal,
            verification_results=verification_results,
            problem=problem,
            elapsed_time=0.5,
        )

        # Low confidence � REFINE
        assert decision.status == HaltingStatus.REFINE

    def test_evaluate_halting_timeouts_on_budget(self, arbiter):
        """Test that evaluate_halting() returns TIMEOUT when budget exceeded."""
        proposal = SolutionProposal(
            proposal_id="test_007",
            problem_id="prob_001",
            engine_id="engine_1",
            version=1,
            checksum=0,
            content=torch.tensor([1, 2, 3]),
            content_text="4",
            raw_confidence=0.95,
            calibrated_confidence=0.95,
        )

        verification_results = [
            VerificationResult(
                verifier_type=Mock(),
                proposal_id="test_007",
                passed=True,
                score=1.0,
            )
        ]

        problem = Problem(
            problem_id="prob_001",
            domain="math",
            input_text="2 + 2 = ?",
            input_tokens=torch.tensor([1, 2, 3]),
            ground_truth="4",
            time_budget=1.0,  # 1 second budget
        )

        decision = arbiter.evaluate_halting(
            proposal=proposal,
            verification_results=verification_results,
            problem=problem,
            elapsed_time=2.0,  # Exceeded budget
        )

        # Budget exceeded � TIMEOUT
        assert decision.status == HaltingStatus.TIMEOUT


# ============================================================================
# Test Component Integration
# ============================================================================

class TestComponentIntegration:
    """Test that all components work together correctly."""

    def test_end_to_end_simple_problem(self):
        """Test complete flow: Blackboard + Engine + Arbiter."""
        from unittest.mock import Mock
        from src.data.tokenizer import RetNetTokenizer
        from src.models.retnet.backbone import RetNetBackbone

        # Create mock components
        mock_tokenizer = Mock(spec=RetNetTokenizer)
        mock_tokenizer.vocab_size = 50000
        mock_tokenizer.tokenizer = Mock()
        mock_tokenizer.tokenizer.eos_token_id = 0
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3])
        mock_tokenizer.decode = Mock(return_value="4")

        mock_backbone = Mock(spec=RetNetBackbone)
        mock_backbone.d_model = 512
        mock_backbone.vocab_size = 50000
        mock_backbone.embed = Mock()
        mock_backbone.embed.weight = torch.randn(50000, 512)
        mock_backbone.forward_train = Mock(return_value=torch.randn(1, 5, 512))

        # Create components
        blackboard = Blackboard()

        engine = SimpleLEngine(
            engine_id="test_engine",
            backbone=mock_backbone,
            tokenizer=mock_tokenizer,
            device="cpu",
        )
        engine.output_head.forward = lambda h: torch.randn(h.shape[0], h.shape[1], 50000)

        arbiter = ArbiterVerifier(
            verifiers=[MathSymbolicVerifier()],
            confidence_threshold=0.8,
        )

        # Create problem
        problem = Problem(
            problem_id="integration_001",
            domain="math",
            input_text="2 + 2 = ?",
            input_tokens=torch.tensor([1, 2, 3]),
            ground_truth="4",
            time_budget=30.0,
        )

        # Execute flow
        # 1. Engine generates proposal
        proposal = engine.solve(problem, blackboard)
        assert proposal is not None

        # 2. Write to blackboard
        blackboard.write_proposal(engine.engine_id, proposal)

        # 3. Verify with arbiter
        verification_results = arbiter.verify_proposal(proposal, problem)
        assert len(verification_results) > 0

        # 4. Evaluate halting
        proposal.verification_results = verification_results
        proposal.verified = all(vr.passed for vr in verification_results)

        # Mock high confidence
        proposal.calibrated_confidence = 0.95

        decision = arbiter.evaluate_halting(
            proposal=proposal,
            verification_results=verification_results,
            problem=problem,
            elapsed_time=0.5,
        )

        # 5. Commit if successful
        if decision.status == HaltingStatus.COMMIT:
            success = blackboard.commit_answer(proposal)
            assert success is True

            # Verify answer is committed
            answer = blackboard.get_answer()
            assert answer is not None
            assert answer.proposal_id == proposal.proposal_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
