"""Unit tests for Titan-HRM Arbiter (T025).

Tests coverage:
- T023: ArbiterVerifier.verify_proposal()
  - Parallel verification with multiple verifiers
  - Timeout handling
  - Error handling
- T024: ArbiterVerifier.evaluate_halting()
  - COMMIT: confidence ≥ τ AND all verifiers pass
  - REFINE: confidence < τ OR any verifier fails
  - TIMEOUT: budget exceeded
- Fast-pass policy
  - Commits first verified proposal within 500ms
  - Falls back to best if timeout expires
- Mock verifiers for controlled test scenarios

All tests use fixtures for clean setup and teardown.
"""

import time
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from src.models.titans.arbiter import ArbiterVerifier
from src.models.titans.data_model import (
    HaltingDecision,
    HaltingStatus,
    Problem,
    SolutionProposal,
    VerificationResult,
    VerifierType,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_problem():
    """Create a mock Problem for testing."""
    return Problem(
        problem_id="test_problem_001",
        domain="math",
        input_text="What is 2 + 2?",
        input_tokens=torch.tensor([1, 2, 3, 4]),
        expected_format="text",
        time_budget=30.0,
        ground_truth="4",
    )


@pytest.fixture
def mock_proposal():
    """Create a mock SolutionProposal for testing."""
    return SolutionProposal(
        proposal_id="test_proposal_001",
        problem_id="test_problem_001",
        engine_id="engine_1",
        version=1,
        checksum=12345,
        content=torch.tensor([5, 6, 7, 8]),
        content_text="4",
        raw_confidence=0.9,
        calibrated_confidence=0.85,
        timestamp=time.time(),
        latency=0.5,
    )


@pytest.fixture
def mock_passing_verifier():
    """Create a mock verifier that always passes."""
    verifier = Mock()
    verifier.verify = Mock(return_value=VerificationResult(
        verifier_type=VerifierType.MATH_SYMBOLIC,
        proposal_id="test_proposal_001",
        passed=True,
        score=1.0,
        execution_time=0.1,
    ))
    return verifier


@pytest.fixture
def mock_failing_verifier():
    """Create a mock verifier that always fails."""
    verifier = Mock()
    verifier.verify = Mock(return_value=VerificationResult(
        verifier_type=VerifierType.MATH_NUMERIC,
        proposal_id="test_proposal_001",
        passed=False,
        score=0.0,
        error_message="Answer incorrect",
        execution_time=0.1,
    ))
    return verifier


@pytest.fixture
def mock_slow_verifier():
    """Create a mock verifier that is slow but passes."""
    def slow_verify(*args, **kwargs):
        time.sleep(0.6)  # Slower than fast-pass timeout
        return VerificationResult(
            verifier_type=VerifierType.MATH_SYMBOLIC,
            proposal_id=kwargs.get('proposal_id', 'test'),
            passed=True,
            score=1.0,
            execution_time=0.6,
        )

    verifier = Mock()
    verifier.verify = Mock(side_effect=slow_verify)
    return verifier


# ============================================================================
# T023: ArbiterVerifier.verify_proposal() Tests
# ============================================================================

class TestArbiterVerifierInit:
    """Tests for ArbiterVerifier initialization."""

    def test_init_valid_params(self, mock_passing_verifier):
        """Test initialization with valid parameters."""
        arbiter = ArbiterVerifier(
            verifiers=[mock_passing_verifier],
            confidence_threshold=0.8,
            fast_pass_timeout=0.5,
            max_verification_timeout=5.0,
            max_workers=2,
        )

        assert arbiter.verifiers == [mock_passing_verifier]
        assert arbiter.confidence_threshold == 0.8
        assert arbiter.fast_pass_timeout == 0.5
        assert arbiter.max_verification_timeout == 5.0
        assert arbiter.max_workers == 2

    def test_init_default_params(self, mock_passing_verifier):
        """Test initialization with default parameters."""
        arbiter = ArbiterVerifier(verifiers=[mock_passing_verifier])

        assert arbiter.confidence_threshold == 0.8
        assert arbiter.fast_pass_timeout == 0.5
        assert arbiter.max_verification_timeout == 5.0
        assert arbiter.max_workers == 2

    def test_init_empty_verifiers(self):
        """Test initialization with empty verifiers list raises error."""
        with pytest.raises(ValueError, match="verifiers list cannot be empty"):
            ArbiterVerifier(verifiers=[])

    def test_init_invalid_confidence_threshold(self, mock_passing_verifier):
        """Test initialization with invalid confidence threshold."""
        with pytest.raises(ValueError, match="confidence_threshold must be in"):
            ArbiterVerifier(verifiers=[mock_passing_verifier], confidence_threshold=1.5)

        with pytest.raises(ValueError, match="confidence_threshold must be in"):
            ArbiterVerifier(verifiers=[mock_passing_verifier], confidence_threshold=-0.1)

    def test_init_invalid_fast_pass_timeout(self, mock_passing_verifier):
        """Test initialization with invalid fast-pass timeout."""
        with pytest.raises(ValueError, match="fast_pass_timeout must be > 0"):
            ArbiterVerifier(verifiers=[mock_passing_verifier], fast_pass_timeout=0)

        with pytest.raises(ValueError, match="fast_pass_timeout must be > 0"):
            ArbiterVerifier(verifiers=[mock_passing_verifier], fast_pass_timeout=-1)

    def test_init_invalid_max_workers(self, mock_passing_verifier):
        """Test initialization with invalid max workers."""
        with pytest.raises(ValueError, match="max_workers must be > 0"):
            ArbiterVerifier(verifiers=[mock_passing_verifier], max_workers=0)


class TestVerifyProposal:
    """Tests for verify_proposal method."""

    def test_verify_proposal_single_verifier_pass(self, mock_passing_verifier, mock_proposal, mock_problem):
        """Test verification with single passing verifier."""
        arbiter = ArbiterVerifier(verifiers=[mock_passing_verifier])

        results = arbiter.verify_proposal(mock_proposal, mock_problem)

        assert len(results) == 1
        assert results[0].passed is True
        assert results[0].score == 1.0
        assert results[0].verifier_type == VerifierType.MATH_SYMBOLIC

        # Verify that the verifier was called with correct args
        mock_passing_verifier.verify.assert_called_once_with(
            proposal_id=mock_proposal.proposal_id,
            answer=mock_proposal.content_text,
            ground_truth=mock_problem.ground_truth,
        )

    def test_verify_proposal_single_verifier_fail(self, mock_failing_verifier, mock_proposal, mock_problem):
        """Test verification with single failing verifier."""
        arbiter = ArbiterVerifier(verifiers=[mock_failing_verifier])

        results = arbiter.verify_proposal(mock_proposal, mock_problem)

        assert len(results) == 1
        assert results[0].passed is False
        assert results[0].score == 0.0
        assert results[0].error_message == "Answer incorrect"

    def test_verify_proposal_multiple_verifiers_all_pass(
        self, mock_passing_verifier, mock_proposal, mock_problem
    ):
        """Test verification with multiple passing verifiers."""
        # Create two passing verifiers
        verifier1 = Mock()
        verifier1.verify = Mock(return_value=VerificationResult(
            verifier_type=VerifierType.MATH_SYMBOLIC,
            proposal_id=mock_proposal.proposal_id,
            passed=True,
            score=1.0,
            execution_time=0.1,
        ))

        verifier2 = Mock()
        verifier2.verify = Mock(return_value=VerificationResult(
            verifier_type=VerifierType.MATH_NUMERIC,
            proposal_id=mock_proposal.proposal_id,
            passed=True,
            score=1.0,
            execution_time=0.1,
        ))

        arbiter = ArbiterVerifier(verifiers=[verifier1, verifier2])

        results = arbiter.verify_proposal(mock_proposal, mock_problem)

        assert len(results) == 2
        assert all(r.passed for r in results)
        assert results[0].verifier_type == VerifierType.MATH_SYMBOLIC
        assert results[1].verifier_type == VerifierType.MATH_NUMERIC

    def test_verify_proposal_multiple_verifiers_mixed_results(
        self, mock_passing_verifier, mock_failing_verifier, mock_proposal, mock_problem
    ):
        """Test verification with mixed pass/fail results."""
        arbiter = ArbiterVerifier(verifiers=[mock_passing_verifier, mock_failing_verifier])

        results = arbiter.verify_proposal(mock_proposal, mock_problem)

        assert len(results) == 2
        assert results[0].passed is True
        assert results[1].passed is False

    def test_verify_proposal_missing_ground_truth(self, mock_passing_verifier, mock_proposal):
        """Test verification fails when ground truth is missing."""
        problem_no_truth = Problem(
            problem_id="test",
            domain="math",
            input_text="What is 2 + 2?",
            input_tokens=torch.tensor([1, 2, 3]),
            ground_truth=None,  # Missing
        )

        arbiter = ArbiterVerifier(verifiers=[mock_passing_verifier])

        with pytest.raises(ValueError, match="ground_truth is required"):
            arbiter.verify_proposal(mock_proposal, problem_no_truth)

    def test_verify_proposal_parallel_execution(self, mock_proposal, mock_problem):
        """Test that verifiers run in parallel (performance check)."""
        # Create verifiers that sleep briefly
        def slow_verify(*args, **kwargs):
            time.sleep(0.2)
            return VerificationResult(
                verifier_type=VerifierType.MATH_SYMBOLIC,
                proposal_id=kwargs.get('proposal_id', 'test'),
                passed=True,
                score=1.0,
                execution_time=0.2,
            )

        verifier1 = Mock()
        verifier1.verify = Mock(side_effect=slow_verify)

        verifier2 = Mock()
        verifier2.verify = Mock(side_effect=slow_verify)

        arbiter = ArbiterVerifier(verifiers=[verifier1, verifier2], max_workers=2)

        start = time.time()
        results = arbiter.verify_proposal(mock_proposal, mock_problem)
        elapsed = time.time() - start

        # If parallel, should take ~0.2s (not 0.4s)
        # Allow some overhead
        assert elapsed < 0.35, f"Expected parallel execution (<0.35s), got {elapsed:.3f}s"
        assert len(results) == 2

    def test_verify_proposal_verifier_exception(self, mock_proposal, mock_problem):
        """Test handling of verifier exceptions."""
        error_verifier = Mock()
        error_verifier.verify = Mock(side_effect=RuntimeError("Verifier crashed"))
        error_verifier.__class__.__name__ = "MathSymbolicVerifier"

        arbiter = ArbiterVerifier(verifiers=[error_verifier])

        results = arbiter.verify_proposal(mock_proposal, mock_problem)

        assert len(results) == 1
        assert results[0].passed is False
        assert "Verifier exception" in results[0].error_message
        assert "Verifier crashed" in results[0].error_message


# ============================================================================
# T024: ArbiterVerifier.evaluate_halting() Tests
# ============================================================================

class TestEvaluateHalting:
    """Tests for evaluate_halting method."""

    def test_evaluate_halting_commit_success(self, mock_passing_verifier, mock_proposal, mock_problem):
        """Test COMMIT when confidence ≥ τ and all verifiers pass."""
        arbiter = ArbiterVerifier(
            verifiers=[mock_passing_verifier],
            confidence_threshold=0.8,
        )

        # High confidence proposal
        mock_proposal.calibrated_confidence = 0.85

        # All verifiers pass
        verification_results = [
            VerificationResult(
                verifier_type=VerifierType.MATH_SYMBOLIC,
                proposal_id=mock_proposal.proposal_id,
                passed=True,
                score=1.0,
                execution_time=0.1,
            )
        ]

        decision = arbiter.evaluate_halting(
            proposal=mock_proposal,
            verification_results=verification_results,
            problem=mock_problem,
            elapsed_time=5.0,
        )

        assert decision.status == HaltingStatus.COMMIT
        assert decision.selected_proposal == mock_proposal.proposal_id
        assert decision.confidence_met is True
        assert decision.verification_passed is True
        assert decision.budget_exceeded is False
        assert decision.confidence_threshold == 0.8

    def test_evaluate_halting_refine_low_confidence(self, mock_passing_verifier, mock_proposal, mock_problem):
        """Test REFINE when confidence < τ."""
        arbiter = ArbiterVerifier(
            verifiers=[mock_passing_verifier],
            confidence_threshold=0.8,
        )

        # Low confidence
        mock_proposal.calibrated_confidence = 0.6

        # Verifiers pass
        verification_results = [
            VerificationResult(
                verifier_type=VerifierType.MATH_SYMBOLIC,
                proposal_id=mock_proposal.proposal_id,
                passed=True,
                score=1.0,
                execution_time=0.1,
            )
        ]

        decision = arbiter.evaluate_halting(
            proposal=mock_proposal,
            verification_results=verification_results,
            problem=mock_problem,
            elapsed_time=5.0,
        )

        assert decision.status == HaltingStatus.REFINE
        assert decision.selected_proposal is None
        assert decision.confidence_met is False
        assert decision.verification_passed is True
        assert decision.budget_exceeded is False

    def test_evaluate_halting_refine_verification_failed(self, mock_failing_verifier, mock_proposal, mock_problem):
        """Test REFINE when verification fails."""
        arbiter = ArbiterVerifier(
            verifiers=[mock_failing_verifier],
            confidence_threshold=0.8,
        )

        # High confidence
        mock_proposal.calibrated_confidence = 0.9

        # Verifiers fail
        verification_results = [
            VerificationResult(
                verifier_type=VerifierType.MATH_SYMBOLIC,
                proposal_id=mock_proposal.proposal_id,
                passed=False,
                score=0.0,
                error_message="Incorrect answer",
                execution_time=0.1,
            )
        ]

        decision = arbiter.evaluate_halting(
            proposal=mock_proposal,
            verification_results=verification_results,
            problem=mock_problem,
            elapsed_time=5.0,
        )

        assert decision.status == HaltingStatus.REFINE
        assert decision.selected_proposal is None
        assert decision.confidence_met is True
        assert decision.verification_passed is False
        assert decision.budget_exceeded is False

    def test_evaluate_halting_timeout_budget_exceeded(self, mock_passing_verifier, mock_proposal, mock_problem):
        """Test TIMEOUT when budget exceeded."""
        arbiter = ArbiterVerifier(
            verifiers=[mock_passing_verifier],
            confidence_threshold=0.8,
        )

        # High confidence
        mock_proposal.calibrated_confidence = 0.9

        # Verifiers pass
        verification_results = [
            VerificationResult(
                verifier_type=VerifierType.MATH_SYMBOLIC,
                proposal_id=mock_proposal.proposal_id,
                passed=True,
                score=1.0,
                execution_time=0.1,
            )
        ]

        # Budget exceeded (problem.time_budget = 30.0)
        decision = arbiter.evaluate_halting(
            proposal=mock_proposal,
            verification_results=verification_results,
            problem=mock_problem,
            elapsed_time=35.0,  # > 30.0
        )

        assert decision.status == HaltingStatus.TIMEOUT
        assert decision.budget_exceeded is True
        # When verified and budget exceeded, can still return proposal
        assert decision.selected_proposal == mock_proposal.proposal_id

    def test_evaluate_halting_timeout_unverified(self, mock_failing_verifier, mock_proposal, mock_problem):
        """Test TIMEOUT when budget exceeded and verification failed."""
        arbiter = ArbiterVerifier(
            verifiers=[mock_failing_verifier],
            confidence_threshold=0.8,
        )

        mock_proposal.calibrated_confidence = 0.9

        verification_results = [
            VerificationResult(
                verifier_type=VerifierType.MATH_SYMBOLIC,
                proposal_id=mock_proposal.proposal_id,
                passed=False,
                score=0.0,
                execution_time=0.1,
            )
        ]

        decision = arbiter.evaluate_halting(
            proposal=mock_proposal,
            verification_results=verification_results,
            problem=mock_problem,
            elapsed_time=35.0,
        )

        assert decision.status == HaltingStatus.TIMEOUT
        assert decision.budget_exceeded is True
        assert decision.verification_passed is False
        # No proposal selected when verification failed
        assert decision.selected_proposal is None

    def test_evaluate_halting_multiple_verifiers_all_must_pass(self, mock_proposal, mock_problem):
        """Test that ALL verifiers must pass for verification_passed=True."""
        arbiter = ArbiterVerifier(
            verifiers=[Mock(), Mock()],  # Dummy verifiers
            confidence_threshold=0.8,
        )

        mock_proposal.calibrated_confidence = 0.9

        # One passes, one fails
        verification_results = [
            VerificationResult(
                verifier_type=VerifierType.MATH_SYMBOLIC,
                proposal_id=mock_proposal.proposal_id,
                passed=True,
                score=1.0,
                execution_time=0.1,
            ),
            VerificationResult(
                verifier_type=VerifierType.MATH_NUMERIC,
                proposal_id=mock_proposal.proposal_id,
                passed=False,
                score=0.0,
                execution_time=0.1,
            ),
        ]

        decision = arbiter.evaluate_halting(
            proposal=mock_proposal,
            verification_results=verification_results,
            problem=mock_problem,
            elapsed_time=5.0,
        )

        assert decision.status == HaltingStatus.REFINE
        assert decision.verification_passed is False

    def test_evaluate_halting_empty_verification_results(self, mock_proposal, mock_problem):
        """Test behavior with empty verification results."""
        arbiter = ArbiterVerifier(
            verifiers=[Mock()],
            confidence_threshold=0.8,
        )

        mock_proposal.calibrated_confidence = 0.9

        decision = arbiter.evaluate_halting(
            proposal=mock_proposal,
            verification_results=[],  # Empty
            problem=mock_problem,
            elapsed_time=5.0,
        )

        # Empty results → verification_passed = False
        assert decision.verification_passed is False
        assert decision.status == HaltingStatus.REFINE


# ============================================================================
# Fast-Pass Policy Tests
# ============================================================================

class TestFastPassPolicy:
    """Tests for fast-pass policy."""

    def test_fast_pass_commits_first_verified(self, mock_passing_verifier, mock_problem):
        """Test fast-pass commits first verified proposal within timeout."""
        arbiter = ArbiterVerifier(
            verifiers=[mock_passing_verifier],
            confidence_threshold=0.8,
            fast_pass_timeout=0.5,
        )

        # Create proposals
        start_time = time.time()
        proposals = [
            SolutionProposal(
                proposal_id=f"proposal_{i}",
                problem_id=mock_problem.problem_id,
                engine_id=f"engine_{i}",
                version=i,
                checksum=i * 1000,
                content=torch.tensor([i]),
                content_text="4",
                raw_confidence=0.8 + i * 0.01,
                calibrated_confidence=0.8 + i * 0.01,
                timestamp=start_time + i * 0.01,
                latency=0.1,
            )
            for i in range(3)
        ]

        selected, decision = arbiter.verify_and_decide_fast_pass(
            proposals=proposals,
            problem=mock_problem,
            start_time=start_time,
        )

        # Should select first proposal (oldest timestamp)
        assert selected.proposal_id == "proposal_0"
        assert decision.status == HaltingStatus.COMMIT
        assert selected.verified is True

    def test_fast_pass_picks_best_after_timeout(self, mock_slow_verifier, mock_problem):
        """Test fast-pass picks best by confidence after timeout."""
        arbiter = ArbiterVerifier(
            verifiers=[mock_slow_verifier],
            confidence_threshold=0.8,
            fast_pass_timeout=0.3,  # Short timeout
        )

        start_time = time.time()
        proposals = [
            SolutionProposal(
                proposal_id="proposal_low",
                problem_id=mock_problem.problem_id,
                engine_id="engine_1",
                version=1,
                checksum=1000,
                content=torch.tensor([1]),
                content_text="4",
                raw_confidence=0.7,
                calibrated_confidence=0.7,
                timestamp=start_time,
                latency=0.1,
            ),
            SolutionProposal(
                proposal_id="proposal_high",
                problem_id=mock_problem.problem_id,
                engine_id="engine_2",
                version=2,
                checksum=2000,
                content=torch.tensor([2]),
                content_text="4",
                raw_confidence=0.95,
                calibrated_confidence=0.95,
                timestamp=start_time + 0.01,
                latency=0.1,
            ),
        ]

        selected, decision = arbiter.verify_and_decide_fast_pass(
            proposals=proposals,
            problem=mock_problem,
            start_time=start_time,
        )

        # Should select proposal_high (highest confidence)
        assert selected.proposal_id == "proposal_high"
        assert selected.calibrated_confidence == 0.95

    def test_fast_pass_rejects_low_confidence(self, mock_passing_verifier, mock_problem):
        """Test fast-pass rejects proposals below confidence threshold."""
        arbiter = ArbiterVerifier(
            verifiers=[mock_passing_verifier],
            confidence_threshold=0.8,
            fast_pass_timeout=0.5,
        )

        start_time = time.time()
        proposals = [
            SolutionProposal(
                proposal_id="proposal_low",
                problem_id=mock_problem.problem_id,
                engine_id="engine_1",
                version=1,
                checksum=1000,
                content=torch.tensor([1]),
                content_text="4",
                raw_confidence=0.6,
                calibrated_confidence=0.6,  # Below threshold
                timestamp=start_time,
                latency=0.1,
            ),
        ]

        selected, decision = arbiter.verify_and_decide_fast_pass(
            proposals=proposals,
            problem=mock_problem,
            start_time=start_time,
        )

        # Should not commit (low confidence)
        assert decision.status == HaltingStatus.REFINE
        assert decision.confidence_met is False

    def test_fast_pass_empty_proposals(self, mock_passing_verifier, mock_problem):
        """Test fast-pass with empty proposals list."""
        arbiter = ArbiterVerifier(
            verifiers=[mock_passing_verifier],
            confidence_threshold=0.8,
            fast_pass_timeout=0.5,
        )

        selected, decision = arbiter.verify_and_decide_fast_pass(
            proposals=[],
            problem=mock_problem,
            start_time=time.time(),
        )

        assert selected is None
        assert decision is None

    def test_fast_pass_updates_proposal_verification_state(self, mock_passing_verifier, mock_problem):
        """Test that fast-pass updates proposal.verified and verification_results."""
        arbiter = ArbiterVerifier(
            verifiers=[mock_passing_verifier],
            confidence_threshold=0.8,
            fast_pass_timeout=0.5,
        )

        start_time = time.time()
        proposal = SolutionProposal(
            proposal_id="test_proposal",
            problem_id=mock_problem.problem_id,
            engine_id="engine_1",
            version=1,
            checksum=1000,
            content=torch.tensor([1]),
            content_text="4",
            raw_confidence=0.9,
            calibrated_confidence=0.9,
            timestamp=start_time,
            latency=0.1,
            verified=False,  # Initially unverified
            verification_results=None,
        )

        selected, decision = arbiter.verify_and_decide_fast_pass(
            proposals=[proposal],
            problem=mock_problem,
            start_time=start_time,
        )

        # Should be verified now
        assert selected.verified is True
        assert selected.verification_results is not None
        assert len(selected.verification_results) > 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestArbiterIntegration:
    """Integration tests combining verify + evaluate."""

    def test_full_workflow_commit(self, mock_passing_verifier, mock_proposal, mock_problem):
        """Test full workflow: verify → evaluate → COMMIT."""
        arbiter = ArbiterVerifier(
            verifiers=[mock_passing_verifier],
            confidence_threshold=0.8,
        )

        # High confidence proposal
        mock_proposal.calibrated_confidence = 0.9

        # Step 1: Verify
        verification_results = arbiter.verify_proposal(mock_proposal, mock_problem)

        # Step 2: Evaluate
        decision = arbiter.evaluate_halting(
            proposal=mock_proposal,
            verification_results=verification_results,
            problem=mock_problem,
            elapsed_time=5.0,
        )

        # Should commit
        assert decision.status == HaltingStatus.COMMIT
        assert all(r.passed for r in verification_results)

    def test_full_workflow_refine(self, mock_failing_verifier, mock_proposal, mock_problem):
        """Test full workflow: verify → evaluate → REFINE."""
        arbiter = ArbiterVerifier(
            verifiers=[mock_failing_verifier],
            confidence_threshold=0.8,
        )

        mock_proposal.calibrated_confidence = 0.9

        verification_results = arbiter.verify_proposal(mock_proposal, mock_problem)
        decision = arbiter.evaluate_halting(
            proposal=mock_proposal,
            verification_results=verification_results,
            problem=mock_problem,
            elapsed_time=5.0,
        )

        assert decision.status == HaltingStatus.REFINE
        assert not all(r.passed for r in verification_results)

    def test_full_workflow_timeout(self, mock_passing_verifier, mock_proposal, mock_problem):
        """Test full workflow: verify → evaluate → TIMEOUT."""
        arbiter = ArbiterVerifier(
            verifiers=[mock_passing_verifier],
            confidence_threshold=0.8,
        )

        mock_proposal.calibrated_confidence = 0.9

        verification_results = arbiter.verify_proposal(mock_proposal, mock_problem)
        decision = arbiter.evaluate_halting(
            proposal=mock_proposal,
            verification_results=verification_results,
            problem=mock_problem,
            elapsed_time=35.0,  # > time_budget
        )

        assert decision.status == HaltingStatus.TIMEOUT
        assert decision.budget_exceeded is True
