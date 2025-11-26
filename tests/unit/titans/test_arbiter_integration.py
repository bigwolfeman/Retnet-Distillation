"""Integration tests for Arbiter with real verifiers.

Tests the complete flow:
- ArbiterVerifier + MathSymbolicVerifier + MathNumericVerifier
- Real verification scenarios
- Fast-pass policy with actual verifiers
"""

import time

import pytest
import torch

from src.models.titans.arbiter import ArbiterVerifier
from src.models.titans.data_model import HaltingStatus, Problem, SolutionProposal
from src.models.titans.verifiers import MathNumericVerifier, MathSymbolicVerifier


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def math_problem():
    """Create a real math problem."""
    return Problem(
        problem_id="math_001",
        domain="math",
        input_text="Calculate: 2 + 2",
        input_tokens=torch.tensor([1, 2, 3, 4]),
        expected_format="text",
        time_budget=30.0,
        ground_truth="4",
    )


@pytest.fixture
def real_arbiter():
    """Create arbiter with real math verifiers."""
    symbolic = MathSymbolicVerifier(tolerance=1e-6)
    numeric = MathNumericVerifier(tolerance=1e-6)

    return ArbiterVerifier(
        verifiers=[symbolic, numeric],
        confidence_threshold=0.8,
        fast_pass_timeout=0.5,
        max_verification_timeout=5.0,
        max_workers=2,
    )


# ============================================================================
# Integration Tests
# ============================================================================

class TestArbiterWithRealVerifiers:
    """Test arbiter with real math verifiers."""

    def test_correct_answer_commits(self, real_arbiter, math_problem):
        """Test that correct answer gets verified and commits."""
        proposal = SolutionProposal(
            proposal_id="proposal_correct",
            problem_id=math_problem.problem_id,
            engine_id="engine_1",
            version=1,
            checksum=1000,
            content=torch.tensor([4]),
            content_text="4",  # Correct answer
            raw_confidence=0.9,
            calibrated_confidence=0.9,
            timestamp=time.time(),
            latency=0.1,
        )

        # Verify
        results = real_arbiter.verify_proposal(proposal, math_problem)

        # Both verifiers should pass
        assert len(results) == 2
        assert all(r.passed for r in results)

        # Evaluate halting
        decision = real_arbiter.evaluate_halting(
            proposal=proposal,
            verification_results=results,
            problem=math_problem,
            elapsed_time=5.0,
        )

        assert decision.status == HaltingStatus.COMMIT
        assert decision.selected_proposal == proposal.proposal_id
        assert decision.confidence_met is True
        assert decision.verification_passed is True

    def test_incorrect_answer_refines(self, real_arbiter, math_problem):
        """Test that incorrect answer gets rejected and refines."""
        proposal = SolutionProposal(
            proposal_id="proposal_wrong",
            problem_id=math_problem.problem_id,
            engine_id="engine_1",
            version=1,
            checksum=1000,
            content=torch.tensor([5]),
            content_text="5",  # Wrong answer
            raw_confidence=0.9,
            calibrated_confidence=0.9,
            timestamp=time.time(),
            latency=0.1,
        )

        # Verify
        results = real_arbiter.verify_proposal(proposal, math_problem)

        # Both verifiers should fail
        assert len(results) == 2
        assert all(not r.passed for r in results)

        # Evaluate halting
        decision = real_arbiter.evaluate_halting(
            proposal=proposal,
            verification_results=results,
            problem=math_problem,
            elapsed_time=5.0,
        )

        assert decision.status == HaltingStatus.REFINE
        assert decision.selected_proposal is None
        assert decision.verification_passed is False

    def test_low_confidence_refines(self, real_arbiter, math_problem):
        """Test that low confidence causes REFINE even if answer is correct."""
        proposal = SolutionProposal(
            proposal_id="proposal_low_conf",
            problem_id=math_problem.problem_id,
            engine_id="engine_1",
            version=1,
            checksum=1000,
            content=torch.tensor([4]),
            content_text="4",  # Correct answer
            raw_confidence=0.5,
            calibrated_confidence=0.5,  # Low confidence
            timestamp=time.time(),
            latency=0.1,
        )

        # Verify
        results = real_arbiter.verify_proposal(proposal, math_problem)

        # Verifiers pass
        assert all(r.passed for r in results)

        # But halting should REFINE due to low confidence
        decision = real_arbiter.evaluate_halting(
            proposal=proposal,
            verification_results=results,
            problem=math_problem,
            elapsed_time=5.0,
        )

        assert decision.status == HaltingStatus.REFINE
        assert decision.confidence_met is False
        assert decision.verification_passed is True

    def test_timeout_with_correct_answer(self, real_arbiter, math_problem):
        """Test TIMEOUT status when budget exceeded but answer correct."""
        proposal = SolutionProposal(
            proposal_id="proposal_timeout",
            problem_id=math_problem.problem_id,
            engine_id="engine_1",
            version=1,
            checksum=1000,
            content=torch.tensor([4]),
            content_text="4",
            raw_confidence=0.9,
            calibrated_confidence=0.9,
            timestamp=time.time(),
            latency=0.1,
        )

        # Verify
        results = real_arbiter.verify_proposal(proposal, math_problem)

        # Evaluate with budget exceeded
        decision = real_arbiter.evaluate_halting(
            proposal=proposal,
            verification_results=results,
            problem=math_problem,
            elapsed_time=35.0,  # > time_budget
        )

        assert decision.status == HaltingStatus.TIMEOUT
        assert decision.budget_exceeded is True
        # Still returns proposal if verified
        assert decision.selected_proposal == proposal.proposal_id

    def test_parallel_verification_performance(self, real_arbiter, math_problem):
        """Test that verifiers run in parallel (performance check)."""
        proposal = SolutionProposal(
            proposal_id="proposal_perf",
            problem_id=math_problem.problem_id,
            engine_id="engine_1",
            version=1,
            checksum=1000,
            content=torch.tensor([4]),
            content_text="4",
            raw_confidence=0.9,
            calibrated_confidence=0.9,
            timestamp=time.time(),
            latency=0.1,
        )

        # Measure verification time
        start = time.time()
        results = real_arbiter.verify_proposal(proposal, math_problem)
        elapsed = time.time() - start

        # Should complete quickly in parallel
        assert len(results) == 2
        assert elapsed < 1.0, f"Verification took {elapsed:.3f}s (should be < 1s)"


class TestFastPassWithRealVerifiers:
    """Test fast-pass policy with real verifiers."""

    def test_fast_pass_commits_correct_answer(self, real_arbiter, math_problem):
        """Test fast-pass commits correct answer immediately."""
        start_time = time.time()

        proposals = [
            SolutionProposal(
                proposal_id="proposal_1",
                problem_id=math_problem.problem_id,
                engine_id="engine_1",
                version=1,
                checksum=1000,
                content=torch.tensor([4]),
                content_text="4",  # Correct
                raw_confidence=0.9,
                calibrated_confidence=0.9,
                timestamp=start_time,
                latency=0.1,
            ),
            SolutionProposal(
                proposal_id="proposal_2",
                problem_id=math_problem.problem_id,
                engine_id="engine_2",
                version=2,
                checksum=2000,
                content=torch.tensor([5]),
                content_text="5",  # Wrong
                raw_confidence=0.95,
                calibrated_confidence=0.95,
                timestamp=start_time + 0.01,
                latency=0.1,
            ),
        ]

        selected, decision = real_arbiter.verify_and_decide_fast_pass(
            proposals=proposals,
            problem=math_problem,
            start_time=start_time,
        )

        # Should select first correct proposal
        assert selected.proposal_id == "proposal_1"
        assert decision.status == HaltingStatus.COMMIT
        assert selected.verified is True

    def test_fast_pass_picks_best_when_none_verified_quickly(self, real_arbiter, math_problem):
        """Test fast-pass picks best by confidence when verification is slow."""
        start_time = time.time()

        # Both wrong, so fast-pass won't commit either
        proposals = [
            SolutionProposal(
                proposal_id="proposal_low",
                problem_id=math_problem.problem_id,
                engine_id="engine_1",
                version=1,
                checksum=1000,
                content=torch.tensor([5]),
                content_text="5",  # Wrong
                raw_confidence=0.6,
                calibrated_confidence=0.6,
                timestamp=start_time,
                latency=0.1,
            ),
            SolutionProposal(
                proposal_id="proposal_high",
                problem_id=math_problem.problem_id,
                engine_id="engine_2",
                version=2,
                checksum=2000,
                content=torch.tensor([3]),
                content_text="3",  # Wrong
                raw_confidence=0.95,
                calibrated_confidence=0.95,
                timestamp=start_time + 0.01,
                latency=0.1,
            ),
        ]

        selected, decision = real_arbiter.verify_and_decide_fast_pass(
            proposals=proposals,
            problem=math_problem,
            start_time=start_time,
        )

        # Should pick highest confidence
        assert selected.proposal_id == "proposal_high"
        # But should REFINE since verification failed
        assert decision.status == HaltingStatus.REFINE


class TestComplexMathProblems:
    """Test arbiter with more complex math problems."""

    def test_algebraic_expression(self, real_arbiter):
        """Test verification of algebraic expressions."""
        problem = Problem(
            problem_id="algebra_001",
            domain="math",
            input_text="Simplify: x^2 - 1",
            input_tokens=torch.tensor([1, 2, 3]),
            expected_format="text",
            time_budget=30.0,
            ground_truth="(x-1)*(x+1)",
        )

        proposal = SolutionProposal(
            proposal_id="algebra_proposal",
            problem_id=problem.problem_id,
            engine_id="engine_1",
            version=1,
            checksum=1000,
            content=torch.tensor([1]),
            content_text="(x-1)*(x+1)",  # Correct
            raw_confidence=0.9,
            calibrated_confidence=0.9,
            timestamp=time.time(),
            latency=0.1,
        )

        results = real_arbiter.verify_proposal(proposal, problem)

        # Symbolic verifier should pass
        symbolic_result = next(r for r in results if r.verifier_type.value == "math_symbolic")
        assert symbolic_result.passed is True

    def test_numeric_approximation(self, real_arbiter):
        """Test verification of numeric approximations."""
        problem = Problem(
            problem_id="numeric_001",
            domain="math",
            input_text="Calculate: pi to 5 decimals",
            input_tokens=torch.tensor([1, 2, 3]),
            expected_format="text",
            time_budget=30.0,
            ground_truth="3.14159",
        )

        proposal = SolutionProposal(
            proposal_id="numeric_proposal",
            problem_id=problem.problem_id,
            engine_id="engine_1",
            version=1,
            checksum=1000,
            content=torch.tensor([1]),
            content_text="3.14159",  # Correct
            raw_confidence=0.9,
            calibrated_confidence=0.9,
            timestamp=time.time(),
            latency=0.1,
        )

        results = real_arbiter.verify_proposal(proposal, problem)

        # Both verifiers should pass
        assert all(r.passed for r in results)

        decision = real_arbiter.evaluate_halting(
            proposal=proposal,
            verification_results=results,
            problem=problem,
            elapsed_time=5.0,
        )

        assert decision.status == HaltingStatus.COMMIT
