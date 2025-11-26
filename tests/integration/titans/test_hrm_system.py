"""Integration tests for HRMSystem (T027).

Tests the complete HRM solve loop:
- Simple math problem (should COMMIT)
- Unsolvable problem (should TIMEOUT)
- Low confidence problem (should REFINE then COMMIT)
- Halting within configured budget

Uses real but simple math problems with ground truth.
"""

import pytest
import torch
from unittest.mock import Mock, patch

from src.models.titans.hrm_system import HRMSystem, HRMResult
from src.models.titans.l_engine import SimpleLEngine
from src.models.titans.arbiter import ArbiterVerifier
from src.models.titans.verifiers import MathSymbolicVerifier, MathNumericVerifier
from src.models.titans.blackboard import Blackboard
from src.models.titans.data_model import Problem, HaltingStatus
from src.data.tokenizer import RetNetTokenizer


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = Mock(spec=RetNetTokenizer)
    tokenizer.vocab_size = 50000
    tokenizer.tokenizer = Mock()
    tokenizer.tokenizer.eos_token_id = 0

    def mock_encode(text, **kwargs):
        return [1, 2, 3, 4, 5]

    def mock_decode(tokens, **kwargs):
        # For simple math tests, return answer based on tokens
        return "4"  # Default answer for 2+2

    tokenizer.encode = Mock(side_effect=mock_encode)
    tokenizer.decode = Mock(side_effect=mock_decode)

    return tokenizer


@pytest.fixture
def mock_backbone():
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
def simple_l_engine(mock_backbone, mock_tokenizer):
    """Create a SimpleLEngine with mocked dependencies."""
    engine = SimpleLEngine(
        engine_id="test_engine",
        backbone=mock_backbone,
        tokenizer=mock_tokenizer,
        max_generation_length=50,
        temperature=1.0,
        device="cpu",
    )

    # Mock output head
    def mock_output_head_forward(hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        return torch.randn(batch_size, seq_len, 50000)

    engine.output_head.forward = mock_output_head_forward
    return engine


@pytest.fixture
def arbiter_verifier():
    """Create an ArbiterVerifier with math verifiers."""
    verifiers = [
        MathSymbolicVerifier(tolerance=1e-6),
        MathNumericVerifier(tolerance=1e-6),
    ]
    return ArbiterVerifier(
        verifiers=verifiers,
        confidence_threshold=0.8,
        fast_pass_timeout=0.5,
    )


@pytest.fixture
def hrm_system(simple_l_engine, arbiter_verifier):
    """Create an HRMSystem instance."""
    return HRMSystem(
        engine=simple_l_engine,
        arbiter=arbiter_verifier,
        max_iterations=3,
    )


# ============================================================================
# Test T027-1: Simple Math Problem (Should COMMIT)
# ============================================================================

def test_solve_simple_math_commits(hrm_system, mock_tokenizer):
    """Test solving a simple math problem that should COMMIT.

    Given: 2 + 2 = ?
    Expected: Answer "4", COMMIT status
    """
    # Set tokenizer to return correct answer
    mock_tokenizer.decode = Mock(return_value="4")

    problem = Problem(
        problem_id="simple_math_001",
        domain="math",
        input_text="2 + 2 = ?",
        input_tokens=torch.tensor([1, 2, 3]),
        ground_truth="4",
        time_budget=30.0,
    )

    # Mock high confidence
    with patch.object(hrm_system.engine, 'get_confidence', return_value=0.95):
        result = hrm_system.solve(problem)

    # Should COMMIT with correct answer
    assert result.status == HaltingStatus.COMMIT
    assert result.answer is not None
    assert result.answer_text == "4"
    assert result.iterations == 1  # Should succeed on first try
    assert result.total_time > 0


def test_solve_result_includes_metadata(hrm_system, mock_tokenizer):
    """Test that solve result includes all metadata."""
    mock_tokenizer.decode = Mock(return_value="4")

    problem = Problem(
        problem_id="meta_test_001",
        domain="math",
        input_text="2 + 2 = ?",
        input_tokens=torch.tensor([1, 2, 3]),
        ground_truth="4",
        time_budget=30.0,
    )

    with patch.object(hrm_system.engine, 'get_confidence', return_value=0.95):
        result = hrm_system.solve(problem)

    # Check metadata
    assert isinstance(result, HRMResult)
    assert result.total_time >= 0
    assert result.iterations >= 1
    assert result.proposals_generated >= 1
    assert result.halting_decision is not None
    assert len(result.blackboard_events) > 0


# ============================================================================
# Test T027-2: Unsolvable Problem (Should TIMEOUT)
# ============================================================================

def test_solve_unsolvable_problem_timeouts(hrm_system, mock_tokenizer):
    """Test solving an unsolvable problem (wrong answer, low confidence).

    Given: Problem with wrong answer and low confidence
    Expected: TIMEOUT status (cannot verify)
    """
    # Return wrong answer
    mock_tokenizer.decode = Mock(return_value="wrong_answer")

    problem = Problem(
        problem_id="unsolvable_001",
        domain="math",
        input_text="What is the meaning of life?",
        input_tokens=torch.tensor([1, 2, 3]),
        ground_truth="42",  # Has ground truth but engine returns wrong answer
        time_budget=1.0,  # Short budget
    )

    # Mock low confidence
    with patch.object(hrm_system.engine, 'get_confidence', return_value=0.3):
        result = hrm_system.solve(problem)

    # Should TIMEOUT because cannot verify (wrong answer + low confidence)
    assert result.status == HaltingStatus.TIMEOUT
    # No answer should be committed (verification failed)
    assert result.answer is None or not result.answer.verified


def test_solve_exceeds_time_budget(hrm_system, mock_tokenizer):
    """Test that solve respects time budget."""
    mock_tokenizer.decode = Mock(return_value="wrong")

    problem = Problem(
        problem_id="timeout_001",
        domain="math",
        input_text="2 + 2 = ?",
        input_tokens=torch.tensor([1, 2, 3]),
        ground_truth="4",
        time_budget=0.01,  # Very short budget (10ms)
    )

    # Mock slow generation
    original_solve = hrm_system.engine.solve
    def slow_solve(*args, **kwargs):
        import time
        time.sleep(0.02)  # Exceed budget
        return original_solve(*args, **kwargs)

    with patch.object(hrm_system.engine, 'solve', side_effect=slow_solve):
        result = hrm_system.solve(problem)

    # Should timeout
    assert result.status == HaltingStatus.TIMEOUT
    assert result.total_time >= problem.time_budget


# ============================================================================
# Test T027-3: Low Confidence (Should REFINE then COMMIT)
# ============================================================================

def test_solve_low_confidence_refines(hrm_system, mock_tokenizer):
    """Test solving with low initial confidence, then improving.

    Given: Initial low confidence (0.5), then high confidence (0.9)
    Expected: REFINE on first attempt, COMMIT on second
    """
    mock_tokenizer.decode = Mock(return_value="4")

    problem = Problem(
        problem_id="refine_001",
        domain="math",
        input_text="2 + 2 = ?",
        input_tokens=torch.tensor([1, 2, 3]),
        ground_truth="4",
        time_budget=30.0,
    )

    # Mock confidence progression: low � high
    confidence_values = [0.5, 0.95]  # First low, then high
    confidence_iter = iter(confidence_values)

    def mock_get_confidence(*args, **kwargs):
        return next(confidence_iter, 0.95)

    with patch.object(hrm_system.engine, 'get_confidence', side_effect=mock_get_confidence):
        result = hrm_system.solve(problem)

    # Should eventually COMMIT after refinement
    assert result.status == HaltingStatus.COMMIT
    assert result.answer is not None
    assert result.iterations >= 2  # Required refinement
    assert result.proposals_generated >= 2


# ============================================================================
# Test T027-4: Halting Within Budget
# ============================================================================

def test_solve_respects_max_iterations(hrm_system, mock_tokenizer):
    """Test that solve respects max_iterations limit."""
    # Always return wrong answer with low confidence
    mock_tokenizer.decode = Mock(return_value="wrong")

    problem = Problem(
        problem_id="maxiter_001",
        domain="math",
        input_text="2 + 2 = ?",
        input_tokens=torch.tensor([1, 2, 3]),
        ground_truth="4",
        time_budget=30.0,
    )

    # Mock low confidence always
    with patch.object(hrm_system.engine, 'get_confidence', return_value=0.3):
        result = hrm_system.solve(problem)

    # Should hit max iterations
    assert result.iterations == hrm_system.max_iterations
    assert result.status == HaltingStatus.TIMEOUT  # Max iterations = timeout
    assert result.proposals_generated == hrm_system.max_iterations


def test_solve_halts_early_on_success(hrm_system, mock_tokenizer):
    """Test that solve halts early when successful (doesn't use all iterations)."""
    mock_tokenizer.decode = Mock(return_value="4")

    problem = Problem(
        problem_id="early_halt_001",
        domain="math",
        input_text="2 + 2 = ?",
        input_tokens=torch.tensor([1, 2, 3]),
        ground_truth="4",
        time_budget=30.0,
    )

    # Set max_iterations high
    hrm_system.max_iterations = 10

    with patch.object(hrm_system.engine, 'get_confidence', return_value=0.95):
        result = hrm_system.solve(problem)

    # Should halt early (iteration 1), not use all 10
    assert result.status == HaltingStatus.COMMIT
    assert result.iterations == 1
    assert result.iterations < hrm_system.max_iterations


# ============================================================================
# Test Blackboard Integration
# ============================================================================

def test_solve_writes_to_blackboard(hrm_system, mock_tokenizer):
    """Test that solve writes proposals to blackboard."""
    mock_tokenizer.decode = Mock(return_value="4")

    problem = Problem(
        problem_id="bb_test_001",
        domain="math",
        input_text="2 + 2 = ?",
        input_tokens=torch.tensor([1, 2, 3]),
        ground_truth="4",
        time_budget=30.0,
    )

    with patch.object(hrm_system.engine, 'get_confidence', return_value=0.95):
        result = hrm_system.solve(problem)

    # Check blackboard events
    events = result.blackboard_events
    assert len(events) > 0

    # Should have proposal_written and answer_committed events
    event_types = [e['type'] for e in events]
    assert 'proposal_written' in event_types
    assert 'answer_committed' in event_types


def test_solve_commits_to_blackboard(hrm_system, mock_tokenizer):
    """Test that successful solve commits answer to blackboard."""
    mock_tokenizer.decode = Mock(return_value="4")

    # Use a shared blackboard
    blackboard = Blackboard()
    hrm_system.blackboard = blackboard

    problem = Problem(
        problem_id="commit_test_001",
        domain="math",
        input_text="2 + 2 = ?",
        input_tokens=torch.tensor([1, 2, 3]),
        ground_truth="4",
        time_budget=30.0,
    )

    with patch.object(hrm_system.engine, 'get_confidence', return_value=0.95):
        result = hrm_system.solve(problem)

    # Blackboard should have committed answer
    assert blackboard.get_answer() is not None
    assert blackboard.get_answer().content_text == "4"
    assert blackboard.answer_version > 0


# ============================================================================
# Test Error Handling
# ============================================================================

def test_solve_handles_engine_failure(hrm_system):
    """Test that solve handles engine failures gracefully."""
    problem = Problem(
        problem_id="error_test_001",
        domain="math",
        input_text="2 + 2 = ?",
        input_tokens=torch.tensor([1, 2, 3]),
        ground_truth="4",
        time_budget=30.0,
    )

    # Mock engine to raise exception
    with patch.object(hrm_system.engine, 'solve', side_effect=RuntimeError("Engine error")):
        with pytest.raises(RuntimeError):
            hrm_system.solve(problem)


def test_solve_handles_invalid_problem():
    """Test that HRMSystem validates problem inputs."""
    from src.models.retnet.backbone import RetNetBackbone
    import torch.nn as nn

    mock_backbone = Mock(spec=RetNetBackbone)
    mock_backbone.d_model = 512
    mock_backbone.vocab_size = 50000
    mock_backbone.embed = Mock(spec=nn.Embedding)
    mock_backbone.embed.weight = torch.randn(50000, 512)

    mock_tokenizer = Mock(spec=RetNetTokenizer)
    mock_tokenizer.vocab_size = 50000

    engine = SimpleLEngine("test", mock_backbone, mock_tokenizer, device="cpu")
    arbiter = ArbiterVerifier(verifiers=[MathSymbolicVerifier()], confidence_threshold=0.8)
    hrm = HRMSystem(engine, arbiter)

    # Invalid domain should be caught by Problem validation
    with pytest.raises(ValueError):
        problem = Problem(
            problem_id="invalid",
            domain="invalid_domain",  # Invalid
            input_text="test",
            input_tokens=torch.tensor([1, 2, 3]),
        )


# ============================================================================
# Test Iteration Tracking
# ============================================================================

def test_solve_tracks_iterations_correctly(hrm_system, mock_tokenizer):
    """Test that solve correctly tracks iteration count."""
    mock_tokenizer.decode = Mock(return_value="wrong")

    problem = Problem(
        problem_id="iter_track_001",
        domain="math",
        input_text="2 + 2 = ?",
        input_tokens=torch.tensor([1, 2, 3]),
        ground_truth="4",
        time_budget=30.0,
    )

    hrm_system.max_iterations = 5

    # Always wrong answer, low confidence
    with patch.object(hrm_system.engine, 'get_confidence', return_value=0.3):
        result = hrm_system.solve(problem)

    # Should use all 5 iterations
    assert result.iterations == 5
    assert result.proposals_generated == 5


def test_solve_updates_halting_decision_iteration(hrm_system, mock_tokenizer):
    """Test that halting decision tracks iteration number."""
    mock_tokenizer.decode = Mock(return_value="4")

    problem = Problem(
        problem_id="iter_decision_001",
        domain="math",
        input_text="2 + 2 = ?",
        input_tokens=torch.tensor([1, 2, 3]),
        ground_truth="4",
        time_budget=30.0,
    )

    # Succeed on second iteration
    confidence_values = [0.3, 0.95]
    confidence_iter = iter(confidence_values)

    with patch.object(hrm_system.engine, 'get_confidence', side_effect=lambda *a, **k: next(confidence_iter)):
        result = hrm_system.solve(problem)

    # Halting decision should show iteration 1 (0-indexed)
    assert result.halting_decision.iteration == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
