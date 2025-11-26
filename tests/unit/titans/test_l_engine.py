"""Unit tests for SimpleLEngine implementation.

Tests per tasks.md T018:
- Test solve() returns a valid SolutionProposal with all required fields
- Test confidence is in [0,1] range
- Test proposal includes content_text and tokens
- Mock the RetNet backbone to avoid requiring trained weights
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch

from src.models.titans.l_engine import SimpleLEngine
from src.models.titans.data_model import Problem, SolutionProposal
from src.models.titans.blackboard import Blackboard
from src.models.retnet.backbone import RetNetBackbone, RetNetOutputHead
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

    # Mock encode to return simple token sequence
    def mock_encode(text, **kwargs):
        return [1, 2, 3, 4, 5]

    # Mock decode to return sample text
    def mock_decode(tokens, **kwargs):
        return "42"

    tokenizer.encode = Mock(side_effect=mock_encode)
    tokenizer.decode = Mock(side_effect=mock_decode)

    return tokenizer


@pytest.fixture
def mock_backbone():
    """Create a mock RetNet backbone."""
    backbone = Mock(spec=RetNetBackbone)
    backbone.d_model = 512
    backbone.vocab_size = 50000

    # Mock embed layer
    backbone.embed = Mock(spec=nn.Embedding)
    backbone.embed.weight = torch.randn(50000, 512)

    # Mock forward_train to return dummy hidden states
    def mock_forward_train(input_ids, segment_ids=None):
        batch_size, seq_len = input_ids.shape
        return torch.randn(batch_size, seq_len, 512)

    backbone.forward_train = Mock(side_effect=mock_forward_train)

    return backbone


@pytest.fixture
def sample_problem():
    """Create a sample problem."""
    return Problem(
        problem_id="test_prob_001",
        domain="math",
        input_text="What is 2 + 2?",
        input_tokens=torch.tensor([1, 2, 3, 4, 5]),
        difficulty=0.5,
        expected_format="text",
        time_budget=30.0,
    )


@pytest.fixture
def l_engine(mock_backbone, mock_tokenizer):
    """Create a SimpleLEngine with mocked dependencies."""
    engine = SimpleLEngine(
        engine_id="test_engine",
        backbone=mock_backbone,
        tokenizer=mock_tokenizer,
        max_generation_length=50,
        temperature=1.0,
        top_p=0.95,
        device="cpu",
    )

    # Mock the output head's forward method to avoid computation
    # We can't replace output_head itself (it's a nn.Module), but we can mock its __call__
    def mock_output_head_forward(hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        return torch.randn(batch_size, seq_len, 50000)

    engine.output_head.forward = mock_output_head_forward

    return engine


@pytest.fixture
def blackboard():
    """Create a fresh blackboard."""
    return Blackboard()


# ============================================================================
# Test solve() Returns Valid SolutionProposal
# ============================================================================

def test_solve_returns_solution_proposal(l_engine, sample_problem):
    """Test that solve() returns a SolutionProposal instance."""
    proposal = l_engine.solve(sample_problem)

    assert isinstance(proposal, SolutionProposal)
    assert proposal is not None


def test_solve_proposal_has_all_required_fields(l_engine, sample_problem):
    """Test that solve() returns proposal with all required fields."""
    proposal = l_engine.solve(sample_problem)

    # Identity fields
    assert proposal.proposal_id is not None
    assert len(proposal.proposal_id) > 0
    assert proposal.problem_id == sample_problem.problem_id
    assert proposal.engine_id == l_engine.engine_id

    # Versioning
    assert proposal.version == 0  # Not yet assigned by blackboard
    assert proposal.checksum == 0  # Not yet computed

    # Content
    assert proposal.content is not None
    assert isinstance(proposal.content, torch.Tensor)
    assert len(proposal.content) > 0
    assert proposal.content_text is not None
    assert isinstance(proposal.content_text, str)
    assert len(proposal.content_text) > 0

    # Confidence
    assert proposal.raw_confidence is not None
    assert proposal.calibrated_confidence is not None

    # Metadata
    assert proposal.timestamp > 0
    assert proposal.latency >= 0
    assert proposal.cost >= 0
    assert proposal.was_constrained == False
    assert proposal.verified == False


def test_solve_proposal_id_is_unique(l_engine, sample_problem):
    """Test that solve() generates unique proposal IDs."""
    proposal1 = l_engine.solve(sample_problem)
    proposal2 = l_engine.solve(sample_problem)

    assert proposal1.proposal_id != proposal2.proposal_id


def test_solve_uses_problem_tokens(l_engine, sample_problem):
    """Test that solve() uses problem tokens when available."""
    proposal = l_engine.solve(sample_problem)

    # Backbone should have been called
    assert l_engine.backbone.forward_train.called


def test_solve_encodes_text_when_no_tokens(l_engine, mock_tokenizer):
    """Test that solve() encodes text when problem has no tokens."""
    problem = Problem(
        problem_id="test_prob_002",
        domain="math",
        input_text="What is 3 + 3?",
        input_tokens=torch.tensor([]),  # Empty tokens
    )

    proposal = l_engine.solve(problem)

    # Tokenizer encode should have been called
    assert mock_tokenizer.encode.called
    assert proposal is not None


def test_solve_with_blackboard_context(l_engine, sample_problem, blackboard):
    """Test that solve() accepts blackboard parameter."""
    # For MVP, blackboard is not used, but parameter should be accepted
    proposal = l_engine.solve(sample_problem, blackboard=blackboard)

    assert proposal is not None
    assert isinstance(proposal, SolutionProposal)


# ============================================================================
# Test Confidence in [0,1] Range
# ============================================================================

def test_confidence_in_valid_range(l_engine, sample_problem):
    """Test that confidence scores are in [0,1] range."""
    proposal = l_engine.solve(sample_problem)

    assert 0.0 <= proposal.raw_confidence <= 1.0
    assert 0.0 <= proposal.calibrated_confidence <= 1.0


def test_get_confidence_with_logits(l_engine):
    """Test get_confidence() with valid logits (Phase 4: returns ConfidenceScore)."""
    # Create sample logits
    logits = torch.tensor([1.0, 3.0, 2.0, 0.5])

    confidence_score = l_engine.get_confidence(logits)

    # Phase 4: now returns ConfidenceScore object
    assert 0.0 <= confidence_score.raw <= 1.0
    assert 0.0 <= confidence_score.calibrated <= 1.0
    assert confidence_score.raw > 0.5  # Should be high since max logit is clear


def test_get_confidence_with_none(l_engine):
    """Test get_confidence() with None logits returns neutral (Phase 4: ConfidenceScore)."""
    confidence_score = l_engine.get_confidence(None)

    # Phase 4: returns ConfidenceScore object
    assert confidence_score.raw == 0.5
    assert confidence_score.calibrated == 0.5


def test_get_confidence_returns_max_probability(l_engine):
    """Test that get_confidence() returns max softmax probability (Phase 4: ConfidenceScore)."""
    # Logits with clear max
    logits = torch.tensor([10.0, 0.0, 0.0, 0.0])

    confidence_score = l_engine.get_confidence(logits)

    # Should be close to 1.0 due to large logit difference
    assert confidence_score.raw > 0.99


def test_confidence_handles_edge_cases(l_engine):
    """Test confidence computation handles edge cases (Phase 4: ConfidenceScore)."""
    # All equal logits (uniform distribution)
    logits_uniform = torch.tensor([1.0, 1.0, 1.0, 1.0])
    conf_uniform = l_engine.get_confidence(logits_uniform)
    assert 0.0 <= conf_uniform.raw <= 1.0
    assert 0.0 <= conf_uniform.calibrated <= 1.0

    # Very negative logits
    logits_negative = torch.tensor([-1000.0, -1001.0, -1002.0])
    conf_negative = l_engine.get_confidence(logits_negative)
    assert 0.0 <= conf_negative.raw <= 1.0
    assert 0.0 <= conf_negative.calibrated <= 1.0


# ============================================================================
# Test Proposal Content (Text and Tokens)
# ============================================================================

def test_proposal_includes_content_text(l_engine, sample_problem):
    """Test that proposal includes decoded content text."""
    proposal = l_engine.solve(sample_problem)

    assert proposal.content_text is not None
    assert isinstance(proposal.content_text, str)
    assert len(proposal.content_text) > 0


def test_proposal_includes_tokens(l_engine, sample_problem):
    """Test that proposal includes token tensor."""
    proposal = l_engine.solve(sample_problem)

    assert proposal.content is not None
    assert isinstance(proposal.content, torch.Tensor)
    assert proposal.content.dim() == 1  # Should be 1D tensor
    assert len(proposal.content) > 0


def test_content_text_decoded_from_tokens(l_engine, sample_problem, mock_tokenizer):
    """Test that content_text is decoded from generated tokens."""
    proposal = l_engine.solve(sample_problem)

    # Tokenizer decode should have been called
    assert mock_tokenizer.decode.called

    # Call args should include generated tokens
    call_args = mock_tokenizer.decode.call_args
    assert call_args is not None


# ============================================================================
# Test Generation Logic
# ============================================================================

def test_generate_produces_tokens(l_engine):
    """Test that _generate() produces output tokens."""
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])

    output_tokens, logits = l_engine._generate(input_ids)

    assert output_tokens is not None
    assert isinstance(output_tokens, torch.Tensor)
    assert len(output_tokens) > 0


def test_generate_respects_max_length(l_engine):
    """Test that generation respects max_generation_length."""
    l_engine.max_generation_length = 10
    input_ids = torch.tensor([[1, 2, 3]])

    output_tokens, _ = l_engine._generate(input_ids)

    # Should not exceed max length
    assert len(output_tokens) <= 10


def test_generate_stops_at_eos(l_engine, mock_tokenizer):
    """Test that generation stops at EOS token."""
    # Set up mock to return EOS after a few tokens
    def mock_forward_with_eos(input_ids, segment_ids=None):
        batch_size, seq_len = input_ids.shape
        # Return hidden states that will lead to EOS after 3 iterations
        return torch.randn(batch_size, seq_len, 512)

    l_engine.backbone.forward_train = Mock(side_effect=mock_forward_with_eos)

    # Mock output head to return logits favoring EOS token (0) after 3 steps
    call_count = [0]
    def mock_output_with_eos(hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        logits = torch.randn(batch_size, seq_len, 50000)

        # After 3 calls, make EOS token have highest logit
        if call_count[0] >= 3:
            logits[0, -1, 0] = 100.0  # EOS token

        call_count[0] += 1
        return logits

    l_engine.output_head.forward = mock_output_with_eos

    input_ids = torch.tensor([[1, 2, 3]])
    output_tokens, _ = l_engine._generate(input_ids)

    # Should have stopped at EOS
    assert len(output_tokens) <= 3


# ============================================================================
# Test Recalibrate Stub
# ============================================================================

def test_recalibrate_exists(l_engine):
    """Test that recalibrate() method exists."""
    assert hasattr(l_engine, 'recalibrate')
    assert callable(l_engine.recalibrate)


def test_recalibrate_accepts_calibration_data(l_engine):
    """Test that recalibrate() accepts calibration data."""
    # Should not raise error (stub implementation)
    l_engine.recalibrate(calibration_data={'test': 'data'})
    l_engine.recalibrate(calibration_data=None)
    l_engine.recalibrate()


# ============================================================================
# Test Metadata and Cost Tracking
# ============================================================================

def test_proposal_has_timestamp(l_engine, sample_problem):
    """Test that proposal includes timestamp."""
    import time
    before = time.time()
    proposal = l_engine.solve(sample_problem)
    after = time.time()

    assert proposal.timestamp >= before
    assert proposal.timestamp <= after


def test_proposal_has_latency(l_engine, sample_problem):
    """Test that proposal includes latency measurement."""
    proposal = l_engine.solve(sample_problem)

    assert proposal.latency > 0
    assert isinstance(proposal.latency, float)


def test_proposal_has_cost(l_engine, sample_problem):
    """Test that proposal includes cost estimate."""
    proposal = l_engine.solve(sample_problem)

    assert proposal.cost > 0
    assert isinstance(proposal.cost, float)


def test_cost_reflects_token_count(l_engine, sample_problem):
    """Test that cost is based on token count."""
    proposal = l_engine.solve(sample_problem)

    # Cost should be at least the number of generated tokens
    assert proposal.cost >= len(proposal.content)


# ============================================================================
# Test Forward Pass (for training)
# ============================================================================

def test_forward_returns_logits(l_engine):
    """Test that forward() returns logits for training."""
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])

    logits = l_engine.forward(input_ids)

    assert logits is not None
    assert isinstance(logits, torch.Tensor)
    assert logits.dim() == 3  # [batch, seq, vocab]


def test_forward_output_shape(l_engine):
    """Test that forward() returns correct shape."""
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    logits = l_engine.forward(input_ids)

    assert logits.shape[0] == batch_size
    assert logits.shape[1] == seq_len
    assert logits.shape[2] == l_engine.tokenizer.vocab_size


# ============================================================================
# Test Integration with Blackboard
# ============================================================================

def test_proposal_can_be_written_to_blackboard(l_engine, sample_problem, blackboard):
    """Test that generated proposal can be written to blackboard."""
    proposal = l_engine.solve(sample_problem)

    # Should be able to write to blackboard without error
    blackboard.write_proposal(l_engine.engine_id, proposal)

    # Verify it's in blackboard
    proposals = blackboard.read_proposals()
    assert l_engine.engine_id in proposals
    assert proposals[l_engine.engine_id].proposal_id == proposal.proposal_id


def test_multiple_solves_can_coexist(l_engine, blackboard):
    """Test that multiple engines can write proposals."""
    # Create multiple problems
    problems = [
        Problem(
            problem_id=f"prob_{i}",
            domain="math",
            input_text=f"Problem {i}",
            input_tokens=torch.tensor([i, i+1, i+2]),
        )
        for i in range(3)
    ]

    # Solve each and write to blackboard
    for i, problem in enumerate(problems):
        # Create different engine IDs
        engine_id = f"engine_{i}"
        l_engine.engine_id = engine_id

        proposal = l_engine.solve(problem)
        blackboard.write_proposal(engine_id, proposal)

    # All proposals should be in blackboard
    proposals = blackboard.read_proposals()
    assert len(proposals) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
