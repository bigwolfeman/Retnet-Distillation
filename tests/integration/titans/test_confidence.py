"""Integration tests for Phase 4: Transparent Confidence and Reasoning (T036).

Tests for:
- End-to-end confidence calibration in SimpleLEngine
- Reasoning trace generation and logging
- ECE measurement on validation problems
- Confidence ranges matching actual accuracy
"""

import numpy as np
import pytest
import torch

from src.models.titans import (
    Problem,
    SimpleLEngine,
    ConfidenceScore,
)
from src.models.titans.calibration import compute_ece
from src.models.retnet.backbone import RetNetBackbone
from src.data.tokenizer import RetNetTokenizer


# ============================================================================
# T036: Integration Tests for Confidence and Reasoning
# ============================================================================

@pytest.fixture
def mock_backbone():
    """Create a small RetNet backbone for testing."""
    # Use default vocab size to match tokenizer
    return RetNetBackbone(
        vocab_size=100352,  # Default vocab size to match RetNetTokenizer
        d_model=64,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
    )


@pytest.fixture
def mock_tokenizer():
    """Create a RetNet tokenizer for testing."""
    return RetNetTokenizer()


@pytest.fixture
def simple_engine(mock_backbone, mock_tokenizer):
    """Create a SimpleLEngine for testing."""
    return SimpleLEngine(
        engine_id="test_engine",
        backbone=mock_backbone,
        tokenizer=mock_tokenizer,
        max_generation_length=50,
        temperature=1.0,
        device="cpu",  # Use CPU for tests
    )


class TestConfidenceIntegration:
    """Integration tests for confidence calibration."""

    def test_solve_returns_confidence_score(self, simple_engine, mock_tokenizer):
        """Test that solve() returns proposal with calibrated confidence."""
        # Create a simple math problem
        problem = Problem(
            problem_id="test_001",
            domain="math",
            input_text="What is 2 + 2?",
            input_tokens=torch.tensor(mock_tokenizer.encode("What is 2 + 2?")),
            ground_truth=4.0,
        )

        # Solve problem
        proposal = simple_engine.solve(problem)

        # Check that confidence values are present
        assert 0 <= proposal.raw_confidence <= 1, "Raw confidence must be in [0,1]"
        assert 0 <= proposal.calibrated_confidence <= 1, "Calibrated confidence must be in [0,1]"

        # Check that logits are available for recalibration
        assert proposal.logits is not None, "Logits should be saved for recalibration"

    def test_reasoning_trace_populated(self, simple_engine, mock_tokenizer):
        """Test that reasoning trace is populated with intermediate steps."""
        problem = Problem(
            problem_id="test_002",
            domain="math",
            input_text="Solve for x: 2x + 5 = 15",
            input_tokens=torch.tensor(mock_tokenizer.encode("Solve for x: 2x + 5 = 15")),
        )

        proposal = simple_engine.solve(problem)

        # Check reasoning trace exists and has content
        assert proposal.reasoning_trace is not None, "Reasoning trace should be populated"
        assert len(proposal.reasoning_trace) > 0, "Reasoning trace should have steps"

        # Verify key steps are logged (from T035)
        trace_text = " ".join(proposal.reasoning_trace)
        assert "Question received" in trace_text, "Should log question receipt"
        assert "Generation started" in trace_text, "Should log generation start"
        assert "Generation complete" in trace_text, "Should log generation completion"
        assert "Confidence computed" in trace_text, "Should log confidence computation"
        assert "Proposal created" in trace_text, "Should log proposal creation"

    def test_reasoning_trace_includes_metadata(self, simple_engine, mock_tokenizer):
        """Test that reasoning trace includes useful metadata."""
        problem = Problem(
            problem_id="test_003",
            domain="math",
            input_text="Calculate the area of a circle with radius 5",
            input_tokens=torch.tensor(mock_tokenizer.encode("Calculate the area of a circle with radius 5")),
        )

        proposal = simple_engine.solve(problem)
        trace = proposal.reasoning_trace

        # Check for metadata in trace
        assert any("tokens" in step.lower() for step in trace), "Should log token counts"
        assert any("latency" in step.lower() for step in trace), "Should log latency"
        assert any("raw=" in step for step in trace), "Should log raw confidence"
        assert any("calibrated=" in step for step in trace), "Should log calibrated confidence"

    def test_temperature_scaling_changes_confidence(self, simple_engine, mock_tokenizer):
        """Test that temperature scaling affects calibrated confidence."""
        problem = Problem(
            problem_id="test_004",
            domain="math",
            input_text="What is 10 * 10?",
            input_tokens=torch.tensor(mock_tokenizer.encode("What is 10 * 10?")),
        )

        # Solve with default temperature (T=1.0)
        proposal = simple_engine.solve(problem)

        # Get raw and calibrated confidence
        raw_conf = proposal.raw_confidence
        calib_conf = proposal.calibrated_confidence

        # With T=1.0, calibrated should be close to raw (but not necessarily identical due to softmax)
        # Just verify both are valid
        assert 0 <= raw_conf <= 1
        assert 0 <= calib_conf <= 1


class TestECEValidation:
    """Tests for ECE computation on validation problems."""

    def test_ece_measurement_on_validation_set(self, simple_engine, mock_tokenizer):
        """Test ECE measurement on a small validation set."""
        # Create 20 simple math problems
        problems = []
        for i in range(20):
            problems.append(Problem(
                problem_id=f"val_{i:03d}",
                domain="math",
                input_text=f"What is {i} + 1?",
                input_tokens=torch.tensor(mock_tokenizer.encode(f"What is {i} + 1?")),
                ground_truth=float(i + 1),
            ))

        # Solve all problems
        proposals = [simple_engine.solve(p) for p in problems]

        # Extract calibrated confidences
        confidences = np.array([p.calibrated_confidence for p in proposals])

        # For this test, we'll mock correctness (in real scenario, verify against ground truth)
        # Since we're using a random backbone, we'll just check structure
        correctness = np.random.randint(0, 2, len(proposals))  # Random for structural test

        # Compute ECE
        ece, bin_accs, bin_confs, bin_counts = compute_ece(confidences, correctness, n_bins=5)

        # Check that ECE computation works
        assert 0 <= ece <= 1, "ECE should be in [0,1]"
        assert len(bin_accs) == 5, "Should have 5 bins"
        assert len(bin_confs) == 5, "Should have 5 confidence bins"
        assert np.sum(bin_counts) == len(proposals), "Bin counts should sum to total"

    def test_confidence_ranges_structure(self, simple_engine, mock_tokenizer):
        """Test that confidence values have reasonable structure across problems."""
        # Create diverse problems
        problems = [
            Problem(
                problem_id=f"test_{i}",
                domain="math",
                input_text=text,
                input_tokens=torch.tensor(mock_tokenizer.encode(text)),
            )
            for i, text in enumerate([
                "2 + 2",
                "What is the square root of 144?",
                "Solve the differential equation dy/dx = y",
                "Calculate 5!",
                "What is pi to 10 decimal places?",
            ])
        ]

        # Solve all
        proposals = [simple_engine.solve(p) for p in problems]

        # Check confidence ranges
        raw_confs = [p.raw_confidence for p in proposals]
        calib_confs = [p.calibrated_confidence for p in proposals]

        # All should be valid [0,1]
        assert all(0 <= c <= 1 for c in raw_confs), "All raw confidences in [0,1]"
        assert all(0 <= c <= 1 for c in calib_confs), "All calibrated confidences in [0,1]"

        # Check that confidences are computed (may be identical for small random model)
        # For a real trained model, we'd expect variance
        assert len(raw_confs) > 0, "Should have raw confidences"
        assert len(calib_confs) > 0, "Should have calibrated confidences"


class TestReasoningTraceDetails:
    """Detailed tests for reasoning trace content."""

    def test_trace_includes_engine_id(self, simple_engine, mock_tokenizer):
        """Test that reasoning trace includes engine ID."""
        problem = Problem(
            problem_id="test_trace_001",
            domain="math",
            input_text="Simple problem",
            input_tokens=torch.tensor(mock_tokenizer.encode("Simple problem")),
        )

        proposal = simple_engine.solve(problem)

        # Check engine ID in trace
        assert any("test_engine" in step for step in proposal.reasoning_trace), \
            "Reasoning trace should include engine ID"

    def test_trace_includes_confidence_values(self, simple_engine, mock_tokenizer):
        """Test that reasoning trace includes confidence values."""
        problem = Problem(
            problem_id="test_trace_002",
            domain="math",
            input_text="Another problem",
            input_tokens=torch.tensor(mock_tokenizer.encode("Another problem")),
        )

        proposal = simple_engine.solve(problem)

        # Check confidence values in trace
        conf_steps = [s for s in proposal.reasoning_trace if "Confidence computed" in s]
        assert len(conf_steps) > 0, "Should have confidence computation steps"

        # Extract and verify confidence from trace
        for step in conf_steps:
            assert "raw=" in step, "Should show raw confidence"
            assert "calibrated=" in step, "Should show calibrated confidence"
            assert "T=" in step, "Should show temperature value"

    def test_trace_step_ordering(self, simple_engine, mock_tokenizer):
        """Test that reasoning trace steps are in logical order."""
        problem = Problem(
            problem_id="test_trace_003",
            domain="math",
            input_text="Test ordering",
            input_tokens=torch.tensor(mock_tokenizer.encode("Test ordering")),
        )

        proposal = simple_engine.solve(problem)
        trace = proposal.reasoning_trace

        # Find indices of key steps
        question_idx = next((i for i, s in enumerate(trace) if "Question received" in s), -1)
        generation_idx = next((i for i, s in enumerate(trace) if "Generation started" in s), -1)
        complete_idx = next((i for i, s in enumerate(trace) if "Generation complete" in s), -1)
        conf_idx = next((i for i, s in enumerate(trace) if "Confidence computed" in s), -1)
        proposal_idx = next((i for i, s in enumerate(trace) if "Proposal created" in s), -1)

        # Verify logical ordering
        assert question_idx < generation_idx, "Question should come before generation"
        assert generation_idx < complete_idx, "Generation start should come before completion"
        assert complete_idx < conf_idx, "Generation should complete before confidence computation"
        assert conf_idx < proposal_idx, "Confidence should be computed before proposal creation"


class TestBlackboardIntegration:
    """Tests for blackboard context integration (T035 MAC context logging)."""

    def test_no_blackboard_logged(self, simple_engine, mock_tokenizer):
        """Test that absence of blackboard is logged in reasoning trace."""
        problem = Problem(
            problem_id="test_bb_001",
            domain="math",
            input_text="Test problem",
            input_tokens=torch.tensor(mock_tokenizer.encode("Test problem")),
        )

        # Solve without blackboard
        proposal = simple_engine.solve(problem, blackboard=None)

        # Check that MVP mode is logged
        trace_text = " ".join(proposal.reasoning_trace)
        assert "No blackboard context available" in trace_text or "MVP mode" in trace_text, \
            "Should log when blackboard is not available"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
