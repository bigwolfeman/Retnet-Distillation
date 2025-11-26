"""Unit tests for Titan-HRM math verifiers (T021).

Tests coverage:
- T019: MathSymbolicVerifier
  - Correct answers (symbolic and numeric)
  - Incorrect answers
  - Parse errors
  - SymPy edge cases
- T020: MathNumericVerifier
  - Correct numeric answers
  - Tolerance edge cases
  - Type conversions
  - Error handling
- Timeout handling (5 seconds max)

All tests use simple cases that don't require external datasets.
"""

import time
from unittest.mock import patch

import pytest
import sympy

from src.models.titans.data_model import VerificationResult, VerifierType
from src.models.titans.verifiers import MathNumericVerifier, MathSymbolicVerifier


# ============================================================================
# T019: MathSymbolicVerifier Tests
# ============================================================================

class TestMathSymbolicVerifier:
    """Tests for SymPy-based symbolic verifier."""

    def test_init_valid_tolerance(self):
        """Test initialization with valid tolerance."""
        verifier = MathSymbolicVerifier(tolerance=1e-6)
        assert verifier.tolerance == 1e-6

    def test_init_invalid_tolerance(self):
        """Test initialization with invalid tolerance raises error."""
        with pytest.raises(ValueError, match="tolerance must be >= 0"):
            MathSymbolicVerifier(tolerance=-1)

    def test_verify_correct_symbolic_simple(self):
        """Test verification of correct symbolic answer - simple expression."""
        verifier = MathSymbolicVerifier()
        result = verifier.verify(
            proposal_id="test_001",
            answer="2 + 2",
            ground_truth="4",
        )

        assert isinstance(result, VerificationResult)
        assert result.verifier_type == VerifierType.MATH_SYMBOLIC
        assert result.proposal_id == "test_001"
        assert result.passed is True
        assert result.score == 1.0
        assert result.error_message is None
        assert result.execution_time > 0

    def test_verify_correct_symbolic_algebraic(self):
        """Test verification of correct symbolic answer - algebraic."""
        verifier = MathSymbolicVerifier()
        result = verifier.verify(
            proposal_id="test_002",
            answer="x**2 - 1",
            ground_truth="(x-1)*(x+1)",
        )

        assert result.passed is True
        assert result.score == 1.0
        assert result.error_message is None

    def test_verify_correct_symbolic_trigonometric(self):
        """Test verification with trigonometric identities."""
        verifier = MathSymbolicVerifier()
        result = verifier.verify(
            proposal_id="test_003",
            answer="sin(x)**2 + cos(x)**2",
            ground_truth="1",
        )

        assert result.passed is True
        assert result.score == 1.0

    def test_verify_correct_numeric_within_tolerance(self):
        """Test verification of numeric answer within tolerance."""
        verifier = MathSymbolicVerifier(tolerance=1e-6)
        result = verifier.verify(
            proposal_id="test_004",
            answer="3.14159265",
            ground_truth="3.14159266",  # Diff: 1e-8
        )

        assert result.passed is True
        assert result.score == 1.0

    def test_verify_incorrect_answer(self):
        """Test verification of incorrect answer."""
        verifier = MathSymbolicVerifier()
        result = verifier.verify(
            proposal_id="test_005",
            answer="5",
            ground_truth="4",
        )

        assert result.passed is False
        assert result.score == 0.0
        assert "Numerical difference" in result.error_message

    def test_verify_incorrect_symbolic(self):
        """Test verification of incorrect symbolic answer."""
        verifier = MathSymbolicVerifier()
        result = verifier.verify(
            proposal_id="test_006",
            answer="x**2 + 1",
            ground_truth="(x-1)*(x+1)",  # Actually x**2 - 1
        )

        assert result.passed is False
        assert result.score == 0.0

    def test_verify_parse_error_answer(self):
        """Test verification when answer fails to parse."""
        verifier = MathSymbolicVerifier()
        # Use a dictionary as input - SymPy can't parse this
        result = verifier.verify(
            proposal_id="test_007",
            answer={"invalid": "dict"},
            ground_truth="4",
        )

        assert result.passed is False
        assert result.score == 0.0
        # Error message could be parse error or evaluation error
        assert result.error_message is not None

    def test_verify_parse_error_ground_truth(self):
        """Test verification when ground truth fails to parse."""
        verifier = MathSymbolicVerifier()
        # Use a list as input - SymPy can't parse this
        result = verifier.verify(
            proposal_id="test_008",
            answer="4",
            ground_truth=[1, 2, 3],
        )

        assert result.passed is False
        assert result.score == 0.0
        # Error message could be parse error or evaluation error
        assert result.error_message is not None

    def test_verify_nan_result(self):
        """Test verification when expression evaluates to NaN."""
        verifier = MathSymbolicVerifier()
        result = verifier.verify(
            proposal_id="test_009",
            answer="0/0",
            ground_truth="1",
        )

        assert result.passed is False
        assert result.score == 0.0
        assert "NaN or Inf" in result.error_message

    def test_verify_infinity_result(self):
        """Test verification when expression evaluates to infinity."""
        verifier = MathSymbolicVerifier()
        result = verifier.verify(
            proposal_id="test_010",
            answer="1/0",
            ground_truth="1",
        )

        assert result.passed is False
        assert result.score == 0.0

    def test_verify_complex_numbers(self):
        """Test verification with complex numbers."""
        verifier = MathSymbolicVerifier()
        result = verifier.verify(
            proposal_id="test_011",
            answer="I",  # SymPy's imaginary unit
            ground_truth="sqrt(-1)",
        )

        assert result.passed is True
        assert result.score == 1.0

    def test_verify_tolerance_boundary(self):
        """Test verification at tolerance boundary."""
        tolerance = 1e-6
        verifier = MathSymbolicVerifier(tolerance=tolerance)

        # Just within tolerance
        result = verifier.verify(
            proposal_id="test_012",
            answer="1.0000009",
            ground_truth="1.0",
        )
        assert result.passed is True

        # Just outside tolerance
        result = verifier.verify(
            proposal_id="test_013",
            answer="1.000002",
            ground_truth="1.0",
        )
        assert result.passed is False

    def test_verify_timeout_handling(self):
        """Test timeout handling for long-running verification."""
        verifier = MathSymbolicVerifier()

        # Mock a slow sympy.simplify operation to trigger timeout
        original_simplify = sympy.simplify

        def slow_simplify(expr):
            time.sleep(6)  # Exceed 5 second timeout
            return original_simplify(expr)

        with patch('sympy.simplify', side_effect=slow_simplify):
            result = verifier.verify(
                proposal_id="test_014",
                answer="2 + 2",
                ground_truth="4",
            )

            assert result.passed is False
            assert result.score == 0.0
            assert "timeout" in result.error_message.lower()
            # Execution time might be slightly over 5 seconds due to threading overhead
            assert result.execution_time >= 4.9

    def test_verify_numeric_types(self):
        """Test verification with different numeric types."""
        verifier = MathSymbolicVerifier()

        # Integer
        result = verifier.verify(proposal_id="test_015", answer=42, ground_truth=42)
        assert result.passed is True

        # Float
        result = verifier.verify(proposal_id="test_016", answer=3.14, ground_truth=3.14)
        assert result.passed is True

        # Mixed types
        result = verifier.verify(proposal_id="test_017", answer=42, ground_truth=42.0)
        assert result.passed is True


# ============================================================================
# T020: MathNumericVerifier Tests
# ============================================================================

class TestMathNumericVerifier:
    """Tests for numeric comparison verifier."""

    def test_init_valid_tolerance(self):
        """Test initialization with valid tolerance."""
        verifier = MathNumericVerifier(tolerance=1e-6)
        assert verifier.tolerance == 1e-6
        assert verifier.relative_tolerance is None

    def test_init_with_relative_tolerance(self):
        """Test initialization with relative tolerance."""
        verifier = MathNumericVerifier(tolerance=1e-6, relative_tolerance=1e-3)
        assert verifier.tolerance == 1e-6
        assert verifier.relative_tolerance == 1e-3

    def test_init_invalid_tolerance(self):
        """Test initialization with invalid tolerance raises error."""
        with pytest.raises(ValueError, match="tolerance must be >= 0"):
            MathNumericVerifier(tolerance=-1)

    def test_init_invalid_relative_tolerance(self):
        """Test initialization with invalid relative tolerance raises error."""
        with pytest.raises(ValueError, match="relative_tolerance must be >= 0"):
            MathNumericVerifier(tolerance=1e-6, relative_tolerance=-1)

    def test_verify_correct_integer(self):
        """Test verification of correct integer answer."""
        verifier = MathNumericVerifier()
        result = verifier.verify(
            proposal_id="test_101",
            answer=42,
            ground_truth=42,
        )

        assert isinstance(result, VerificationResult)
        assert result.verifier_type == VerifierType.MATH_NUMERIC
        assert result.proposal_id == "test_101"
        assert result.passed is True
        assert result.score == 1.0
        assert result.error_message is None
        assert result.execution_time >= 0  # May be 0 on fast systems

    def test_verify_correct_float(self):
        """Test verification of correct float answer."""
        verifier = MathNumericVerifier()
        result = verifier.verify(
            proposal_id="test_102",
            answer=3.14159,
            ground_truth=3.14159,
        )

        assert result.passed is True
        assert result.score == 1.0

    def test_verify_correct_complex(self):
        """Test verification of correct complex answer."""
        verifier = MathNumericVerifier()
        result = verifier.verify(
            proposal_id="test_103",
            answer=3 + 4j,
            ground_truth=3 + 4j,
        )

        assert result.passed is True
        assert result.score == 1.0

    def test_verify_string_to_int(self):
        """Test verification with string that parses to int."""
        verifier = MathNumericVerifier()
        result = verifier.verify(
            proposal_id="test_104",
            answer="42",
            ground_truth=42,
        )

        assert result.passed is True
        assert result.score == 1.0

    def test_verify_string_to_float(self):
        """Test verification with string that parses to float."""
        verifier = MathNumericVerifier()
        result = verifier.verify(
            proposal_id="test_105",
            answer="3.14159",
            ground_truth=3.14159,
        )

        assert result.passed is True
        assert result.score == 1.0

    def test_verify_string_to_complex(self):
        """Test verification with string that parses to complex."""
        verifier = MathNumericVerifier()
        result = verifier.verify(
            proposal_id="test_106",
            answer="3+4j",
            ground_truth=3 + 4j,
        )

        assert result.passed is True
        assert result.score == 1.0

    def test_verify_incorrect_answer(self):
        """Test verification of incorrect numeric answer."""
        verifier = MathNumericVerifier()
        result = verifier.verify(
            proposal_id="test_107",
            answer=42,
            ground_truth=43,
        )

        assert result.passed is False
        assert result.score == 0.0
        assert "Absolute difference" in result.error_message

    def test_verify_within_absolute_tolerance(self):
        """Test verification within absolute tolerance."""
        verifier = MathNumericVerifier(tolerance=1e-6)
        result = verifier.verify(
            proposal_id="test_108",
            answer=1.0000005,
            ground_truth=1.0,
        )

        assert result.passed is True
        assert result.score == 1.0

    def test_verify_outside_absolute_tolerance(self):
        """Test verification outside absolute tolerance."""
        verifier = MathNumericVerifier(tolerance=1e-6)
        result = verifier.verify(
            proposal_id="test_109",
            answer=1.00001,
            ground_truth=1.0,
        )

        assert result.passed is False
        assert result.score == 0.0

    def test_verify_within_relative_tolerance(self):
        """Test verification within relative tolerance for large values."""
        verifier = MathNumericVerifier(tolerance=1e-6, relative_tolerance=1e-3)
        result = verifier.verify(
            proposal_id="test_110",
            answer=1000.5,  # Relative error: 0.5/1000 = 0.0005 < 1e-3
            ground_truth=1000.0,
        )

        assert result.passed is True
        assert result.score == 1.0

    def test_verify_tolerance_edge_case_exact_boundary(self):
        """Test verification at exact tolerance boundary."""
        tolerance = 1e-6
        verifier = MathNumericVerifier(tolerance=tolerance)

        # Exactly at tolerance (should pass)
        result = verifier.verify(
            proposal_id="test_111",
            answer=1.0 + tolerance,
            ground_truth=1.0,
        )
        assert result.passed is True

    def test_verify_tolerance_edge_case_just_outside(self):
        """Test verification just outside tolerance."""
        tolerance = 1e-6
        verifier = MathNumericVerifier(tolerance=tolerance)

        # Just outside tolerance
        result = verifier.verify(
            proposal_id="test_112",
            answer=1.0 + tolerance * 1.1,
            ground_truth=1.0,
        )
        assert result.passed is False

    def test_verify_parse_error(self):
        """Test verification when parsing fails."""
        verifier = MathNumericVerifier()
        result = verifier.verify(
            proposal_id="test_113",
            answer="not_a_number",
            ground_truth=42,
        )

        assert result.passed is False
        assert result.score == 0.0
        assert "Failed to parse to numeric" in result.error_message

    def test_verify_nan_answer(self):
        """Test verification with NaN answer."""
        verifier = MathNumericVerifier()
        result = verifier.verify(
            proposal_id="test_114",
            answer=float('nan'),
            ground_truth=1.0,
        )

        assert result.passed is False
        assert result.score == 0.0
        assert "NaN or Inf" in result.error_message

    def test_verify_inf_answer(self):
        """Test verification with infinity answer."""
        verifier = MathNumericVerifier()
        result = verifier.verify(
            proposal_id="test_115",
            answer=float('inf'),
            ground_truth=1.0,
        )

        assert result.passed is False
        assert result.score == 0.0
        assert "NaN or Inf" in result.error_message

    def test_verify_negative_zero(self):
        """Test verification with negative zero."""
        verifier = MathNumericVerifier()
        result = verifier.verify(
            proposal_id="test_116",
            answer=-0.0,
            ground_truth=0.0,
        )

        assert result.passed is True
        assert result.score == 1.0

    def test_verify_mixed_int_float(self):
        """Test verification with mixed int/float types."""
        verifier = MathNumericVerifier()

        # Int vs float
        result = verifier.verify(
            proposal_id="test_117",
            answer=42,
            ground_truth=42.0,
        )
        assert result.passed is True

        # Float vs int
        result = verifier.verify(
            proposal_id="test_118",
            answer=42.0,
            ground_truth=42,
        )
        assert result.passed is True

    def test_verify_timeout_handling(self):
        """Test timeout handling for numeric verification."""
        verifier = MathNumericVerifier()

        # Create a custom verifier with a slow implementation
        def slow_verify_impl(self, proposal_id, answer, ground_truth, start_time):
            time.sleep(6)  # Exceed 5 second timeout
            return VerificationResult(
                verifier_type=VerifierType.MATH_NUMERIC,
                proposal_id=proposal_id,
                passed=True,
                score=1.0,
                execution_time=6.0,
            )

        # Patch the internal implementation to be slow
        with patch.object(MathNumericVerifier, '_verify_impl', slow_verify_impl):
            result = verifier.verify(
                proposal_id="test_119",
                answer=42,
                ground_truth=42,
            )

            assert result.passed is False
            assert result.score == 0.0
            assert "timeout" in result.error_message.lower()
            # Execution time might be slightly over 5 seconds due to threading overhead
            assert result.execution_time >= 4.9

    def test_verify_very_small_numbers(self):
        """Test verification with very small numbers."""
        verifier = MathNumericVerifier(tolerance=1e-10)
        result = verifier.verify(
            proposal_id="test_120",
            answer=1e-15,
            ground_truth=1e-15,
        )

        assert result.passed is True
        assert result.score == 1.0

    def test_verify_very_large_numbers(self):
        """Test verification with very large numbers."""
        verifier = MathNumericVerifier(tolerance=1e-6, relative_tolerance=1e-9)
        result = verifier.verify(
            proposal_id="test_121",
            answer=1e15,
            ground_truth=1e15,
        )

        assert result.passed is True
        assert result.score == 1.0

    def test_verify_complex_tolerance(self):
        """Test verification with complex numbers and tolerance."""
        verifier = MathNumericVerifier(tolerance=1e-6)

        # Within tolerance
        result = verifier.verify(
            proposal_id="test_122",
            answer=3.0 + 4.0j,
            ground_truth=3.0000005 + 4.0000005j,
        )
        assert result.passed is True

        # Outside tolerance
        result = verifier.verify(
            proposal_id="test_123",
            answer=3.0 + 4.0j,
            ground_truth=3.001 + 4.001j,
        )
        assert result.passed is False


# ============================================================================
# Integration Tests
# ============================================================================

class TestVerifiersIntegration:
    """Integration tests comparing both verifiers."""

    def test_symbolic_vs_numeric_simple(self):
        """Test that both verifiers agree on simple numeric cases."""
        symbolic = MathSymbolicVerifier(tolerance=1e-6)
        numeric = MathNumericVerifier(tolerance=1e-6)

        test_cases = [
            (42, 42, True),
            (3.14, 3.14, True),
            (42, 43, False),
        ]

        for answer, truth, expected in test_cases:
            sym_result = symbolic.verify("sym", answer, truth)
            num_result = numeric.verify("num", answer, truth)

            assert sym_result.passed == expected
            assert num_result.passed == expected
            assert sym_result.passed == num_result.passed

    def test_symbolic_advantage_algebraic(self):
        """Test that symbolic verifier handles algebraic expressions."""
        symbolic = MathSymbolicVerifier()

        # Symbolic can handle this
        result = symbolic.verify(
            "test",
            answer="x**2 - 1",
            ground_truth="(x-1)*(x+1)",
        )
        assert result.passed is True

    def test_execution_time_recorded(self):
        """Test that both verifiers record execution time."""
        symbolic = MathSymbolicVerifier()
        numeric = MathNumericVerifier()

        sym_result = symbolic.verify("sym", "2+2", "4")
        num_result = numeric.verify("num", 42, 42)

        # Execution time should be recorded (may be 0 on fast systems, but should be >= 0)
        assert sym_result.execution_time >= 0
        assert num_result.execution_time >= 0
        # Should be reasonably fast (< 5 seconds)
        assert sym_result.execution_time < 5.0
        assert num_result.execution_time < 5.0
