"""
Edge case tests for input validation (F-002, F-004, F-005, F-006).

Tests specifically for security vulnerabilities:
- F-002: DoS via unbounded memory allocation (sequence length limit)
- F-004: Batch size validation
- F-005: topk validation
- F-006: Token ID upper bound validation
"""

import pytest
from pydantic import ValidationError

from src.distillation.schemas import (
    TopKRequest,
    MAX_SEQUENCE_LENGTH,
    MAX_BATCH_SIZE,
    MIN_TOPK,
    MAX_TOPK
)


class TestSecurityValidation:
    """Test security-critical validation."""

    def test_f002_sequence_length_dos_prevention(self):
        """Test F-002: Prevent DoS via unbounded memory allocation."""
        # Attack: Send extremely long sequence to exhaust memory
        # Expected: Rejected with clear error message

        # Valid: At the limit
        valid_request = TopKRequest(
            input_ids=[[1] * MAX_SEQUENCE_LENGTH]
        )
        assert len(valid_request.input_ids[0]) == MAX_SEQUENCE_LENGTH

        # Invalid: Over the limit (DoS attempt)
        with pytest.raises(ValueError) as exc_info:
            TopKRequest(
                input_ids=[[1] * (MAX_SEQUENCE_LENGTH + 1)]
            )
        assert f"exceeds maximum {MAX_SEQUENCE_LENGTH}" in str(exc_info.value)

        # Attack: Multiple sequences, each at max length
        # Should still be allowed (batch size is separate limit)
        multi_max_request = TopKRequest(
            input_ids=[[1] * MAX_SEQUENCE_LENGTH, [2] * MAX_SEQUENCE_LENGTH]
        )
        assert len(multi_max_request.input_ids) == 2

    def test_f004_batch_size_dos_prevention(self):
        """Test F-004: Prevent DoS via excessive batch size."""
        # Attack: Send huge batch to exhaust memory/compute
        # Expected: Rejected with clear error message

        # Valid: At the limit
        valid_request = TopKRequest(
            input_ids=[[1, 2, 3]] * MAX_BATCH_SIZE
        )
        assert len(valid_request.input_ids) == MAX_BATCH_SIZE

        # Invalid: Over the limit (DoS attempt)
        with pytest.raises(ValueError) as exc_info:
            TopKRequest(
                input_ids=[[1, 2, 3]] * (MAX_BATCH_SIZE + 1)
            )
        assert f"exceeds maximum {MAX_BATCH_SIZE}" in str(exc_info.value)

        # Attack: Combination of max batch size + long sequences
        # Both limits should be enforced independently
        with pytest.raises(ValueError) as exc_info:
            TopKRequest(
                input_ids=[[1] * (MAX_SEQUENCE_LENGTH + 1)] * MAX_BATCH_SIZE
            )
        assert "exceeds maximum" in str(exc_info.value)

    def test_f005_topk_validation(self):
        """Test F-005: topk parameter validation."""
        # Attack: Request extremely large topk to exhaust memory
        # Expected: Rejected at schema level

        # Valid: Within bounds
        valid_request = TopKRequest(
            input_ids=[[1, 2, 3]],
            topk=MAX_TOPK
        )
        assert valid_request.topk == MAX_TOPK

        # Invalid: topk too small
        with pytest.raises(ValueError) as exc_info:
            TopKRequest(
                input_ids=[[1, 2, 3]],
                topk=0
            )
        assert f"topk must be between {MIN_TOPK} and {MAX_TOPK}" in str(exc_info.value)

        # Invalid: topk too large
        with pytest.raises(ValueError) as exc_info:
            TopKRequest(
                input_ids=[[1, 2, 3]],
                topk=MAX_TOPK + 1
            )
        assert f"topk must be between {MIN_TOPK} and {MAX_TOPK}" in str(exc_info.value)

    def test_f006_token_id_validation_negative(self):
        """Test F-006: Token ID validation (negative IDs)."""
        # Attack: Send negative token IDs (could cause index errors)
        # Expected: Rejected at schema level

        # Valid: Non-negative token IDs
        valid_request = TopKRequest(
            input_ids=[[0, 1, 2, 3, 100, 1000]]
        )
        assert all(tid >= 0 for seq in valid_request.input_ids for tid in seq)

        # Invalid: Negative token ID
        with pytest.raises(ValueError) as exc_info:
            TopKRequest(
                input_ids=[[1, 2, -1, 3]]
            )
        assert "negative token ID" in str(exc_info.value)

        # Invalid: Multiple negative IDs
        with pytest.raises(ValueError) as exc_info:
            TopKRequest(
                input_ids=[[-1, -2, -3]]
            )
        assert "negative token ID" in str(exc_info.value)

    def test_combined_attack_scenarios(self):
        """Test combinations of invalid inputs (defense in depth)."""
        # Attack 1: Max batch + max sequence + large topk
        # All individual limits should be enforced
        with pytest.raises(ValueError):
            TopKRequest(
                input_ids=[[1] * (MAX_SEQUENCE_LENGTH + 1)] * (MAX_BATCH_SIZE + 1),
                topk=MAX_TOPK + 1
            )

        # Attack 2: Invalid token IDs + oversized batch
        with pytest.raises(ValueError):
            TopKRequest(
                input_ids=[[-1, -2]] * (MAX_BATCH_SIZE + 1)
            )

        # Attack 3: Empty sequences in oversized batch
        with pytest.raises(ValueError):
            TopKRequest(
                input_ids=[[]] * (MAX_BATCH_SIZE + 1)
            )

    def test_edge_case_minimum_values(self):
        """Test minimum valid values."""
        # Minimum valid request: 1 sequence, 1 token, topk=1
        min_request = TopKRequest(
            input_ids=[[0]],
            topk=1
        )
        assert len(min_request.input_ids) == 1
        assert len(min_request.input_ids[0]) == 1
        assert min_request.topk == 1

    def test_edge_case_maximum_values(self):
        """Test maximum valid values."""
        # Maximum valid request (resource-intensive but legal)
        max_request = TopKRequest(
            input_ids=[[1] * MAX_SEQUENCE_LENGTH] * MAX_BATCH_SIZE,
            topk=MAX_TOPK
        )
        assert len(max_request.input_ids) == MAX_BATCH_SIZE
        assert all(len(seq) == MAX_SEQUENCE_LENGTH for seq in max_request.input_ids)
        assert max_request.topk == MAX_TOPK

    def test_error_messages_are_clear(self):
        """Test that validation errors provide clear, actionable messages."""
        # Error messages should help users understand what went wrong

        # Batch size error
        try:
            TopKRequest(input_ids=[[1]] * (MAX_BATCH_SIZE + 1))
        except ValueError as e:
            assert "Batch size" in str(e)
            assert str(MAX_BATCH_SIZE) in str(e)
            assert "exceeds" in str(e)

        # Sequence length error
        try:
            TopKRequest(input_ids=[[1] * (MAX_SEQUENCE_LENGTH + 1)])
        except ValueError as e:
            assert "length" in str(e)
            assert str(MAX_SEQUENCE_LENGTH) in str(e)
            assert "exceeds" in str(e)

        # topk error
        try:
            TopKRequest(input_ids=[[1, 2, 3]], topk=MAX_TOPK + 1)
        except ValueError as e:
            assert "topk" in str(e)
            assert str(MAX_TOPK) in str(e)
            assert "between" in str(e)

        # Negative token ID error
        try:
            TopKRequest(input_ids=[[1, -2, 3]])
        except ValueError as e:
            assert "negative token ID" in str(e)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
