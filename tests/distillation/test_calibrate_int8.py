"""
Unit tests for int8 calibration test script.

Tests:
- CalibrationDatasetLoader functionality
- Int8Calibrator CE computation
- Report generation
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from src.distillation.calibrate_int8_topk import (
    CalibrationDatasetLoader,
    Int8Calibrator,
    generate_report
)
from src.distillation.schemas import TopKResponse


class TestCalibrationDatasetLoader:
    """Test CalibrationDatasetLoader class."""

    def test_synthetic_generation(self):
        """Test synthetic sequence generation."""
        loader = CalibrationDatasetLoader(
            data_path=None,
            max_length=100,
            seed=42
        )

        sequences = loader.load_sequences(num_sequences=10)

        assert len(sequences) == 10
        assert all(len(seq) == 100 for seq in sequences)
        assert all(isinstance(token_id, int) for seq in sequences for token_id in seq)

    def test_synthetic_reproducibility(self):
        """Test that synthetic generation is reproducible with same seed."""
        loader1 = CalibrationDatasetLoader(data_path=None, max_length=50, seed=42)
        loader2 = CalibrationDatasetLoader(data_path=None, max_length=50, seed=42)

        seq1 = loader1.load_sequences(5)
        seq2 = loader2.load_sequences(5)

        assert seq1 == seq2

    def test_load_jsonl(self, tmp_path):
        """Test loading from JSONL file."""
        # Create test JSONL file
        jsonl_file = tmp_path / "test.jsonl"
        with open(jsonl_file, 'w') as f:
            for i in range(5):
                f.write(json.dumps({"text": f"Test sequence {i} with some content"}) + "\n")

        # Mock tokenizer to avoid downloading
        with patch('src.distillation.calibrate_int8_topk.AutoTokenizer') as mock_tokenizer:
            mock_tok = Mock()
            mock_tok.encode.return_value = list(range(100))  # Return fixed sequence
            mock_tokenizer.from_pretrained.return_value = mock_tok

            loader = CalibrationDatasetLoader(
                data_path=jsonl_file,
                max_length=100,
                seed=42
            )

            sequences = loader.load_sequences(5)

            assert len(sequences) == 5
            assert all(len(seq) == 100 for seq in sequences)

    def test_load_text(self, tmp_path):
        """Test loading from plain text file."""
        # Create test text file
        text_file = tmp_path / "test.txt"
        with open(text_file, 'w') as f:
            for i in range(5):
                f.write(f"Test sequence {i} with some content\n")

        # Mock tokenizer
        with patch('src.distillation.calibrate_int8_topk.AutoTokenizer') as mock_tokenizer:
            mock_tok = Mock()
            mock_tok.encode.return_value = list(range(100))
            mock_tokenizer.from_pretrained.return_value = mock_tok

            loader = CalibrationDatasetLoader(
                data_path=text_file,
                max_length=100,
                seed=42
            )

            sequences = loader.load_sequences(5)

            assert len(sequences) == 5

    def test_fallback_to_synthetic(self, tmp_path):
        """Test fallback to synthetic when data file not found."""
        loader = CalibrationDatasetLoader(
            data_path=tmp_path / "nonexistent.jsonl",
            max_length=100,
            seed=42
        )

        sequences = loader.load_sequences(10)

        # Should generate synthetic data
        assert len(sequences) == 10
        assert all(len(seq) == 100 for seq in sequences)


class TestInt8Calibrator:
    """Test Int8Calibrator class."""

    def test_ce_loss_computation(self):
        """Test CE loss computation from TopKResponse."""
        # Create mock client
        mock_client = Mock()

        calibrator = Int8Calibrator(mock_client)

        # Create mock response
        B, L, K = 2, 10, 5

        response = TopKResponse(
            indices=[
                [[1, 2, 3, 4, 5] for _ in range(L)]
                for _ in range(B)
            ],
            values_int8=[
                [[100, 80, 60, 40, 20] for _ in range(L)]
                for _ in range(B)
            ],
            scale=[
                [0.01 for _ in range(L)]
                for _ in range(B)
            ],
            other_mass=[
                [0.05 for _ in range(L)]
                for _ in range(B)
            ],
            batch_size=B,
            num_positions=L,
            k=K,
            return_dtype="int8"
        )

        # Create target sequences
        target_sequences = [
            [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            [2, 3, 4, 5, 1, 2, 3, 4, 5, 1]
        ]

        # Compute CE loss
        ce_loss = calibrator._compute_ce_loss(response, target_sequences)

        # Should return a scalar float
        assert isinstance(ce_loss, float)
        assert ce_loss >= 0.0
        assert not np.isnan(ce_loss)
        assert not np.isinf(ce_loss)

    def test_calibration_run_mock(self):
        """Test calibration run with mocked teacher client."""
        # Create mock client
        mock_client = Mock()

        # Mock responses
        B, L, K = 2, 10, 128

        mock_response = TopKResponse(
            indices=[
                [[i for i in range(K)] for _ in range(L)]
                for _ in range(B)
            ],
            values_int8=[
                [[127 - i for i in range(K)] for _ in range(L)]
                for _ in range(B)
            ],
            scale=[
                [0.01 for _ in range(L)]
                for _ in range(B)
            ],
            other_mass=[
                [0.05 for _ in range(L)]
                for _ in range(B)
            ],
            batch_size=B,
            num_positions=L,
            k=K,
            return_dtype="int8"
        )

        mock_client.query_topk.return_value = mock_response

        calibrator = Int8Calibrator(mock_client)

        # Create test sequences
        sequences = [
            [i % 1000 for i in range(L)]
            for _ in range(B)
        ]

        # Run calibration
        result = calibrator.run_calibration(sequences, topk=K)

        # Verify report structure
        assert "ce_fp32" in result
        assert "ce_int8" in result
        assert "delta" in result
        assert "status" in result
        assert "threshold" in result
        assert "recommendations" in result

        # Verify types
        assert isinstance(result["ce_fp32"], float)
        assert isinstance(result["ce_int8"], float)
        assert isinstance(result["delta"], float)
        assert result["status"] in ["PASS", "FAIL"]

        # Verify client was called twice (fp32 and int8)
        assert mock_client.query_topk.call_count == 2

    def test_ce_loss_with_known_distribution(self):
        """Test CE loss computation with a known probability distribution."""
        mock_client = Mock()
        calibrator = Int8Calibrator(mock_client)

        # Create a simple test case with known probabilities
        # Let's say we have 3 top-k tokens with logits [2.0, 1.0, 0.0]
        # After softmax: [0.6652, 0.2447, 0.0900] (unnormalized)
        # If other_mass = 0.1, then top-k probs should be rescaled by 0.9:
        # [0.5987, 0.2202, 0.0810], and tail gets 0.1

        B, L, K = 1, 3, 3

        # Use logits that are easy to compute: [2.0, 1.0, 0.0]
        # With scale=0.01, we need int8 values of [200, 100, 0]
        # But int8 range is [-128, 127], so we use [127, 63, 0] and adjust scale
        # scale = 2.0 / 127 ≈ 0.01575
        scale_val = 2.0 / 127.0

        response = TopKResponse(
            indices=[
                [[10, 20, 30] for _ in range(L)]
            ],
            values_int8=[
                [[127, 63, 0] for _ in range(L)]  # Corresponds to logits [2.0, 0.992, 0.0]
            ],
            scale=[
                [scale_val for _ in range(L)]
            ],
            other_mass=[
                [0.1 for _ in range(L)]
            ],
            batch_size=B,
            num_positions=L,
            k=K,
            return_dtype="int8"
        )

        # Target sequence where:
        # - Position 0 predicts token at position 1
        # - Position 1 predicts token at position 2
        # We'll make position 0 predict token 10 (in top-k), position 1 predict token 999 (in tail)
        target_sequences = [
            [0, 10, 999]  # Doesn't matter what position 0 is, we predict 10 and 999
        ]

        ce_loss = calibrator._compute_ce_loss(response, target_sequences)

        # Verify it's a valid CE loss
        assert isinstance(ce_loss, float)
        assert ce_loss >= 0.0
        assert not np.isnan(ce_loss)
        assert not np.isinf(ce_loss)

        # The CE loss should be reasonable (between 0 and 10 for this test)
        assert 0.0 <= ce_loss <= 10.0


class TestReportGeneration:
    """Test report generation functions."""

    def test_generate_report(self, tmp_path):
        """Test report generation with JSON output."""
        calibration_result = {
            "ce_fp32": 2.5,
            "ce_int8": 2.502,
            "delta": 0.002,
            "threshold": 0.001,
            "status": "FAIL",
            "num_sequences": 128,
            "topk": 128,
            "temperature": 1.0,
            "recommendations": [
                "CE delta (0.002000) exceeds threshold (0.001). Recommend increasing topk from 128 to 256."
            ]
        }

        output_file = tmp_path / "report.json"

        report = generate_report(calibration_result, output_file)

        # Verify JSON file was created
        assert output_file.exists()

        # Verify JSON content
        with open(output_file) as f:
            saved_data = json.load(f)

        assert saved_data == calibration_result

        # Verify human-readable report
        assert "FAIL" in report
        assert "2.500000" in report
        assert "2.502000" in report
        assert "0.002000" in report

    def test_generate_report_pass(self):
        """Test report generation for passing calibration."""
        calibration_result = {
            "ce_fp32": 2.5,
            "ce_int8": 2.5005,
            "delta": 0.0005,
            "threshold": 0.001,
            "status": "PASS",
            "num_sequences": 128,
            "topk": 128,
            "temperature": 1.0,
            "recommendations": []
        }

        report = generate_report(calibration_result)

        assert "PASS" in report
        assert "✓" in report


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
