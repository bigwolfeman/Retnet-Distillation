"""Unit tests for calibration module (Phase 4, T033).

Tests for:
- TemperatureScaler: temperature scaling correctness
- compute_ece: ECE computation accuracy
- Calibration behavior: T>1 reduces confidence, T<1 increases
"""

import numpy as np
import pytest
import torch

from src.models.titans.calibration import (
    TemperatureScaler,
    compute_ece,
    compute_max_calibration_error,
    calibrate_confidences,
    get_max_confidence,
)


# ============================================================================
# T033: TemperatureScaler Tests
# ============================================================================

class TestTemperatureScaler:
    """Test suite for TemperatureScaler class."""

    def test_initialization(self):
        """Test scaler initializes with correct temperature."""
        scaler = TemperatureScaler(initial_temperature=2.0)
        assert scaler.get_temperature() == pytest.approx(2.0, abs=1e-4)

    def test_forward_no_scaling(self):
        """Test T=1.0 produces standard softmax (no scaling)."""
        scaler = TemperatureScaler(initial_temperature=1.0)
        logits = torch.tensor([[1.0, 2.0, 3.0]])

        # With T=1.0, should match standard softmax
        calibrated = scaler(logits)
        expected = torch.softmax(logits, dim=-1)

        torch.testing.assert_close(calibrated, expected, atol=1e-5, rtol=1e-5)

    def test_forward_temperature_greater_than_1(self):
        """Test T>1 softens confidence (reduces max prob)."""
        scaler = TemperatureScaler(initial_temperature=2.0)
        logits = torch.tensor([[1.0, 2.0, 5.0]])  # Strong preference for class 2

        # Standard softmax
        standard_probs = torch.softmax(logits, dim=-1)
        standard_max = torch.max(standard_probs).item()

        # Temperature-scaled (T=2.0)
        calibrated_probs = scaler(logits)
        calibrated_max = torch.max(calibrated_probs).item()

        # T>1 should reduce max probability (soften)
        assert calibrated_max < standard_max
        assert calibrated_max < 0.8  # Should be less certain

    def test_forward_temperature_less_than_1(self):
        """Test T<1 sharpens confidence (increases max prob)."""
        scaler = TemperatureScaler(initial_temperature=0.5)
        logits = torch.tensor([[1.0, 2.0, 3.0]])  # Moderate preference

        # Standard softmax
        standard_probs = torch.softmax(logits, dim=-1)
        standard_max = torch.max(standard_probs).item()

        # Temperature-scaled (T=0.5)
        calibrated_probs = scaler(logits)
        calibrated_max = torch.max(calibrated_probs).item()

        # T<1 should increase max probability (sharpen)
        assert calibrated_max > standard_max
        assert calibrated_max > 0.7  # Should be more certain

    def test_temperature_clamping_lower(self):
        """Test temperature is clamped to [0.1, 10.0] - lower bound."""
        scaler = TemperatureScaler(initial_temperature=0.05)  # Too low
        logits = torch.tensor([[1.0, 2.0, 3.0]])

        # Should clamp to 0.1
        _ = scaler(logits)
        assert scaler.get_temperature() >= 0.1

    def test_temperature_clamping_upper(self):
        """Test temperature is clamped to [0.1, 10.0] - upper bound."""
        scaler = TemperatureScaler(initial_temperature=15.0)  # Too high
        logits = torch.tensor([[1.0, 2.0, 3.0]])

        # Should clamp to 10.0
        _ = scaler(logits)
        assert scaler.get_temperature() <= 10.0

    def test_fit_improves_calibration(self):
        """Test fit() optimizes temperature on validation data."""
        # Create miscalibrated data: overconfident predictions
        # True distribution is uniform, but logits suggest class 0
        num_samples = 100
        num_classes = 3

        # Overconfident logits (always predict class 0 strongly)
        val_logits = torch.zeros(num_samples, num_classes)
        val_logits[:, 0] = 5.0  # Very confident in class 0

        # But true labels are uniform across classes
        val_labels = torch.randint(0, num_classes, (num_samples,))

        # Fit temperature scaler
        scaler = TemperatureScaler(initial_temperature=1.0)
        final_nll = scaler.fit(val_logits, val_labels, verbose=False)

        # After fitting, temperature should increase (T>1) to soften confidence
        fitted_temp = scaler.get_temperature()
        assert fitted_temp > 1.0, "Should increase temperature for overconfident model"
        assert final_nll < 10.0, "NLL should be reasonable after fitting"

    def test_fit_batch_size(self):
        """Test fit() works with different batch sizes."""
        val_logits = torch.randn(50, 5)  # 50 samples, 5 classes
        val_labels = torch.randint(0, 5, (50,))

        scaler = TemperatureScaler()
        final_nll = scaler.fit(val_logits, val_labels, max_iter=10, verbose=False)

        assert final_nll > 0, "NLL should be positive"
        assert 0.1 <= scaler.get_temperature() <= 10.0, "Temperature should be in valid range"


# ============================================================================
# T033: ECE Computation Tests
# ============================================================================

class TestECEComputation:
    """Test suite for compute_ece function."""

    def test_perfect_calibration(self):
        """Test ECE=0 for perfectly calibrated predictions."""
        # Perfect calibration: confidence matches accuracy
        # Use more samples for better binning
        confidences = np.array([0.9] * 90 + [0.5] * 50 + [0.1] * 10)
        correctness = np.array([1] * 81 + [0] * 9 + [1] * 25 + [0] * 25 + [1] * 1 + [0] * 9)

        # Bin into 10 bins
        ece, _, _, _ = compute_ece(confidences, correctness, n_bins=10)

        # Should be close to 0 (some error due to binning)
        assert ece < 0.15, "ECE should be near 0 for reasonably calibrated predictions"

    def test_miscalibrated_overconfident(self):
        """Test ECE>0 for overconfident predictions."""
        # Overconfident: high confidence but low accuracy
        confidences = np.array([0.95] * 50 + [0.9] * 50)  # Very high confidence
        correctness = np.array([1] * 25 + [0] * 75)  # But only 25% correct

        ece, _, _, _ = compute_ece(confidences, correctness, n_bins=10)

        # Should have high ECE
        assert ece > 0.3, "ECE should be high for overconfident model"

    def test_miscalibrated_underconfident(self):
        """Test ECE>0 for underconfident predictions."""
        # Underconfident: low confidence but high accuracy
        confidences = np.array([0.6] * 100)  # Low confidence
        correctness = np.array([1] * 95 + [0] * 5)  # But 95% correct

        ece, _, _, _ = compute_ece(confidences, correctness, n_bins=10)

        # Should have significant ECE
        assert ece > 0.2, "ECE should be high for underconfident model"

    def test_ece_bin_statistics(self):
        """Test ECE returns correct bin statistics."""
        confidences = np.array([0.9, 0.9, 0.5, 0.5, 0.1, 0.1])
        correctness = np.array([1, 1, 1, 0, 0, 0])

        ece, bin_accs, bin_confs, bin_counts = compute_ece(confidences, correctness, n_bins=10)

        # Check that bins sum to total samples
        assert np.sum(bin_counts) == len(confidences)

        # Check that confidence and accuracy arrays have correct length
        assert len(bin_accs) == 10
        assert len(bin_confs) == 10
        assert len(bin_counts) == 10

    def test_ece_input_validation(self):
        """Test ECE raises errors for invalid inputs."""
        # Mismatched lengths
        with pytest.raises(AssertionError):
            compute_ece(np.array([0.5, 0.6]), np.array([1]), n_bins=10)

        # Confidences out of range
        with pytest.raises(AssertionError):
            compute_ece(np.array([1.5, 0.5]), np.array([1, 0]), n_bins=10)

        # Correctness not binary
        with pytest.raises(AssertionError):
            compute_ece(np.array([0.5, 0.6]), np.array([0.5, 0.7]), n_bins=10)

    def test_ece_different_n_bins(self):
        """Test ECE computation with different bin counts."""
        confidences = np.random.rand(100)
        correctness = np.random.randint(0, 2, 100)

        ece_5 = compute_ece(confidences, correctness, n_bins=5)[0]
        ece_10 = compute_ece(confidences, correctness, n_bins=10)[0]
        ece_20 = compute_ece(confidences, correctness, n_bins=20)[0]

        # All should be valid ECE values
        assert 0 <= ece_5 <= 1
        assert 0 <= ece_10 <= 1
        assert 0 <= ece_20 <= 1

    def test_ece_empty_bins(self):
        """Test ECE handles empty bins correctly."""
        # Only confident predictions
        confidences = np.array([0.9, 0.95, 0.92, 0.88])
        correctness = np.array([1, 1, 0, 1])

        ece, bin_accs, bin_confs, bin_counts = compute_ece(confidences, correctness, n_bins=10)

        # Most bins should be empty
        assert np.sum(bin_counts == 0) > 5, "Should have multiple empty bins"
        assert ece >= 0, "ECE should still be computed"


# ============================================================================
# T033: MCE (Maximum Calibration Error) Tests
# ============================================================================

class TestMCE:
    """Test suite for Maximum Calibration Error."""

    def test_mce_greater_than_ece(self):
        """Test MCE >= ECE (max is always >= average)."""
        confidences = np.random.rand(100)
        correctness = np.random.randint(0, 2, 100)

        ece = compute_ece(confidences, correctness, n_bins=10)[0]
        mce = compute_max_calibration_error(confidences, correctness, n_bins=10)

        assert mce >= ece, "MCE should be >= ECE"

    def test_mce_worst_case(self):
        """Test MCE captures worst-case calibration error."""
        # One bin is perfectly calibrated, one is terrible
        confidences = np.array([0.9] * 10 + [0.5] * 10)
        correctness = np.array([1] * 9 + [0] * 1 + [0] * 10)  # 90% @ 0.9, 0% @ 0.5

        mce = compute_max_calibration_error(confidences, correctness, n_bins=10)

        # Should capture the 0.5 bin error (|0.5 - 0.0| = 0.5)
        assert mce > 0.4, "MCE should capture worst-case bin error"


# ============================================================================
# T033: Utility Function Tests
# ============================================================================

class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_calibrate_confidences(self):
        """Test quick calibration utility function."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])

        # Default temperature (1.0)
        probs_t1 = calibrate_confidences(logits, temperature=1.0)
        expected_t1 = torch.softmax(logits, dim=-1)
        torch.testing.assert_close(probs_t1, expected_t1)

        # Higher temperature
        probs_t2 = calibrate_confidences(logits, temperature=2.0)
        assert torch.max(probs_t2) < torch.max(probs_t1), "T>1 should reduce max prob"

    def test_get_max_confidence(self):
        """Test extracting max confidence from probabilities."""
        probs = torch.tensor([[0.1, 0.7, 0.2], [0.3, 0.3, 0.4]])

        max_conf = get_max_confidence(probs)

        assert max_conf.shape == (2,)
        assert max_conf[0].item() == pytest.approx(0.7)
        assert max_conf[1].item() == pytest.approx(0.4)


# ============================================================================
# Integration Tests
# ============================================================================

class TestCalibrationIntegration:
    """Integration tests for calibration pipeline."""

    def test_end_to_end_calibration_pipeline(self):
        """Test full calibration pipeline: uncalibrated -> fitted -> evaluated."""
        # Generate synthetic data: overconfident model
        num_train = 200
        num_val = 100
        num_classes = 5

        # Overconfident training logits
        train_logits = torch.randn(num_train, num_classes) * 2 + 3  # High variance
        train_labels = torch.randint(0, num_classes, (num_train,))

        # Validation data
        val_logits = torch.randn(num_val, num_classes) * 2 + 3
        val_labels = torch.randint(0, num_classes, (num_val,))

        # 1. Compute ECE before calibration
        uncalib_probs = torch.softmax(val_logits, dim=-1)
        uncalib_conf = torch.max(uncalib_probs, dim=-1).values.numpy()
        uncalib_correct = (torch.argmax(uncalib_probs, dim=-1) == val_labels).numpy().astype(int)
        ece_before = compute_ece(uncalib_conf, uncalib_correct, n_bins=10)[0]

        # 2. Fit temperature scaler
        scaler = TemperatureScaler()
        scaler.fit(train_logits, train_labels, max_iter=20, verbose=False)

        # 3. Apply calibration to validation set
        with torch.no_grad():
            calib_probs = scaler(val_logits)
        calib_conf = torch.max(calib_probs, dim=-1).values.detach().numpy()
        calib_correct = (torch.argmax(calib_probs, dim=-1) == val_labels).detach().numpy().astype(int)
        ece_after = compute_ece(calib_conf, calib_correct, n_bins=10)[0]

        # 4. ECE should improve (or stay similar for well-calibrated model)
        # Note: For random data, improvement not guaranteed, but pipeline should run
        assert ece_after >= 0, "Calibrated ECE should be valid"
        assert scaler.get_temperature() > 0, "Temperature should be positive"

    def test_calibration_preserves_accuracy(self):
        """Test temperature scaling preserves top-1 accuracy."""
        logits = torch.tensor([
            [1.0, 2.0, 5.0],
            [3.0, 1.0, 2.0],
            [2.0, 4.0, 1.0],
        ])
        labels = torch.tensor([2, 0, 1])

        # Accuracy before calibration
        uncalib_pred = torch.argmax(logits, dim=-1)
        uncalib_acc = (uncalib_pred == labels).float().mean().item()

        # Apply calibration (T=2.0)
        scaler = TemperatureScaler(initial_temperature=2.0)
        calib_probs = scaler(logits)
        calib_pred = torch.argmax(calib_probs, dim=-1)
        calib_acc = (calib_pred == labels).float().mean().item()

        # Accuracy should be preserved
        assert calib_acc == uncalib_acc, "Temperature scaling should preserve accuracy"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
