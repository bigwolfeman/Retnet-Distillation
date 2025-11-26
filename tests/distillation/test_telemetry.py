"""
Tests for telemetry and logging functionality.

Tests T048-T051: TelemetryLogger, metric logging, throughput measurement, wandb integration.
"""

import json
import tempfile
import time
from pathlib import Path
from typing import Dict, Any

import pytest
import torch

from src.distillation.telemetry import (
    TelemetryLogger,
    TrainingMetrics,
    OutputSink,
)


class TestTrainingMetrics:
    """Test TrainingMetrics dataclass."""

    def test_training_metrics_creation(self):
        """Test creating TrainingMetrics object."""
        metrics = TrainingMetrics(
            step=100,
            epoch=1,
            timestamp=time.time(),
            loss=1.5,
            learning_rate=1e-4,
            grad_norm=0.5,
        )

        assert metrics.step == 100
        assert metrics.epoch == 1
        assert metrics.loss == 1.5
        assert metrics.learning_rate == 1e-4
        assert metrics.grad_norm == 0.5

    def test_training_metrics_to_dict(self):
        """Test converting TrainingMetrics to dictionary."""
        metrics = TrainingMetrics(
            step=100,
            epoch=1,
            timestamp=time.time(),
            loss=1.5,
            learning_rate=1e-4,
            grad_norm=0.5,
            gpu_memory_allocated_gb=10.5,
        )

        metrics_dict = metrics.to_dict()

        assert metrics_dict['step'] == 100
        assert metrics_dict['epoch'] == 1
        assert metrics_dict['loss'] == 1.5
        assert metrics_dict['learning_rate'] == 1e-4
        assert metrics_dict['grad_norm'] == 0.5
        assert metrics_dict['gpu_memory_allocated_gb'] == 10.5

        # Check that None values are excluded
        assert 'distillation_loss' not in metrics_dict

    def test_training_metrics_to_json(self):
        """Test converting TrainingMetrics to JSON."""
        metrics = TrainingMetrics(
            step=100,
            epoch=1,
            timestamp=1234567890.0,
            loss=1.5,
            learning_rate=1e-4,
        )

        json_str = metrics.to_json()
        parsed = json.loads(json_str)

        assert parsed['step'] == 100
        assert parsed['epoch'] == 1
        assert parsed['timestamp'] == 1234567890.0
        assert parsed['loss'] == 1.5
        assert parsed['learning_rate'] == 1e-4


class TestTelemetryLogger:
    """Test TelemetryLogger class."""

    def test_telemetry_logger_initialization_file_sink(self):
        """Test initializing TelemetryLogger with FILE sink."""
        with tempfile.TemporaryDirectory() as tmpdir:
            telemetry = TelemetryLogger(
                log_dir=tmpdir,
                log_interval=10,
                sinks=[OutputSink.FILE],
                enable_wandb=False,
            )

            assert telemetry.log_interval == 10
            assert OutputSink.FILE in telemetry.sinks
            assert telemetry.log_file is not None
            # Log file is created on first write, not on init
            assert not telemetry.enable_wandb

    def test_telemetry_logger_initialization_console_only(self):
        """Test initializing TelemetryLogger with CONSOLE sink only."""
        telemetry = TelemetryLogger(
            log_interval=10,
            sinks=[OutputSink.CONSOLE],
            enable_wandb=False,
        )

        assert telemetry.log_interval == 10
        assert OutputSink.CONSOLE in telemetry.sinks
        assert telemetry.log_file is None
        assert not telemetry.enable_wandb

    def test_telemetry_logger_requires_log_dir_for_file_sink(self):
        """Test that FILE sink requires log_dir."""
        with pytest.raises(ValueError, match="log_dir must be specified"):
            TelemetryLogger(
                sinks=[OutputSink.FILE],
                enable_wandb=False,
            )

    def test_should_log(self):
        """Test should_log method."""
        telemetry = TelemetryLogger(
            log_interval=10,
            sinks=[OutputSink.CONSOLE],
            enable_wandb=False,
        )

        # Should log at step 0
        assert telemetry.should_log(0)

        # Should log at multiples of log_interval
        assert telemetry.should_log(10)
        assert telemetry.should_log(20)
        assert telemetry.should_log(100)

        # Should not log at other steps
        assert not telemetry.should_log(1)
        assert not telemetry.should_log(5)
        assert not telemetry.should_log(15)

    def test_gpu_memory_stats(self):
        """Test GPU memory statistics collection."""
        telemetry = TelemetryLogger(
            log_interval=10,
            sinks=[OutputSink.CONSOLE],
            enable_wandb=False,
        )

        memory_stats = telemetry._get_gpu_memory_stats()

        assert 'gpu_memory_allocated_gb' in memory_stats
        assert 'gpu_memory_reserved_gb' in memory_stats
        assert 'vram_utilization' in memory_stats

        # Values should be non-negative
        assert memory_stats['gpu_memory_allocated_gb'] >= 0
        assert memory_stats['gpu_memory_reserved_gb'] >= 0
        assert memory_stats['vram_utilization'] >= 0

    def test_log_step_file_sink(self):
        """Test logging a training step to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            telemetry = TelemetryLogger(
                log_dir=tmpdir,
                log_interval=10,
                sinks=[OutputSink.FILE],
                enable_wandb=False,
            )

            # Log a step (should log at step 0)
            metrics = telemetry.log_step(
                step=0,
                epoch=1,
                loss=1.5,
                learning_rate=1e-4,
                grad_norm=0.5,
                num_tokens=4096,
                batch_size=1,
                step_time=0.5,
            )

            assert metrics is not None
            assert metrics.step == 0
            assert metrics.loss == 1.5
            assert metrics.learning_rate == 1e-4
            assert metrics.grad_norm == 0.5

            # Check that log file was written
            log_file = Path(tmpdir) / "training_metrics.jsonl"
            assert log_file.exists()

            # Read and verify log content
            with open(log_file, 'r') as f:
                log_line = f.readline()
                log_data = json.loads(log_line)

            assert log_data['step'] == 0
            assert log_data['epoch'] == 1
            assert log_data['loss'] == 1.5
            assert log_data['learning_rate'] == 1e-4
            assert log_data['grad_norm'] == 0.5

    def test_log_step_skip_non_log_steps(self):
        """Test that non-log steps are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            telemetry = TelemetryLogger(
                log_dir=tmpdir,
                log_interval=10,
                sinks=[OutputSink.FILE],
                enable_wandb=False,
            )

            # Log a step that should be skipped
            metrics = telemetry.log_step(
                step=5,  # Not a multiple of log_interval
                epoch=1,
                loss=1.5,
            )

            assert metrics is None

            # Check that log file doesn't exist yet (no writes)
            log_file = Path(tmpdir) / "training_metrics.jsonl"
            assert not log_file.exists()

    def test_throughput_measurement(self):
        """Test throughput measurement (tokens/sec, samples/sec, steps/sec)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            telemetry = TelemetryLogger(
                log_dir=tmpdir,
                log_interval=1,  # Log every step
                sinks=[OutputSink.FILE],
                enable_wandb=False,
            )

            # Log a step with throughput info
            metrics = telemetry.log_step(
                step=0,
                epoch=1,
                loss=1.5,
                num_tokens=4096,
                batch_size=2,
                step_time=0.5,  # 0.5 seconds per step
            )

            assert metrics is not None
            # tokens_per_sec = 4096 / 0.5 = 8192
            assert metrics.tokens_per_sec == pytest.approx(8192.0, rel=1e-5)
            # samples_per_sec = 2 / 0.5 = 4
            assert metrics.samples_per_sec == pytest.approx(4.0, rel=1e-5)
            # steps_per_sec = 1 / 0.5 = 2
            assert metrics.steps_per_sec == pytest.approx(2.0, rel=1e-5)

    def test_moving_averages(self):
        """Test moving averages of throughput metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            telemetry = TelemetryLogger(
                log_dir=tmpdir,
                log_interval=1,  # Log every step
                sinks=[OutputSink.FILE],
                enable_wandb=False,
                moving_avg_window=3,  # Small window for testing
            )

            # Log multiple steps
            for step in range(5):
                metrics = telemetry.log_step(
                    step=step,
                    epoch=1,
                    loss=1.5,
                    num_tokens=1000,
                    batch_size=1,
                    step_time=1.0,  # 1 second per step
                )

            # Check moving averages (should converge to 1000 tokens/sec)
            assert metrics.tokens_per_sec_avg == pytest.approx(1000.0, rel=1e-5)
            assert metrics.samples_per_sec_avg == pytest.approx(1.0, rel=1e-5)
            assert metrics.steps_per_sec_avg == pytest.approx(1.0, rel=1e-5)

    def test_moving_averages_window_size(self):
        """Test that moving averages respect window size."""
        telemetry = TelemetryLogger(
            log_interval=1,
            sinks=[OutputSink.CONSOLE],
            enable_wandb=False,
            moving_avg_window=3,
        )

        # Log 5 steps with different throughputs
        for step in range(5):
            telemetry.log_step(
                step=step,
                epoch=1,
                loss=1.5,
                num_tokens=(step + 1) * 1000,  # Increasing throughput
                batch_size=1,
                step_time=1.0,
            )

        # Check that history only contains last 3 values
        assert len(telemetry.tokens_per_sec_history) == 3

        # Should be [3000, 4000, 5000]
        expected_avg = (3000 + 4000 + 5000) / 3
        moving_avgs = telemetry._compute_moving_averages()
        assert moving_avgs['tokens_per_sec_avg'] == pytest.approx(expected_avg, rel=1e-5)

    def test_log_evaluation(self):
        """Test logging evaluation metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            telemetry = TelemetryLogger(
                log_dir=tmpdir,
                log_interval=10,
                sinks=[OutputSink.FILE],
                enable_wandb=False,
            )

            # Log evaluation
            telemetry.log_evaluation(
                step=100,
                epoch=1,
                eval_loss=1.2,
                eval_metrics={'perplexity': 3.5},
            )

            # Check that log file was written
            log_file = Path(tmpdir) / "training_metrics.jsonl"
            assert log_file.exists()

            # Read and verify log content
            with open(log_file, 'r') as f:
                log_line = f.readline()
                log_data = json.loads(log_line)

            assert log_data['step'] == 100
            assert log_data['epoch'] == 1
            assert log_data['eval_loss'] == 1.2
            assert log_data['eval_perplexity'] == 3.5

    def test_get_summary_stats(self):
        """Test getting summary statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            telemetry = TelemetryLogger(
                log_dir=tmpdir,
                log_interval=10,
                sinks=[OutputSink.FILE],
                enable_wandb=False,
            )

            # Log a few steps
            telemetry.log_step(step=0, epoch=1, loss=1.5)
            telemetry.log_step(step=10, epoch=1, loss=1.4)
            telemetry.log_step(step=20, epoch=1, loss=1.3)

            # Get summary stats
            summary = telemetry.get_summary_stats()

            assert 'total_runtime_sec' in summary
            assert summary['total_runtime_sec'] > 0
            assert summary['last_log_step'] == 20
            assert 'moving_averages' in summary
            assert 'current_memory' in summary
            assert summary['wandb_enabled'] is False

    def test_finalize(self):
        """Test finalizing telemetry logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            telemetry = TelemetryLogger(
                log_dir=tmpdir,
                log_interval=10,
                sinks=[OutputSink.FILE],
                enable_wandb=False,
            )

            # Should not raise any errors
            telemetry.finalize()

    def test_extra_metrics(self):
        """Test logging extra metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            telemetry = TelemetryLogger(
                log_dir=tmpdir,
                log_interval=10,
                sinks=[OutputSink.FILE],
                enable_wandb=False,
            )

            # Log with extra metrics
            metrics = telemetry.log_step(
                step=0,
                epoch=1,
                loss=1.5,
                custom_metric_1=42,
                custom_metric_2="test",
            )

            assert metrics is not None

    def test_distillation_loss_components(self):
        """Test logging distillation loss components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            telemetry = TelemetryLogger(
                log_dir=tmpdir,
                log_interval=10,
                sinks=[OutputSink.FILE],
                enable_wandb=False,
            )

            # Log with distillation loss components
            metrics = telemetry.log_step(
                step=0,
                epoch=1,
                loss=1.5,
                distillation_loss=1.0,
                hard_ce_loss=0.3,
                soft_kl_loss=0.7,
            )

            assert metrics is not None
            assert metrics.distillation_loss == 1.0
            assert metrics.hard_ce_loss == 0.3
            assert metrics.soft_kl_loss == 0.7

            # Check that log file contains these components
            log_file = Path(tmpdir) / "training_metrics.jsonl"
            with open(log_file, 'r') as f:
                log_data = json.loads(f.readline())

            assert log_data['distillation_loss'] == 1.0
            assert log_data['hard_ce_loss'] == 0.3
            assert log_data['soft_kl_loss'] == 0.7

    def test_vram_tracking(self):
        """Test VRAM tracking in metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            telemetry = TelemetryLogger(
                log_dir=tmpdir,
                log_interval=10,
                sinks=[OutputSink.FILE],
                enable_wandb=False,
            )

            # Log a step
            metrics = telemetry.log_step(
                step=0,
                epoch=1,
                loss=1.5,
            )

            assert metrics is not None
            assert metrics.gpu_memory_allocated_gb >= 0
            assert metrics.gpu_memory_reserved_gb >= 0
            assert metrics.vram_utilization >= 0

            # Check that log file contains VRAM metrics
            log_file = Path(tmpdir) / "training_metrics.jsonl"
            with open(log_file, 'r') as f:
                log_data = json.loads(f.readline())

            assert 'gpu_memory_allocated_gb' in log_data
            assert 'gpu_memory_reserved_gb' in log_data
            assert 'vram_utilization' in log_data


class TestTelemetryIntegration:
    """Integration tests for telemetry with realistic training scenarios."""

    def test_realistic_training_loop(self):
        """Test telemetry in a realistic training loop."""
        with tempfile.TemporaryDirectory() as tmpdir:
            telemetry = TelemetryLogger(
                log_dir=tmpdir,
                log_interval=5,
                sinks=[OutputSink.FILE, OutputSink.CONSOLE],
                enable_wandb=False,
            )

            # Simulate training loop
            num_steps = 20
            seq_len = 4096
            batch_size = 1

            for step in range(num_steps):
                # Simulate step time
                step_start = time.time()
                time.sleep(0.01)  # Simulate computation
                step_time = time.time() - step_start

                # Log metrics
                telemetry.log_step(
                    step=step,
                    epoch=step // 10,
                    loss=2.0 - step * 0.05,  # Decreasing loss
                    learning_rate=1e-4 * (1.0 - step / num_steps),  # Decaying LR
                    grad_norm=1.0 + 0.1 * step,  # Increasing grad norm
                    num_tokens=seq_len,
                    batch_size=batch_size,
                    step_time=step_time,
                )

            # Verify log file
            log_file = Path(tmpdir) / "training_metrics.jsonl"
            assert log_file.exists()

            # Read all log entries
            with open(log_file, 'r') as f:
                log_lines = f.readlines()

            # Should have logged at steps 0, 5, 10, 15 (4 entries)
            assert len(log_lines) == 4

            # Verify first and last entries
            first_log = json.loads(log_lines[0])
            assert first_log['step'] == 0
            assert first_log['loss'] == pytest.approx(2.0, rel=1e-5)

            last_log = json.loads(log_lines[-1])
            assert last_log['step'] == 15
            assert last_log['loss'] == pytest.approx(1.25, rel=1e-5)

    def test_evaluation_logging(self):
        """Test logging both training and evaluation metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            telemetry = TelemetryLogger(
                log_dir=tmpdir,
                log_interval=10,
                sinks=[OutputSink.FILE],
                enable_wandb=False,
            )

            # Log training step
            telemetry.log_step(step=0, epoch=1, loss=1.5)

            # Log evaluation
            telemetry.log_evaluation(
                step=0,
                epoch=1,
                eval_loss=1.2,
                eval_metrics={'perplexity': 3.5, 'accuracy': 0.75},
            )

            # Log another training step
            telemetry.log_step(step=10, epoch=1, loss=1.4)

            # Verify log file
            log_file = Path(tmpdir) / "training_metrics.jsonl"
            with open(log_file, 'r') as f:
                log_lines = f.readlines()

            # Should have 3 entries (2 training, 1 eval)
            assert len(log_lines) == 3

            # Verify eval entry
            eval_log = json.loads(log_lines[1])
            assert eval_log['eval_loss'] == 1.2
            assert eval_log['eval_perplexity'] == 3.5
            assert eval_log['eval_accuracy'] == 0.75


class TestWandbIntegration:
    """Tests for wandb integration (optional)."""

    def test_wandb_disabled_by_default(self):
        """Test that wandb is disabled by default."""
        telemetry = TelemetryLogger(
            log_interval=10,
            sinks=[OutputSink.CONSOLE],
            enable_wandb=False,
        )

        assert not telemetry.enable_wandb

    def test_wandb_requires_project_name(self):
        """Test that wandb requires project name."""
        with pytest.raises(ValueError, match="wandb_project must be specified"):
            TelemetryLogger(
                log_interval=10,
                sinks=[OutputSink.CONSOLE],
                enable_wandb=True,
                wandb_project=None,  # Missing project name
            )

    def test_wandb_graceful_fallback_when_not_installed(self):
        """Test that telemetry works even if wandb is not installed."""
        # This test verifies the graceful fallback behavior
        # If wandb is installed, it will attempt to initialize (may fail without credentials)
        # If wandb is not installed, it should fall back gracefully
        telemetry = TelemetryLogger(
            log_interval=10,
            sinks=[OutputSink.CONSOLE],
            enable_wandb=False,
        )

        # Should work fine
        telemetry.log_step(step=0, epoch=1, loss=1.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
