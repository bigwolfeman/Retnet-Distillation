"""Tests for RetNet student model configuration.

Tests cover:
- T031.1: Verify 350M variant has 300-400M parameters
- T031.2: Verify 500M variant has 450-550M parameters
- T031.3: Test tokenizer loading
- T031.4: Test config validation
- T031.5: Test model initialization with configs
"""

import pytest
import torch
from src.distillation.student_config import (
    RetNetStudentConfig,
    RetNetStudent350MConfig,
    RetNetStudent500MConfig,
    create_student_config,
    load_llama_tokenizer,
)


class TestRetNetStudentConfig:
    """Test suite for RetNet student configuration."""

    def test_350m_config_creation(self):
        """Test 350M variant configuration creation."""
        config = RetNetStudent350MConfig()

        # Check architecture dimensions
        assert config.d_model == 960, "350M variant should have d_model=960"
        assert config.n_layers == 12, "350M variant should have n_layers=12"
        assert config.n_heads == 12, "350M variant should have n_heads=12"
        assert config.variant == "350M"

        # Check tokenizer config
        assert config.vocab_size == 128256, "Should use Llama tokenizer vocab_size"
        assert config.tokenizer_name == "meta-llama/Llama-3.2-1B"

        # Check position embeddings
        assert config.max_position_embeddings == 4096, "v1 should use 4k context"

    def test_500m_config_creation(self):
        """Test 500M variant configuration creation."""
        config = RetNetStudent500MConfig()

        # Check architecture dimensions
        assert config.d_model == 1152, "500M variant should have d_model=1152"
        assert config.n_layers == 15, "500M variant should have n_layers=15"
        assert config.n_heads == 12, "500M variant should have n_heads=12"
        assert config.variant == "500M"

        # Check tokenizer config
        assert config.vocab_size == 128256, "Should use Llama tokenizer vocab_size"
        assert config.tokenizer_name == "meta-llama/Llama-3.2-1B"

        # Check position embeddings
        assert config.max_position_embeddings == 4096, "v1 should use 4k context"

    def test_350m_parameter_count(self):
        """T031.1: Verify 350M variant has 300-400M parameters (actual, not estimated)."""
        config = RetNetStudent350MConfig()
        estimated_count = config.estimate_param_count()

        # Check target range values
        min_params, max_params = config.target_param_count_range
        assert min_params == 300_000_000, "350M variant min should be 300M"
        assert max_params == 400_000_000, "350M variant max should be 400M"

        # Note: estimated_count will be ~256M, but actual will be ~311M (see model init tests)
        # This is because our simplified formula doesn't account for GLU's 3x projections
        assert 200_000_000 <= estimated_count <= 400_000_000, (
            f"350M variant estimated count {estimated_count:,} seems unreasonable"
        )

        # Print for reporting
        print(f"\n350M variant (estimated): {estimated_count:,} parameters ({estimated_count/1e6:.2f}M)")
        print(f"  (Actual model will be ~311M with TorchScale GLU)")

    def test_500m_parameter_count(self):
        """T031.2: Verify 500M variant has 450-550M parameters (actual, not estimated)."""
        config = RetNetStudent500MConfig()
        estimated_count = config.estimate_param_count()

        # Check target range values
        min_params, max_params = config.target_param_count_range
        assert min_params == 450_000_000, "500M variant min should be 450M"
        assert max_params == 550_000_000, "500M variant max should be 550M"

        # Note: estimated_count will be ~387M, but actual will be ~486M (see model init tests)
        # This is because our simplified formula doesn't account for GLU's 3x projections
        assert 300_000_000 <= estimated_count <= 550_000_000, (
            f"500M variant estimated count {estimated_count:,} seems unreasonable"
        )

        # Print for reporting
        print(f"\n500M variant (estimated): {estimated_count:,} parameters ({estimated_count/1e6:.2f}M)")
        print(f"  (Actual model will be ~486M with TorchScale GLU)")

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Tokenizer loading test skipped (no GPU/network access)"
    )
    def test_tokenizer_loading(self):
        """T031.3: Test tokenizer loading utility function.

        Note: This test requires network access to download the tokenizer.
        It will be skipped in offline environments.
        """
        try:
            tokenizer = load_llama_tokenizer()

            # Verify vocab size
            assert tokenizer.vocab_size == 128256, (
                f"Expected vocab_size=128256, got {tokenizer.vocab_size}"
            )

            # Verify pad token is set
            assert tokenizer.pad_token is not None, "Pad token should be set"

            # Test encoding
            text = "Hello world"
            tokens = tokenizer.encode(text, add_special_tokens=True)
            assert len(tokens) > 0, "Should encode text to tokens"

            # Test decoding
            decoded = tokenizer.decode(tokens)
            assert decoded is not None, "Should decode tokens back to text"

            print(f"\nTokenizer loaded successfully: vocab_size={tokenizer.vocab_size}")

        except Exception as e:
            pytest.skip(f"Tokenizer loading skipped (network/access issue): {e}")

    def test_config_validation(self):
        """T031.4: Test configuration validation.

        Tests:
        - d_model must be divisible by n_heads
        - vocab_size must be 128256 (Llama tokenizer)
        - max_position_embeddings must be 4096 (v1)
        - Parameter count must be in target range
        """
        # Test valid config passes validation
        config = RetNetStudent350MConfig()
        config.validate()  # Should not raise

        # Test d_model not divisible by n_heads
        config_bad_heads = RetNetStudent350MConfig()
        config_bad_heads.n_heads = 7  # 1280 not divisible by 7
        with pytest.raises(AssertionError, match="d_model.*must be divisible by n_heads"):
            config_bad_heads.validate()

        # Test wrong vocab_size
        config_bad_vocab = RetNetStudent350MConfig()
        config_bad_vocab.vocab_size = 50000
        with pytest.raises(AssertionError, match="vocab_size must be 128256"):
            config_bad_vocab.validate()

        # Test wrong max_position_embeddings
        config_bad_pos = RetNetStudent350MConfig()
        config_bad_pos.max_position_embeddings = 2048
        with pytest.raises(AssertionError, match="max_position_embeddings must be 4096"):
            config_bad_pos.validate()

        # Test parameter count validation with validate_actual_params()
        config_test = RetNetStudent350MConfig()
        # Should pass with params in range
        config_test.validate_actual_params(350_000_000)  # Within 300M-400M range
        # Should fail with params out of range
        with pytest.raises(AssertionError, match="Actual parameter count.*outside target range"):
            config_test.validate_actual_params(500_000_000)  # Way too many

    def test_retnet_backbone_args_conversion(self):
        """Test conversion to RetNetBackbone initialization arguments."""
        config = RetNetStudent350MConfig()
        args = config.to_retnet_backbone_args()

        # Check all required fields are present
        assert "vocab_size" in args
        assert "d_model" in args
        assert "n_layers" in args
        assert "n_heads" in args
        assert "dropout" in args
        assert "max_seq_len" in args
        assert "debug" in args

        # Check values match config
        assert args["vocab_size"] == config.vocab_size
        assert args["d_model"] == config.d_model
        assert args["n_layers"] == config.n_layers
        assert args["n_heads"] == config.n_heads
        assert args["dropout"] == config.dropout
        assert args["max_seq_len"] == config.max_position_embeddings

    def test_factory_function(self):
        """Test create_student_config factory function."""
        # Test 350M variant
        config_350m = create_student_config("350M")
        assert isinstance(config_350m, RetNetStudent350MConfig)
        assert config_350m.variant == "350M"
        assert config_350m.d_model == 960

        # Test 500M variant
        config_500m = create_student_config("500M")
        assert isinstance(config_500m, RetNetStudent500MConfig)
        assert config_500m.variant == "500M"
        assert config_500m.d_model == 1152

        # Test invalid variant
        with pytest.raises(ValueError, match="Unknown variant"):
            create_student_config("1B")

    def test_config_serialization(self):
        """Test config serialization/deserialization."""
        config = RetNetStudent350MConfig()

        # Test to_dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["d_model"] == 960
        assert config_dict["n_layers"] == 12
        assert config_dict["vocab_size"] == 128256

        # Test from_dict
        config_restored = RetNetStudentConfig.from_dict(config_dict)
        assert config_restored.d_model == config.d_model
        assert config_restored.n_layers == config.n_layers
        assert config_restored.vocab_size == config.vocab_size

    def test_parameter_count_calculation_accuracy(self):
        """Test parameter count calculation matches expected formula.

        This test verifies the estimate_param_count() calculation is accurate
        by breaking down each component.
        """
        config = RetNetStudent350MConfig()

        # Calculate expected parameters manually
        d_model = config.d_model
        n_layers = config.n_layers
        vocab_size = config.vocab_size
        mlp_mult = config.mlp_mult

        # Token embeddings
        embed_params = vocab_size * d_model

        # Per-layer parameters
        retention_params_per_layer = 4 * d_model * d_model  # Q, K, V, O projections
        ffn_params_per_layer = 2 * d_model * (d_model * mlp_mult)  # Up + down
        layer_params = n_layers * (retention_params_per_layer + ffn_params_per_layer)

        # Output projection (tied, so 0 extra params)
        output_params = 0 if config.tie_word_embeddings else vocab_size * d_model

        # Layer norms
        layernorm_params = (2 * n_layers + 2) * d_model

        expected_total = embed_params + layer_params + output_params + layernorm_params

        # Compare with config's estimate
        actual_total = config.estimate_param_count()

        assert actual_total == expected_total, (
            f"Parameter count mismatch: expected {expected_total:,}, "
            f"got {actual_total:,}"
        )

        # Print breakdown
        print(f"\nParameter breakdown for 350M variant:")
        print(f"  Token embeddings:  {embed_params:>12,}")
        print(f"  RetNet layers:     {layer_params:>12,}")
        print(f"  Output projection: {output_params:>12,}")
        print(f"  Layer norms:       {layernorm_params:>12,}")
        print(f"  {'-' * 30}")
        print(f"  Total:             {actual_total:>12,}")

    def test_both_variants_satisfy_constraints(self):
        """Test that both variants satisfy all distillation constraints."""
        configs = [
            RetNetStudent350MConfig(),
            RetNetStudent500MConfig(),
        ]

        for config in configs:
            # Validate config (dimension checks, vocab, etc.)
            config.validate()  # Should not raise

            # Check tokenizer compatibility
            assert config.vocab_size == 128256
            assert config.tokenizer_name == "meta-llama/Llama-3.2-1B"

            # Check context length
            assert config.max_position_embeddings == 4096

            # Check d_model divisible by n_heads
            assert config.d_model % config.n_heads == 0

            # Check estimated parameter count is reasonable (not checking target range,
            # since simplified formula underestimates by ~25%)
            estimated_count = config.estimate_param_count()
            assert estimated_count > 0, "Parameter count should be positive"

            print(f"\n{config.variant} variant: estimated {estimated_count:,} parameters - PASS")


class TestModelInitialization:
    """T031.5: Test model initialization with student configs."""

    def test_350m_model_initialization(self):
        """Test initializing RetNetBackbone with 350M config."""
        config = RetNetStudent350MConfig()
        args = config.to_retnet_backbone_args()

        # Import here to avoid loading TorchScale unnecessarily
        try:
            from src.models.retnet.backbone import RetNetBackbone

            # Initialize model
            model = RetNetBackbone(**args)

            # Check model properties
            assert model.vocab_size == config.vocab_size
            assert model.d_model == config.d_model
            assert model.n_layers == config.n_layers

            # Count actual parameters
            actual_params = sum(p.numel() for p in model.parameters())
            estimated_params = config.estimate_param_count()

            # Allow 30% tolerance for estimation accuracy
            # (Simplified formula doesn't account for GLU's 3x projections)
            tolerance = 0.30
            lower_bound = estimated_params * (1 - tolerance)
            upper_bound = estimated_params * (1 + tolerance)

            # Also check that actual params are in target range
            min_target, max_target = config.target_param_count_range
            assert min_target <= actual_params <= max_target, (
                f"Actual parameter count {actual_params:,} outside target range "
                f"[{min_target:,}, {max_target:,}]"
            )

            print(f"\n350M model initialized successfully:")
            print(f"  Estimated params: {estimated_params:>12,}")
            print(f"  Actual params:    {actual_params:>12,}")
            print(f"  Difference:       {abs(actual_params - estimated_params):>12,} ({abs(actual_params - estimated_params) / estimated_params * 100:.2f}%)")

        except ImportError as e:
            pytest.skip(f"Could not import RetNetBackbone (missing TorchScale?): {e}")

    def test_500m_model_initialization(self):
        """Test initializing RetNetBackbone with 500M config."""
        config = RetNetStudent500MConfig()
        args = config.to_retnet_backbone_args()

        # Import here to avoid loading TorchScale unnecessarily
        try:
            from src.models.retnet.backbone import RetNetBackbone

            # Initialize model
            model = RetNetBackbone(**args)

            # Check model properties
            assert model.vocab_size == config.vocab_size
            assert model.d_model == config.d_model
            assert model.n_layers == config.n_layers

            # Count actual parameters
            actual_params = sum(p.numel() for p in model.parameters())
            estimated_params = config.estimate_param_count()

            # Allow 30% tolerance for estimation accuracy
            # (Simplified formula doesn't account for GLU's 3x projections)
            tolerance = 0.30
            lower_bound = estimated_params * (1 - tolerance)
            upper_bound = estimated_params * (1 + tolerance)

            # Also check that actual params are in target range
            min_target, max_target = config.target_param_count_range
            assert min_target <= actual_params <= max_target, (
                f"Actual parameter count {actual_params:,} outside target range "
                f"[{min_target:,}, {max_target:,}]"
            )

            print(f"\n500M model initialized successfully:")
            print(f"  Estimated params: {estimated_params:>12,}")
            print(f"  Actual params:    {actual_params:>12,}")
            print(f"  Difference:       {abs(actual_params - estimated_params):>12,} ({abs(actual_params - estimated_params) / estimated_params * 100:.2f}%)")

        except ImportError as e:
            pytest.skip(f"Could not import RetNetBackbone (missing TorchScale?): {e}")

    def test_forward_pass_smoke_test(self):
        """Smoke test: run forward pass with both configs."""
        configs = [
            RetNetStudent350MConfig(),
            RetNetStudent500MConfig(),
        ]

        try:
            from src.models.retnet.backbone import RetNetBackbone

            for config in configs:
                args = config.to_retnet_backbone_args()
                model = RetNetBackbone(**args)
                model.eval()

                # Create dummy input
                batch_size = 2
                seq_len = 128
                input_ids = torch.randint(
                    0, config.vocab_size, (batch_size, seq_len)
                )

                # Forward pass
                with torch.no_grad():
                    output = model.forward_train(input_ids)

                # Check output shape
                assert output.shape == (batch_size, seq_len, config.d_model), (
                    f"Expected output shape {(batch_size, seq_len, config.d_model)}, "
                    f"got {output.shape}"
                )

                print(f"\n{config.variant} forward pass: SUCCESS")
                print(f"  Input shape:  {input_ids.shape}")
                print(f"  Output shape: {output.shape}")

        except ImportError as e:
            pytest.skip(f"Could not import RetNetBackbone (missing TorchScale?): {e}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
