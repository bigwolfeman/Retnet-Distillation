"""
Shared fixtures for unit tests.

This module provides common test fixtures for RheaNet retention tests
and Titans architecture tests.
"""

import pytest
import torch


# ============================================================================
# RheaNet Retention Fixtures
# ============================================================================

@pytest.fixture
def synthetic_sequence():
    """
    Random sequence for retention tests.

    Returns:
        torch.Tensor: [B=2, T=128, D=64] random sequence
    """
    return torch.randn(2, 128, 64)


@pytest.fixture
def retention_config():
    """
    Test configuration for retention.

    Returns:
        TitanMACConfig with retention enabled and test-friendly parameters
    """
    from src.models.titans.titan_config import TitanMACConfig

    return TitanMACConfig(
        d_model=64,
        n_heads=4,
        d_head=16,
        n_layers=4,
        d_ff=256,
        use_retention=True,
        retention_block_len=16,
        n_persistent_tokens=8,
        n_memory_tokens=16,
        mixer_window_size=128,
    )


@pytest.fixture
def retention_state():
    """
    Initial retention state for testing.

    Returns:
        RetentionState: Zero-initialized state [B=2, H=4, d_head=16, d_head=16]
    """
    from src.models.titans.retention_block import RetentionState

    return RetentionState.create_initial(
        batch_size=2,
        n_heads=4,
        d_head=16,
        device=torch.device('cpu'),
        dtype=torch.float32
    )


@pytest.fixture
def long_sequence():
    """
    Long sequence for streaming and chunk tests.

    Returns:
        torch.Tensor: [B=1, T=1024, D=64] long sequence
    """
    return torch.randn(1, 1024, 64)


@pytest.fixture
def chunked_sequence():
    """
    Sequence split into chunks for chunk-wise testing.

    Returns:
        list of torch.Tensor: 8 chunks of [B=1, T=128, D=64]
    """
    full_seq = torch.randn(1, 1024, 64)
    chunk_size = 128
    chunks = []
    for i in range(0, 1024, chunk_size):
        chunks.append(full_seq[:, i:i+chunk_size, :])
    return chunks


# ============================================================================
# Memory and MAC Fixtures
# ============================================================================

@pytest.fixture
def memory_config():
    """
    Configuration for memory tests.

    Returns:
        TitanMACConfig with memory enabled
    """
    from src.models.titans.titan_config import TitanMACConfig

    return TitanMACConfig(
        d_model=64,
        n_heads=4,
        d_head=16,
        n_layers=4,
        d_ff=256,
        use_retention=True,
        n_persistent_tokens=8,
        n_memory_tokens=16,
        num_memory_layers=2,
    )


@pytest.fixture
def persistent_tokens():
    """
    Learnable persistent tokens for MAC tests.

    Returns:
        torch.Tensor: [B=2, N_p=8, D=64] persistent tokens
    """
    return torch.randn(2, 8, 64)


@pytest.fixture
def memory_retrievals():
    """
    Mock memory retrievals for MAC tests.

    Returns:
        torch.Tensor: [B=2, N_ℓ=16, D=64] memory vectors
    """
    return torch.randn(2, 16, 64)


# ============================================================================
# Device Fixtures
# ============================================================================

@pytest.fixture
def device():
    """
    Get available device (CUDA if available, else CPU).

    Returns:
        torch.device: Available device
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def cuda_only(device):
    """
    Skip test if CUDA is not available.

    Usage:
        def test_gpu_feature(cuda_only):
            # This test will be skipped on CPU-only systems
            pass
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


# ============================================================================
# Numerical Testing Fixtures
# ============================================================================

@pytest.fixture
def fp32_tolerance():
    """
    Tolerance for FP32 equivalence tests.

    Returns:
        float: 1e-5 (max absolute error for fp32)
    """
    return 1e-5


@pytest.fixture
def bf16_tolerance():
    """
    Tolerance for BF16 equivalence tests.

    Returns:
        float: 5e-3 (max absolute error for bf16)
    """
    return 5e-3


@pytest.fixture
def equivalence_test_lengths():
    """
    Sequence lengths for equivalence testing.

    Returns:
        list of int: [1, 16, 64, 128, 512, 1024]
    """
    return [1, 16, 64, 128, 512, 1024]


# ============================================================================
# Benchmark Fixtures
# ============================================================================

@pytest.fixture
def benchmark_lengths():
    """
    Sequence lengths for benchmarking.

    Returns:
        list of int: [128, 512, 2048, 4096]
    """
    return [128, 512, 2048, 4096]


@pytest.fixture
def warmup_iterations():
    """
    Number of warmup iterations for benchmarks.

    Returns:
        int: 3 warmup iterations
    """
    return 3


@pytest.fixture
def benchmark_iterations():
    """
    Number of benchmark iterations for timing.

    Returns:
        int: 10 iterations for averaging
    """
    return 10
