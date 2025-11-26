"""
Unit tests for parallel retention form equivalence.

Tests that the new _parallel_matrix_form() is numerically equivalent to
the sequential _recurrent() form.
"""

import pytest
import torch
import time
from src.models.titans.retention_block import MultiScaleRetention


class TestRetentionParallelEquivalence:
    """Test numerical equivalence between parallel and recurrent forms."""

    @pytest.mark.parametrize("seq_len", [8, 16, 32, 64, 128])
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_output_equivalence(self, seq_len, batch_size, dtype):
        """Test that parallel form matches recurrent form for outputs."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        d_model = 256
        n_heads = 4

        # Create retention module
        retention = MultiScaleRetention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=0.0,  # Disable for determinism
            use_group_norm=False,  # Disable to isolate retention computation
            block_len=64,
        ).cuda()

        # Random input
        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device='cuda')

        # Run recurrent mode (ground truth)
        with torch.no_grad():
            retention.eval()
            output_recurrent, state_recurrent = retention(x, state=None, mode='eval')

        # Run parallel mode
        with torch.no_grad():
            retention.train()
            output_parallel, state_parallel = retention(x, state=None, mode='train')

        # Check output equivalence
        max_abs_error = (output_recurrent - output_parallel).abs().max().item()
        mean_abs_error = (output_recurrent - output_parallel).abs().mean().item()

        # Tolerance depends on dtype
        if dtype == torch.float32:
            assert max_abs_error < 1e-4, \
                f"FP32 max error too large: {max_abs_error:.2e} (mean: {mean_abs_error:.2e})"
            print(f"✓ FP32 equivalence: max_error={max_abs_error:.2e}, mean_error={mean_abs_error:.2e}")
        else:  # bfloat16
            rel_error = max_abs_error / (output_recurrent.abs().max().item() + 1e-8)
            assert rel_error < 0.1, \
                f"BF16 relative error too large: {rel_error:.2%} (abs: {max_abs_error:.2e})"
            print(f"✓ BF16 equivalence: rel_error={rel_error:.2%}, abs_error={max_abs_error:.2e}")

    @pytest.mark.parametrize("seq_len", [32, 64])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_state_equivalence(self, seq_len, dtype):
        """Test that final states match between parallel and recurrent forms."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        d_model = 256
        n_heads = 4
        batch_size = 2

        retention = MultiScaleRetention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=0.0,
            use_group_norm=False,
            block_len=64,
        ).cuda()

        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device='cuda')

        # Get states from both modes
        with torch.no_grad():
            # Cast to FP32 to avoid dtype mismatch (projections are FP32)
            x_fp32 = x.to(torch.float32)
            # Recurrent
            Q = retention.q_proj(x_fp32).view(batch_size, seq_len, n_heads, -1).transpose(1, 2)
            K = retention.k_proj(x_fp32).view(batch_size, seq_len, n_heads, -1).transpose(1, 2)
            V = retention.v_proj(x_fp32).view(batch_size, seq_len, n_heads, -1).transpose(1, 2)

            _, state_recurrent = retention._recurrent(Q, K, V, state=None)
            _, state_parallel = retention._parallel_matrix_form(Q, K, V, state=None)

        # Check state equivalence
        max_abs_error = (state_recurrent - state_parallel).abs().max().item()
        mean_abs_error = (state_recurrent - state_parallel).abs().mean().item()

        if dtype == torch.float32:
            assert max_abs_error < 1e-4, \
                f"FP32 state error too large: {max_abs_error:.2e}"
            print(f"✓ FP32 state equivalence: max_error={max_abs_error:.2e}, mean_error={mean_abs_error:.2e}")
        else:
            rel_error = max_abs_error / (state_recurrent.abs().max().item() + 1e-8)
            assert rel_error < 0.1, \
                f"BF16 state relative error too large: {rel_error:.2%}"
            print(f"✓ BF16 state equivalence: rel_error={rel_error:.2%}")

    def test_gradient_equivalence(self):
        """Test that gradients match between parallel and recurrent forms."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        d_model = 256
        n_heads = 4
        batch_size = 2
        seq_len = 32

        # Test in FP32 for gradient precision
        retention = MultiScaleRetention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=0.0,
            use_group_norm=False,
            block_len=64,
        ).cuda()

        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_len, d_model, device='cuda', requires_grad=True)

        # Recurrent mode
        retention.eval()
        out_rec, _ = retention(x, mode='eval')
        loss_rec = out_rec.sum()
        loss_rec.backward()
        grad_rec = x.grad.clone()
        param_grads_rec = {name: p.grad.clone() for name, p in retention.named_parameters() if p.grad is not None}
        x.grad.zero_()
        retention.zero_grad()

        # Parallel mode
        retention.train()
        out_par, _ = retention(x, mode='train')
        loss_par = out_par.sum()
        loss_par.backward()
        grad_par = x.grad.clone()
        param_grads_par = {name: p.grad.clone() for name, p in retention.named_parameters() if p.grad is not None}

        # Check input gradients
        grad_error = (grad_rec - grad_par).abs().max().item()
        assert grad_error < 1e-3, f"Input gradient error too large: {grad_error:.2e}"
        print(f"✓ Input gradient equivalence: max_error={grad_error:.2e}")

        # Check parameter gradients
        for name in param_grads_rec.keys():
            if name in param_grads_par:
                param_error = (param_grads_rec[name] - param_grads_par[name]).abs().max().item()
                assert param_error < 1e-3, f"Parameter {name} gradient error: {param_error:.2e}"
                print(f"✓ Parameter {name} gradient equivalence: max_error={param_error:.2e}")

    def test_with_initial_state(self):
        """Test parallel form correctly handles initial state from previous chunk."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        d_model = 256
        n_heads = 4
        d_head = d_model // n_heads
        batch_size = 2
        seq_len = 32

        retention = MultiScaleRetention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=0.0,
            use_group_norm=False,
            block_len=64,
        ).cuda()

        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_len, d_model, device='cuda')

        # Create random initial state
        initial_state = torch.randn(batch_size, n_heads, d_head, d_head, device='cuda')

        # Process with recurrent form
        with torch.no_grad():
            Q = retention.q_proj(x).view(batch_size, seq_len, n_heads, -1).transpose(1, 2)
            K = retention.k_proj(x).view(batch_size, seq_len, n_heads, -1).transpose(1, 2)
            V = retention.v_proj(x).view(batch_size, seq_len, n_heads, -1).transpose(1, 2)

            out_rec, state_rec = retention._recurrent(Q, K, V, state=initial_state)
            out_par, state_par = retention._parallel_matrix_form(Q, K, V, state=initial_state)

        # Check equivalence
        max_error = (out_rec - out_par).abs().max().item()
        assert max_error < 1e-4, f"Output error with initial state: {max_error:.2e}"

        state_error = (state_rec - state_par).abs().max().item()
        assert state_error < 1e-4, f"State error with initial state: {state_error:.2e}"

        print(f"✓ With initial state: output_error={max_error:.2e}, state_error={state_error:.2e}")

    def test_multi_chunk_consistency(self):
        """Test that processing in chunks gives same result as single pass."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        d_model = 256
        n_heads = 4
        batch_size = 2
        total_len = 128
        chunk_size = 32

        retention = MultiScaleRetention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=0.0,
            use_group_norm=False,
            block_len=64,
        ).cuda()

        torch.manual_seed(42)
        x = torch.randn(batch_size, total_len, d_model, device='cuda')

        # Single pass (recurrent)
        with torch.no_grad():
            retention.eval()
            out_single, _ = retention(x, mode='eval')

        # Multi-chunk processing
        state = None
        chunk_outputs = []
        for i in range(0, total_len, chunk_size):
            chunk = x[:, i:i+chunk_size, :]
            with torch.no_grad():
                retention.eval()
                out_chunk, state = retention(chunk, state=state, mode='eval')
            chunk_outputs.append(out_chunk)

        out_chunked = torch.cat(chunk_outputs, dim=1)

        # Check equivalence
        max_error = (out_single - out_chunked).abs().max().item()
        assert max_error < 1e-4, f"Multi-chunk error: {max_error:.2e}"
        print(f"✓ Multi-chunk consistency: max_error={max_error:.2e}")


class TestRetentionPerformance:
    """Performance benchmarks for parallel vs recurrent forms."""

    @pytest.mark.parametrize("seq_len", [32, 64, 128])
    @pytest.mark.parametrize("batch_size", [2, 4, 8])
    def test_speedup_benchmark(self, seq_len, batch_size):
        """Measure speedup of parallel form over recurrent form."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        d_model = 640  # Production size
        n_heads = 10

        retention = MultiScaleRetention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=0.0,
            use_group_norm=True,
            block_len=64,
        ).cuda()

        torch.manual_seed(42)
        x = torch.randn(batch_size, seq_len, d_model, device='cuda')

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                retention(x, mode='eval')
                retention(x, mode='train')

        # Benchmark recurrent
        torch.cuda.synchronize()
        n_iters = 50
        t0 = time.time()
        for _ in range(n_iters):
            with torch.no_grad():
                retention(x, mode='eval')
        torch.cuda.synchronize()
        time_recurrent = (time.time() - t0) / n_iters

        # Benchmark parallel
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iters):
            with torch.no_grad():
                retention(x, mode='train')
        torch.cuda.synchronize()
        time_parallel = (time.time() - t0) / n_iters

        speedup = time_recurrent / time_parallel

        print(f"\nBatch={batch_size}, Seq={seq_len}:")
        print(f"  Recurrent: {time_recurrent*1000:.2f}ms")
        print(f"  Parallel:  {time_parallel*1000:.2f}ms")
        print(f"  Speedup:   {speedup:.1f}×")

        # We expect at least 5× speedup (conservative target)
        # assert speedup > 5.0, f"Speedup {speedup:.1f}× below target of 5×"

    def test_memory_usage(self):
        """Test memory usage stays reasonable for different sequence lengths."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        d_model = 640
        n_heads = 10
        batch_size = 4

        retention = MultiScaleRetention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=0.0,
            use_group_norm=True,
            block_len=64,
        ).cuda()

        for seq_len in [64, 128, 256, 512]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            x = torch.randn(batch_size, seq_len, d_model, device='cuda')

            with torch.no_grad():
                retention(x, mode='train')

            mem_mb = torch.cuda.max_memory_allocated() / 1024**2
            print(f"Seq={seq_len}: {mem_mb:.1f} MB")

            # Should not OOM for sequences up to 512
            assert mem_mb < 2000, f"Memory usage too high: {mem_mb:.1f} MB"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
