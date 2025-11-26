"""
Test and benchmark GPU FSM lookup table implementation.

This test verifies that the GPU-accelerated FSM produces identical results
to the original CPU implementation, and measures the speedup.
"""

import sys
from pathlib import Path
import time
import torch

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.structure_fsm import StructureFSM, StructureState
from model.tokenizer import get_tokenizer


def test_correctness():
    """Verify GPU FSM produces identical results to CPU FSM."""
    print("=" * 80)
    print("CORRECTNESS TEST: GPU FSM vs CPU FSM")
    print("=" * 80)

    tokenizer = get_tokenizer()
    vocab_size = len(tokenizer)

    # Create both implementations
    fsm_gpu = StructureFSM(tokenizer=tokenizer, use_gpu_lut=True)
    fsm_cpu = StructureFSM(tokenizer=tokenizer, use_gpu_lut=False)

    # Test configurations
    batch_sizes = [1, 4, 8]
    seq_lens = [64, 128, 512]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\nDevice: {device}")
    print(f"Vocab size: {vocab_size}")

    all_passed = True

    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            # Generate random states
            states = torch.randint(0, len(StructureState), (batch_size, seq_len), device=device)

            # Build masks with both implementations
            mask_gpu = fsm_gpu.build_allowed_mask(states, vocab_size, device)
            mask_cpu = fsm_cpu.build_allowed_mask(states, vocab_size, device)

            # Check equality
            matches = torch.equal(mask_gpu, mask_cpu)

            status = "✓" if matches else "✗ FAILED"
            print(f"  {status} batch={batch_size}, seq_len={seq_len}: shapes {mask_gpu.shape}")

            if not matches:
                all_passed = False
                # Find first mismatch
                diff = mask_gpu != mask_cpu
                if diff.any():
                    b, t, v = torch.where(diff)
                    print(f"    First mismatch at batch={b[0]}, pos={t[0]}, vocab={v[0]}")
                    print(f"    GPU: {mask_gpu[b[0], t[0], v[0]]}, CPU: {mask_cpu[b[0], t[0], v[0]]}")

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL CORRECTNESS TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 80)

    return all_passed


def benchmark_performance():
    """Benchmark GPU FSM vs CPU FSM performance."""
    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARK: GPU FSM vs CPU FSM")
    print("=" * 80)

    tokenizer = get_tokenizer()
    vocab_size = len(tokenizer)

    # Create both implementations
    fsm_gpu = StructureFSM(tokenizer=tokenizer, use_gpu_lut=True)
    fsm_cpu = StructureFSM(tokenizer=tokenizer, use_gpu_lut=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Training-like configuration
    batch_size = 8
    seq_len = 512
    num_iterations = 100

    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Iterations: {num_iterations}")

    # Generate test data
    states = torch.randint(0, len(StructureState), (batch_size, seq_len), device=device)

    # Warmup
    print("\nWarming up...")
    for _ in range(10):
        _ = fsm_gpu.build_allowed_mask(states, vocab_size, device)
        _ = fsm_cpu.build_allowed_mask(states, vocab_size, device)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    # Benchmark GPU implementation
    print("\nBenchmarking GPU FSM...")
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(num_iterations):
        mask_gpu = fsm_gpu.build_allowed_mask(states, vocab_size, device)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    gpu_time = time.time() - start_time
    gpu_time_per_iter = (gpu_time / num_iterations) * 1000  # ms

    # Benchmark CPU implementation
    print("Benchmarking CPU FSM...")
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(num_iterations):
        mask_cpu = fsm_cpu.build_allowed_mask(states, vocab_size, device)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    cpu_time = time.time() - start_time
    cpu_time_per_iter = (cpu_time / num_iterations) * 1000  # ms

    # Results
    speedup = cpu_time_per_iter / gpu_time_per_iter

    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)
    print(f"CPU FSM:  {cpu_time_per_iter:.2f} ms/iter")
    print(f"GPU FSM:  {gpu_time_per_iter:.2f} ms/iter")
    print(f"Speedup:  {speedup:.2f}x")
    print(f"Saved:    {cpu_time_per_iter - gpu_time_per_iter:.2f} ms/iter")
    print("=" * 80)

    return speedup


def test_state_transitions():
    """Test that FSM state transitions work correctly with GPU FSM."""
    print("\n" + "=" * 80)
    print("STATE TRANSITION TEST WITH GPU FSM")
    print("=" * 80)

    fsm = StructureFSM(use_gpu_lut=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = len(fsm.tokenizer)

    # Valid sequence: <Q>...<A><ANS>95</ANS><ANS_SPLIT>9 5</ANS_SPLIT><SEP>
    test_tokens = [
        fsm.token_ids['<Q>'],
        fsm.token_ids['<A>'],
        fsm.token_ids['<ANS>'],
        fsm.tokenizer.convert_tokens_to_ids('9'),
        fsm.tokenizer.convert_tokens_to_ids('5'),
        fsm.token_ids['</ANS>'],
        fsm.token_ids['<ANS_SPLIT>'],
        fsm.tokenizer.convert_tokens_to_ids('9'),
        fsm.space_token,
        fsm.tokenizer.convert_tokens_to_ids('5'),
        fsm.token_ids['</ANS_SPLIT>'],
        fsm.token_ids['<SEP>'],
    ]

    # Compute states
    token_tensor = torch.tensor([test_tokens], device=device)  # [1, seq_len]
    states = fsm.compute_states_from_tokens(token_tensor)

    print(f"\nToken sequence ({len(test_tokens)} tokens):")
    for i, (token_id, state_id) in enumerate(zip(test_tokens, states[0].cpu().tolist())):
        token_str = fsm.tokenizer.decode([token_id], skip_special_tokens=False)
        state = StructureState(state_id)
        print(f"  [{i:2d}] Token='{token_str:15s}' State={state.name}")

    # Check that mask allows valid next tokens
    print("\nValidating allowed tokens at each position...")
    mask = fsm.build_allowed_mask(states, vocab_size, device)

    all_valid = True
    for i in range(len(test_tokens) - 1):
        current_token = test_tokens[i]
        next_token = test_tokens[i + 1]
        is_allowed = mask[0, i, next_token].item()

        if not is_allowed:
            token_str = fsm.tokenizer.decode([next_token], skip_special_tokens=False)
            print(f"  ✗ Position {i}: next token '{token_str}' should be allowed but isn't!")
            all_valid = False

    if all_valid:
        print("  ✓ All next tokens are correctly allowed")

    print("=" * 80)

    return all_valid


def main():
    """Run all tests and benchmarks."""
    print("\n" + "=" * 80)
    print("GPU FSM LOOKUP TABLE TEST SUITE")
    print("=" * 80)

    # Check device
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("\nWarning: CUDA not available, running on CPU")
        print("Performance improvement will be minimal without GPU")

    # Run tests
    correctness_passed = test_correctness()
    transition_passed = test_state_transitions()
    speedup = benchmark_performance()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Correctness:      {'✓ PASS' if correctness_passed else '✗ FAIL'}")
    print(f"State transitions: {'✓ PASS' if transition_passed else '✗ FAIL'}")
    print(f"Speedup:          {speedup:.2f}x")
    print("=" * 80)

    if correctness_passed and transition_passed:
        print("\n✓ GPU FSM is ready for production use!")
        print(f"  Expected speedup: ~{speedup:.1f}x faster than CPU FSM")
        print(f"  To enable: StructureFSM(use_gpu_lut=True) [default]")
        print(f"  To disable: StructureFSM(use_gpu_lut=False)")
    else:
        print("\n✗ GPU FSM has issues, please investigate")
        return 1

    return 0


if __name__ == "__main__":
    # Fix Unicode encoding on Windows
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    exit_code = main()
    sys.exit(exit_code)
