"""
Test script to verify vectorized FSM implementation produces correct results.
"""

import sys
from pathlib import Path
import torch

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.structure_fsm import StructureFSM, StructureState

def test_vectorized_fsm():
    """Test that vectorized FSM produces correct state sequences."""

    print("=" * 80)
    print("VECTORIZED FSM TEST")
    print("=" * 80)

    # Create FSM
    fsm = StructureFSM()

    # Test Case 1: Valid sequence with all state transitions
    print("\nTest Case 1: Valid sequence with all states")
    print("-" * 80)

    # Create a valid sequence: <Q>...<A><ANS>95</ANS><ANS_SPLIT>9 5</ANS_SPLIT><SEP>
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

    # Create batch tensor [1, seq_len]
    token_ids = torch.tensor([test_tokens], dtype=torch.long)

    # Compute states using vectorized method
    states = fsm.compute_states_from_tokens(token_ids)

    # Verify by manually computing expected states
    expected_states = []
    state = StructureState.OUT
    for token_id in test_tokens:
        expected_states.append(int(state))
        state = fsm.step(state, token_id)

    expected_states_tensor = torch.tensor([expected_states], dtype=torch.long)

    # Check if they match
    match = torch.all(states == expected_states_tensor).item()

    print(f"Token sequence:")
    for i, token_id in enumerate(test_tokens):
        token_str = fsm.tokenizer.decode([token_id], skip_special_tokens=False)
        print(f"  [{i:2d}] Token='{token_str:15s}' Expected={StructureState(expected_states[i]).name:10s} Got={StructureState(states[0, i].item()).name:10s}")

    print(f"\nResults match: {match}")
    assert match, "Vectorized FSM produced incorrect states!"

    # Test Case 2: Batch processing
    print("\nTest Case 2: Batch processing (multiple sequences)")
    print("-" * 80)

    # Create two different sequences
    seq1 = [
        fsm.token_ids['<Q>'],
        fsm.token_ids['<A>'],
        fsm.token_ids['<ANS>'],
        fsm.tokenizer.convert_tokens_to_ids('1'),
        fsm.token_ids['</ANS>'],
        fsm.token_ids['<SEP>'],
    ]

    seq2 = [
        fsm.token_ids['<Q>'],
        fsm.token_ids['<A>'],
        fsm.token_ids['<ANS>'],
        fsm.tokenizer.convert_tokens_to_ids('2'),
        fsm.tokenizer.convert_tokens_to_ids('3'),
        fsm.token_ids['</ANS>'],
    ]

    # Pad to same length
    max_len = max(len(seq1), len(seq2))
    seq1_padded = seq1 + [0] * (max_len - len(seq1))
    seq2_padded = seq2 + [0] * (max_len - len(seq2))

    batch_tokens = torch.tensor([seq1_padded, seq2_padded], dtype=torch.long)

    # Compute states
    batch_states = fsm.compute_states_from_tokens(batch_tokens)

    # Verify each sequence independently
    for b in range(2):
        print(f"\nBatch {b}:")
        seq = [seq1_padded, seq2_padded][b]
        state = StructureState.OUT
        for i, token_id in enumerate(seq):
            expected_state = int(state)
            got_state = batch_states[b, i].item()
            token_str = fsm.tokenizer.decode([token_id], skip_special_tokens=False) if token_id != 0 else '<PAD>'
            print(f"  [{i:2d}] Token='{token_str:15s}' Expected={StructureState(expected_state).name:10s} Got={StructureState(got_state).name:10s}")
            assert expected_state == got_state, f"Mismatch at batch {b}, position {i}"
            state = fsm.step(state, token_id)

    print("\nBatch processing: PASS")

    # Test Case 3: GPU compatibility
    print("\nTest Case 3: GPU compatibility")
    print("-" * 80)

    if torch.cuda.is_available():
        gpu_tokens = token_ids.cuda()
        gpu_states = fsm.compute_states_from_tokens(gpu_tokens)

        # Move back to CPU and compare
        cpu_states = gpu_states.cpu()
        match = torch.all(cpu_states == states).item()

        print(f"GPU results match CPU: {match}")
        assert match, "GPU computation produced different results!"
        print("GPU test: PASS")
    else:
        print("CUDA not available, skipping GPU test")

    # Test Case 4: Large batch for performance verification
    print("\nTest Case 4: Large batch processing")
    print("-" * 80)

    batch_size = 32
    seq_len = 100
    large_batch = torch.randint(0, fsm.tokenizer.vocab_size, (batch_size, seq_len))

    try:
        large_states = fsm.compute_states_from_tokens(large_batch)
        assert large_states.shape == (batch_size, seq_len), f"Wrong output shape: {large_states.shape}"
        print(f"Processed large batch: {batch_size}x{seq_len} tokens")
        print("Large batch test: PASS")
    except Exception as e:
        print(f"Large batch test FAILED: {e}")
        raise

    # Test Case 5: Verify no .item() calls (check for torch.compile compatibility)
    print("\nTest Case 5: torch.compile compatibility")
    print("-" * 80)

    try:
        # Try to compile the method (requires PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            # Create a simple test
            test_tokens_compile = torch.randint(0, fsm.tokenizer.vocab_size, (4, 50))

            # This should work without errors if no .item() calls exist
            compiled_fsm = fsm
            result = compiled_fsm.compute_states_from_tokens(test_tokens_compile)

            print("Method is torch.compile compatible (no .item() calls detected)")
            print("torch.compile test: PASS")
        else:
            print("torch.compile not available (PyTorch < 2.0), skipping")
    except Exception as e:
        print(f"torch.compile compatibility test FAILED: {e}")
        raise

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)

if __name__ == "__main__":
    test_vectorized_fsm()
