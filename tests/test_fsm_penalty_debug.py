"""
Debug script to test FSM penalty computation with actual training data.
"""

import sys
from pathlib import Path
import torch

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from model.tokenizer import get_tokenizer
from utils.structure_fsm import StructureFSM, compute_fsm_penalty

def main():
    # Initialize
    tokenizer = get_tokenizer()
    fsm = StructureFSM(tokenizer)

    print(f"Tokenizer vocab_size: {tokenizer.vocab_size}")
    print(f"Tokenizer len: {len(tokenizer)}")
    print(f"Token 0: {repr(tokenizer.decode([0]))}")
    print(f"Token 49154 (<ANS>): {repr(tokenizer.decode([49154]))}")
    print()

    # Create a simple test sequence: <Q>1<A><ANS>1</ANS><SEP>
    input_ids = torch.tensor([[
        49158,  # <Q>
        58,     # 1
        49159,  # <A>
        49154,  # <ANS>
        58,     # 1
        49155,  # </ANS>
        49160,  # <SEP>
    ]])

    # Labels: mask question, supervise answer
    labels = torch.tensor([[
        -100,   # <Q> masked
        -100,   # 1 masked
        49159,  # <A> supervised
        49154,  # <ANS> supervised
        58,     # 1 supervised
        49155,  # </ANS> supervised
        49160,  # <SEP> supervised
    ]])

    # Create mock logits (all zeros except token 0 = EOS which is high)
    vocab_size = 49180  # Match model
    logits = torch.zeros(1, 7, vocab_size)
    logits[:, :, 0] = 100.0  # Make EOS very likely

    print("Input IDs:", input_ids[0].tolist())
    print("Labels:", labels[0].tolist())
    print("Logits shape:", logits.shape)
    print()

    # Compute FSM states
    states = fsm.compute_states_from_tokens(input_ids)
    print(f"FSM states: {states[0].tolist()}")
    print()

    # Compute FSM penalty
    loss_fsm, stats = compute_fsm_penalty(
        logits=logits,
        labels=labels,
        fsm=fsm,
        input_ids=input_ids,
        weight=0.1
    )

    print(f"FSM penalty loss: {loss_fsm.item():.4f}")
    print(f"FSM stats: {stats}")
    print()

    # Check what tokens are allowed at each position
    print("Allowed tokens check:")
    allowed_mask = fsm.build_allowed_mask(states, vocab_size, logits.device)
    for pos in range(7):
        state = states[0, pos].item()
        is_eos_allowed = allowed_mask[0, pos, 0].item()
        num_allowed = allowed_mask[0, pos].sum().item()
        print(f"  Pos {pos} (state={state}): EOS allowed={is_eos_allowed}, total_allowed={num_allowed}/{vocab_size}")

    print("\nDebug FSM allowed_tokens:")
    from utils.structure_fsm import StructureState
    for state_enum in StructureState:
        allowed_set = fsm.allowed_tokens[state_enum]
        print(f"  State {state_enum.name} ({int(state_enum)}): {len(allowed_set)} tokens")
        if len(allowed_set) < 20:
            print(f"    Tokens: {sorted(list(allowed_set))}")
        else:
            print(f"    First 10: {sorted(list(allowed_set))[:10]}")
            print(f"    Last 10: {sorted(list(allowed_set))[-10:]}")

    print("\nDone!")

if __name__ == "__main__":
    main()
