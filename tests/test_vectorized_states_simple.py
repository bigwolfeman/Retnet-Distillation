"""
Simple test to verify state computation is correct.
"""

import sys
from pathlib import Path
import torch

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.structure_fsm import StructureFSM, StructureState

def main():
    fsm = StructureFSM()

    # Valid sequence: <Q>...<A><ANS>95</ANS><ANS_SPLIT>9 5</ANS_SPLIT><SEP>
    test_tokens = [
        fsm.token_ids['<Q>'],        # 0: should be OUT before, OUT after
        fsm.token_ids['<A>'],        # 1: should be OUT before, OUT after
        fsm.token_ids['<ANS>'],      # 2: should be OUT before, IN_ANS after
        fsm.tokenizer.convert_tokens_to_ids('9'),        # 3: should be IN_ANS before, IN_ANS after
        fsm.tokenizer.convert_tokens_to_ids('5'),        # 4: should be IN_ANS before, IN_ANS after
        fsm.token_ids['</ANS>'],     # 5: should be IN_ANS before, DONE after
        fsm.token_ids['<ANS_SPLIT>'], # 6: should be DONE before, IN_SPLIT after
        fsm.tokenizer.convert_tokens_to_ids('9'),        # 7: should be IN_SPLIT before, IN_SPLIT after
        fsm.space_token,              # 8: should be IN_SPLIT before, IN_SPLIT after
        fsm.tokenizer.convert_tokens_to_ids('5'),        # 9: should be IN_SPLIT before, IN_SPLIT after
        fsm.token_ids['</ANS_SPLIT>'], # 10: should be IN_SPLIT before, DONE after
        fsm.token_ids['<SEP>'],       # 11: should be DONE before, OUT after
    ]

    token_tensor = torch.tensor([test_tokens])
    states = fsm.compute_states_from_tokens(token_tensor)

    print("State verification (state BEFORE processing each token):")
    print("-" * 80)

    # Expected states BEFORE processing each token
    expected = [
        StructureState.OUT,      # 0: before <Q>
        StructureState.OUT,      # 1: before <A>
        StructureState.OUT,      # 2: before <ANS>
        StructureState.IN_ANS,   # 3: before 9
        StructureState.IN_ANS,   # 4: before 5
        StructureState.IN_ANS,   # 5: before </ANS>
        StructureState.DONE,     # 6: before <ANS_SPLIT>
        StructureState.IN_SPLIT, # 7: before 9
        StructureState.IN_SPLIT, # 8: before space
        StructureState.IN_SPLIT, # 9: before 5
        StructureState.IN_SPLIT, # 10: before </ANS_SPLIT>
        StructureState.DONE,     # 11: before <SEP>
    ]

    all_match = True
    for i, (token_id, expected_state) in enumerate(zip(test_tokens, expected)):
        got_state = StructureState(states[0, i].item())
        token_str = fsm.tokenizer.decode([token_id], skip_special_tokens=False)
        match = "✓" if got_state == expected_state else "✗"
        print(f"{match} [{i:2d}] Before '{token_str:15s}': Expected={expected_state.name:10s} Got={got_state.name:10s}")

        if got_state != expected_state:
            all_match = False

    print("-" * 80)
    if all_match:
        print("✓ All states match expected values!")
    else:
        print("✗ Some states don't match!")

    return all_match

if __name__ == "__main__":
    # Fix Unicode encoding on Windows
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    success = main()
    sys.exit(0 if success else 1)
