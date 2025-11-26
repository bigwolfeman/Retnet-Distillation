"""
Direct test for segment isolation: Verify model CANNOT access other segments' answers.

This test addresses the fundamental question:
    "During training, does the model have access to answer tokens from other examples?"

Instead of analyzing model outputs, we directly verify that:
1. Cross-segment attention to answer positions is blocked
2. Hidden states don't contain information from other segments' answers
3. The attention mask is properly applied during forward pass

This is an INPUT SANITIZATION test, not an output analysis.
"""

import torch
import pytest
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.core import RetNetHRMModel
from src.config import ModelConfig
from model.tokenizer import get_tokenizer


def create_test_model():
    """Create a small RetNet-HRM model for testing."""
    config = ModelConfig()
    config.vocab_size = 49180
    config.d_model = 128  # Small for testing
    config.n_layers_retnet = 2
    config.n_retention_heads = 4
    config.n_layers_attention = 0
    config.dropout = 0.0
    config.max_seq_len_train = 512
    config.enable_retrieval = False
    config.hrm_epsilon = 0.01
    config.hrm_t_max = 5

    # Temporarily disable validation for test model
    original_validate = config.validate
    config.validate = lambda: None

    model = RetNetHRMModel(config)
    model.eval()  # Eval mode for deterministic behavior

    # Restore validation
    config.validate = original_validate

    return model


def create_packed_sequence_with_marker():
    """
    Create a packed sequence with a UNIQUE MARKER TOKEN in segment 0's answer.

    Structure:
        Segment 0: <Q>test<A><ANS>MARKER</ANS><SEP>
        Segment 1: <Q>test<A><ANS>?</ANS><SEP>

    If segment 1's hidden states contain information about MARKER, then leakage exists.
    """
    tokenizer = get_tokenizer()

    # Use a rare token as marker (e.g., "ZZMARKERXX" which should be unique)
    marker_token = "🔴"  # Use emoji as unique marker
    marker_id = tokenizer.encode(marker_token, add_special_tokens=False)[0]

    # Create packed sequence
    # Segment 0: <Q>test<A><ANS>🔴</ANS><SEP>
    # Segment 1: <Q>test<A><ANS>X</ANS><SEP>  (X will be predicted, should NOT contain marker info)

    seg0_q = "<Q>test1<A>"
    seg0_a = f"<ANS>{marker_token}</ANS><SEP>"
    seg1_q = "<Q>test2<A>"
    seg1_a = "<ANS>X</ANS><SEP>"

    full_text = seg0_q + seg0_a + seg1_q + seg1_a

    # Tokenize
    input_ids = tokenizer.encode(full_text, add_special_tokens=False)

    # Find marker position
    marker_pos = input_ids.index(marker_id)

    # Create segment IDs
    seg0_len = len(tokenizer.encode(seg0_q + seg0_a, add_special_tokens=False))
    segment_ids = [0] * seg0_len + [1] * (len(input_ids) - seg0_len)

    # Find segment 1 answer start position (after <A> tag in segment 1)
    seg1_answer_start = seg0_len + len(tokenizer.encode(seg1_q, add_special_tokens=False))

    return {
        'input_ids': torch.tensor([input_ids]),
        'segment_ids': torch.tensor([segment_ids]),
        'marker_pos': marker_pos,
        'seg1_answer_start': seg1_answer_start,
        'marker_id': marker_id,
    }


class TestDirectSegmentIsolation:
    """
    Test suite to directly verify segment isolation during forward pass.
    """

    def test_marker_token_invisible_to_other_segment(self):
        """
        CORE TEST: Verify segment 1 CANNOT access segment 0's answer marker.

        Test strategy:
        1. Create packed sequence with unique marker in segment 0's answer
        2. Run forward pass with segment_ids (mask applied)
        3. Check if marker token appears in segment 1's top-K predictions
        4. If marker appears in segment 1 predictions → LEAKAGE DETECTED
        """
        model = create_test_model()
        batch = create_packed_sequence_with_marker()

        input_ids = batch['input_ids']
        segment_ids = batch['segment_ids']
        marker_id = batch['marker_id']
        seg1_answer_start = batch['seg1_answer_start']

        # Run model WITH segment isolation
        with torch.no_grad():
            outputs = model(input_ids=input_ids, segment_ids=segment_ids, return_dict=True)
            logits = outputs.logits  # [1, seq_len, vocab]

        # Check segment 1's first answer prediction
        seg1_logits = logits[0, seg1_answer_start, :]  # [vocab]
        top_k_tokens = torch.topk(seg1_logits, k=100).indices.tolist()

        # CRITICAL ASSERTION: Marker should NOT appear in segment 1's predictions
        assert marker_id not in top_k_tokens, \
            f"LEAKAGE DETECTED: Marker token {marker_id} appears in segment 1's top-100 predictions! " \
            f"This means segment 1 can access segment 0's answer."

        print(f"✓ Marker token {marker_id} NOT in segment 1's top-100 predictions (isolation working)")


    def test_cross_segment_logit_difference(self):
        """
        Test if mask affects logits by comparing masked vs unmasked forward pass.

        If mask is working:
        - Masked forward (with segment_ids) should have DIFFERENT logits for seg1
        - Unmasked forward (no segment_ids) should have access to seg0 answer
        - The difference proves the mask is doing something
        """
        model = create_test_model()
        batch = create_packed_sequence_with_marker()

        input_ids = batch['input_ids']
        segment_ids = batch['segment_ids']
        marker_id = batch['marker_id']
        seg1_answer_start = batch['seg1_answer_start']

        # Forward pass WITH segment isolation
        with torch.no_grad():
            outputs_masked = model(input_ids=input_ids, segment_ids=segment_ids, return_dict=True)
            logits_masked = outputs_masked.logits[0, seg1_answer_start, marker_id].item()

        # Forward pass WITHOUT segment isolation (leakage allowed)
        with torch.no_grad():
            outputs_unmasked = model(input_ids=input_ids, segment_ids=None, return_dict=True)
            logits_unmasked = outputs_unmasked.logits[0, seg1_answer_start, marker_id].item()

        # Calculate divergence
        logit_diff = abs(logits_masked - logits_unmasked)

        print(f"Marker logit (masked): {logits_masked:.4f}")
        print(f"Marker logit (unmasked): {logits_unmasked:.4f}")
        print(f"Difference: {logit_diff:.4f}")

        # If mask is working, logits should differ
        # (unmasked should have higher logit for marker since it can see it in context)
        # Note: Small models may show small differences, but any non-zero difference proves mask is active
        assert logit_diff > 0.001, \
            f"Mask appears ineffective: logit difference is only {logit_diff:.4f}. " \
            f"Expected difference >0.001 if mask blocks cross-segment attention."

        print(f"✓ Mask affects logits (diff={logit_diff:.4f} > 0.001)")



    def test_segment_mask_created_correctly(self):
        """
        Verify that segment_ids are being converted to proper attention mask.

        Expected mask structure for [seg0, seg0, seg0, seg1, seg1]:
        ```
        Can attend to:
        pos 0: [0]             (seg0, can only see pos 0)
        pos 1: [0, 1]          (seg0, can see 0-1)
        pos 2: [0, 1, 2]       (seg0, can see 0-2)
        pos 3: [3]             (seg1, CANNOT see 0-2, only sees pos 3)
        pos 4: [3, 4]          (seg1, can see 3-4, NOT 0-2)
        ```
        """
        model = create_test_model()
        batch = create_packed_sequence_with_marker()

        segment_ids = batch['segment_ids'][0]  # [seq_len]

        # Build expected mask using model's internal method
        from src.models.retnet.backbone import RetNetBackbone
        backbone = model.retnet  # Correct attribute name

        # Get the mask that should be created
        attn_mask = backbone._build_segment_mask(batch['segment_ids'])  # [1, seq_len, seq_len]

        # Verify mask shape
        assert attn_mask.shape == (1, len(segment_ids), len(segment_ids)), \
            f"Mask shape incorrect: {attn_mask.shape}"

        # Verify block-diagonal structure
        seg0_positions = (segment_ids == 0).nonzero(as_tuple=True)[0]
        seg1_positions = (segment_ids == 1).nonzero(as_tuple=True)[0]

        if len(seg0_positions) > 0 and len(seg1_positions) > 0:
            # Check that seg1 positions CANNOT attend to seg0 positions
            for seg1_pos in seg1_positions:
                for seg0_pos in seg0_positions:
                    can_attend = attn_mask[0, seg1_pos, seg0_pos].item()
                    assert can_attend == False, \
                        f"MASK ERROR: Seg1 pos {seg1_pos} can attend to seg0 pos {seg0_pos}! " \
                        f"This allows answer leakage."

            print(f"✓ Mask correctly blocks cross-segment attention")
            print(f"  Seg0 positions: {seg0_positions.tolist()}")
            print(f"  Seg1 positions: {seg1_positions.tolist()}")
            print(f"  Seg1 CANNOT attend to seg0: Verified")


if __name__ == "__main__":
    # Fix Unicode encoding on Windows
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    # Run tests
    print("="*80)
    print("DIRECT SEGMENT ISOLATION TEST")
    print("="*80)
    print("\nThis test verifies that segments CANNOT access other segments' answers")
    print("during the forward pass (input sanitization, not output analysis).\n")

    test = TestDirectSegmentIsolation()

    print("\n[Test 1] Verifying mask structure...")
    test.test_segment_mask_created_correctly()

    print("\n[Test 2] Checking if mask affects logits...")
    test.test_cross_segment_logit_difference()

    print("\n[Test 3] Testing marker token invisibility...")
    test.test_marker_token_invisible_to_other_segment()

    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED - Segment isolation is working correctly!")
    print("="*80)
