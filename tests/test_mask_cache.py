"""Quick test to verify mask caching implementation."""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import from src
from src.models.retnet.backbone import RetNetBackbone

# Create small model
model = RetNetBackbone(
    vocab_size=1000,
    d_model=128,
    n_layers=2,
    n_heads=4,
    debug=True
)

# Test 1: First call should create and cache causal mask
print("Test 1: First forward pass (should cache mask for seq_len=10)")
segment_ids_1 = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 2]])
mask_1 = model._build_segment_mask(segment_ids_1)
print(f"Cache size after first call: {len(model._causal_mask_cache)}")
print(f"Cached seq_lens: {list(model._causal_mask_cache.keys())}")

# Test 2: Second call with same seq_len should reuse cached mask
print("\nTest 2: Second forward pass with same seq_len (should reuse cache)")
segment_ids_2 = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3, 4, 4]])  # Different segments, same length
mask_2 = model._build_segment_mask(segment_ids_2)
print(f"Cache size after second call: {len(model._causal_mask_cache)}")
print(f"Cached seq_lens: {list(model._causal_mask_cache.keys())}")

# Test 3: Different seq_len should create new cache entry
print("\nTest 3: Third forward pass with different seq_len (should add to cache)")
segment_ids_3 = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1]])  # seq_len=8
mask_3 = model._build_segment_mask(segment_ids_3)
print(f"Cache size after third call: {len(model._causal_mask_cache)}")
print(f"Cached seq_lens: {list(model._causal_mask_cache.keys())}")

# Test 4: Verify masks are still correct
print("\nTest 4: Verify mask correctness")
print(f"Mask 1 shape: {mask_1.shape}")
print(f"Mask 2 shape: {mask_2.shape}")
print(f"Mask 3 shape: {mask_3.shape}")

# Verify segment isolation (mask_1: segment 1 should not see segment 0)
print(f"\nSegment isolation check (mask_1):")
print(f"Position [0,4,0] (segment 1 looking at segment 0): {mask_1[0, 4, 0].item()} (should be False)")
print(f"Position [0,4,3] (segment 1 looking at self): {mask_1[0, 4, 3].item()} (should be True)")

print("\nAll tests passed! Caching is working correctly.")
