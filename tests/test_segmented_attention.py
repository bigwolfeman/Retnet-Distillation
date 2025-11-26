"""
Test suite for SegmentedAttention module

Tests basic functionality of the extracted SegmentedAttention class
to ensure it works correctly in the Titan-HRM project.
"""

import pytest
import torch
import sys
import os
import importlib.util

# Load the attention module directly to avoid import path issues
def load_attention_module():
    project_root = os.path.dirname(os.path.dirname(__file__))
    attention_path = os.path.join(project_root, "src", "models", "titans", "attention.py")
    rope_path = os.path.join(project_root, "src", "models", "attention", "rope.py")

    # Load rope module first
    rope_spec = importlib.util.spec_from_file_location("rope", rope_path)
    rope_module = importlib.util.module_from_spec(rope_spec)
    sys.modules['rope'] = rope_module
    rope_spec.loader.exec_module(rope_module)

    # Create a mock parent module structure
    sys.modules['models'] = type(sys)('models')
    sys.modules['models.attention'] = type(sys)('models.attention')
    sys.modules['models.attention'].rope = rope_module

    # Now load attention module
    spec = importlib.util.spec_from_file_location("attention", attention_path)
    attention_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(attention_module)

    return attention_module

# Load module
attention = load_attention_module()

SegmentedAttention = attention.SegmentedAttention
AttnIntermediates = attention.AttnIntermediates
exists = attention.exists
default = attention.default
pad_at_dim = attention.pad_at_dim
pad_and_segment_with_inverse = attention.pad_and_segment_with_inverse
create_mac_block_mask = attention.create_mac_block_mask


class TestHelperFunctions:
    """Test helper functions"""

    def test_exists(self):
        assert exists(1) == True
        assert exists(0) == True
        assert exists(None) == False
        assert exists("") == True

    def test_default(self):
        assert default(None, 5) == 5
        assert default(3, 5) == 3
        assert default(0, 5) == 0

    def test_pad_at_dim(self):
        t = torch.randn(2, 4, 8)
        padded = pad_at_dim(t, (1, 2), dim=1)
        assert padded.shape == (2, 7, 8)

    def test_pad_and_segment_with_inverse(self):
        batch, seq_len, dim = 2, 17, 64
        segment_len = 8
        seq = torch.randn(batch, seq_len, dim)

        # Test with folding
        segmented, inverse = pad_and_segment_with_inverse(seq, segment_len, fold_into_batch=True)
        assert segmented.shape[1] == segment_len  # Segments have correct length
        reconstructed = inverse(segmented)
        assert reconstructed.shape == seq.shape

        # Test without folding
        segmented, inverse = pad_and_segment_with_inverse(seq, segment_len, fold_into_batch=False)
        reconstructed = inverse(segmented)
        assert reconstructed.shape == seq.shape


class TestSegmentedAttention:
    """Test SegmentedAttention class"""

    @pytest.fixture
    def attention_config(self):
        """Standard attention configuration"""
        return {
            'dim': 512,
            'segment_len': 16,
            'num_persist_mem_tokens': 4,
            'num_longterm_mem_tokens': 0,
            'dim_head': 64,
            'heads': 8,
            'sliding': False,
            'accept_value_residual': False,
            'use_flex_attn': False  # Disable for testing
        }

    def test_init(self, attention_config):
        """Test initialization"""
        attn = SegmentedAttention(**attention_config)
        assert attn.segment_len == 16
        assert attn.num_persist_mem_tokens == 4
        assert attn.persistent_memory.shape == (2, 8, 4, 64)  # (kv, heads, mem_tokens, dim_head)

    def test_forward_basic(self, attention_config):
        """Test basic forward pass"""
        attn = SegmentedAttention(**attention_config)
        batch, seq_len = 2, 32
        seq = torch.randn(batch, seq_len, attention_config['dim'])

        output, intermediates = attn(seq)

        assert output.shape == seq.shape
        assert isinstance(intermediates, AttnIntermediates)
        assert intermediates.value_residual is not None
        assert intermediates.cached_key_values is not None

    def test_forward_with_value_residual(self):
        """Test forward with value residual (MAC coupling)"""
        # First layer without value residual
        config1 = {
            'dim': 512,
            'segment_len': 16,
            'num_persist_mem_tokens': 4,
            'dim_head': 64,
            'heads': 8,
            'sliding': False,
            'accept_value_residual': False,  # First layer doesn't accept residual
            'use_flex_attn': False
        }
        attn1 = SegmentedAttention(**config1)
        batch, seq_len = 2, 32
        seq = torch.randn(batch, seq_len, config1['dim'])

        # First pass to get value residual
        _, intermediates1 = attn1(seq)
        value_residual = intermediates1.value_residual

        # Second layer accepts value residual
        config2 = config1.copy()
        config2['accept_value_residual'] = True
        attn2 = SegmentedAttention(**config2)

        # Second pass with value residual
        output, intermediates2 = attn2(seq, value_residual=value_residual)

        assert output.shape == seq.shape
        assert intermediates2.value_residual is not None

    def test_forward_with_padding(self, attention_config):
        """Test forward with sequence that needs padding"""
        attn = SegmentedAttention(**attention_config)
        batch, seq_len = 2, 25  # Not multiple of segment_len (16)
        seq = torch.randn(batch, seq_len, attention_config['dim'])

        output, _ = attn(seq)

        # Output should be same length as input (padding removed)
        assert output.shape == seq.shape

    def test_forward_inference(self, attention_config):
        """Test inference mode with cache"""
        attn = SegmentedAttention(**attention_config)
        batch = 2
        dim = attention_config['dim']
        heads = attention_config['heads']
        dim_head = attention_config['dim_head']

        # Single token input for inference
        token = torch.randn(batch, 1, dim)

        # Initialize cache
        cache_k = torch.randn(batch, heads, 10, dim_head)
        cache_v = torch.randn(batch, heads, 10, dim_head)
        cache = (cache_k, cache_v)

        output, intermediates = attn(token, cache=cache)

        assert output.shape == (batch, 1, dim)
        assert isinstance(intermediates, AttnIntermediates)

        # Check cache was updated
        new_k, new_v = intermediates.cached_key_values
        assert new_k.shape[-2] == 11  # Added one position
        assert new_v.shape[-2] == 11

    def test_sliding_attention(self, attention_config):
        """Test with sliding window attention"""
        attention_config['sliding'] = True
        attn = SegmentedAttention(**attention_config)
        batch, seq_len = 2, 32
        seq = torch.randn(batch, seq_len, attention_config['dim'])

        output, _ = attn(seq)

        assert output.shape == seq.shape

    def test_longterm_memory_tokens(self, attention_config):
        """Test with long-term memory tokens"""
        attention_config['num_longterm_mem_tokens'] = 4
        attn = SegmentedAttention(**attention_config)
        batch, seq_len = 2, 32
        seq = torch.randn(batch, seq_len, attention_config['dim'])

        output, _ = attn(seq)

        assert output.shape == seq.shape
        assert attn.total_segment_len == 20  # segment_len (16) + num_longterm_mem_tokens (4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_flex_attention(self, attention_config):
        """Test flex attention on CUDA"""
        attention_config['use_flex_attn'] = True
        attn = SegmentedAttention(**attention_config).cuda()
        batch, seq_len = 2, 32
        seq = torch.randn(batch, seq_len, attention_config['dim']).cuda()

        output, _ = attn(seq)

        assert output.shape == seq.shape
        assert output.is_cuda


class TestCreateMacBlockMask:
    """Test MAC block mask creation"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_create_mac_block_mask_basic(self):
        """Test basic MAC block mask creation"""
        try:
            from torch.nn.attention.flex_attention import create_block_mask
            seq_len = 32
            window_size = 16
            persist_mem_len = 4

            mask = create_mac_block_mask(seq_len, window_size, persist_mem_len, sliding=False)
            assert mask is not None
        except ImportError:
            pytest.skip("Flex attention not available")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_create_mac_block_mask_sliding(self):
        """Test MAC block mask with sliding window"""
        try:
            from torch.nn.attention.flex_attention import create_block_mask
            seq_len = 32
            window_size = 16
            persist_mem_len = 4

            mask = create_mac_block_mask(seq_len, window_size, persist_mem_len, sliding=True)
            assert mask is not None
        except ImportError:
            pytest.skip("Flex attention not available")


def test_attn_intermediates_namedtuple():
    """Test AttnIntermediates namedtuple"""
    value_residual = torch.randn(2, 8, 32, 64)
    cached_kv = (torch.randn(2, 8, 32, 64), torch.randn(2, 8, 32, 64))

    intermediates = AttnIntermediates(value_residual, cached_kv)

    assert intermediates.value_residual is value_residual
    assert intermediates.cached_key_values is cached_kv


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
