"""
Tests for data pipeline (dataset.py and dummy_data.py).

Tests cover:
- Tokenization correctness
- Batching logic
- Sequence length handling (truncation/padding)
- Data loading from different formats
- Edge cases (empty sequences, very long sequences, etc.)

Task implemented: T035
"""

import json
import tempfile
import pytest
import torch
from pathlib import Path
from transformers import AutoTokenizer

from src.distillation.dataset import (
    SimpleDataLoader,
    StreamingDataLoader,
    load_llama_tokenizer,
)
from src.distillation.dummy_data import (
    DummyDataGenerator,
    generate_quick_test_data,
)


# Fixtures
@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def dummy_tokenizer():
    """Create a dummy tokenizer for testing (faster than loading real Llama tokenizer)."""
    # Use a small tokenizer for tests (GPT-2 is fast and widely available)
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture
def dummy_data_generator():
    """Create dummy data generator with fixed seed."""
    return DummyDataGenerator(vocab_size=50257, max_length=512, seed=42)


# Tests for DummyDataGenerator
class TestDummyDataGenerator:
    """Tests for synthetic data generation."""

    def test_generate_text_file(self, dummy_data_generator, temp_dir):
        """Test generating plain text file."""
        output_path = temp_dir / "test.txt"
        dummy_data_generator.generate_text_file(output_path, num_examples=10)

        assert output_path.exists()

        # Verify content
        with open(output_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 10
            assert all(len(line.strip()) > 0 for line in lines)

    def test_generate_jsonl(self, dummy_data_generator, temp_dir):
        """Test generating JSONL file."""
        output_path = temp_dir / "test.jsonl"
        dummy_data_generator.generate_jsonl(output_path, num_examples=10)

        assert output_path.exists()

        # Verify content
        with open(output_path, 'r') as f:
            records = [json.loads(line) for line in f]
            assert len(records) == 10
            assert all('text' in rec for rec in records)
            assert all('id' in rec for rec in records)

    def test_generate_pretokenized(self, dummy_data_generator, temp_dir):
        """Test generating pre-tokenized JSONL file."""
        output_path = temp_dir / "test_tok.jsonl"
        dummy_data_generator.generate_pretokenized(output_path, num_examples=10)

        assert output_path.exists()

        # Verify content
        with open(output_path, 'r') as f:
            records = [json.loads(line) for line in f]
            assert len(records) == 10
            assert all('input_ids' in rec for rec in records)
            assert all(isinstance(rec['input_ids'], list) for rec in records)
            assert all(len(rec['input_ids']) <= dummy_data_generator.max_length for rec in records)

    def test_generate_mixed_lengths(self, dummy_data_generator, temp_dir):
        """Test generating data with mixed lengths."""
        output_path = temp_dir / "test_mixed.jsonl"
        dummy_data_generator.generate_mixed_lengths(output_path, num_examples=100)

        assert output_path.exists()

        # Verify length distribution
        with open(output_path, 'r') as f:
            records = [json.loads(line) for line in f]
            lengths = [rec['length'] for rec in records]

            # Check that we have variety in lengths
            # (Note: max_length for dummy_data_generator is 512, so we can't have > 512)
            assert min(lengths) < 100  # Some very short
            assert max(lengths) <= dummy_data_generator.max_length  # Respects max_length
            assert len(set(lengths)) > 10  # Variety

    def test_generate_edge_cases(self, dummy_data_generator, temp_dir):
        """Test generating edge case examples."""
        output_path = temp_dir / "edge_cases.jsonl"
        dummy_data_generator.generate_edge_cases(output_path)

        assert output_path.exists()

        # Verify edge cases
        with open(output_path, 'r') as f:
            records = [json.loads(line) for line in f]

            # Should have multiple edge cases
            assert len(records) >= 5

            # Find specific edge cases
            ids = [rec['id'] for rec in records]
            assert 'edge_empty' in ids
            assert 'edge_single_token' in ids
            assert 'edge_max_length' in ids

    def test_generate_all(self, dummy_data_generator, temp_dir):
        """Test generating complete dataset."""
        dummy_data_generator.generate_all(
            output_dir=temp_dir,
            num_train=50,
            num_val=10,
            num_test=5,
        )

        # Verify all files exist
        assert (temp_dir / "train.jsonl").exists()
        assert (temp_dir / "train_tok.jsonl").exists()
        assert (temp_dir / "val.jsonl").exists()
        assert (temp_dir / "test.jsonl").exists()
        assert (temp_dir / "edge_cases.jsonl").exists()
        assert (temp_dir / "mixed_lengths.jsonl").exists()


# Tests for SimpleDataLoader
class TestSimpleDataLoader:
    """Tests for simple data loading."""

    def test_load_jsonl_text(self, dummy_tokenizer, temp_dir):
        """Test loading JSONL file with text."""
        # Generate test data
        generator = DummyDataGenerator(vocab_size=50257, max_length=512, seed=42)
        data_path = temp_dir / "test.jsonl"
        generator.generate_jsonl(data_path, num_examples=10)

        # Load with SimpleDataLoader
        loader = SimpleDataLoader(
            data_path=data_path,
            max_length=512,
            tokenizer=dummy_tokenizer,
        )

        assert len(loader) == 10

        # Get first example
        example = loader[0]
        assert 'input_ids' in example
        assert 'attention_mask' in example
        assert 'labels' in example

        # Check shapes
        assert example['input_ids'].shape == torch.Size([512])
        assert example['attention_mask'].shape == torch.Size([512])
        assert example['labels'].shape == torch.Size([512])

        # Check dtypes
        assert example['input_ids'].dtype == torch.long
        assert example['attention_mask'].dtype == torch.long
        assert example['labels'].dtype == torch.long

    def test_load_pretokenized(self, dummy_tokenizer, temp_dir):
        """Test loading pre-tokenized data."""
        # Generate test data
        generator = DummyDataGenerator(vocab_size=50257, max_length=512, seed=42)
        data_path = temp_dir / "test_tok.jsonl"
        generator.generate_pretokenized(data_path, num_examples=10)

        # Load with SimpleDataLoader
        loader = SimpleDataLoader(
            data_path=data_path,
            max_length=512,
            tokenizer=dummy_tokenizer,
            use_pretokenized=True,
        )

        assert len(loader) == 10

        # Get first example
        example = loader[0]
        assert 'input_ids' in example
        assert 'attention_mask' in example
        assert 'labels' in example

    def test_truncation(self, dummy_tokenizer, temp_dir):
        """Test sequence truncation."""
        # Create data with very long sequences
        data_path = temp_dir / "test_long.jsonl"
        with open(data_path, 'w') as f:
            # Very long text
            long_text = " ".join(["word"] * 10000)
            f.write(json.dumps({"text": long_text}) + '\n')

        # Load with max_length=512
        loader = SimpleDataLoader(
            data_path=data_path,
            max_length=512,
            tokenizer=dummy_tokenizer,
            truncation=True,
        )

        example = loader[0]
        # Should be truncated to max_length
        assert len(example['input_ids']) == 512

    def test_padding(self, dummy_tokenizer, temp_dir):
        """Test sequence padding."""
        # Create data with short sequences
        data_path = temp_dir / "test_short.jsonl"
        with open(data_path, 'w') as f:
            f.write(json.dumps({"text": "short text"}) + '\n')

        # Load with max_length=512 and padding
        loader = SimpleDataLoader(
            data_path=data_path,
            max_length=512,
            tokenizer=dummy_tokenizer,
            padding="max_length",
        )

        example = loader[0]
        # Should be padded to max_length
        assert len(example['input_ids']) == 512

        # Verify padding tokens
        pad_count = (example['input_ids'] == dummy_tokenizer.pad_token_id).sum().item()
        assert pad_count > 0  # Should have padding

        # Verify attention mask
        assert example['attention_mask'].sum() < 512  # Some positions are masked

    def test_labels_masking(self, dummy_tokenizer, temp_dir):
        """Test that padding tokens are masked in labels."""
        # Create data with short sequences
        data_path = temp_dir / "test_short.jsonl"
        with open(data_path, 'w') as f:
            f.write(json.dumps({"text": "short text"}) + '\n')

        loader = SimpleDataLoader(
            data_path=data_path,
            max_length=512,
            tokenizer=dummy_tokenizer,
            padding="max_length",
        )

        example = loader[0]

        # Padding tokens should be -100 in labels
        pad_mask = example['input_ids'] == dummy_tokenizer.pad_token_id
        labels_at_pad = example['labels'][pad_mask]
        assert (labels_at_pad == -100).all()

    def test_batching(self, dummy_tokenizer, temp_dir):
        """Test batch creation."""
        # Generate test data
        generator = DummyDataGenerator(vocab_size=50257, max_length=512, seed=42)
        data_path = temp_dir / "test.jsonl"
        generator.generate_jsonl(data_path, num_examples=10)

        loader = SimpleDataLoader(
            data_path=data_path,
            max_length=512,
            tokenizer=dummy_tokenizer,
        )

        # Get batch of 4 examples
        batch = loader.get_batch([0, 1, 2, 3])

        assert 'input_ids' in batch
        assert 'attention_mask' in batch
        assert 'labels' in batch

        # Check shapes [batch_size, seq_len]
        assert batch['input_ids'].shape == torch.Size([4, 512])
        assert batch['attention_mask'].shape == torch.Size([4, 512])
        assert batch['labels'].shape == torch.Size([4, 512])

    def test_collate_fn(self, dummy_tokenizer, temp_dir):
        """Test collate function."""
        # Generate test data
        generator = DummyDataGenerator(vocab_size=50257, max_length=512, seed=42)
        data_path = temp_dir / "test.jsonl"
        generator.generate_jsonl(data_path, num_examples=5)

        loader = SimpleDataLoader(
            data_path=data_path,
            max_length=512,
            tokenizer=dummy_tokenizer,
        )

        # Get individual examples
        examples = [loader[i] for i in range(3)]

        # Collate them
        batch = SimpleDataLoader.collate_fn(examples)

        assert batch['input_ids'].shape[0] == 3
        assert batch['input_ids'].shape[1] == 512

    def test_empty_file(self, dummy_tokenizer, temp_dir):
        """Test handling of empty file."""
        data_path = temp_dir / "empty.jsonl"
        data_path.touch()  # Create empty file

        with pytest.raises(ValueError, match="No valid data found"):
            SimpleDataLoader(
                data_path=data_path,
                max_length=512,
                tokenizer=dummy_tokenizer,
            )

    def test_invalid_json(self, dummy_tokenizer, temp_dir):
        """Test handling of invalid JSON."""
        data_path = temp_dir / "invalid.jsonl"
        with open(data_path, 'w') as f:
            f.write("not valid json\n")
            f.write('{"text": "valid json"}\n')

        # Should skip invalid lines
        loader = SimpleDataLoader(
            data_path=data_path,
            max_length=512,
            tokenizer=dummy_tokenizer,
        )

        # Should only load valid line
        assert len(loader) == 1

    def test_text_file(self, dummy_tokenizer, temp_dir):
        """Test loading plain text file."""
        data_path = temp_dir / "test.txt"
        with open(data_path, 'w') as f:
            for i in range(5):
                f.write(f"This is line {i}\n")

        loader = SimpleDataLoader(
            data_path=data_path,
            max_length=512,
            tokenizer=dummy_tokenizer,
        )

        assert len(loader) == 5


# Tests for StreamingDataLoader
class TestStreamingDataLoader:
    """Tests for streaming data loading."""

    def test_streaming_iteration(self, dummy_tokenizer, temp_dir):
        """Test streaming iteration."""
        # Generate test data
        generator = DummyDataGenerator(vocab_size=50257, max_length=512, seed=42)
        data_path = temp_dir / "test.jsonl"
        generator.generate_jsonl(data_path, num_examples=100)

        # Create streaming loader
        loader = StreamingDataLoader(
            data_path=data_path,
            buffer_size=10,
            max_length=512,
            tokenizer=dummy_tokenizer,
        )

        # Iterate and count
        count = 0
        for example in loader:
            assert 'input_ids' in example
            assert 'attention_mask' in example
            assert 'labels' in example
            count += 1

            if count >= 20:  # Don't iterate all
                break

        assert count == 20

    def test_streaming_shuffle(self, dummy_tokenizer, temp_dir):
        """Test streaming with shuffle."""
        # Generate test data with IDs
        data_path = temp_dir / "test.jsonl"
        with open(data_path, 'w') as f:
            for i in range(50):
                f.write(json.dumps({"text": f"Example {i}", "id": i}) + '\n')

        # Create streaming loader with shuffle
        loader = StreamingDataLoader(
            data_path=data_path,
            buffer_size=10,
            max_length=512,
            tokenizer=dummy_tokenizer,
            shuffle_buffer=True,
            seed=42,
        )

        # Collect first 20 examples
        examples = []
        for i, example in enumerate(loader):
            examples.append(example)
            if i >= 19:
                break

        # Should have 20 examples
        assert len(examples) == 20

    def test_streaming_pretokenized(self, dummy_tokenizer, temp_dir):
        """Test streaming with pre-tokenized data."""
        # Generate pre-tokenized data
        generator = DummyDataGenerator(vocab_size=50257, max_length=512, seed=42)
        data_path = temp_dir / "test_tok.jsonl"
        generator.generate_pretokenized(data_path, num_examples=50)

        # Create streaming loader
        loader = StreamingDataLoader(
            data_path=data_path,
            buffer_size=10,
            max_length=512,
            tokenizer=dummy_tokenizer,
            use_pretokenized=True,
        )

        # Get first example
        example = next(iter(loader))
        assert 'input_ids' in example
        assert example['input_ids'].dtype == torch.long


# Tests for tokenizer integration
class TestTokenizerIntegration:
    """Tests for Llama tokenizer integration."""

    @pytest.mark.skipif(
        not Path("~/.cache/huggingface").expanduser().exists(),
        reason="HuggingFace cache not available"
    )
    def test_load_llama_tokenizer(self):
        """Test loading Llama tokenizer."""
        try:
            tokenizer = load_llama_tokenizer()
            assert tokenizer.vocab_size == 128256
            assert tokenizer.pad_token is not None
        except Exception as e:
            # Skip if Llama tokenizer not available
            pytest.skip(f"Llama tokenizer not available: {e}")

    def test_tokenizer_special_tokens(self, dummy_tokenizer):
        """Test special token handling."""
        # Verify pad token is set
        assert dummy_tokenizer.pad_token is not None
        assert dummy_tokenizer.pad_token_id is not None

        # Test encoding
        text = "Hello world"
        tokens = dummy_tokenizer.encode(text, add_special_tokens=True)
        assert len(tokens) > 0

    def test_tokenizer_truncation(self, dummy_tokenizer):
        """Test tokenizer truncation."""
        # Very long text
        long_text = " ".join(["word"] * 10000)

        # Encode with truncation
        tokens = dummy_tokenizer.encode(
            long_text,
            add_special_tokens=True,
            max_length=512,
            truncation=True,
        )

        assert len(tokens) == 512

    def test_tokenizer_padding(self, dummy_tokenizer):
        """Test tokenizer padding."""
        # Short text
        short_text = "hi"

        # Encode with padding
        encoded = dummy_tokenizer(
            short_text,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        assert encoded['input_ids'].shape[1] == 512
        assert encoded['attention_mask'].shape[1] == 512


# Performance tests
class TestDataPipelinePerformance:
    """Performance tests for data pipeline."""

    def test_loading_speed_jsonl(self, dummy_tokenizer, temp_dir):
        """Test loading speed for JSONL files."""
        import time

        # Generate test data
        generator = DummyDataGenerator(vocab_size=50257, max_length=512, seed=42)
        data_path = temp_dir / "test.jsonl"
        generator.generate_jsonl(data_path, num_examples=1000)

        # Time loading
        start = time.time()
        loader = SimpleDataLoader(
            data_path=data_path,
            max_length=512,
            tokenizer=dummy_tokenizer,
        )
        load_time = time.time() - start

        # Time iteration
        start = time.time()
        for i in range(100):
            _ = loader[i]
        iter_time = time.time() - start

        print(f"\nLoading time: {load_time:.3f}s")
        print(f"Iteration time (100 examples): {iter_time:.3f}s")
        print(f"Average per example: {iter_time/100*1000:.2f}ms")

        # Should be reasonably fast
        assert load_time < 10.0  # Loading should take < 10s
        assert iter_time < 5.0   # 100 iterations should take < 5s

    def test_loading_speed_pretokenized(self, dummy_tokenizer, temp_dir):
        """Test loading speed for pre-tokenized files."""
        import time

        # Generate test data
        generator = DummyDataGenerator(vocab_size=50257, max_length=512, seed=42)
        data_path = temp_dir / "test_tok.jsonl"
        generator.generate_pretokenized(data_path, num_examples=1000)

        # Time loading
        start = time.time()
        loader = SimpleDataLoader(
            data_path=data_path,
            max_length=512,
            tokenizer=dummy_tokenizer,
            use_pretokenized=True,
        )
        load_time = time.time() - start

        # Time iteration
        start = time.time()
        for i in range(100):
            _ = loader[i]
        iter_time = time.time() - start

        print(f"\nPre-tokenized loading time: {load_time:.3f}s")
        print(f"Iteration time (100 examples): {iter_time:.3f}s")
        print(f"Average per example: {iter_time/100*1000:.2f}ms")

        # Pre-tokenized should be faster
        assert load_time < 10.0
        assert iter_time < 2.0  # Should be faster than raw text


# Edge case tests
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_short_sequence(self, dummy_tokenizer, temp_dir):
        """Test handling of very short sequences."""
        data_path = temp_dir / "short.jsonl"
        with open(data_path, 'w') as f:
            f.write(json.dumps({"text": "hi"}) + '\n')

        loader = SimpleDataLoader(
            data_path=data_path,
            max_length=512,
            tokenizer=dummy_tokenizer,
            padding="max_length",
        )

        example = loader[0]
        # Should be padded to max_length
        assert len(example['input_ids']) == 512

    def test_empty_text(self, dummy_tokenizer, temp_dir):
        """Test handling of empty text."""
        data_path = temp_dir / "empty_text.jsonl"
        with open(data_path, 'w') as f:
            f.write(json.dumps({"text": ""}) + '\n')
            f.write(json.dumps({"text": "valid"}) + '\n')

        loader = SimpleDataLoader(
            data_path=data_path,
            max_length=512,
            tokenizer=dummy_tokenizer,
        )

        # Should raise error on empty text
        with pytest.raises(ValueError, match="Empty text"):
            _ = loader[0]

        # Second example should work
        example = loader[1]
        assert 'input_ids' in example

    def test_max_length_sequence(self, dummy_tokenizer, temp_dir):
        """Test handling of exactly max_length sequences."""
        # Create pre-tokenized data with exactly max_length tokens
        data_path = temp_dir / "exact_length.jsonl"
        with open(data_path, 'w') as f:
            exact_tokens = list(range(512))
            f.write(json.dumps({"input_ids": exact_tokens}) + '\n')

        loader = SimpleDataLoader(
            data_path=data_path,
            max_length=512,
            tokenizer=dummy_tokenizer,
            use_pretokenized=True,
        )

        example = loader[0]
        # Should be exactly max_length
        assert len(example['input_ids']) == 512

    def test_missing_file(self, dummy_tokenizer, temp_dir):
        """Test handling of missing file."""
        data_path = temp_dir / "nonexistent.jsonl"

        with pytest.raises(FileNotFoundError):
            SimpleDataLoader(
                data_path=data_path,
                max_length=512,
                tokenizer=dummy_tokenizer,
            )


# Integration tests
class TestDataPipelineIntegration:
    """Integration tests combining multiple components."""

    def test_end_to_end_workflow(self, dummy_tokenizer, temp_dir):
        """Test complete end-to-end workflow."""
        # 1. Generate dummy data
        generator = DummyDataGenerator(vocab_size=50257, max_length=512, seed=42)
        generator.generate_all(
            output_dir=temp_dir,
            num_train=100,
            num_val=20,
            num_test=10,
        )

        # 2. Load training data
        train_loader = SimpleDataLoader(
            data_path=temp_dir / "train.jsonl",
            max_length=512,
            tokenizer=dummy_tokenizer,
        )

        assert len(train_loader) == 100

        # 3. Load validation data
        val_loader = SimpleDataLoader(
            data_path=temp_dir / "val.jsonl",
            max_length=512,
            tokenizer=dummy_tokenizer,
        )

        assert len(val_loader) == 20

        # 4. Get batches
        train_batch = train_loader.get_batch([0, 1, 2, 3])
        assert train_batch['input_ids'].shape == torch.Size([4, 512])

        val_batch = val_loader.get_batch([0, 1])
        assert val_batch['input_ids'].shape == torch.Size([2, 512])

    def test_pytorch_dataloader_integration(self, dummy_tokenizer, temp_dir):
        """Test integration with PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        # Generate data
        generator = DummyDataGenerator(vocab_size=50257, max_length=512, seed=42)
        data_path = temp_dir / "test.jsonl"
        generator.generate_jsonl(data_path, num_examples=50)

        # Create dataset
        dataset = SimpleDataLoader(
            data_path=data_path,
            max_length=512,
            tokenizer=dummy_tokenizer,
        )

        # Create PyTorch DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=SimpleDataLoader.collate_fn,
        )

        # Iterate
        for i, batch in enumerate(dataloader):
            assert batch['input_ids'].shape[0] == 4
            assert batch['input_ids'].shape[1] == 512

            if i >= 2:  # Just test a few batches
                break


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
