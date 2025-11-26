#!/usr/bin/env python3
"""
Test tokenization with Llama-3.2-1B tokenizer on FineWeb-Edu data.
"""

import json
from pathlib import Path
from transformers import AutoTokenizer

# Configuration
TOKENIZER_NAME = "meta-llama/Llama-3.2-1B"
MAX_SEQ_LENGTH = 4096
SAMPLE_FILE = Path("/mnt/BigAssDrive/00projects/00DeepNet/000Distill-Titan-Retnet-HRM/data/raw_fineweb/train/train.jsonl")

def main():
    print("="*60)
    print("Testing Tokenization with Llama-3.2-1B")
    print("="*60)

    # Load tokenizer
    print(f"\nLoading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    print(f"Vocab size: {tokenizer.vocab_size:,}")
    print(f"Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")

    # Test on sample data
    print(f"\nTesting on samples from: {SAMPLE_FILE}")

    stats = {
        'num_samples': 0,
        'total_tokens': 0,
        'sequences_created': 0,
        'tokens_truncated': 0,
        'min_tokens': float('inf'),
        'max_tokens': 0
    }

    with open(SAMPLE_FILE, 'r') as f:
        for i, line in enumerate(f):
            if i >= 100:  # Test first 100 samples
                break

            sample = json.loads(line)
            text = sample['text']

            # Tokenize
            tokens = tokenizer.encode(text, add_special_tokens=True)
            num_tokens = len(tokens)

            stats['num_samples'] += 1
            stats['total_tokens'] += num_tokens
            stats['min_tokens'] = min(stats['min_tokens'], num_tokens)
            stats['max_tokens'] = max(stats['max_tokens'], num_tokens)

            # Calculate how many sequences this would create
            if num_tokens <= MAX_SEQ_LENGTH:
                stats['sequences_created'] += 1
            else:
                # Would be split into multiple sequences
                num_sequences = (num_tokens + MAX_SEQ_LENGTH - 1) // MAX_SEQ_LENGTH
                stats['sequences_created'] += num_sequences
                stats['tokens_truncated'] += num_tokens % MAX_SEQ_LENGTH

            # Print first few examples
            if i < 3:
                print(f"\n--- Sample {i+1} ---")
                print(f"Text length: {len(text)} chars")
                print(f"Tokens: {num_tokens}")
                print(f"Sequences: {max(1, (num_tokens + MAX_SEQ_LENGTH - 1) // MAX_SEQ_LENGTH)}")
                print(f"Text preview: {text[:200]}...")

    # Print statistics
    print("\n" + "="*60)
    print("Tokenization Statistics (first 100 samples)")
    print("="*60)
    print(f"Samples tested: {stats['num_samples']}")
    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"Avg tokens/sample: {stats['total_tokens'] / stats['num_samples']:.1f}")
    print(f"Min/Max tokens: {stats['min_tokens']:,} / {stats['max_tokens']:,}")
    print(f"Sequences created: {stats['sequences_created']}")
    print(f"Avg tokens/sequence: {stats['total_tokens'] / stats['sequences_created']:.1f}")

    # Estimate full dataset
    print("\n" + "="*60)
    print("Full Dataset Estimates (50,000 training samples)")
    print("="*60)
    avg_tokens = stats['total_tokens'] / stats['num_samples']
    total_tokens_estimate = avg_tokens * 50000
    sequences_estimate = (total_tokens_estimate // MAX_SEQ_LENGTH)

    print(f"Estimated total tokens: {total_tokens_estimate:,.0f}")
    print(f"Estimated sequences: {sequences_estimate:,.0f}")
    print(f"Estimated training steps (batch_size=8): {sequences_estimate / 8:,.0f}")
    print(f"Estimated training steps (batch_size=16): {sequences_estimate / 16:,.0f}")
    print(f"Estimated training steps (batch_size=32): {sequences_estimate / 32:,.0f}")

    print("\n" + "="*60)
    print("Tokenization test PASSED!")
    print("="*60)
    print("\nNext steps:")
    print("1. Preprocess and tokenize full dataset")
    print("2. Create parquet shards for efficient loading")
    print("3. Update training config to point to new data")

if __name__ == "__main__":
    main()
