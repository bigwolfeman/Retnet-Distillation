"""
Dummy/synthetic data generator for testing data pipeline.

This module generates synthetic training data for testing the distillation
pipeline without requiring actual training datasets. Useful for:
- Integration tests
- Performance benchmarking
- Pipeline validation
- Development/debugging

Task implemented: T034
"""

import json
import random
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import torch


class DummyDataGenerator:
    """Generate synthetic training data for testing.

    Supports multiple formats:
    - Raw text (plain strings)
    - JSONL (JSON objects with 'text' field)
    - Pre-tokenized (JSONL with 'input_ids' field)

    Example:
        >>> generator = DummyDataGenerator(vocab_size=128256, max_length=4096)
        >>> generator.generate_jsonl("data/dummy_train.jsonl", num_examples=1000)
        >>> generator.generate_pretokenized("data/dummy_train_tok.jsonl", num_examples=1000)
    """

    def __init__(
        self,
        vocab_size: int = 128256,
        max_length: int = 4096,
        min_length: int = 128,
        seed: Optional[int] = None,
    ):
        """Initialize dummy data generator.

        Args:
            vocab_size: Vocabulary size (default: 128256 for Llama)
            max_length: Maximum sequence length
            min_length: Minimum sequence length
            seed: Random seed for reproducibility
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.min_length = min_length
        self.seed = seed

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        # Sample phrases for text generation
        self.sample_phrases = [
            "The quick brown fox jumps over the lazy dog.",
            "In the beginning was the Word, and the Word was with God.",
            "To be or not to be, that is the question.",
            "All that glitters is not gold.",
            "A journey of a thousand miles begins with a single step.",
            "Knowledge is power.",
            "Time is money.",
            "Practice makes perfect.",
            "Actions speak louder than words.",
            "The early bird catches the worm.",
            "Better late than never.",
            "Two wrongs don't make a right.",
            "When in Rome, do as the Romans do.",
            "The pen is mightier than the sword.",
            "Hope for the best, prepare for the worst.",
        ]

    def _generate_random_text(self, target_length: Optional[int] = None) -> str:
        """Generate random text by repeating sample phrases.

        Args:
            target_length: Target character length (optional)

        Returns:
            Random text string
        """
        if target_length is None:
            # Random length between min and max (approximate character count)
            target_length = random.randint(self.min_length * 4, self.max_length * 4)

        text_parts = []
        current_length = 0

        while current_length < target_length:
            phrase = random.choice(self.sample_phrases)
            text_parts.append(phrase)
            current_length += len(phrase) + 1  # +1 for space

        return " ".join(text_parts)

    def _generate_random_tokens(
        self,
        seq_length: Optional[int] = None,
        special_token_prob: float = 0.05,
    ) -> List[int]:
        """Generate random token IDs.

        Args:
            seq_length: Sequence length (optional, random if None)
            special_token_prob: Probability of special tokens (BOS/EOS/PAD)

        Returns:
            List of token IDs
        """
        if seq_length is None:
            seq_length = random.randint(self.min_length, self.max_length)

        tokens = []

        # Add BOS token (typically 1 for Llama)
        if random.random() < special_token_prob:
            tokens.append(1)

        # Generate content tokens
        while len(tokens) < seq_length - 1:  # -1 to leave room for EOS
            # Sample from vocab (avoid special tokens 0-3 most of the time)
            if random.random() < special_token_prob:
                token_id = random.randint(0, 3)  # Special tokens
            else:
                token_id = random.randint(4, self.vocab_size - 1)  # Content tokens

            tokens.append(token_id)

        # Add EOS token (typically 2 for Llama)
        if random.random() < special_token_prob:
            tokens.append(2)

        return tokens

    def generate_text_file(
        self,
        output_path: Union[str, Path],
        num_examples: int = 1000,
    ) -> None:
        """Generate plain text file (one sequence per line).

        Args:
            output_path: Output file path
            num_examples: Number of examples to generate
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for _ in range(num_examples):
                text = self._generate_random_text()
                f.write(text + '\n')

        print(f"Generated {num_examples} text examples to {output_path}")

    def generate_jsonl(
        self,
        output_path: Union[str, Path],
        num_examples: int = 1000,
        text_field: str = "text",
        include_metadata: bool = True,
    ) -> None:
        """Generate JSONL file with text data.

        Args:
            output_path: Output file path
            num_examples: Number of examples to generate
            text_field: Field name for text
            include_metadata: Whether to include metadata fields
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for i in range(num_examples):
                text = self._generate_random_text()

                record = {text_field: text}

                if include_metadata:
                    record.update({
                        'id': f"dummy_{i:06d}",
                        'length': len(text),
                        'source': 'synthetic',
                    })

                f.write(json.dumps(record) + '\n')

        print(f"Generated {num_examples} JSONL examples to {output_path}")

    def generate_pretokenized(
        self,
        output_path: Union[str, Path],
        num_examples: int = 1000,
        pretokenized_field: str = "input_ids",
        include_metadata: bool = True,
        variable_length: bool = True,
    ) -> None:
        """Generate JSONL file with pre-tokenized data.

        Args:
            output_path: Output file path
            num_examples: Number of examples to generate
            pretokenized_field: Field name for token IDs
            include_metadata: Whether to include metadata fields
            variable_length: Whether to use variable sequence lengths
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for i in range(num_examples):
                if variable_length:
                    seq_length = random.randint(self.min_length, self.max_length)
                else:
                    seq_length = self.max_length

                tokens = self._generate_random_tokens(seq_length)

                record = {pretokenized_field: tokens}

                if include_metadata:
                    record.update({
                        'id': f"dummy_tok_{i:06d}",
                        'length': len(tokens),
                        'source': 'synthetic',
                    })

                f.write(json.dumps(record) + '\n')

        print(f"Generated {num_examples} pre-tokenized examples to {output_path}")

    def generate_mixed_lengths(
        self,
        output_path: Union[str, Path],
        num_examples: int = 1000,
        length_distribution: Optional[Dict[str, float]] = None,
    ) -> None:
        """Generate data with specific length distribution.

        Useful for testing edge cases:
        - Very short sequences (< 100 tokens)
        - Short sequences (100-512 tokens)
        - Medium sequences (512-2048 tokens)
        - Long sequences (2048-4096 tokens)

        Args:
            output_path: Output file path
            num_examples: Number of examples to generate
            length_distribution: Distribution of lengths (default: uniform)
        """
        if length_distribution is None:
            # Default: uniform distribution across length bins
            length_distribution = {
                'very_short': 0.25,  # < 100 tokens
                'short': 0.25,       # 100-512 tokens
                'medium': 0.25,      # 512-2048 tokens
                'long': 0.25,        # 2048-4096 tokens
            }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Define length ranges (adjust based on max_length)
        # Ensure ranges are valid (min < max)
        length_ranges = {
            'very_short': (10, min(100, self.max_length // 4)),
            'short': (min(100, self.max_length // 4), min(512, self.max_length // 2)),
            'medium': (min(512, self.max_length // 2), min(2048, self.max_length * 3 // 4)),
            'long': (min(2048, self.max_length * 3 // 4), self.max_length),
        }

        # Filter out invalid ranges (where min >= max)
        valid_ranges = {k: v for k, v in length_ranges.items() if v[0] < v[1]}

        with open(output_path, 'w', encoding='utf-8') as f:
            for i in range(num_examples):
                # Sample length category from valid ranges only
                if valid_ranges:
                    category = random.choice(list(valid_ranges.keys()))
                    min_len, max_len = valid_ranges[category]
                else:
                    # Fallback: use full range
                    min_len, max_len = self.min_length, self.max_length

                seq_length = random.randint(min_len, max_len)

                # Generate tokens
                tokens = self._generate_random_tokens(seq_length)

                record = {
                    'input_ids': tokens,
                    'id': f"dummy_mixed_{i:06d}",
                    'length': len(tokens),
                    'category': category,
                    'source': 'synthetic',
                }

                f.write(json.dumps(record) + '\n')

        print(f"Generated {num_examples} mixed-length examples to {output_path}")
        print(f"  Distribution: {length_distribution}")

    def generate_edge_cases(
        self,
        output_path: Union[str, Path],
    ) -> None:
        """Generate edge case examples for testing.

        Includes:
        - Empty sequences (edge case, should be handled gracefully)
        - Single token sequences
        - Exactly max_length sequences
        - All special tokens
        - All padding tokens

        Args:
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        edge_cases = []

        # Empty sequence (0 tokens)
        edge_cases.append({
            'input_ids': [],
            'id': 'edge_empty',
            'description': 'Empty sequence',
        })

        # Single token
        edge_cases.append({
            'input_ids': [100],
            'id': 'edge_single_token',
            'description': 'Single token',
        })

        # Exactly max_length
        edge_cases.append({
            'input_ids': list(range(self.max_length)),
            'id': 'edge_max_length',
            'description': f'Exactly max_length ({self.max_length})',
        })

        # All special tokens (BOS, EOS, PAD, UNK)
        edge_cases.append({
            'input_ids': [0, 1, 2, 3] * 100,
            'id': 'edge_all_special',
            'description': 'All special tokens',
        })

        # Very long (exceeds max_length, should be truncated)
        edge_cases.append({
            'input_ids': list(range(self.max_length + 1000)),
            'id': 'edge_too_long',
            'description': f'Exceeds max_length by 1000 tokens',
        })

        with open(output_path, 'w', encoding='utf-8') as f:
            for case in edge_cases:
                f.write(json.dumps(case) + '\n')

        print(f"Generated {len(edge_cases)} edge case examples to {output_path}")

    def generate_all(
        self,
        output_dir: Union[str, Path],
        num_train: int = 1000,
        num_val: int = 200,
        num_test: int = 100,
    ) -> None:
        """Generate complete dummy dataset with train/val/test splits.

        Creates:
        - train.jsonl: Raw text training data
        - train_tok.jsonl: Pre-tokenized training data
        - val.jsonl: Raw text validation data
        - test.jsonl: Raw text test data
        - edge_cases.jsonl: Edge case examples
        - mixed_lengths.jsonl: Mixed length distribution

        Args:
            output_dir: Output directory
            num_train: Number of training examples
            num_val: Number of validation examples
            num_test: Number of test examples
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating dummy dataset in {output_dir}")
        print("=" * 80)

        # Training data (raw text)
        self.generate_jsonl(
            output_dir / "train.jsonl",
            num_examples=num_train,
        )

        # Training data (pre-tokenized)
        self.generate_pretokenized(
            output_dir / "train_tok.jsonl",
            num_examples=num_train,
        )

        # Validation data
        self.generate_jsonl(
            output_dir / "val.jsonl",
            num_examples=num_val,
        )

        # Test data
        self.generate_jsonl(
            output_dir / "test.jsonl",
            num_examples=num_test,
        )

        # Edge cases
        self.generate_edge_cases(
            output_dir / "edge_cases.jsonl",
        )

        # Mixed lengths
        self.generate_mixed_lengths(
            output_dir / "mixed_lengths.jsonl",
            num_examples=500,
        )

        print("=" * 80)
        print(f"Dummy dataset generated successfully in {output_dir}")
        print(f"  Train: {num_train} examples (raw + pre-tokenized)")
        print(f"  Val: {num_val} examples")
        print(f"  Test: {num_test} examples")
        print(f"  Edge cases: 5 examples")
        print(f"  Mixed lengths: 500 examples")


def generate_quick_test_data(
    output_path: Union[str, Path] = "data/dummy/quick_test.jsonl",
    num_examples: int = 50,
) -> None:
    """Quick helper to generate small test dataset.

    Args:
        output_path: Output file path
        num_examples: Number of examples (default: 50)
    """
    generator = DummyDataGenerator(seed=42)
    generator.generate_jsonl(output_path, num_examples=num_examples)
    print(f"Quick test data ready at {output_path}")


# Export public API
__all__ = [
    'DummyDataGenerator',
    'generate_quick_test_data',
]


# Main function for CLI usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate dummy/synthetic data for testing")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/dummy",
        help="Output directory for dummy data",
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=1000,
        help="Number of training examples",
    )
    parser.add_argument(
        "--num-val",
        type=int,
        default=200,
        help="Number of validation examples",
    )
    parser.add_argument(
        "--num-test",
        type=int,
        default=100,
        help="Number of test examples",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Generate complete dataset
    generator = DummyDataGenerator(
        max_length=args.max_length,
        seed=args.seed,
    )

    generator.generate_all(
        output_dir=args.output_dir,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
    )

    print(f"\nUsage example:")
    print(f"  from src.distillation.dataset import SimpleDataLoader")
    print(f"  loader = SimpleDataLoader('{args.output_dir}/train.jsonl', max_length={args.max_length})")
    print(f"  print(len(loader))  # {args.num_train}")
