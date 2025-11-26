#!/usr/bin/env python3
"""Dataset Preparation Script for Knowledge Distillation.

Downloads, filters, and formats datasets for distillation training.
Outputs JSONL format expected by the distillation pipeline.

Usage:
    python scripts/prepare_datasets.py --mode all --output-dir data/unlabeled
    python scripts/prepare_datasets.py --mode test --output-dir data/test --num-examples 100
    python scripts/prepare_datasets.py --mode text --max-examples 1000000 --output-dir data/unlabeled

Datasets:
    - Text: FineWeb-Edu (high quality), WikiText-103
    - Code: The Stack (Python, filtered), CodeParrot
    - Math: OpenWebMath, MATH dataset

Output Format:
    JSONL with {"text": "...", "domain": "text|code|math", "num_tokens": int}
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Iterator
from tqdm import tqdm

try:
    from datasets import load_dataset, Dataset
    from transformers import AutoTokenizer
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Run: pip install datasets transformers")
    exit(1)

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.tokenizer_utils import load_tokenizer_with_fallback, get_tokenizer_info


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetPreparer:
    """Prepares datasets for knowledge distillation."""

    def __init__(
        self,
        output_dir: str,
        tokenizer_name: str = "meta-llama/Llama-3.2-1B",
        hf_token: Optional[str] = None,
        min_tokens: int = 512,
        max_tokens: int = 4096,
        seed: int = 42,
    ):
        """Initialize dataset preparer.

        Args:
            output_dir: Directory to save prepared datasets
            tokenizer_name: HuggingFace tokenizer name
            hf_token: HuggingFace token for gated models (optional)
            min_tokens: Minimum tokens per example (filter shorter)
            max_tokens: Maximum tokens per example (chunk longer)
            seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.seed = seed

        logger.info(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = load_tokenizer_with_fallback(
            preferred_tokenizer=tokenizer_name,
            hf_token=hf_token,
            trust_remote_code=True,
        )

        # Log tokenizer info
        tokenizer_info = get_tokenizer_info(self.tokenizer)
        logger.info(f"Loaded tokenizer: {tokenizer_info['name_or_path']}")
        logger.info(f"  Vocab size: {tokenizer_info['vocab_size']}")

        self.stats = {
            "text": {"total": 0, "filtered": 0, "chunks": 0},
            "code": {"total": 0, "filtered": 0, "chunks": 0},
            "math": {"total": 0, "filtered": 0, "chunks": 0},
        }

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def chunk_text(self, text: str, domain: str) -> List[Dict]:
        """Chunk long text into max_tokens segments.

        Args:
            text: Input text
            domain: Domain label (text/code/math)

        Returns:
            List of chunks as dicts
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) <= self.max_tokens:
            return [{
                "text": text,
                "domain": domain,
                "num_tokens": len(tokens)
            }]

        # Chunk into overlapping segments (10% overlap for context)
        chunks = []
        stride = int(self.max_tokens * 0.9)

        for i in range(0, len(tokens), stride):
            chunk_tokens = tokens[i:i + self.max_tokens]
            if len(chunk_tokens) >= self.min_tokens:
                chunk_text = self.tokenizer.decode(chunk_tokens)
                chunks.append({
                    "text": chunk_text,
                    "domain": domain,
                    "num_tokens": len(chunk_tokens)
                })

            if i + self.max_tokens >= len(tokens):
                break

        self.stats[domain]["chunks"] += len(chunks)
        return chunks

    def prepare_text_dataset(
        self,
        max_examples: Optional[int] = None,
        dataset_name: str = "HuggingFaceFW/fineweb-edu",
        split: str = "train",
    ) -> str:
        """Prepare text dataset (FineWeb-Edu or WikiText).

        Args:
            max_examples: Maximum number of examples to process
            dataset_name: HuggingFace dataset name
            split: Dataset split

        Returns:
            Path to output JSONL file
        """
        logger.info(f"Preparing text dataset: {dataset_name}")
        output_path = self.output_dir / "text" / f"{dataset_name.replace('/', '_')}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if dataset_name == "HuggingFaceFW/fineweb-edu":
                # FineWeb-Edu is large, use streaming
                dataset = load_dataset(
                    dataset_name,
                    name="sample-10BT",  # Use 10B token sample
                    split=split,
                    streaming=True
                )
            else:
                dataset = load_dataset(dataset_name, split=split, streaming=True)
        except Exception as e:
            logger.error(f"Failed to load {dataset_name}: {e}")
            logger.info("Falling back to WikiText-103")
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)

        with open(output_path, "w") as f:
            count = 0
            for example in tqdm(dataset, desc="Processing text"):
                if max_examples and count >= max_examples:
                    break

                # Extract text field (different datasets use different keys)
                text = example.get("text") or example.get("content") or ""

                if not text or len(text.strip()) < 100:
                    continue

                self.stats["text"]["total"] += 1

                # Chunk if needed
                chunks = self.chunk_text(text, "text")

                for chunk in chunks:
                    if chunk["num_tokens"] >= self.min_tokens:
                        f.write(json.dumps(chunk) + "\n")
                        count += 1
                        self.stats["text"]["filtered"] += 1

        logger.info(f"Saved {count} text examples to {output_path}")
        return str(output_path)

    def prepare_code_dataset(
        self,
        max_examples: Optional[int] = None,
        language: str = "python",
    ) -> str:
        """Prepare code dataset (The Stack).

        Args:
            max_examples: Maximum number of examples
            language: Programming language to filter

        Returns:
            Path to output JSONL file
        """
        logger.info(f"Preparing code dataset: {language}")
        output_path = self.output_dir / "code" / f"{language}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Try The Stack v2 (dedup version)
            dataset = load_dataset(
                "bigcode/the-stack-dedup",
                data_dir=f"data/{language}",
                split="train",
                streaming=True
            )
        except Exception as e:
            logger.warning(f"Failed to load The Stack: {e}")
            logger.info("Using CodeParrot fallback")
            try:
                dataset = load_dataset(
                    "codeparrot/github-code",
                    languages=[language],
                    split="train",
                    streaming=True
                )
            except Exception as e2:
                logger.error(f"Failed to load CodeParrot: {e2}")
                logger.info("Creating synthetic code examples")
                return self._create_synthetic_code(output_path, max_examples or 1000)

        with open(output_path, "w") as f:
            count = 0
            for example in tqdm(dataset, desc="Processing code"):
                if max_examples and count >= max_examples:
                    break

                code = example.get("content") or example.get("code") or ""

                if not code or len(code.strip()) < 100:
                    continue

                self.stats["code"]["total"] += 1

                # Filter out very short files
                if code.count("\n") < 10:
                    continue

                chunks = self.chunk_text(code, "code")

                for chunk in chunks:
                    if chunk["num_tokens"] >= self.min_tokens:
                        f.write(json.dumps(chunk) + "\n")
                        count += 1
                        self.stats["code"]["filtered"] += 1

        logger.info(f"Saved {count} code examples to {output_path}")
        return str(output_path)

    def prepare_math_dataset(
        self,
        max_examples: Optional[int] = None,
    ) -> str:
        """Prepare math dataset (OpenWebMath).

        Args:
            max_examples: Maximum number of examples

        Returns:
            Path to output JSONL file
        """
        logger.info("Preparing math dataset: OpenWebMath")
        output_path = self.output_dir / "math" / "openwebmath.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            dataset = load_dataset(
                "open-web-math/open-web-math",
                split="train",
                streaming=True
            )
        except Exception as e:
            logger.warning(f"Failed to load OpenWebMath: {e}")
            logger.info("Using MATH dataset fallback")
            try:
                dataset = load_dataset("hendrycks/math", split="train")
                return self._prepare_math_fallback(dataset, output_path, max_examples)
            except Exception as e2:
                logger.error(f"Failed to load MATH: {e2}")
                logger.info("Creating synthetic math examples")
                return self._create_synthetic_math(output_path, max_examples or 1000)

        with open(output_path, "w") as f:
            count = 0
            for example in tqdm(dataset, desc="Processing math"):
                if max_examples and count >= max_examples:
                    break

                text = example.get("text") or example.get("content") or ""

                if not text or len(text.strip()) < 100:
                    continue

                self.stats["math"]["total"] += 1

                chunks = self.chunk_text(text, "math")

                for chunk in chunks:
                    if chunk["num_tokens"] >= self.min_tokens:
                        f.write(json.dumps(chunk) + "\n")
                        count += 1
                        self.stats["math"]["filtered"] += 1

        logger.info(f"Saved {count} math examples to {output_path}")
        return str(output_path)

    def _prepare_math_fallback(
        self,
        dataset: Dataset,
        output_path: Path,
        max_examples: Optional[int]
    ) -> str:
        """Prepare MATH dataset as fallback."""
        with open(output_path, "w") as f:
            count = 0
            for example in tqdm(dataset, desc="Processing MATH"):
                if max_examples and count >= max_examples:
                    break

                # Format as Q: ... A: ...
                problem = example.get("problem", "")
                solution = example.get("solution", "")

                if not problem or not solution:
                    continue

                text = f"Problem: {problem}\n\nSolution: {solution}"

                self.stats["math"]["total"] += 1
                chunks = self.chunk_text(text, "math")

                for chunk in chunks:
                    if chunk["num_tokens"] >= self.min_tokens:
                        f.write(json.dumps(chunk) + "\n")
                        count += 1
                        self.stats["math"]["filtered"] += 1

        return str(output_path)

    def _create_synthetic_code(self, output_path: Path, num_examples: int) -> str:
        """Create synthetic code examples for testing."""
        synthetic_code = [
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n",
            "class Node:\n    def __init__(self, value):\n        self.value = value\n        self.next = None\n",
            "import numpy as np\n\ndef matrix_multiply(A, B):\n    return np.dot(A, B)\n",
            "from typing import List\n\ndef quicksort(arr: List[int]) -> List[int]:\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n",
        ]

        with open(output_path, "w") as f:
            for i in range(num_examples):
                code = synthetic_code[i % len(synthetic_code)]
                # Repeat to meet token requirements
                code = code * (self.min_tokens // 50 + 1)

                f.write(json.dumps({
                    "text": code,
                    "domain": "code",
                    "num_tokens": self.count_tokens(code)
                }) + "\n")

        logger.info(f"Created {num_examples} synthetic code examples")
        return str(output_path)

    def _create_synthetic_math(self, output_path: Path, num_examples: int) -> str:
        """Create synthetic math examples for testing."""
        synthetic_math = [
            "Theorem: The sum of angles in a triangle is 180 degrees.\n\nProof: Let ABC be a triangle. Draw a line through A parallel to BC...",
            "Problem: Solve for x: 2x + 5 = 15\n\nSolution: 2x = 10, therefore x = 5",
            "The Pythagorean theorem states that in a right triangle, a² + b² = c², where c is the hypotenuse.",
            "Calculate the derivative of f(x) = x³ + 2x² - 5x + 1\n\nf'(x) = 3x² + 4x - 5",
        ]

        with open(output_path, "w") as f:
            for i in range(num_examples):
                text = synthetic_math[i % len(synthetic_math)]
                # Repeat to meet token requirements
                text = text * (self.min_tokens // 50 + 1)

                f.write(json.dumps({
                    "text": text,
                    "domain": "math",
                    "num_tokens": self.count_tokens(text)
                }) + "\n")

        logger.info(f"Created {num_examples} synthetic math examples")
        return str(output_path)

    def create_test_dataset(self, num_examples: int = 100) -> Dict[str, str]:
        """Create small test dataset for pipeline validation.

        Args:
            num_examples: Number of examples per domain

        Returns:
            Dict mapping domain to output path
        """
        logger.info(f"Creating test dataset with {num_examples} examples per domain")

        test_dir = self.output_dir / "test"
        test_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Text: Use WikiText-103 (small, fast to download)
        try:
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
            output_path = test_dir / "text.jsonl"

            with open(output_path, "w") as f:
                count = 0
                for example in dataset:
                    if count >= num_examples:
                        break

                    text = example["text"]
                    if len(text.strip()) < 100:
                        continue

                    chunks = self.chunk_text(text, "text")
                    for chunk in chunks[:1]:  # One chunk per example
                        if chunk["num_tokens"] >= self.min_tokens:
                            f.write(json.dumps(chunk) + "\n")
                            count += 1

            paths["text"] = str(output_path)
            logger.info(f"Created {count} test text examples")
        except Exception as e:
            logger.error(f"Failed to create test text dataset: {e}")

        # Code: Synthetic for speed
        paths["code"] = self._create_synthetic_code(
            test_dir / "code.jsonl",
            num_examples
        )

        # Math: Synthetic for speed
        paths["math"] = self._create_synthetic_math(
            test_dir / "math.jsonl",
            num_examples
        )

        return paths

    def create_train_val_split(
        self,
        input_files: List[str],
        val_ratio: float = 0.05,
    ) -> None:
        """Split dataset into train/val.

        Args:
            input_files: List of input JSONL files
            val_ratio: Fraction for validation set
        """
        import random
        random.seed(self.seed)

        for input_file in input_files:
            input_path = Path(input_file)
            domain = input_path.parent.name

            train_path = self.output_dir / "train" / domain / input_path.name
            val_path = self.output_dir / "val" / domain / input_path.name

            train_path.parent.mkdir(parents=True, exist_ok=True)
            val_path.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"Splitting {input_file} into train/val")

            # Load all examples
            examples = []
            with open(input_path) as f:
                for line in f:
                    examples.append(line)

            # Shuffle and split
            random.shuffle(examples)
            split_idx = int(len(examples) * (1 - val_ratio))

            # Write splits
            with open(train_path, "w") as f:
                f.writelines(examples[:split_idx])

            with open(val_path, "w") as f:
                f.writelines(examples[split_idx:])

            logger.info(f"Train: {split_idx}, Val: {len(examples) - split_idx}")

    def print_statistics(self) -> None:
        """Print dataset statistics."""
        logger.info("\n" + "="*60)
        logger.info("Dataset Preparation Statistics")
        logger.info("="*60)

        total_examples = 0
        for domain, stats in self.stats.items():
            logger.info(f"\n{domain.upper()}:")
            logger.info(f"  Total processed: {stats['total']}")
            logger.info(f"  After filtering: {stats['filtered']}")
            logger.info(f"  Chunks created: {stats['chunks']}")
            total_examples += stats['filtered']

        logger.info(f"\nTotal examples: {total_examples}")
        logger.info("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare datasets for knowledge distillation"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "text", "code", "math", "test"],
        default="test",
        help="Which datasets to prepare"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/unlabeled",
        help="Output directory for prepared datasets"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum examples per domain (None = unlimited)"
    )
    parser.add_argument(
        "--num-test-examples",
        type=int,
        default=100,
        help="Number of test examples per domain"
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=512,
        help="Minimum tokens per example"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens per example (will chunk longer)"
    )
    parser.add_argument(
        "--create-splits",
        action="store_true",
        help="Create train/val splits (95/5)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="HuggingFace tokenizer name"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token for gated models (or set HF_TOKEN env var)"
    )

    args = parser.parse_args()

    preparer = DatasetPreparer(
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer,
        hf_token=args.hf_token,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
    )

    files_to_split = []

    if args.mode == "test":
        logger.info("Creating test dataset for pipeline validation...")
        paths = preparer.create_test_dataset(args.num_test_examples)
        for domain, path in paths.items():
            logger.info(f"  {domain}: {path}")

    elif args.mode in ["all", "text"]:
        path = preparer.prepare_text_dataset(max_examples=args.max_examples)
        files_to_split.append(path)

        if args.mode == "text":
            preparer.print_statistics()
            if args.create_splits:
                preparer.create_train_val_split(files_to_split)
            return

    if args.mode in ["all", "code"]:
        path = preparer.prepare_code_dataset(max_examples=args.max_examples)
        files_to_split.append(path)

        if args.mode == "code":
            preparer.print_statistics()
            if args.create_splits:
                preparer.create_train_val_split(files_to_split)
            return

    if args.mode in ["all", "math"]:
        path = preparer.prepare_math_dataset(max_examples=args.max_examples)
        files_to_split.append(path)

        if args.mode == "math":
            preparer.print_statistics()
            if args.create_splits:
                preparer.create_train_val_split(files_to_split)
            return

    preparer.print_statistics()

    if args.create_splits and files_to_split:
        preparer.create_train_val_split(files_to_split)


if __name__ == "__main__":
    main()
