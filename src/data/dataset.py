"""Example dataset loading for RetNet-HRM.

Implements GSM8k dataset loading via HuggingFace datasets as an example.
Formats examples for language modeling task.
"""

from datasets import load_dataset
from typing import Dict, Any, Optional
from torch.utils.data import Dataset


class GSM8kDataset(Dataset):
    """GSM8k dataset wrapper for language modeling.

    Formats mathematical reasoning examples as Q: ... A: ... for
    next-token prediction training.
    """

    def __init__(
        self,
        tokenizer: Any,
        split: str = "train",
        max_length: int = 32768,
        max_examples: Optional[int] = None,
    ):
        """Initialize dataset.

        Args:
            tokenizer: Tokenizer for encoding text
            split: Dataset split ("train" or "test")
            max_length: Maximum sequence length
            max_examples: Limit number of examples (for testing)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = load_dataset("gsm8k", "main")[split]

        if max_examples is not None:
            self.dataset = self.dataset.select(range(min(max_examples, len(self.dataset))))

        self.split = split

    def __len__(self) -> int:
        """Number of examples in dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get example at index.

        Args:
            idx: Example index

        Returns:
            Dict with 'text' key containing formatted example
        """
        example = self.dataset[idx]

        # Format as Q: ... A: ... for language modeling
        text = f"Q: {example['question']}\nA: {example['answer']}"

        # Return raw text - tokenization is handled by DataCollator
        return {"text": text}

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics.

        Returns:
            Dict with size, split info
        """
        return {
            "split": self.split,
            "size": len(self),
            "dataset_name": "gsm8k",
        }


# Alias for compatibility (capital K vs lowercase k)
GSM8KDataset = GSM8kDataset
