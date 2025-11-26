"""Batch collation for RetNet-HRM training.

Handles padding, truncation, and attention mask creation for variable-length
sequences. Supports sequences up to 32768 tokens (training) / 65536 (inference).
"""

import torch
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class DataCollator:
    """Collates variable-length sequences into batches.

    Handles:
    - Padding/truncation to max_length
    - Attention mask creation
    - Label generation for language modeling

    Attributes:
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        pad_to_multiple_of: Pad length to multiple of this value
    """

    tokenizer: Any  # RetNetTokenizer
    max_length: int = 32768
    pad_to_multiple_of: int = 8  # For efficient GPU computation

    def __call__(self, examples: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        """Collate examples into batch.

        Args:
            examples: List of dicts with 'text' key

        Returns:
            Dict with input_ids, attention_mask, labels tensors
        """
        # Extract texts
        texts = [ex["text"] for ex in examples]

        # Tokenize batch
        batch = self.tokenizer.batch_encode(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Pad to multiple (for efficiency)
        if self.pad_to_multiple_of is not None:
            batch = self._pad_to_multiple(batch, self.pad_to_multiple_of)

        # Create labels for language modeling (shifted input_ids)
        # Labels are the same as input_ids, model handles shifting internally
        batch["labels"] = batch["input_ids"].clone()

        # Mask padding tokens in labels (-100 = ignore index)
        if "attention_mask" in batch:
            batch["labels"][batch["attention_mask"] == 0] = -100

        return batch

    def _pad_to_multiple(
        self,
        batch: Dict[str, torch.Tensor],
        multiple: int,
    ) -> Dict[str, torch.Tensor]:
        """Pad tensors to multiple of given value.

        Args:
            batch: Batch dict with tensors
            multiple: Pad to multiple of this value

        Returns:
            Padded batch
        """
        seq_len = batch["input_ids"].size(1)
        padded_len = ((seq_len + multiple - 1) // multiple) * multiple

        if padded_len == seq_len:
            return batch  # Already multiple

        padding_len = padded_len - seq_len

        # Pad input_ids
        batch["input_ids"] = torch.nn.functional.pad(
            batch["input_ids"],
            (0, padding_len),
            value=self.tokenizer.tokenizer.pad_token_id or 0,
        )

        # Pad attention_mask
        if "attention_mask" in batch:
            batch["attention_mask"] = torch.nn.functional.pad(
                batch["attention_mask"],
                (0, padding_len),
                value=0,  # Padding positions have mask=0
            )

        return batch


def create_collator(tokenizer: Any, max_length: int = 32768) -> DataCollator:
    """Create data collator instance.

    Args:
        tokenizer: RetNetTokenizer instance
        max_length: Maximum sequence length

    Returns:
        DataCollator instance
    """
    return DataCollator(
        tokenizer=tokenizer,
        max_length=max_length,
    )
