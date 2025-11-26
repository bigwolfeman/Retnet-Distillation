"""Tokenizer wrapper for RetNet-HRM.

Uses CodeGen tokenizer from HuggingFace with additional landmark tokens.
Implements requirements from research.md Decision 10.
"""

from transformers import AutoTokenizer
from typing import List, Dict, Any, Union, Optional
import torch


class RetNetTokenizer:
    """Wrapper around HuggingFace tokenizer with landmark token support.

    Uses CodeGen vocabulary (code-optimized BPE) with 256 additional
    special tokens reserved for retrieval landmark pointers.
    """

    def __init__(
        self,
        pretrained_model: str = "Salesforce/codegen-2B-mono",
        max_length: int = 128000,  # Stretch goal support
        num_landmark_tokens: int = 256,
    ):
        """Initialize tokenizer.

        Args:
            pretrained_model: HuggingFace model ID for tokenizer
            max_length: Maximum sequence length
            num_landmark_tokens: Number of special tokens for landmarks
        """
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

        # Configure for long contexts
        self.tokenizer.model_max_length = max_length
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "right"

        # Set pad_token if not already set (use eos_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add special landmark tokens
        landmark_tokens = [f"<LANDMARK_{i}>" for i in range(num_landmark_tokens)]
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": landmark_tokens
        })

        self.vocab_size = len(self.tokenizer)
        self.landmark_token_ids = [
            self.tokenizer.convert_tokens_to_ids(tok) for tok in landmark_tokens
        ]

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
    ) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: Add BOS/EOS tokens
            max_length: Maximum length (uses default if None)
            truncation: Truncate to max_length

        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            max_length=max_length or self.tokenizer.model_max_length,
            truncation=truncation,
        )

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: Token IDs (list or tensor)
            skip_special_tokens: Skip special tokens in output

        Returns:
            Decoded text string
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

    def batch_encode(
        self,
        texts: List[str],
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """Batch encode texts.

        Args:
            texts: List of input texts
            padding: Pad to max length in batch
            truncation: Truncate to max_length
            max_length: Maximum length
            return_tensors: Return as PyTorch tensors

        Returns:
            Dict with input_ids, attention_mask tensors
        """
        return self.tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=max_length or self.tokenizer.model_max_length,
            return_tensors=return_tensors,
        )

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size

    def save_pretrained(self, save_directory: str):
        """Save tokenizer to directory."""
        self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, load_directory: str) -> "RetNetTokenizer":
        """Load tokenizer from directory."""
        instance = cls.__new__(cls)
        instance.tokenizer = AutoTokenizer.from_pretrained(load_directory)
        instance.vocab_size = len(instance.tokenizer)
        # Reconstruct landmark token IDs
        landmark_tokens = [f"<LANDMARK_{i}>" for i in range(256)]
        instance.landmark_token_ids = [
            instance.tokenizer.convert_tokens_to_ids(tok) for tok in landmark_tokens
        ]
        return instance


def load_tokenizer(pretrained_model: str = "Salesforce/codegen-2B-mono") -> RetNetTokenizer:
    """Convenience function to load tokenizer.

    Args:
        pretrained_model: HuggingFace model ID

    Returns:
        RetNetTokenizer instance
    """
    return RetNetTokenizer(pretrained_model=pretrained_model)


# Alias for compatibility
get_tokenizer = load_tokenizer
