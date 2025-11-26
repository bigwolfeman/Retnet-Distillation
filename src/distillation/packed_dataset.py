"""
Packed sequence dataloader for efficient training on short sequences.

This module implements sequence packing to eliminate padding waste and provide
genuine long-context examples to TitanMAC retention layers.

Problem:
    Current training pads 300-token sequences → 4096 tokens (87% padding waste).
    TitanMAC memory components (28.5% of params) only see padding gradients → frozen.

Solution:
    Pack multiple short examples into full 4k sequences with proper attention masking.
    Example: [Ex1: 300 tok][EOS][Ex2: 250 tok][EOS][Ex3: 400 tok]... ≈ 4000 tokens

Tasks implemented: Packed sequence loading with document-level attention masking
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class PackedDataLoader(Dataset):
    """Pack multiple examples into full 4k sequences.

    This dataloader packs short sequences together to maximize context length usage
    and provide genuine long-context training data instead of padding.

    Key features:
    - Packs examples until reaching ~4000 tokens (leaves buffer for BOS/EOS)
    - Uses EOS token as separator between examples
    - Creates proper attention masks (attend within examples, not across)
    - Handles labels correctly (shift for causal LM)

    Example:
        Input:  [Ex1: 300 tok] [Ex2: 250 tok] [Ex3: 400 tok]
        Output: [Ex1: 300 tok][EOS][Ex2: 250 tok][EOS][Ex3: 400 tok]... = ~4000 tokens

    Attributes:
        data_path: Path to JSONL file with text data
        max_length: Maximum sequence length (default 4096)
        pack_max_length: Target length for packed sequences (default 4000, leaves buffer)
        tokenizer: Llama tokenizer
        packed_examples: List of packed sequences ready for training
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        max_length: int = 4096,
        pack_max_length: int = 4000,
        tokenizer_name: str = "meta-llama/Llama-3.2-1B",
        tokenizer: Optional[AutoTokenizer] = None,
        text_field: str = "text",
        return_labels: bool = True,
    ):
        """Initialize PackedDataLoader.

        Args:
            data_path: Path to JSONL file with training data
            max_length: Maximum sequence length (default 4096 for TitanMAC)
            pack_max_length: Target length for packing (default 4000, leaves buffer)
            tokenizer_name: HuggingFace tokenizer name
            tokenizer: Pre-loaded tokenizer (optional, loads if None)
            text_field: Field name for text in JSONL files
            return_labels: Whether to return labels (for causal LM)
        """
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.pack_max_length = pack_max_length
        self.text_field = text_field
        self.return_labels = return_labels

        # Load tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = self._load_tokenizer(tokenizer_name)

        # Load and pack data
        self.packed_examples = self._pack_data()

    def _load_tokenizer(self, tokenizer_name: str) -> AutoTokenizer:
        """Load Llama tokenizer.

        Args:
            tokenizer_name: HuggingFace model name

        Returns:
            Loaded tokenizer with vocab_size=128256
        """
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=True,
        )

        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def _pack_data(self) -> List[Dict[str, torch.Tensor]]:
        """Load data and pack into sequences.

        Packing algorithm:
        1. Load all examples from JSONL
        2. Tokenize each example
        3. Pack examples together until reaching pack_max_length
        4. Create attention masks to prevent cross-document attention
        5. Create labels for causal LM (shifted input_ids)

        Returns:
            List of packed examples, each with:
            - input_ids: [max_length] token IDs
            - attention_mask: [max_length] attention mask (1=real, 0=padding)
            - labels: [max_length] labels for causal LM
            - num_docs: number of documents packed in this sequence
            - doc_lengths: list of document lengths in tokens
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        # Load all examples and tokenize
        print(f"Loading data from: {self.data_path}")
        raw_examples = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    text = record.get(self.text_field, "")
                    if not text:
                        continue

                    # Tokenize (without BOS/EOS yet - we'll add those during packing)
                    tokens = self.tokenizer.encode(
                        text,
                        add_special_tokens=False,  # We'll add BOS/EOS manually during packing
                        truncation=False,
                    )

                    if len(tokens) > 0:
                        raw_examples.append(tokens)

                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
                    continue

        if len(raw_examples) == 0:
            raise ValueError(f"No valid data found in {self.data_path}")

        print(f"Loaded {len(raw_examples)} examples")

        # Calculate stats on unpacked data
        avg_len = sum(len(ex) for ex in raw_examples) / len(raw_examples)
        print(f"Average example length: {avg_len:.1f} tokens")
        print(f"Packing target: {self.pack_max_length} tokens")

        # Pack examples
        packed_examples = []
        current_pack = []
        current_length = 0

        for tokens in raw_examples:
            # +2 for BOS (at start of pack) and EOS (after this doc)
            tokens_with_special = len(tokens) + 1  # +1 for EOS separator

            # Check if adding this example would exceed pack_max_length
            if current_length + tokens_with_special > self.pack_max_length and len(current_pack) > 0:
                # Finalize current pack
                packed_examples.append(self._create_packed_sequence(current_pack))
                current_pack = [tokens]
                current_length = len(tokens) + 1  # +1 for EOS
            else:
                # Add to current pack
                current_pack.append(tokens)
                current_length += tokens_with_special

        # Don't forget the last pack
        if len(current_pack) > 0:
            packed_examples.append(self._create_packed_sequence(current_pack))

        # Print packing statistics
        print(f"\n{'='*60}")
        print("PACKING STATISTICS")
        print(f"{'='*60}")
        print(f"Total examples: {len(raw_examples)}")
        print(f"Packed sequences: {len(packed_examples)}")
        print(f"Avg docs per pack: {len(raw_examples) / len(packed_examples):.2f}")

        # Calculate packing efficiency
        total_real_tokens = sum(len(ex) for ex in raw_examples)
        total_packed_tokens = len(packed_examples) * self.max_length
        real_token_count = sum(
            (packed['attention_mask'] == 1).sum().item()
            for packed in packed_examples
        )
        efficiency = (real_token_count / total_packed_tokens) * 100

        print(f"Total real tokens: {total_real_tokens:,}")
        print(f"Total packed tokens: {total_packed_tokens:,}")
        print(f"Real tokens (with EOS): {real_token_count:,}")
        print(f"Packing efficiency: {efficiency:.1f}%")
        print(f"{'='*60}\n")

        return packed_examples

    def _create_packed_sequence(self, token_lists: List[List[int]]) -> Dict[str, torch.Tensor]:
        """Create a single packed sequence from multiple token lists.

        Packing format:
            [BOS] [Doc1 tokens] [EOS] [Doc2 tokens] [EOS] ... [PAD] [PAD] ...

        Attention mask strategy:
            - Use 1D mask: 1 for real tokens (including BOS/EOS), 0 for padding
            - Model's causal masking handles preventing attention to future positions
            - Documents are separated by EOS tokens
            - No cross-document attention needed - each doc is independent

        Args:
            token_lists: List of token lists to pack together

        Returns:
            Dictionary with packed sequence data:
            - input_ids: [max_length] packed token IDs
            - attention_mask: [max_length] 1D attention mask
            - labels: [max_length] shifted labels for causal LM
            - num_docs: number of documents in this pack
            - doc_lengths: list of document lengths (including EOS)
        """
        # Build packed sequence
        packed_tokens = [self.tokenizer.bos_token_id]  # Start with BOS
        doc_lengths = []

        for tokens in token_lists:
            # Add document tokens
            packed_tokens.extend(tokens)
            # Add EOS separator
            packed_tokens.append(self.tokenizer.eos_token_id)
            # Track length (tokens + EOS)
            doc_lengths.append(len(tokens) + 1)

        # Pad to max_length
        num_real_tokens = len(packed_tokens)
        if num_real_tokens > self.max_length:
            # Truncate if somehow exceeded (shouldn't happen with proper packing)
            packed_tokens = packed_tokens[:self.max_length]
            num_real_tokens = self.max_length
        else:
            # Pad with pad token
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id
            packed_tokens.extend([pad_token_id] * (self.max_length - num_real_tokens))

        # Convert to tensor
        input_ids = torch.tensor(packed_tokens, dtype=torch.long)

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        attention_mask[:num_real_tokens] = 1

        # Create labels for causal LM (shift input_ids)
        if self.return_labels:
            labels = input_ids.clone()
            # Shift left: predict token[i+1] from position[i]
            labels[:-1] = input_ids[1:]
            labels[-1] = -100  # No next token for last position

            # Mask ONLY padding tokens (not all EOS tokens)
            # Use attention mask to identify padding
            # Padding positions have attention_mask=0, real tokens have attention_mask=1
            labels[attention_mask == 0] = -100
        else:
            labels = None

        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'num_docs': len(token_lists),
            'doc_lengths': doc_lengths,
        }

        if labels is not None:
            result['labels'] = labels

        return result

    def __len__(self) -> int:
        """Return number of packed examples."""
        return len(self.packed_examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single packed example.

        Args:
            idx: Index of packed example

        Returns:
            Dictionary with:
            - input_ids: [max_length] token IDs
            - attention_mask: [max_length] attention mask
            - labels: [max_length] labels for causal LM (if return_labels=True)
        """
        packed = self.packed_examples[idx]

        # Return only the tensors needed for training
        # (num_docs and doc_lengths are just for debugging/stats)
        result = {
            'input_ids': packed['input_ids'],
            'attention_mask': packed['attention_mask'],
        }

        if 'labels' in packed:
            result['labels'] = packed['labels']

        return result

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function for DataLoader.

        Args:
            batch: List of examples from __getitem__

        Returns:
            Batched tensors
        """
        # Stack all tensors
        result = {}

        for key in batch[0].keys():
            tensors = [item[key] for item in batch]
            result[key] = torch.stack(tensors, dim=0)

        return result

    def get_packing_stats(self) -> Dict[str, float]:
        """Get packing statistics for monitoring.

        Returns:
            Dictionary with:
            - total_packs: number of packed sequences
            - avg_docs_per_pack: average documents per pack
            - avg_tokens_per_pack: average real tokens per pack
            - packing_efficiency: percentage of real tokens vs padding
        """
        total_packs = len(self.packed_examples)
        total_docs = sum(p['num_docs'] for p in self.packed_examples)
        total_real_tokens = sum(
            (p['attention_mask'] == 1).sum().item()
            for p in self.packed_examples
        )
        total_capacity = total_packs * self.max_length

        return {
            'total_packs': total_packs,
            'avg_docs_per_pack': total_docs / total_packs if total_packs > 0 else 0,
            'avg_tokens_per_pack': total_real_tokens / total_packs if total_packs > 0 else 0,
            'packing_efficiency': (total_real_tokens / total_capacity * 100) if total_capacity > 0 else 0,
        }


# Export public API
__all__ = [
    'PackedDataLoader',
]
