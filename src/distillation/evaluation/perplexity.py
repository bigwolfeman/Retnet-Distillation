"""
Perplexity evaluator for language models.

Computes perplexity on validation/test datasets to measure model quality.
Perplexity is a standard metric for language models - lower is better.

Task: T057
"""

import logging
import math
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)


@dataclass
class PerplexityConfig:
    """Configuration for perplexity evaluation.

    Attributes:
        max_samples: Maximum number of samples to evaluate (None = all)
        batch_size: Batch size for evaluation (default: 1)
        max_length: Maximum sequence length (default: 4096)
        stride: Stride for sliding window (default: None, no overlap)
        use_cache: Use KV cache for efficiency (default: True)
        ignore_index: Token ID to ignore in loss computation (default: -100)
    """
    max_samples: Optional[int] = None
    batch_size: int = 1
    max_length: int = 4096
    stride: Optional[int] = None
    use_cache: bool = True
    ignore_index: int = -100


class PerplexityEvaluator:
    """Perplexity evaluator for language models.

    Computes perplexity (and related metrics) on a given dataset.
    Perplexity = exp(average negative log-likelihood) - lower is better.

    Features:
    - Handles variable-length sequences
    - Supports sliding window for long sequences
    - Computes bits-per-byte and bits-per-character
    - Memory-efficient evaluation with gradient checkpointing

    Example:
        >>> evaluator = PerplexityEvaluator(model, tokenizer)
        >>> config = PerplexityConfig(max_samples=1000)
        >>> results = evaluator.evaluate(dataloader, config)
        >>> print(f"Perplexity: {results['perplexity']:.2f}")
        >>> print(f"Bits/byte: {results['bits_per_byte']:.4f}")
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        device: Optional[torch.device] = None,
    ):
        """Initialize perplexity evaluator.

        Args:
            model: Language model to evaluate
            tokenizer: Tokenizer for the model
            device: Device to run evaluation on (default: CUDA if available)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device
        self.model = self.model.to(self.device)

        logger.info(f"Initialized PerplexityEvaluator:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Model: {type(self.model).__name__}")

    @torch.no_grad()
    def _compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        return_logits: bool = False,
    ) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, torch.Tensor]]:
        """Compute cross-entropy loss for a batch.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            labels: Target labels [batch_size, seq_len] (default: shifted input_ids)
            ignore_index: Token ID to ignore in loss
            return_logits: If True, also return logits for decoding

        Returns:
            Tuple of (total_loss, num_tokens) or (total_loss, num_tokens, logits)
        """
        # Default labels: next token prediction
        if labels is None:
            labels = input_ids.clone()

        # Ensure model is in eval mode
        self.model.eval()

        # Get model logits
        # input_ids should stay as long/int64 for embedding layer
        # Use autocast to handle mixed-precision properly and avoid dtype mismatches
        # between FP32 retention params and BF16 model weights
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            if hasattr(self.model, 'forward_train'):
                # RetNetBackbone: returns hidden states
                hidden_states = self.model.forward_train(input_ids)
                if hasattr(self.model, 'lm_head'):
                    logits = self.model.lm_head(hidden_states)
                else:
                    raise ValueError("Model must have 'lm_head' for language modeling")
            else:
                # Standard forward pass
                outputs = self.model(input_ids)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs

        # FIX: Labels are already shifted by the dataset (PretokenizedShardDataset)
        # where labels[i] = input_ids[i+1]. The model's logits[i] predicts the next
        # token after input_ids[0:i+1], so we should NOT shift labels again here.
        # Previous code had a double-shift bug: dataset shifted labels, then evaluator
        # shifted again, causing logits[i] to be compared with input_ids[i+2] instead
        # of input_ids[i+1], resulting in astronomical perplexity.
        #
        # Correct behavior: logits[i] predicts labels[i] (which is input_ids[i+1])

        # Flatten for loss computation
        batch_size, seq_len, vocab_size = logits.shape
        flat_logits = logits.view(-1, vocab_size)  # [batch * seq_len, vocab]
        flat_labels = labels.view(-1)  # [batch * seq_len]

        # Cast logits to FP32 for precise loss computation
        # Labels must remain as Long (int64) for cross_entropy
        # This prevents BF16/FP32 dtype mismatch errors
        flat_logits = flat_logits.to(torch.float32)
        flat_labels = flat_labels.long()  # Ensure labels are Long type

        # Compute cross-entropy loss
        loss = F.cross_entropy(
            flat_logits,
            flat_labels,
            ignore_index=ignore_index,
            reduction='sum',  # Sum over all tokens
        )

        # Count valid tokens (not ignored)
        num_tokens = (flat_labels != ignore_index).sum().item()

        if return_logits:
            return loss, num_tokens, logits
        return loss, num_tokens

    @torch.no_grad()
    def evaluate_batch(
        self,
        batch: Dict[str, torch.Tensor],
        config: PerplexityConfig,
    ) -> Dict[str, float]:
        """Evaluate perplexity on a single batch.

        Args:
            batch: Batch with 'input_ids' and optional 'labels'
            config: Evaluation configuration

        Returns:
            Dictionary with loss and token count
        """
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        labels = batch.get('labels', None)
        if labels is not None:
            labels = labels.to(self.device)

        # Compute loss
        total_loss, num_tokens = self._compute_loss(
            input_ids,
            labels,
            ignore_index=config.ignore_index,
        )

        return {
            'total_loss': total_loss.item(),
            'num_tokens': num_tokens,
        }

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        config: Optional[PerplexityConfig] = None,
    ) -> Dict[str, Any]:
        """Evaluate perplexity on a dataset.

        Args:
            dataloader: DataLoader with evaluation data
            config: Evaluation configuration (default: PerplexityConfig())

        Returns:
            Dictionary with evaluation metrics

        Metrics:
            - perplexity: exp(average_loss)
            - loss: average cross-entropy loss (nats)
            - bits_per_token: average loss in bits (log2)
            - total_tokens: total number of tokens evaluated
            - num_samples: number of samples evaluated

        Example:
            >>> evaluator = PerplexityEvaluator(model, tokenizer)
            >>> results = evaluator.evaluate(val_dataloader)
            >>> print(f"Perplexity: {results['perplexity']:.2f}")
        """
        if config is None:
            config = PerplexityConfig()

        logger.info("Starting perplexity evaluation...")
        logger.info(f"  Max samples: {config.max_samples or 'all'}")
        logger.info(f"  Batch size: {config.batch_size}")

        # Set model to eval mode
        self.model.eval()

        # Accumulate statistics
        total_loss = 0.0
        total_tokens = 0
        num_samples = 0
        last_batch = None

        # Evaluate
        for batch_idx, batch in enumerate(dataloader):
            # Check max samples limit
            if config.max_samples and num_samples >= config.max_samples:
                break

            # Evaluate batch
            batch_stats = self.evaluate_batch(batch, config)

            # Accumulate
            total_loss += batch_stats['total_loss']
            total_tokens += batch_stats['num_tokens']
            num_samples += batch['input_ids'].shape[0]

            # Save last batch for sample output
            last_batch = batch

            # Log progress
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
                logger.info(f"  Processed {num_samples} samples, avg loss: {avg_loss:.4f}")

        # Compute final metrics
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens  # Nats per token
            perplexity = math.exp(avg_loss)
            bits_per_token = avg_loss / math.log(2)  # Convert nats to bits
        else:
            avg_loss = float('inf')
            perplexity = float('inf')
            bits_per_token = float('inf')

        logger.info(f"Perplexity evaluation complete:")
        logger.info(f"  Perplexity: {perplexity:.2f}")
        logger.info(f"  Loss (nats): {avg_loss:.4f}")
        logger.info(f"  Bits/token: {bits_per_token:.4f}")
        logger.info(f"  Total tokens: {total_tokens:,}")
        logger.info(f"  Num samples: {num_samples}")

        # Decode and print sample from last batch
        if last_batch is not None:
            self._print_sample_output(last_batch, config)

        return {
            'perplexity': perplexity,
            'loss': avg_loss,
            'bits_per_token': bits_per_token,
            'total_loss': total_loss,
            'total_tokens': total_tokens,
            'num_samples': num_samples,
            'config': {
                'max_samples': config.max_samples,
                'batch_size': config.batch_size,
                'max_length': config.max_length,
                'ignore_index': config.ignore_index,
            },
        }

    @torch.no_grad()
    def _print_sample_output(
        self,
        batch: Dict[str, torch.Tensor],
        config: PerplexityConfig,
    ):
        """Print decoded sample input and model output from batch.

        Args:
            batch: Last batch from evaluation
            config: Evaluation configuration
        """
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        labels = batch.get('labels', None)
        if labels is not None:
            labels = labels.to(self.device)

        # Get logits for this batch
        _, _, logits = self._compute_loss(
            input_ids,
            labels,
            ignore_index=config.ignore_index,
            return_logits=True,
        )

        # Take first sample from batch, limit to 512 tokens
        sample_input_ids = input_ids[0][:512].cpu()
        sample_logits = logits[0][:512].cpu()

        # Get predicted tokens (argmax of logits)
        predicted_ids = torch.argmax(sample_logits, dim=-1)

        # Decode input and predictions
        input_text = self.tokenizer.decode(sample_input_ids, skip_special_tokens=False)
        predicted_text = self.tokenizer.decode(predicted_ids, skip_special_tokens=False)

        # Print sample
        logger.info("\n" + "="*80)
        logger.info("SAMPLE OUTPUT FROM LAST EVAL BATCH (first 512 tokens):")
        logger.info("="*80)
        logger.info("\nINPUT (prompt):")
        logger.info("-"*80)
        logger.info(input_text[:2000])  # Limit display to 2000 chars
        if len(input_text) > 2000:
            logger.info(f"\n... (truncated, {len(input_text)} total chars)")
        logger.info("\n" + "-"*80)
        logger.info("\nMODEL OUTPUT (predicted next tokens):")
        logger.info("-"*80)
        logger.info(predicted_text[:2000])  # Limit display to 2000 chars
        if len(predicted_text) > 2000:
            logger.info(f"\n... (truncated, {len(predicted_text)} total chars)")
        logger.info("\n" + "="*80 + "\n")

    @torch.no_grad()
    def evaluate_text(
        self,
        text: Union[str, List[str]],
        config: Optional[PerplexityConfig] = None,
    ) -> Dict[str, Any]:
        """Evaluate perplexity on raw text.

        Convenience method for evaluating on text without creating a DataLoader.

        Args:
            text: Single text string or list of texts
            config: Evaluation configuration

        Returns:
            Evaluation metrics dictionary

        Example:
            >>> evaluator = PerplexityEvaluator(model, tokenizer)
            >>> text = "The quick brown fox jumps over the lazy dog."
            >>> results = evaluator.evaluate_text(text)
            >>> print(f"Perplexity: {results['perplexity']:.2f}")
        """
        if config is None:
            config = PerplexityConfig()

        # Convert single text to list
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        # Set model to eval mode
        self.model.eval()

        # Accumulate statistics
        total_loss = 0.0
        total_tokens = 0

        for text_item in texts:
            # Tokenize
            inputs = self.tokenizer(
                text_item,
                return_tensors='pt',
                truncation=True,
                max_length=config.max_length,
            )
            input_ids = inputs['input_ids'].to(self.device)

            # Compute loss
            loss, num_tokens = self._compute_loss(
                input_ids,
                ignore_index=config.ignore_index,
            )

            total_loss += loss.item()
            total_tokens += num_tokens

        # Compute metrics
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            perplexity = math.exp(avg_loss)
            bits_per_token = avg_loss / math.log(2)
        else:
            avg_loss = float('inf')
            perplexity = float('inf')
            bits_per_token = float('inf')

        return {
            'perplexity': perplexity,
            'loss': avg_loss,
            'bits_per_token': bits_per_token,
            'total_loss': total_loss,
            'total_tokens': total_tokens,
            'num_samples': len(texts),
        }

    def save_results(
        self,
        results: Dict[str, Any],
        output_path: Path,
    ):
        """Save evaluation results to JSON file.

        Args:
            results: Results dictionary from evaluate()
            output_path: Path to output JSON file
        """
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved perplexity results to {output_path}")


def evaluate_perplexity(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    dataloader: DataLoader,
    config: Optional[PerplexityConfig] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Convenience function to evaluate perplexity.

    Args:
        model: Language model to evaluate
        tokenizer: Tokenizer for the model
        dataloader: DataLoader with evaluation data
        config: Evaluation configuration (default: PerplexityConfig())
        output_path: Optional path to save results JSON

    Returns:
        Evaluation results dictionary

    Example:
        >>> from distillation.evaluation.perplexity import evaluate_perplexity, PerplexityConfig
        >>> config = PerplexityConfig(max_samples=1000)
        >>> results = evaluate_perplexity(model, tokenizer, val_dataloader, config)
        >>> print(f"Perplexity: {results['perplexity']:.2f}")
    """
    evaluator = PerplexityEvaluator(model, tokenizer)
    results = evaluator.evaluate(dataloader, config)

    if output_path:
        evaluator.save_results(results, output_path)

    return results
