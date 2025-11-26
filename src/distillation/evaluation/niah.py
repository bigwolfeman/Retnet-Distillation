"""
Needle-in-a-Haystack (NIAH) evaluator for testing long-context retrieval.

Tests model's ability to retrieve specific information ("needle") from a long
context ("haystack"). This is a critical capability test for models that need
to handle long sequences.

Task: T056
"""

import logging
import random
import string
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)


@dataclass
class NIAHConfig:
    """Configuration for NIAH evaluation.

    Attributes:
        context_length: Length of context to test (default: 4096)
        num_samples: Number of test samples (default: 100)
        needle_positions: List of relative positions to test (0.0-1.0)
        min_position: Minimum relative position (default: 0.1)
        max_position: Maximum relative position (default: 0.9)
        num_positions: Number of positions to test (default: 5)
        seed: Random seed for reproducibility (default: 42)
    """
    context_length: int = 4096
    num_samples: int = 100
    needle_positions: Optional[List[float]] = None
    min_position: float = 0.1
    max_position: float = 0.9
    num_positions: int = 5
    seed: int = 42

    def __post_init__(self):
        """Generate needle positions if not provided."""
        if self.needle_positions is None:
            # Generate evenly spaced positions between min and max
            self.needle_positions = [
                self.min_position + i * (self.max_position - self.min_position) / (self.num_positions - 1)
                for i in range(self.num_positions)
            ]


class NIAHEvaluator:
    """Needle-in-a-Haystack evaluator for long-context retrieval.

    Tests model's ability to retrieve specific information from long contexts
    at various positions. This is a critical capability for long-context models.

    The test works by:
    1. Generating a long "haystack" of text (typically random or semi-random)
    2. Inserting a "needle" (specific fact/number) at various positions
    3. Prompting the model to retrieve the needle
    4. Measuring retrieval accuracy

    Example:
        >>> evaluator = NIAHEvaluator(model, tokenizer)
        >>> config = NIAHConfig(context_length=4096, num_samples=100)
        >>> results = evaluator.evaluate(config)
        >>> print(f"Overall accuracy: {results['accuracy']:.2%}")
        >>> print(f"Position breakdown: {results['position_accuracies']}")
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        device: Optional[torch.device] = None,
    ):
        """Initialize NIAH evaluator.

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

        logger.info(f"Initialized NIAHEvaluator:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Model: {type(self.model).__name__}")

    def _generate_haystack(
        self,
        length: int,
        random_gen: random.Random,
    ) -> str:
        """Generate haystack text of specified length.

        Creates semi-random text that is realistic enough for the model
        to process, but doesn't contain any specific information.

        Args:
            length: Target length in tokens (approximate)
            random_gen: Random number generator for reproducibility

        Returns:
            Haystack text string
        """
        # Generate random sentences
        # Each sentence is 10-20 words, each word is 4-8 characters
        sentences = []
        current_tokens = 0
        target_tokens = length

        while current_tokens < target_tokens:
            # Generate sentence
            num_words = random_gen.randint(10, 20)
            words = []
            for _ in range(num_words):
                word_len = random_gen.randint(4, 8)
                word = ''.join(random_gen.choices(string.ascii_lowercase, k=word_len))
                words.append(word)

            sentence = ' '.join(words).capitalize() + '.'
            sentences.append(sentence)

            # Estimate tokens (rough: ~1.3 tokens per word)
            current_tokens += int(num_words * 1.3)

        haystack = ' '.join(sentences)
        return haystack

    def _create_needle(
        self,
        needle_id: int,
        random_gen: random.Random,
    ) -> Tuple[str, str]:
        """Create a needle (fact to retrieve) and its answer.

        Args:
            needle_id: Unique ID for this needle
            random_gen: Random number generator

        Returns:
            Tuple of (needle_text, answer)

        Example:
            >>> needle, answer = evaluator._create_needle(42, random_gen)
            >>> # needle: "The magic number is 7329."
            >>> # answer: "7329"
        """
        # Generate a random number as the answer
        answer = str(random_gen.randint(1000, 9999))

        # Create needle text
        needle = f"The magic number is {answer}."

        return needle, answer

    def _insert_needle(
        self,
        haystack: str,
        needle: str,
        position: float,
    ) -> str:
        """Insert needle at specified relative position in haystack.

        Args:
            haystack: Haystack text
            needle: Needle text to insert
            position: Relative position (0.0 = start, 1.0 = end)

        Returns:
            Combined text with needle inserted
        """
        # Split haystack into sentences
        sentences = haystack.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]

        # Calculate insertion point
        insert_idx = int(len(sentences) * position)
        insert_idx = max(0, min(insert_idx, len(sentences)))

        # Insert needle
        sentences.insert(insert_idx, needle)

        # Rejoin
        text = '. '.join(sentences) + '.'
        return text

    def _create_prompt(
        self,
        context: str,
    ) -> str:
        """Create prompt asking for needle retrieval.

        Args:
            context: Full context with needle inserted

        Returns:
            Prompt text
        """
        prompt = f"{context}\n\nQuestion: What is the magic number?\nAnswer:"
        return prompt

    def _extract_answer(
        self,
        generated_text: str,
    ) -> str:
        """Extract answer from model generation.

        Args:
            generated_text: Text generated by model

        Returns:
            Extracted answer (first 4-digit number found)
        """
        # Find first 4-digit number
        import re
        match = re.search(r'\b\d{4}\b', generated_text)
        if match:
            return match.group(0)
        else:
            # Fallback: return first token
            tokens = generated_text.strip().split()
            return tokens[0] if tokens else ""

    @torch.no_grad()
    def _generate(
        self,
        prompt: str,
        max_new_tokens: int = 10,
    ) -> str:
        """Generate answer from model.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer.model_max_length or 4096,
        )
        input_ids = inputs['input_ids'].to(self.device)

        # Generate
        # Handle different model APIs
        if hasattr(self.model, 'generate'):
            # Standard HuggingFace API or TitanMACForDistillation.generate()
            # TitanMACForDistillation.generate() doesn't accept pad_token_id parameter
            # Default behavior is greedy decoding (temperature=1.0, no sampling)
            # Bug #2 fix: Remove pad_token_id to avoid TypeError
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
            )
        else:
            # Fallback: use forward pass and take argmax
            # This is for models that don't have .generate()
            logger.warning("Model does not have .generate(), using argmax fallback")

            # Forward pass
            if hasattr(self.model, 'forward_train'):
                hidden_states = self.model.forward_train(input_ids)
                if hasattr(self.model, 'lm_head'):
                    logits = self.model.lm_head(hidden_states)
                else:
                    raise ValueError("Model must have 'lm_head' or 'generate' method")
            else:
                outputs = self.model(input_ids)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs

            # Take argmax for next token (simple 1-step generation)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            output_ids = torch.cat([input_ids, next_token], dim=1)

        # Decode
        generated_ids = output_ids[0, input_ids.shape[1]:]  # Only new tokens
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated_text

    def evaluate_single(
        self,
        sample_id: int,
        position: float,
        context_length: int,
        random_gen: random.Random,
    ) -> Dict[str, Any]:
        """Evaluate a single NIAH test case.

        Args:
            sample_id: Unique ID for this sample
            position: Relative position for needle (0.0-1.0)
            context_length: Target context length
            random_gen: Random number generator

        Returns:
            Dictionary with results for this sample
        """
        # Generate haystack
        haystack = self._generate_haystack(context_length, random_gen)

        # Create needle
        needle, answer = self._create_needle(sample_id, random_gen)

        # Insert needle
        context = self._insert_needle(haystack, needle, position)

        # Create prompt
        prompt = self._create_prompt(context)

        # Tokenize to get actual length
        tokens = self.tokenizer(prompt, return_tensors="pt")
        actual_length = tokens['input_ids'].shape[1]

        # Generate answer
        generated = self._generate(prompt)

        # Extract answer
        predicted_answer = self._extract_answer(generated)

        # Check correctness
        correct = predicted_answer == answer

        return {
            'sample_id': sample_id,
            'position': position,
            'target_length': context_length,
            'actual_length': actual_length,
            'needle': needle,
            'expected_answer': answer,
            'generated_text': generated,
            'predicted_answer': predicted_answer,
            'correct': correct,
        }

    def evaluate(
        self,
        config: Optional[NIAHConfig] = None,
    ) -> Dict[str, Any]:
        """Run full NIAH evaluation.

        Args:
            config: Evaluation configuration (default: NIAHConfig())

        Returns:
            Dictionary with evaluation results

        Example:
            >>> evaluator = NIAHEvaluator(model, tokenizer)
            >>> results = evaluator.evaluate()
            >>> print(f"Accuracy: {results['accuracy']:.2%}")
            >>> for pos, acc in results['position_accuracies'].items():
            ...     print(f"  Position {pos:.1%}: {acc:.2%}")
        """
        if config is None:
            config = NIAHConfig()

        logger.info("Starting NIAH evaluation...")
        logger.info(f"  Context length: {config.context_length}")
        logger.info(f"  Num samples: {config.num_samples}")
        logger.info(f"  Positions: {config.needle_positions}")

        # Set model to eval mode
        self.model.eval()

        # Initialize random generator
        random_gen = random.Random(config.seed)

        # Run evaluation
        results = []
        total_correct = 0
        position_stats = {pos: {'correct': 0, 'total': 0} for pos in config.needle_positions}

        sample_id = 0
        for _ in range(config.num_samples):
            for position in config.needle_positions:
                # Evaluate single sample
                result = self.evaluate_single(
                    sample_id=sample_id,
                    position=position,
                    context_length=config.context_length,
                    random_gen=random_gen,
                )

                results.append(result)
                sample_id += 1

                # Update stats
                if result['correct']:
                    total_correct += 1
                    position_stats[position]['correct'] += 1
                position_stats[position]['total'] += 1

                # Log progress
                if sample_id % 50 == 0:
                    logger.info(f"  Processed {sample_id} samples...")

        # Compute overall accuracy
        total_samples = len(results)
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        # Compute per-position accuracy
        position_accuracies = {
            pos: stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
            for pos, stats in position_stats.items()
        }

        logger.info(f"NIAH evaluation complete:")
        logger.info(f"  Overall accuracy: {overall_accuracy:.2%}")
        logger.info(f"  Per-position accuracies:")
        for pos, acc in sorted(position_accuracies.items()):
            logger.info(f"    Position {pos:.1%}: {acc:.2%}")

        return {
            'accuracy': overall_accuracy,
            'total_samples': total_samples,
            'total_correct': total_correct,
            'position_accuracies': position_accuracies,
            'position_stats': position_stats,
            'config': {
                'context_length': config.context_length,
                'num_samples': config.num_samples,
                'needle_positions': config.needle_positions,
                'seed': config.seed,
            },
            'samples': results,  # Full results for analysis
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

        logger.info(f"Saved NIAH results to {output_path}")


def evaluate_niah(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    config: Optional[NIAHConfig] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Convenience function to run NIAH evaluation.

    Args:
        model: Language model to evaluate
        tokenizer: Tokenizer for the model
        config: Evaluation configuration (default: NIAHConfig())
        output_path: Optional path to save results JSON

    Returns:
        Evaluation results dictionary

    Example:
        >>> from distillation.evaluation.niah import evaluate_niah, NIAHConfig
        >>> config = NIAHConfig(context_length=4096, num_samples=50)
        >>> results = evaluate_niah(model, tokenizer, config, Path("niah_results.json"))
        >>> print(f"Accuracy: {results['accuracy']:.2%}")
    """
    evaluator = NIAHEvaluator(model, tokenizer)
    results = evaluator.evaluate(config)

    if output_path:
        evaluator.save_results(results, output_path)

    return results
