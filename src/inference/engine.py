"""Inference engine for RetNet-HRM.

Implements streaming generation with:
- Recurrent inference (O(1) memory per layer)
- State management for 64k+ sequences
- Memory monitoring (FR-003)
- Latency tracking (NFR-001)
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Any
import time

from ..models.core import RetNetHRMModel
from .state import ContextState, InferenceStateManager


class InferenceEngine:
    """Inference engine for RetNet-HRM model.

    Features:
    - Recurrent generation with O(1) memory per layer
    - Streaming output support
    - Memory monitoring for 64k+ sequences (FR-002, FR-003)
    - Latency tracking (NFR-001)
    - Temperature and top-p sampling
    """

    def __init__(
        self,
        model: RetNetHRMModel,
        tokenizer,
        device: str = "cuda",
    ):
        """Initialize inference engine.

        Args:
            model: RetNetHRMModel (in eval mode)
            tokenizer: Tokenizer for encoding/decoding
            device: Device for inference
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Move model to device and set eval mode
        self.model.to(device)
        self.model.eval()

        # State manager
        self.state_manager = InferenceStateManager(
            batch_size=1,  # Single sequence generation
            device=device,
            n_layers=model.config.n_layers_retnet,
            d_model=model.config.d_model,
            n_heads=model.config.n_retention_heads,
        )

        # Stats tracking
        self.stats = {
            'total_tokens': 0,
            'avg_ponder_steps': 0.0,
            'latency_ms': 0.0,
            'tokens_per_second': 0.0,
        }

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        stream: bool = False,
    ) -> str:
        """Generate text from prompt.

        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter (optional)
            stream: Stream tokens as they're generated

        Returns:
            Generated text (prompt + completion)
        """
        start_time = time.time()

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        prompt_length = input_ids.size(1)

        # Initialize state
        state = self.state_manager.create_new_state()

        # Process prompt (all tokens at once for efficiency)
        # This builds up the recurrent state
        for i in range(0, prompt_length, 1):  # Process token by token for recurrence
            token_id = input_ids[:, i:i+1]
            outputs = self.model.forward_recurrent(
                input_ids=token_id,
                state=state.get_retnet_state(),
            )
            # Update state
            state.update_retnet_state(outputs.state)
            state.update_position(1)

            # Memory check
            state.validate_memory()

        # Generate new tokens
        generated_tokens = []
        current_token = input_ids[:, -1:]  # Last token of prompt

        for step in range(max_new_tokens):
            # Forward pass (recurrent)
            outputs = self.model.forward_recurrent(
                input_ids=current_token,
                state=state.get_retnet_state(),
            )

            # Update state
            state.update_retnet_state(outputs.state)
            state.update_position(1)

            # Get logits for last token
            logits = outputs.logits[:, -1, :]  # (1, vocab_size)

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature

            # Sample next token
            next_token = self._sample(
                logits=logits,
                top_p=top_p,
                top_k=top_k,
            )

            # Append to generated
            generated_tokens.append(next_token.item())

            # Stream output if requested
            if stream:
                token_text = self.tokenizer.decode([next_token.item()])
                print(token_text, end='', flush=True)

            # Check for EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            # Update current token for next iteration
            current_token = next_token.unsqueeze(0)

            # Memory check
            state.validate_memory()

        # Decode full output
        all_token_ids = input_ids[0].tolist() + generated_tokens
        output_text = self.tokenizer.decode(all_token_ids, skip_special_tokens=True)

        # Compute stats
        elapsed_time = time.time() - start_time
        total_tokens = len(generated_tokens)

        self.stats = {
            'total_tokens': total_tokens,
            'avg_ponder_steps': outputs.num_steps,  # From last step (US3 future)
            'latency_ms': elapsed_time * 1000,
            'tokens_per_second': total_tokens / elapsed_time if elapsed_time > 0 else 0,
            'peak_memory_mb': state.peak_memory_mb,
        }

        return output_text

    def _sample(
        self,
        logits: torch.Tensor,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample next token from logits.

        Implements temperature, top-p (nucleus), and top-k sampling.

        Args:
            logits: Logits tensor (1, vocab_size)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter

        Returns:
            Sampled token ID
        """
        # Top-k sampling
        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        # Top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Keep at least one token
            sorted_indices_to_remove[..., 0] = False

            # Scatter to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        return next_token.squeeze(0)

    def generate_streaming(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ):
        """Generate text with streaming output (yields tokens).

        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Yields:
            Generated tokens (as strings)
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        prompt_length = input_ids.size(1)

        # Initialize state
        state = self.state_manager.create_new_state()

        # Process prompt
        for i in range(prompt_length):
            token_id = input_ids[:, i:i+1]
            outputs = self.model.forward_recurrent(
                input_ids=token_id,
                state=state.get_retnet_state(),
            )
            state.update_retnet_state(outputs.state)
            state.update_position(1)

        # Generate tokens
        current_token = input_ids[:, -1:]

        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.model.forward_recurrent(
                input_ids=current_token,
                state=state.get_retnet_state(),
            )

            # Update state
            state.update_retnet_state(outputs.state)
            state.update_position(1)

            # Sample next token
            logits = outputs.logits[:, -1, :]
            if temperature > 0:
                logits = logits / temperature

            next_token = self._sample(logits, top_p=top_p)

            # Yield token text
            token_text = self.tokenizer.decode([next_token.item()])
            yield token_text

            # Check EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            current_token = next_token.unsqueeze(0)

    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics.

        Returns:
            Stats dict with latency, throughput, memory
        """
        return self.stats.copy()

    def reset(self):
        """Reset inference state for new sequence."""
        self.state_manager.reset()
        self.stats = {
            'total_tokens': 0,
            'avg_ponder_steps': 0.0,
            'latency_ms': 0.0,
            'tokens_per_second': 0.0,
        }
