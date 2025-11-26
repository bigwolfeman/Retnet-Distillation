"""
Direct teacher client that loads model locally and runs inference in-process.

This client provides the fastest execution path for users with sufficient VRAM:
- Loads teacher model directly into memory (BF16 for efficiency)
- Runs inference locally (no network overhead)
- Inference-only mode (no gradients, no backward pass)
- Optional logit caching to disk
- Drop-in compatible with VLLMTeacherClient API

Use cases:
- Big GPU (32GB+): Fastest training with both teacher + student in VRAM
- Cache generation: Load teacher once, cache all logits, then use CachedTeacherClient
- Experimentation: No need to run separate vLLM server

VRAM requirements (approximate):
- 1B model: ~4GB (BF16)
- 3B model: ~12GB (BF16)
- 7B model: ~28GB (BF16)

Example:
    # Direct mode (no caching)
    teacher = DirectTeacherClient("meta-llama/Llama-3.2-1B")
    results = teacher.get_top_k_logits(input_ids=[[1, 2, 3, ...]])

    # Direct + caching
    teacher = DirectTeacherClient(
        "meta-llama/Llama-3.2-1B",
        cache_dir="data/cache/"
    )
    results = teacher.get_top_k_logits(input_ids=[[1, 2, 3, ...]])
    # Logits are returned AND written to cache

Performance:
- Inference: ~10-50ms per sequence (depends on length and model size)
- Memory: Model size + KV cache + activations (~5GB total for 1B model)
- No network overhead
- No serialization overhead
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from .dataset import load_llama_tokenizer

logger = logging.getLogger(__name__)


class DirectTeacherError(Exception):
    """Base exception for DirectTeacherClient errors."""
    pass


class DirectTeacherClient:
    """
    Teacher client that loads model locally and runs inference in-process.

    Features:
    - Loads teacher model in BF16 for memory efficiency
    - Inference-only mode (no gradients)
    - Optional logit caching to disk
    - Top-k extraction and int8 quantization
    - Compatible with VLLMTeacherClient API

    Attributes:
        model_name: HuggingFace model name/path
        device: Device to load model on
        torch_dtype: Model dtype (default: torch.bfloat16)
        topk: Default top-k value
        model: Loaded model (AutoModelForCausalLM)
        tokenizer: Loaded tokenizer (for debugging/validation)
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float32,
        cache_dir: Optional[str] = None,
        topk: int = 512,  # Increased from 128 for better probability coverage
        use_flash_attention: bool = False,  # Disabled by default (requires flash-attn package)
        trust_remote_code: bool = True,
        hf_token: Optional[str] = None,
        adapter_path: Optional[str] = None,  # Optional PEFT adapter path
        tokenizer: Optional[Any] = None,  # Optional pre-loaded tokenizer
    ):
        """
        Initialize DirectTeacherClient.

        Args:
            model_name: HuggingFace model name or path
            device: Device to load model on (default: "cuda")
            torch_dtype: Model dtype (default: torch.bfloat16)
            cache_dir: If set, cache logits to this directory
            topk: Default top-k value (default: 128)
            use_flash_attention: Use Flash Attention 2 if available
            trust_remote_code: Trust remote code for custom models
            hf_token: HuggingFace token for gated models
            adapter_path: Optional path to PEFT adapter (e.g., LoRA weights)
            tokenizer: Optional pre-loaded tokenizer (to ensure consistency)

        Raises:
            DirectTeacherError: If model loading fails
        """
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.topk = topk
        self.adapter_path = adapter_path

        # Track close state
        self._closed = False

        # Buffer reuse for top-k computation (reduces allocator churn)
        # These buffers are lazily allocated on first use and reused across batches
        # Will be reallocated if batch shape changes (different batch_size or seq_len)
        self._topk_probs_buffer = None
        self._topk_mass_buffer = None
        self._other_mass_buffer = None
        self._buffer_shape = None  # Track (batch_size, seq_len, topk) for reallocation detection

        logger.info(f"Initializing DirectTeacherClient: {model_name}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Dtype: {torch_dtype}")
        logger.info(f"  Adapter path: {adapter_path or 'None'}")
        logger.info(f"  Cache dir: {cache_dir or 'None (no caching)'}")

        # Load tokenizer (for validation and debugging)
        if tokenizer is not None:
            logger.info("Using provided tokenizer instance")
            self.tokenizer = tokenizer
        else:
            logger.info("Loading tokenizer from scratch...")
            try:
                self.tokenizer = load_llama_tokenizer(
                    model_name=model_name,
                    adapter_path=adapter_path,
                    hf_token=hf_token,
                    trust_remote_code=trust_remote_code,
                )
                logger.info(f"Tokenizer loaded: vocab_size={len(self.tokenizer)}")
            except Exception as e:
                raise DirectTeacherError(f"Failed to load tokenizer: {e}") from e

        # Load model
        logger.info(f"Loading model {model_name}...")
        try:
            model_kwargs = {
                "torch_dtype": torch_dtype,
                "device_map": device,
                "trust_remote_code": trust_remote_code,
            }

            # Add Flash Attention if requested and available
            if use_flash_attention:
                try:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    logger.info("Attempting to use Flash Attention 2 for KV cache optimization...")

                    # Add HF token if provided
                    if hf_token:
                        model_kwargs["token"] = hf_token

                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        **model_kwargs
                    )
                    logger.info("✅ Teacher model loaded with Flash Attention 2 (saves ~2GB KV cache memory)")

                except Exception as flash_error:
                    logger.warning(f"⚠️  Flash Attention 2 not available, falling back to standard attention: {flash_error}")
                    logger.warning("   To enable Flash Attention 2, install: pip install flash-attn --no-build-isolation")

                    # Retry without flash attention
                    model_kwargs.pop("attn_implementation", None)
                    if hf_token:
                        model_kwargs["token"] = hf_token

                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        **model_kwargs
                    )
                    logger.info("Model loaded with standard attention")
            else:
                # Add HF token if provided
                if hf_token:
                    model_kwargs["token"] = hf_token

                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **model_kwargs
                )
                logger.info("Model loaded with standard attention (Flash Attention 2 not requested)")

            # If adapter tokenizer added PAD (or otherwise changed vocab), align embedding size
            base_vocab = self.model.get_input_embeddings().weight.size(0)
            target_vocab = max(
                base_vocab,
                len(self.tokenizer),
                (self.tokenizer.pad_token_id + 1) if self.tokenizer.pad_token_id is not None else base_vocab,
            )
            if target_vocab != base_vocab:
                logger.info(f"Resizing embeddings {base_vocab}->{target_vocab} to match tokenizer")
                self.model.resize_token_embeddings(target_vocab)

            # Load PEFT adapter if provided
            if adapter_path:
                logger.info(f"Loading PEFT adapter from: {adapter_path}")
                try:
                    from peft import PeftModel

                    # Verify adapter path exists
                    adapter_path_obj = Path(adapter_path)
                    if not adapter_path_obj.exists():
                        raise DirectTeacherError(f"Adapter path does not exist: {adapter_path}")

                    # Load adapter weights onto base model
                    self.model = PeftModel.from_pretrained(
                        self.model,
                        adapter_path,
                        torch_dtype=torch_dtype,
                    )

                    # Merge adapter weights into base model for inference efficiency
                    logger.info("Merging adapter weights into base model...")
                    self.model = self.model.merge_and_unload()

                    logger.info("PEFT adapter loaded and merged successfully")

                except ImportError as e:
                    raise DirectTeacherError(
                        "PEFT library not found. Install with: pip install peft"
                    ) from e
                except Exception as e:
                    raise DirectTeacherError(f"Failed to load PEFT adapter: {e}") from e

            # Set to evaluation mode (disables dropout, etc.)
            self.model.eval()

            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Model loaded: {total_params:,} parameters")

            # Estimate VRAM usage
            param_bytes = total_params * torch_dtype.itemsize if torch_dtype == torch.float16 or torch_dtype == torch.bfloat16 else total_params * 4
            vram_gb = param_bytes / (1024 ** 3)
            logger.info(f"Estimated VRAM usage: {vram_gb:.2f} GB (params only)")

        except DirectTeacherError:
            # Re-raise DirectTeacherError without wrapping
            raise
        except Exception as e:
            raise DirectTeacherError(f"Failed to load model: {e}") from e

        # Initialize cache writer if caching enabled
        self.cache_writer = None
        if self.cache_dir:
            # Import CacheWriter from caching_wrapper module
            from src.distillation.caching_wrapper import CachingTeacherWrapper

            # Note: CachingTeacherWrapper expects to wrap a client, but we'll
            # use its internal buffer mechanism directly via a helper class
            # For now, create a simple buffer that we'll flush manually
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Create a simple cache writer helper
            self.cache_writer = _SimpleCacheWriter(
                output_dir=self.cache_dir,
                shard_size=1000,
            )
            logger.info(f"Cache writer initialized: {self.cache_dir}")

        logger.info("DirectTeacherClient initialized successfully")

    def get_top_k_logits(
        self,
        input_ids: List[List[int]],
        topk: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get top-k logits for input sequences.

        This method is compatible with VLLMTeacherClient.get_prompt_logprobs().

        Args:
            input_ids: List of token ID sequences. Shape: (batch_size, seq_len)
            topk: Number of top logits to return (default: self.topk)

        Returns:
            List of dicts (one per sequence) containing:
                - indices: List[List[int]] - top-k token IDs per position
                - logprobs: List[List[float]] - top-k log probabilities per position
                - tokens: List[str] - token strings (empty for compatibility)
                - top_logprobs: List[Dict] - sparse logprobs dict (empty for compatibility)

        Raises:
            DirectTeacherError: If inference fails
        """
        if topk is None:
            topk = self.topk

        if topk > self.tokenizer.vocab_size:
            logger.warning(f"topk={topk} > vocab_size={self.tokenizer.vocab_size}, capping")
            topk = self.tokenizer.vocab_size

        # Convert to tensor
        try:
            input_ids_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        except Exception as e:
            raise DirectTeacherError(f"Failed to convert input_ids to tensor: {e}") from e

        # Run inference (no gradients)
        try:
            with torch.no_grad():
                outputs = self.model(input_ids_tensor)
                logits = outputs.logits  # [batch, seq_len, vocab_size]

        except Exception as e:
            raise DirectTeacherError(f"Model inference failed: {e}") from e

        # Extract top-k for each position
        results = []
        batch_size, seq_len, vocab_size = logits.shape

        for batch_idx in range(batch_size):
            # Get logits for this sequence: [seq_len, vocab_size]
            seq_logits = logits[batch_idx]

            # Apply log_softmax to get log probabilities
            seq_logprobs = torch.log_softmax(seq_logits, dim=-1)  # [seq_len, vocab_size]

            # Get top-k indices and values for each position
            topk_logprobs, topk_indices = torch.topk(seq_logprobs, k=topk, dim=-1)

            # Convert to lists
            indices = topk_indices.cpu().tolist()  # List[List[int]], shape [seq_len, topk]
            logprobs = topk_logprobs.cpu().tolist()  # List[List[float]], shape [seq_len, topk]

            # MEMORY FIX: Free teacher logits immediately after top-k extraction
            # seq_logprobs is ~512MB full vocab tensor, no longer needed after top-k
            # This saves ~512MB per batch
            del seq_logprobs, topk_logprobs, topk_indices
            # REMOVED: caused 64x/step memory fragmentation
            # torch.cuda.empty_cache()

            result = {
                "indices": indices,
                "logprobs": logprobs,
                "tokens": [],  # Not needed for training, kept for compatibility
                "top_logprobs": [],  # Not needed for training, kept for compatibility
            }

            results.append(result)

        # MEMORY FIX: Free main logits tensor after processing all sequences
        # logits tensor (batch_size × seq_len × vocab_size) can be very large
        del logits, outputs
        # REMOVED: caused 64x/step memory fragmentation
        # torch.cuda.empty_cache()

        # Optional: Write to cache
        if self.cache_writer:
            self._write_to_cache(input_ids, results)

        return results

    def get_prompt_logprobs(
        self,
        input_ids: List[List[int]],
        topk: int = 512,  # Increased from 128 for better probability coverage
        temperature: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Get top-k logprobs for input sequences.

        This is an alias for get_top_k_logits() to match VLLMTeacherClient API.

        Args:
            input_ids: List of token ID sequences
            topk: Number of top logits to return
            temperature: Temperature for softmax (ignored, always 1.0)

        Returns:
            List of dicts with logprobs data per sequence
        """
        if temperature != 1.0:
            logger.warning(
                f"Temperature={temperature} is not supported by DirectTeacherClient. "
                "Using temperature=1.0. Apply temperature in loss function instead."
            )

        return self.get_top_k_logits(input_ids=input_ids, topk=topk)

    def get_top_k_logits_tensors(
        self,
        input_ids: torch.Tensor,
        topk: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        GPU-native tensor API: Get top-k logits as tensors (NO CPU SYNCHRONIZATION).

        This method keeps all data on GPU, eliminating the CPU/GPU round-trips
        that cause performance issues in the list-based API.

        Args:
            input_ids: Token ID tensor already on GPU. Shape: [batch_size, seq_len]
            topk: Number of top logits to return (default: self.topk)

        Returns:
            tuple of 3 tensors (all on GPU, no .cpu() calls):
                - topk_indices: Top-k token IDs per position. Shape: [batch, seq_len, topk]
                - topk_logprobs: Top-k log probabilities. Shape: [batch, seq_len, topk]
                - other_logprob: Log probability of mass NOT in top-k. Shape: [batch, seq_len, 1]
                  NOTE: All outputs are log-probabilities (single source of truth)

        Raises:
            DirectTeacherError: If inference fails
        """
        if topk is None:
            topk = self.topk

        if topk > self.tokenizer.vocab_size:
            logger.warning(f"topk={topk} > vocab_size={self.tokenizer.vocab_size}, capping")
            topk = self.tokenizer.vocab_size

        # Ensure input is on correct device
        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device, non_blocking=True)

        # Run inference (no gradients)
        try:
            with torch.no_grad():
                outputs = self.model(input_ids)
                # Keep teacher forward + top-k math in fp32 to avoid BF16 rounding
                # that can push the top-k mass slightly above 1 and make other_mass negative.
                logits = outputs.logits.float()  # [batch, seq_len, vocab_size]
        except Exception as e:
            raise DirectTeacherError(f"Model inference failed: {e}") from e

        # FIX #3: Optimized log_softmax with explicit memory management
        # PyTorch doesn't have a truly in-place log_softmax, but we use explicit del
        # to help the allocator free and reuse memory more efficiently
        #
        # Memory impact: For batch_size=8, seq_len=2048, vocab_size=32000:
        #   Temporary allocation: 8 * 2048 * 32000 * 4 bytes = 2.1GB
        #   But freed immediately with del, reducing peak memory pressure

        # FIX #3: Memory profiling
        mem_before = torch.cuda.max_memory_allocated()

        # Compute log_softmax and immediately free original tensor
        # Note: PyTorch doesn't have in-place log_softmax, so we explicitly del
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        del logits  # Explicit free helps allocator reuse memory faster
        logits = log_probs  # Reuse variable name (now contains log probabilities)

        # FIX #3: Log memory impact (only once)
        if not hasattr(self, '_logsoft_logged'):
            mem_after = torch.cuda.max_memory_allocated()
            mem_delta = (mem_after - mem_before) / 1e9
            tensor_size = (logits.numel() * 4) / 1e9
            logger.info(
                f"FIX #3: Log_softmax + explicit del: peak memory delta {mem_delta:.3f} GB "
                f"(tensor size: {tensor_size:.2f} GB, freed immediately)"
            )
            self._logsoft_logged = True

        # Get top-k indices and values (stays on GPU)
        topk_logprobs, topk_indices = torch.topk(logits, k=topk, dim=-1)
        # topk_indices: [batch, seq_len, topk]
        # topk_logprobs: [batch, seq_len, topk]

        # Compute "other mass" (probability mass NOT in top-k)
        # This is needed for the distillation loss
        # OPTIMIZATION: Reuse buffers to reduce allocator churn
        batch_size, seq_len, _ = topk_logprobs.shape
        current_shape = (batch_size, seq_len, topk)

        # Check if we need to reallocate buffers (shape changed)
        if self._buffer_shape != current_shape:
            # Allocate new buffers for this shape
            self._topk_probs_buffer = torch.empty(
                (batch_size, seq_len, topk),
                dtype=topk_logprobs.dtype,
                device=self.device
            )
            self._topk_mass_buffer = torch.empty(
                (batch_size, seq_len, 1),
                dtype=topk_logprobs.dtype,
                device=self.device
            )
            self._other_mass_buffer = torch.empty(
                (batch_size, seq_len, 1),
                dtype=topk_logprobs.dtype,
                device=self.device
            )
            self._buffer_shape = current_shape
            logger.debug(f"Allocated top-k buffers for shape {current_shape}")

        # Reuse buffers for computation (in-place operations)
        torch.exp(topk_logprobs, out=self._topk_probs_buffer)  # [batch, seq_len, topk]
        torch.sum(self._topk_probs_buffer, dim=-1, keepdim=True, out=self._topk_mass_buffer)  # [batch, seq_len, 1]
        torch.sub(1.0, self._topk_mass_buffer, out=self._other_mass_buffer)  # [batch, seq_len, 1]

        # Numerically safe conversion to log-probability
        # Clamp to avoid log(0) or log of negative mass when top-k mass slightly exceeds 1.0
        torch.clamp_(self._other_mass_buffer, min=1e-12, max=1.0)
        other_logprob = torch.log(self._other_mass_buffer)  # [batch, seq_len, 1]

        # Make a copy to return (since we'll reuse the buffer)
        other_logprob = other_logprob.clone()

        # Free large intermediate tensors immediately
        del logits, outputs, log_probs

        # Return tensors on GPU - NO .cpu() calls!
        # All outputs are log-probabilities (single source of truth)
        return topk_indices, topk_logprobs, other_logprob

    def _write_to_cache(
        self,
        input_ids: List[List[int]],
        results: List[Dict[str, Any]],
    ):
        """
        Write results to cache.

        Uses the same format as cache_teacher_logits.py for compatibility.

        Args:
            input_ids: Original input IDs
            results: Results from get_top_k_logits()
        """
        try:
            # Convert to cache format (int8 quantization)
            for seq_input_ids, result in zip(input_ids, results):
                # Generate sequence ID (hash of input_ids)
                import hashlib
                id_bytes = np.array(seq_input_ids, dtype=np.int32).tobytes()
                seq_id = f"seq_{hashlib.sha256(id_bytes).hexdigest()[:16]}"

                # Convert to int8 cache format
                cached_seq = self._convert_to_cache_format(seq_id, seq_input_ids, result)

                # Write to cache
                self.cache_writer.write_sequence(cached_seq)

        except Exception as e:
            logger.error(f"Failed to write to cache: {e}")
            # Don't raise - caching is optional

    def _convert_to_cache_format(
        self,
        seq_id: str,
        input_ids: List[int],
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Convert result to cache format with int8 quantization.

        Matches the format used by cache_teacher_logits.py.

        Args:
            seq_id: Sequence ID
            input_ids: Input token IDs
            result: Result from get_top_k_logits()

        Returns:
            Dict in cache format
        """
        indices = result["indices"]
        logprobs = result["logprobs"]

        teacher_indices = []
        teacher_values = []
        teacher_scales = []
        teacher_other = []

        for position_indices, position_logprobs in zip(indices, logprobs):
            if not position_indices:
                # Empty position
                teacher_indices.append([])
                teacher_values.append([])
                teacher_scales.append(0.0)
                teacher_other.append(0.0)
                continue

            # Convert to numpy
            position_indices = np.array(position_indices, dtype=np.int32)
            position_logprobs = np.array(position_logprobs, dtype=np.float32)

            # Convert to probabilities
            position_probs = np.exp(position_logprobs)
            total_prob = position_probs.sum()

            # Other mass (probability not in top-k)
            # FIX: Convert to logit (log-probability) not raw probability
            # This ensures sparse KL loss receives correct teacher distribution
            other_prob = max(1e-8, 1.0 - total_prob)  # Clamp to avoid log(0)
            other_mass = np.log(other_prob)  # Convert to logit

            # Normalize to sum to 1.0
            if total_prob > 0:
                position_probs = position_probs / total_prob * (1.0 - other_prob)

            # Quantize to int8 (0-255)
            max_prob = position_probs.max()
            if max_prob > 0:
                scale_factor = max_prob / 255.0
                quantized_values = np.round(position_probs / scale_factor).astype(np.uint8)
            else:
                scale_factor = 1.0
                quantized_values = np.zeros(len(position_probs), dtype=np.uint8)

            teacher_indices.append(position_indices.tolist())
            teacher_values.append(quantized_values.tolist())
            teacher_scales.append(float(scale_factor))
            teacher_other.append(float(other_mass))

        return {
            "sequence_id": seq_id,
            "input_ids": input_ids,
            "teacher_indices": teacher_indices,
            "teacher_values": teacher_values,
            "teacher_scales": teacher_scales,
            "teacher_other": teacher_other,
        }

    def clear_buffers(self):
        """
        Clear internal buffers used for top-k computation.

        This frees VRAM used by reusable buffers. Buffers will be reallocated
        on next call to get_top_k_logits_tensors().

        Useful for:
        - Freeing memory between training runs
        - Switching to different batch sizes
        - Memory-constrained environments
        """
        if self._topk_probs_buffer is not None:
            del self._topk_probs_buffer
            self._topk_probs_buffer = None

        if self._topk_mass_buffer is not None:
            del self._topk_mass_buffer
            self._topk_mass_buffer = None

        if self._other_mass_buffer is not None:
            del self._other_mass_buffer
            self._other_mass_buffer = None

        self._buffer_shape = None
        logger.debug("Cleared top-k computation buffers")

    def health_check(self) -> bool:
        """
        Check if model is loaded and ready.

        Returns:
            True if model is ready, False otherwise
        """
        try:
            # Try a simple forward pass with dummy input
            dummy_input = torch.tensor([[1, 2, 3]], dtype=torch.long, device=self.device)
            with torch.no_grad():
                _ = self.model(dummy_input)
            return True
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    def close(self):
        """
        Close the client and free resources.

        This unloads the model from VRAM and closes the cache writer.
        """
        if self._closed:
            return

        logger.info("Closing DirectTeacherClient...")

        try:
            # Clear computation buffers
            self.clear_buffers()

            # Close cache writer
            if self.cache_writer:
                self.cache_writer.close()
                logger.info("Cache writer closed")

            # Delete model to free VRAM
            if hasattr(self, 'model'):
                del self.model
                # Force garbage collection
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                logger.info("Model unloaded from VRAM")

            self._closed = True
            logger.info("DirectTeacherClient closed")

        except Exception as e:
            logger.warning(f"Error closing DirectTeacherClient: {e}")
            self._closed = True

    def __del__(self):
        """Finalizer to ensure resources are freed."""
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        try:
            self.close()
        except Exception as e:
            logger.warning(f"Error in __exit__ while closing: {e}")
        return False


class _SimpleCacheWriter:
    """
    Simple cache writer for DirectTeacherClient.

    Buffers sequences and writes them to parquet shards.
    """

    def __init__(self, output_dir: Path, shard_size: int):
        import pandas as pd
        import json
        import time

        self.output_dir = output_dir
        self.shard_size = shard_size
        self.buffer = []
        self.current_shard = 0
        self.total_sequences = 0
        logger.info(f"SimpleCacheWriter initialized: {output_dir}")

    def write_sequence(self, cached_seq: Dict[str, Any]):
        """Write a sequence to cache buffer."""
        self.buffer.append(cached_seq)

        # Auto-flush if buffer is full
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self):
        """Flush buffer to disk."""
        if not self.buffer:
            return

        import pandas as pd

        shard_path = self.output_dir / f"cache_shard_{self.current_shard:04d}.parquet"
        logger.info(f"Flushing cache shard {self.current_shard} ({len(self.buffer)} sequences)")

        # Convert to DataFrame and save
        df = pd.DataFrame(self.buffer)
        df.to_parquet(
            shard_path,
            engine="pyarrow",
            compression="snappy",
            index=False,
        )

        self.total_sequences += len(self.buffer)
        self.current_shard += 1
        self.buffer = []

        logger.info(f"Cache shard saved: {shard_path}")

    def close(self):
        """Close and flush remaining sequences."""
        import json
        import time

        if self.buffer:
            self.flush()

        # Write manifest
        manifest_path = self.output_dir / "manifest.json"
        shard_files = sorted(self.output_dir.glob("cache_shard_*.parquet"))
        total_size = sum(f.stat().st_size for f in shard_files)

        manifest = {
            "total_sequences": self.total_sequences,
            "num_shards": len(shard_files),
            "shard_files": [f.name for f in shard_files],
            "timestamp": time.time(),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
        }

        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Cache writer closed: {self.total_sequences} sequences cached")
