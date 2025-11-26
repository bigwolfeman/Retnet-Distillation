"""RetNet backbone wrapper for RetNet-HRM.

Integrates Microsoft TorchScale RetNetDecoder with custom embeddings and output head.
Supports both parallel training and O(1) recurrent inference modes.

Expert implementation from docs/snippets with TorchScale integration.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from argparse import Namespace
from collections import OrderedDict

# TorchScale import - requires the patched local fork (see ../torchscale)
try:
    from torchscale.architecture.retnet import RetNetDecoder
except ModuleNotFoundError as exc:
    missing = exc.name
    if missing == "torchscale":
        raise ImportError(
            "TorchScale not installed. Install the patched local copy with:\n"
            "pip install -e ../torchscale --no-build-isolation"
        ) from exc
    if missing in {"fairscale", "timm", "einops"}:
        raise ImportError(
            f"TorchScale dependency '{missing}' is missing. "
            "Reinstall TorchScale with dependencies:\n"
            "pip install -e ../torchscale --no-build-isolation"
        ) from exc
    raise


class RetNetBackbone(nn.Module):
    """RetNet backbone with embeddings and output head.

    Features:
    - Parallel training mode (efficient batch processing)
    - Recurrent inference mode (O(1) memory per layer)
    - Chunk-recurrent mode (for 64k-128k sequences)

    Implements FR-005: O(1) memory overhead per layer during inference.
    """

    def __init__(
        self,
        vocab_size: int = 100352,
        d_model: int = 2816,
        n_layers: int = 28,
        n_heads: int = 12,
        dropout: float = 0.0,
        max_seq_len: int = 65536,
        debug: bool = False,
        checkpoint_activations: bool = True,  # FIX #4: Make gradient checkpointing configurable
    ):
        """Initialize RetNet backbone.

        Args:
            vocab_size: Vocabulary size
            d_model: Hidden dimension
            n_layers: Number of RetNet layers
            n_heads: Number of retention heads
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            debug: Enable debug output (segment isolation messages, etc.)
            checkpoint_activations: Enable gradient checkpointing (default: True for memory savings)
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.debug = debug

        # Embeddings (separate from decoder)
        self.embed = nn.Embedding(vocab_size, d_model)

        # Create args namespace for TorchScale RetNetDecoder
        # TorchScale requires many configuration attributes - using comprehensive defaults
        retnet_args = Namespace(
            # Core architecture
            decoder_embed_dim=d_model,
            decoder_retention_heads=n_heads,
            decoder_value_embed_dim=d_model,
            decoder_ffn_embed_dim=d_model * 4,
            decoder_layers=n_layers,
            # Dropout
            dropout=dropout,
            activation_dropout=dropout,
            attention_dropout=dropout,
            drop_path_rate=0.0,
            # Normalization
            decoder_normalize_before=True,
            no_scale_embedding=False,
            layernorm_embedding=False,
            layernorm_eps=1e-5,
            # Activation
            activation_fn='gelu',
            # RetNet specific
            recurrent_chunk_size=512,
            chunkwise_recurrent=True,  # Re-enabled: Will be overridden per forward call
            retention_drop=dropout,
            # Output layer
            no_output_layer=True,
            share_decoder_input_output_embed=False,
            # Architecture variants
            deepnorm=False,
            subln=False,
            multiway=False,
            # MoE (disabled)
            moe_freq=0,
            moe_top1_expert=False,
            moe_expert_count=0,
            moe_gating_use_fp32=False,
            moe_second_expert_policy='all',
            moe_normalize_gate_prob_before_dropping=False,
            moe_eval_capacity_token_fraction=0.25,
            use_xmoe=False,
            # Additional common TorchScale args
            # FIX #4: Use configurable gradient checkpointing
            # Trades ~30% speed for 50% memory savings (critical for staying under 32GB)
            checkpoint_activations=checkpoint_activations,
            fsdp=False,
            ddp_rank=0,
            xpos_rel_pos=False,
            xpos_scale_base=512,
        )

        # Create identity output projection (TorchScale requires it even with no_output_layer=True)
        # We'll handle the actual output projection in RetNetHRMModel
        self.output_proj_identity = nn.Identity()

        # RetNet decoder from TorchScale
        # Pass our embedding layer so decoder can handle embedding
        self.decoder = RetNetDecoder(
            args=retnet_args,
            embed_tokens=self.embed,
            output_projection=self.output_proj_identity,
        )

        # Output normalization
        self.norm = nn.LayerNorm(d_model)

        # Mask cache: LRU cache with size limit to prevent unbounded growth
        # Causal masks are sequence-length dependent but batch-invariant
        # Caching avoids torch.tril() overhead (2-5ms per forward pass)
        # Using OrderedDict for LRU behavior (oldest entries evicted first)
        self._causal_mask_cache = OrderedDict()
        self._max_cache_size = 50  # Max 50 sequence lengths (~200MB at seq_len=2048)

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass (defaults to training mode).

        Args:
            input_ids: Token IDs, shape (batch_size, seq_len)
            segment_ids: Optional segment IDs for packed sequences

        Returns:
            Hidden states, shape (batch_size, seq_len, d_model)
        """
        return self.forward_train(input_ids, segment_ids)

    def forward_train(
        self,
        input_ids: torch.Tensor,
        segment_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass in training mode (parallel).

        Uses TorchScale parallel kernels for efficient batch processing.

        Args:
            input_ids: Token IDs, shape (batch_size, seq_len)
            segment_ids: Segment IDs for packed sequences, shape (batch_size, seq_len)
                        Used to prevent cross-example attention leakage.
                        Each example in packed sequence gets unique segment ID.

        Returns:
            Hidden states, shape (batch_size, seq_len, d_model)
        """
        # Build segment-aware attention mask if segment_ids provided
        # This prevents later examples from attending to earlier examples in packed sequences
        attn_mask = None
        if segment_ids is not None:
            attn_mask = self._build_segment_mask(segment_ids)
            # DEBUG: Verify mask is created (only if debug enabled)
            if self.debug:
                print(f"[SEGMENT ISOLATION] Mask created: shape={attn_mask.shape}, unique_segments={segment_ids.unique().tolist()}")

        # RetNet decoder (handles embedding internally)
        # Returns (output, extra_info) tuple
        # ✅ FIXED: Using custom TorchScale fork with attention_mask support
        # The fork supports 2D/3D/4D attention masks for segment isolation in:
        # - parallel_forward: Full attention matrix masking ✓
        # - recurrent_forward: Per-step state gating ✓
        #
        # WORKAROUND: chunk_recurrent_forward has issues with arbitrary sequence lengths
        # that don't align with chunk_size. Disable chunking when masks are used.
        if attn_mask is not None:
            orig_chunkwise = self.decoder.chunkwise_recurrent
            self.decoder.chunkwise_recurrent = False
            y, _ = self.decoder(input_ids, attention_mask=attn_mask)
            self.decoder.chunkwise_recurrent = orig_chunkwise
        else:
            y, _ = self.decoder(input_ids, attention_mask=attn_mask)

        # Output normalization
        return self.norm(y)

    def _build_segment_mask(self, segment_ids: torch.Tensor) -> torch.Tensor:
        """Build block-diagonal causal attention mask from segment IDs.

        Caching strategy:
        - Causal mask: Cached by sequence length (batch-invariant)
        - Segment mask: Built fresh each time (batch-specific)
        - Final mask: Combines cached causal + fresh segment + padding

        Args:
            segment_ids: Segment IDs, shape (batch_size, seq_len)
                        Values indicate which example each token belongs to.
                        Padding positions have segment_id = -1.

        Returns:
            Attention mask, shape (batch_size, seq_len, seq_len)
            False = do not attend, True = can attend

        Example:
            segment_ids = [[0, 0, 0, 1, 1, 1, -1, -1]]  # 2 examples + padding
            mask[0, 3, 0] = False  # Example 1 cannot see Example 0
            mask[0, 1, 0] = True   # Within same example, can attend to past
        """
        batch_size, seq_len = segment_ids.shape
        device = segment_ids.device

        # Get or create causal mask from LRU cache
        # Causal mask is sequence-length dependent but batch-invariant
        cache_key = seq_len
        if cache_key in self._causal_mask_cache:
            # Cache hit: Move to end (mark as most recently used)
            self._causal_mask_cache.move_to_end(cache_key)
            # Clone cached mask to avoid CUDA graphs "tensor overwritten" error with torch.compile
            causal_mask = self._causal_mask_cache[cache_key].clone()
            # Move to correct device if needed (handles CPU/GPU transfers)
            if causal_mask.device != device:
                causal_mask = causal_mask.to(device)
                self._causal_mask_cache[cache_key] = causal_mask
        else:
            # Cache miss: Build new mask
            # Shape: (seq_len, seq_len)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))

            # Add to cache with LRU eviction
            self._causal_mask_cache[cache_key] = causal_mask

            # Evict oldest entry if cache exceeds max size
            if len(self._causal_mask_cache) > self._max_cache_size:
                # Pop first (oldest) item from OrderedDict
                evicted_key = next(iter(self._causal_mask_cache))
                del self._causal_mask_cache[evicted_key]

        # Create same-segment mask: True where tokens belong to same segment
        # This is batch-specific and must be built fresh each time
        # Shape: (batch_size, seq_len, seq_len)
        same_segment = segment_ids.unsqueeze(-1) == segment_ids.unsqueeze(-2)

        # Combine: can attend only if same segment AND causal
        # This creates block-diagonal causal mask
        attn_mask = same_segment & causal_mask.unsqueeze(0)

        # Mask out padding positions (segment_id == -1)
        padding_mask = segment_ids != -1  # (batch_size, seq_len)
        attn_mask = attn_mask & padding_mask.unsqueeze(-1) & padding_mask.unsqueeze(-2)

        return attn_mask

    @torch.no_grad()
    def forward_recurrent(
        self,
        input_ids: torch.Tensor,
        state: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass in recurrent mode (O(1) inference).

        Implements FR-005: Constant memory overhead per layer.

        Args:
            input_ids: Token IDs, shape (batch_size, chunk_size)
                      chunk_size typically 1 (next token) or small window
            state: List of per-layer retention states (None to initialize)

        Returns:
            Tuple of:
                - Hidden states, shape (batch_size, chunk_size, d_model)
                - Updated state (list of per-layer states)
        """
        # RetNet decoder in recurrent mode (handles embedding internally)
        # TorchScale supports state carry for O(1) memory
        y, new_state = self.decoder(input_ids, incremental_state=state)

        # Output normalization
        return self.norm(y), new_state

    def init_state(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        """Initialize recurrent state for inference.

        Args:
            batch_size: Batch size
            device: Device for states

        Returns:
            List of per-layer retention states (zeros)
        """
        # Initialize state as list of zeros (one per layer)
        # State shape depends on TorchScale implementation
        # Typically: (batch_size, n_heads, d_head) per layer
        state = []
        for _ in range(self.n_layers):
            # Placeholder zero state (TorchScale handles actual shape)
            layer_state = torch.zeros(
                batch_size, self.decoder.num_heads, self.d_model // self.decoder.num_heads,
                device=device
            )
            state.append(layer_state)
        return state


class RetNetEmbeddings(nn.Module):
    """Enhanced embeddings for RetNet with pointer slots for landmarks.

    Includes:
    - Token embeddings
    - Type embeddings (optional, for future multi-modal)
    - Pointer slot reservations for retrieval landmarks
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_pointer_slots: int = 256,
        dropout: float = 0.0,
    ):
        """Initialize embeddings.

        Args:
            vocab_size: Vocabulary size
            d_model: Embedding dimension
            num_pointer_slots: Number of special tokens for landmarks
            dropout: Dropout rate
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        # Token embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model)

        # Type embeddings (reserved for future use)
        self.type_embed = nn.Embedding(2, d_model)  # Default 2 types

        self.dropout = nn.Dropout(dropout)

        # Pointer slots are part of vocabulary (special tokens)
        # Already handled by tokenizer landmark tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute embeddings.

        Args:
            input_ids: Token IDs, shape (batch_size, seq_len)
            type_ids: Optional type IDs, shape (batch_size, seq_len)

        Returns:
            Embeddings, shape (batch_size, seq_len, d_model)
        """
        # Token embeddings
        embeddings = self.token_embed(input_ids)

        # Add type embeddings if provided
        if type_ids is not None:
            embeddings = embeddings + self.type_embed(type_ids)

        # Note: RetNet doesn't use positional encoding
        # Position information is handled by the retention mechanism

        return self.dropout(embeddings)


class RetNetOutputHead(nn.Module):
    """Output head for language modeling.

    Projects hidden states to vocabulary logits.
    Optionally supports kNN-LM logit mixing (disabled by default).
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        tie_weights: bool = True,
        embedding_layer: Optional[nn.Embedding] = None,
    ):
        """Initialize output head.

        Args:
            d_model: Hidden dimension
            vocab_size: Vocabulary size
            tie_weights: Tie with input embeddings (saves memory)
            embedding_layer: Embedding layer to tie with (if tie_weights=True)
        """
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        if tie_weights and embedding_layer is not None:
            # Tie weights with input embeddings (memory efficient)
            self.proj = embedding_layer
            self.tied = True
        else:
            # Separate output projection
            self.proj = nn.Linear(d_model, vocab_size, bias=False)
            self.tied = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        knn_logits: Optional[torch.Tensor] = None,
        knn_lambda: float = 0.0,
    ) -> torch.Tensor:
        """Compute output logits.

        Args:
            hidden_states: Hidden states, shape (batch_size, seq_len, d_model)
            knn_logits: Optional kNN-LM logits (not used in MVP)
            knn_lambda: Mixing weight for kNN logits

        Returns:
            Logits, shape (batch_size, seq_len, vocab_size)
        """
        # Project to vocabulary
        if self.tied:
            # Use tied embedding weights as projection
            logits = torch.nn.functional.linear(hidden_states, self.proj.weight)
        else:
            logits = self.proj(hidden_states)

        # Optional kNN-LM mixing (disabled in MVP per research.md)
        if knn_logits is not None and knn_lambda > 0:
            logits = (1 - knn_lambda) * logits + knn_lambda * knn_logits

        return logits
