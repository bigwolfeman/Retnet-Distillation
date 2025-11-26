"""Core model assembly for RetNet-HRM.

Current implementation:
- US1 (MVP): RetNet backbone + output head for training/inference
- US2: Thin attention band for cross-token fusion (IMPLEMENTED)
- US3: HRM controller + ACT halting for adaptive computation (IMPLEMENTED)

Future phases:
- US4: Retrieval + routing

Implements contracts from model-forward.md.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass

from .retnet import RetNetBackbone, RetNetOutputHead
from .attention import ThinAttentionBand
from .hrm import HRMController, HRMSummarizer, ACTHaltingHead
from .retrieval import (
    LandmarkCompressor,
    RetrievalRegistry,
)
from .routing import GumbelTopKRouter
from ..config import ModelConfig
from ..config.computation_budget import ComputationBudget
from ..retrieval_index import DualEncoder, RetrievalIndex
import os
import sys
from pathlib import Path

# Note: Structure FSM was removed (training-only utility)
# If you need structure validation during training, implement separately


@dataclass
class ModelOutput:
    """Output from forward pass (training mode).

    Fields align with model-forward.md contract.
    """
    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None
    ponder_cost: Optional[torch.Tensor] = None
    avg_steps: Optional[float] = None
    router_stats: Optional[Dict[str, Any]] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    # Loss components for debugging/logging
    ce_loss: Optional[torch.Tensor] = None
    fsm_loss: Optional[torch.Tensor] = None


@dataclass
class InferenceOutput:
    """Output from recurrent inference.

    Fields align with model-forward.md contract.
    """
    logits: torch.Tensor
    state: Any  # ContextState (to be defined in T022)
    halted: bool = False
    num_steps: int = 1


class RetNetHRMModel(nn.Module):
    """RetNet-HRM language model.

    Current implementation (US1-US3):
    - RetNet backbone (parallel training + recurrent inference)
    - Thin attention band for cross-token fusion
    - HRM controller + ACT halting for adaptive computation
    - Output head with tied embeddings
    - Language modeling loss with ponder cost

    Future extensions (US4):
    - Retrieval system + routing for codebase knowledge
    """

    def __init__(self, config: ModelConfig):
        """Initialize model.

        Args:
            config: ModelConfig with architecture parameters
        """
        super().__init__()

        self.config = config

        # Validate config
        config.validate()

        # RetNet backbone
        self.retnet = RetNetBackbone(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers_retnet,
            n_heads=config.n_retention_heads,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len_infer,
            debug=config.debug,
        )

        # Thin attention band (US2) - Optional, controlled by config
        if config.n_layers_attention > 0:
            # Calculate num_heads for attention (must divide d_model)
            # Use 16 heads for d_model=2816 (2816/16=176)
            attention_heads = 16 if config.d_model == 2816 else 12

            self.attention_band = ThinAttentionBand(
                d_model=config.d_model,
                num_heads=attention_heads,
                num_layers=config.n_layers_attention,
                window_size=config.attention_window,
                dropout=config.dropout,
                use_rope=config.use_rope_in_attention,
            )
        else:
            self.attention_band = None

        # HRM Controller + ACT Halting (US3)
        # HRM summarizes RetNet outputs and maintains recurrent state
        self.hrm_summarizer = HRMSummarizer(
            d_model=config.d_model,
            strategy='mean'  # Mean pooling strategy
        )

        # Controller maintains recurrent state across ACT steps
        d_controller = config.d_model // 2  # Reduce controller size for efficiency
        self.hrm_controller = HRMController(
            d_model=config.d_model,
            d_controller=d_controller,
            dropout=config.dropout,
        )

        # ACT halting head predicts when to stop pondering
        self.act_halting = ACTHaltingHead(
            d_input=d_controller,
            epsilon=config.hrm_epsilon,
            bias_init=config.hrm_halting_bias_init,
        )

        # Computation budget configuration
        self.computation_budget = ComputationBudget(
            min_steps=1,
            max_steps=config.hrm_t_max,
            epsilon=config.hrm_epsilon,
            ponder_tau=config.hrm_ponder_tau,
        )

        # Output head (tied with embeddings for memory efficiency)
        self.output_head = RetNetOutputHead(
            d_model=config.d_model,
            vocab_size=config.vocab_size,
            tie_weights=True,
            embedding_layer=self.retnet.embed,
        )

        # Note: Structure FSM removed (was training-only utility for tag validation)
        # If needed for your training, implement structure validation separately
        self.fsm = None
        self.fsm_weight = 0.0

        # Retrieval system (US4) - Optional
        self.dual_encoder = None
        self.workspace_index = None
        self.global_index = None
        self.compressor = None
        self.router = None

        if config.enable_retrieval:
            self._initialize_retrieval_system(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        must_split: Optional[torch.Tensor] = None,  # Per-position FSM conditional flags
        return_dict: bool = True,
        use_act: bool = False,  # Enable ACT loop (US3)
    ) -> ModelOutput:
        """Forward pass in training mode (parallel).

        Implements model-forward.md contract.

        Args:
            input_ids: Token IDs, shape (batch_size, seq_len)
            attention_mask: Attention mask, shape (batch_size, seq_len)
            labels: Target tokens for loss, shape (batch_size, seq_len)
            segment_ids: Segment IDs for packed sequences, shape (batch_size, seq_len)
                        Used to prevent cross-example attention leakage.
            return_dict: Return ModelOutput instead of tuple

        Returns:
            ModelOutput with loss, logits, stats

        Raises:
            RuntimeError: If memory constraint violated (FR-003)
        """
        # Validate inputs
        batch_size, seq_len = input_ids.shape
        assert seq_len <= self.config.max_seq_len_train, \
            f"Sequence length {seq_len} exceeds max_seq_len_train {self.config.max_seq_len_train}"

        # Memory check (FR-003)
        self._check_memory_constraint()

        # RetNet forward (parallel mode) with segment isolation
        hidden_states = self.retnet.forward_train(input_ids, segment_ids=segment_ids)  # (B, T, d)

        if use_act:
            # ACT loop (US3) - adaptive computation
            logits, ponder_cost_tensor, num_steps, retrieval_stats = self._forward_with_act(
                hidden_states, attention_mask
            )
            avg_steps = float(num_steps)
            router_stats = retrieval_stats  # Pass retrieval stats to router_stats
        else:
            # Standard forward (no ACT)
            router_stats = None

            # Retrieval (US4) - Execute if enabled
            landmark_tokens = None
            if self.config.enable_retrieval:
                try:
                    # Generate query from hidden states (simple mean pooling)
                    if attention_mask is not None:
                        # Masked mean pooling
                        mask_expanded = attention_mask.unsqueeze(-1).float()
                        sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
                        sum_mask = mask_expanded.sum(dim=1)
                        query_state = sum_hidden / sum_mask.clamp(min=1e-9)
                    else:
                        # Simple mean
                        query_state = hidden_states.mean(dim=1)  # [batch, d_model]

                    landmark_tokens, router_stats = self._retrieve_and_select(
                        query_hidden_state=query_state,
                        k_retrieve=self.config.retrieval_topk,
                        k_select=4,  # 4 landmarks × 6 tokens = 24
                    )
                except Exception as e:
                    print(f"Warning: Retrieval failed: {e}")
                    landmark_tokens = None

            # Prepare hidden states for attention band
            attn_hidden = hidden_states

            # If landmarks retrieved, prepend them
            if landmark_tokens is not None:
                attn_hidden = torch.cat([landmark_tokens, hidden_states], dim=1)

                # Update attention mask
                if attention_mask is not None:
                    landmark_mask = torch.ones(
                        batch_size, landmark_tokens.shape[1],
                        dtype=attention_mask.dtype, device=hidden_states.device
                    )
                    expanded_mask = torch.cat([landmark_mask, attention_mask], dim=1)
                else:
                    expanded_mask = None

                global_token_indices = list(range(landmark_tokens.shape[1]))
            else:
                expanded_mask = attention_mask
                global_token_indices = None

            # Attention band for cross-token fusion (US2)
            if self.attention_band is not None:
                attn_hidden = self.attention_band(
                    attn_hidden,
                    attention_mask=expanded_mask,
                    global_token_indices=global_token_indices,
                )

            # Remove landmarks if added
            if landmark_tokens is not None:
                attn_hidden = attn_hidden[:, landmark_tokens.shape[1]:, :]

            # Output logits
            logits = self.output_head(attn_hidden)  # (B, T, vocab_size)
            ponder_cost_tensor = None
            avg_steps = None

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift labels (language modeling)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Cross-entropy loss (no label smoothing for FORMAT band)
            # Label smoothing spreads probability to wrong classes, slowing convergence
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.0)
            ce_loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )

            # Note: FSM penalty removed (training-only utility)
            # If you need structure validation during training, implement separately
            loss_fsm = torch.tensor(0.0, device=ce_loss.device)
            fsm_stats = {}

            # Combined loss (just CE loss now, FSM removed)
            loss = ce_loss

            # Store loss components for logging
            ce_loss_component = ce_loss.detach() if ce_loss is not None else None
            fsm_loss_component = loss_fsm.detach() if loss_fsm is not None else None

            # Add ponder cost if using ACT
            if use_act and ponder_cost_tensor is not None:
                ponder_weight = self.computation_budget.get_ponder_loss_weight()
                loss = loss + ponder_weight * ponder_cost_tensor.mean()

        # Convert ponder cost to scalar for logging
        ponder_cost = ponder_cost_tensor.mean().item() if ponder_cost_tensor is not None else None

        # Future: Add router stats here (US4)
        router_stats = None

        if return_dict:
            return ModelOutput(
                loss=loss,
                logits=logits,
                ponder_cost=ponder_cost,
                avg_steps=avg_steps,
                router_stats=router_stats,
                ce_loss=ce_loss_component if labels is not None else None,
                fsm_loss=fsm_loss_component if labels is not None else None,
            )
        else:
            return (loss, logits, ponder_cost, avg_steps, router_stats)

    def forward_recurrent(
        self,
        input_ids: torch.Tensor,
        state: Optional[Any] = None,  # ContextState from T022
        return_dict: bool = True,
        use_retrieval: bool = False,  # Enable retrieval in recurrent mode
    ) -> InferenceOutput:
        """Forward pass in recurrent inference mode.

        Implements model-forward.md contract for O(1) inference.

        Args:
            input_ids: New token IDs, shape (batch_size, chunk_size)
            state: ContextState with recurrent state
            return_dict: Return InferenceOutput instead of tuple
            use_retrieval: Enable retrieval (experimental)

        Returns:
            InferenceOutput with logits and updated state

        Raises:
            RuntimeError: If memory constraint violated (FR-003)
        """
        # Memory check (FR-003)
        self._check_memory_constraint()

        # Extract RetNet state (simplified for MVP)
        retnet_state = state if state is not None else None

        # RetNet recurrent forward
        hidden_states, new_retnet_state = self.retnet.forward_recurrent(
            input_ids,
            state=retnet_state
        )

        # Retrieval (US4) - Experimental in recurrent mode
        # TODO: Implement proper landmark caching in ContextState
        landmark_tokens = None
        if self.config.enable_retrieval and use_retrieval:
            try:
                # Simple query from current hidden states
                query_state = hidden_states.mean(dim=1)  # [batch, d_model]
                landmark_tokens, _ = self._retrieve_and_select(
                    query_hidden_state=query_state,
                    k_retrieve=self.config.retrieval_topk,
                    k_select=4,
                )
            except Exception as e:
                print(f"Warning: Retrieval in recurrent mode failed: {e}")

        # Prepare for attention band
        attn_hidden = hidden_states

        if landmark_tokens is not None:
            attn_hidden = torch.cat([landmark_tokens, hidden_states], dim=1)
            global_token_indices = list(range(landmark_tokens.shape[1]))
        else:
            global_token_indices = None

        # Attention band for cross-token fusion (US2)
        # Note: For recurrent mode, attention band also uses KV cache
        # TODO: Properly manage attention KV cache in ContextState (US2)
        if self.attention_band is not None:
            attn_hidden = self.attention_band(
                attn_hidden,
                global_token_indices=global_token_indices,
            )

        # Remove landmarks if added
        if landmark_tokens is not None:
            attn_hidden = attn_hidden[:, landmark_tokens.shape[1]:, :]

        # Output logits
        logits = self.output_head(attn_hidden)

        # Future: Add ACT halting here (US3)
        halted = False
        num_steps = 1

        if return_dict:
            return InferenceOutput(
                logits=logits,
                state=new_retnet_state,
                halted=halted,
                num_steps=num_steps,
            )
        else:
            return (logits, new_retnet_state, halted, num_steps)

    def _forward_with_act(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, Optional[Dict[str, Any]]]:
        """
        Forward pass with ACT loop (adaptive computation).

        Args:
            hidden_states: RetNet outputs [batch, seq_len, d_model]
            attention_mask: Optional mask [batch, seq_len]

        Returns:
            tuple of (logits, ponder_cost, num_steps, retrieval_stats):
                - logits: Weighted output logits [batch, seq_len, vocab_size]
                - ponder_cost: Expected steps E[N] [batch]
                - num_steps: Maximum steps taken
                - retrieval_stats: Retrieval metrics (if retrieval enabled)
        """
        batch_size, seq_len, d_model = hidden_states.shape
        device = hidden_states.device

        # Initialize accumulators
        accumulated_logits = None
        accumulated_prob = torch.zeros(batch_size, device=device)
        halting_probs_list = []
        halted_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Initialize HRM controller state
        controller_state = None

        # Retrieval stats (if retrieval enabled)
        retrieval_stats = None

        max_steps = self.computation_budget.max_steps
        min_steps = self.computation_budget.min_steps

        for step in range(1, max_steps + 1):
            # Summarize hidden states for controller
            summary = self.hrm_summarizer(hidden_states, attention_mask)  # [batch, d_model]

            # Update HRM controller state
            controller_state, query_vector, output_state = self.hrm_controller(
                summary, prev_state=controller_state
            )

            # Retrieval (US4) - Execute on first step or when needed
            # Use controller state as query for retrieval
            landmark_tokens = None
            if self.config.enable_retrieval and step == 1:  # Retrieve on first step
                try:
                    landmark_tokens, retrieval_stats = self._retrieve_and_select(
                        query_hidden_state=controller_state,  # [batch, d_controller]
                        k_retrieve=self.config.retrieval_topk,
                        k_select=4,  # 4 landmarks × 6 tokens = 24
                    )
                    # landmark_tokens: [batch, 24, d_model]
                except Exception as e:
                    print(f"Warning: Retrieval failed: {e}")
                    landmark_tokens = None

            # Predict halting probability
            halting_prob, new_accumulated_prob, should_halt = self.act_halting(
                controller_state=controller_state,
                accumulated_prob=accumulated_prob,
                step=step,
                max_steps=max_steps,
            )

            # Store halting probability for ponder cost
            halting_probs_list.append(halting_prob.unsqueeze(1))

            # Apply attention band (if enabled)
            step_hidden = hidden_states

            # If landmarks retrieved, prepend them to the sequence
            if landmark_tokens is not None:
                # Concatenate landmarks with hidden states
                step_hidden = torch.cat([landmark_tokens, step_hidden], dim=1)
                # Shape: [batch, 24 + seq_len, d_model]

                # Update attention mask to include landmarks
                if attention_mask is not None:
                    landmark_mask = torch.ones(
                        batch_size, landmark_tokens.shape[1],
                        dtype=attention_mask.dtype, device=device
                    )
                    expanded_mask = torch.cat([landmark_mask, attention_mask], dim=1)
                else:
                    expanded_mask = None

                # Mark landmarks as global tokens (indices 0-23)
                global_token_indices = list(range(landmark_tokens.shape[1]))
            else:
                expanded_mask = attention_mask
                global_token_indices = None

            if self.attention_band is not None:
                step_hidden = self.attention_band(
                    step_hidden,
                    attention_mask=expanded_mask,
                    global_token_indices=global_token_indices,
                )

            # If landmarks were added, remove them before output head
            if landmark_tokens is not None:
                step_hidden = step_hidden[:, landmark_tokens.shape[1]:, :]
                # Back to: [batch, seq_len, d_model]

            # Compute logits for this step
            step_logits = self.output_head(step_hidden)  # [batch, seq_len, vocab]

            # Weight by halting probability
            # Only weight active (non-halted) sequences
            active_mask = ~halted_mask
            weight = halting_prob * active_mask.float()

            # Expand weight to match logits shape
            weight = weight.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1]

            # Accumulate weighted logits
            weighted_logits = step_logits * weight

            if accumulated_logits is None:
                accumulated_logits = weighted_logits
            else:
                accumulated_logits = accumulated_logits + weighted_logits

            # Update state
            accumulated_prob = new_accumulated_prob
            halted_mask = halted_mask | should_halt

            # Early exit if all sequences halted and past min_steps
            if step >= min_steps and halted_mask.all():
                num_steps = step
                break
        else:
            num_steps = max_steps

        # Compute ponder cost E[N]
        halting_probs = torch.cat(halting_probs_list, dim=1)  # [batch, T]
        steps_tensor = torch.arange(1, num_steps + 1, device=device, dtype=torch.float32)
        ponder_cost = self.act_halting.compute_ponder_cost(halting_probs, steps_tensor)

        return accumulated_logits, ponder_cost, num_steps, retrieval_stats

    def _retrieve_and_select(
        self,
        query_hidden_state: torch.Tensor,
        k_retrieve: int = 32,
        k_select: int = 4,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Execute retrieval pipeline: query → search → compress → route.

        Args:
            query_hidden_state: Query representation from HRM controller [batch, d_model]
            k_retrieve: Number of chunks to retrieve (default: 32)
            k_select: Number of landmarks to select (default: 4, gives 4×6=24 tokens)

        Returns:
            Tuple of (selected_landmarks, retrieval_stats):
                - selected_landmarks: Selected landmark tokens [batch, k_select*6, d_model]
                - retrieval_stats: Dict with hit rates, precision, etc.

        Raises:
            RuntimeError: If retrieval system not initialized
        """
        if self.dual_encoder is None or self.compressor is None or self.router is None:
            raise RuntimeError("Retrieval system not initialized. Set enable_retrieval=True in config.")

        batch_size = query_hidden_state.shape[0]
        device = query_hidden_state.device

        # Initialize stats
        retrieval_stats = {
            'hit_rate': 0.0,
            'precision': 0.0,
            'avg_distance': 0.0,
            'cache_hit_rate': 0.0,
        }

        # 1. Encode query using dual encoder
        # Note: HRM controller output is d_model, but encoder expects text/token IDs
        # For now, project controller state to encoder's embedding space
        # TODO: Add learned projection layer
        with torch.no_grad():
            # Temporary: Use controller state directly as query embedding
            # In practice, we'd want to add a projection: d_model → 768
            query_embedding = query_hidden_state  # [batch, d_model]

            # If dimensions don't match, project
            if query_embedding.shape[-1] != 768:
                # Simple linear projection (should be learned)
                if not hasattr(self, 'query_projector'):
                    self.query_projector = nn.Linear(
                        query_embedding.shape[-1], 768, bias=False
                    ).to(device)
                query_embedding = self.query_projector(query_embedding)

            # L2 normalize for cosine similarity
            query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=-1)

        # 2. Search indexes for top-k chunks
        all_landmarks = []
        all_scores = []

        for batch_idx in range(batch_size):
            query_emb_np = query_embedding[batch_idx].cpu().numpy()

            # Search workspace index (if available)
            workspace_chunks = []
            if self.workspace_index is not None:
                try:
                    workspace_results = self.workspace_index.search(
                        query_emb_np, k=k_retrieve // 2
                    )
                    workspace_chunks = workspace_results
                except Exception as e:
                    print(f"Warning: Workspace search failed: {e}")

            # Search global index (if available)
            global_chunks = []
            if self.global_index is not None:
                try:
                    global_results = self.global_index.search(
                        query_emb_np, k=k_retrieve // 2
                    )
                    global_chunks = global_results
                except Exception as e:
                    print(f"Warning: Global search failed: {e}")

            # Combine results
            all_chunks = workspace_chunks + global_chunks
            if len(all_chunks) == 0:
                # No results - create dummy landmarks
                dummy_landmarks = torch.zeros(
                    k_select, 6, self.config.d_model, device=device
                )
                all_landmarks.append(dummy_landmarks)
                all_scores.append(torch.zeros(k_select, device=device))
                continue

            # Sort by score and take top-k
            all_chunks.sort(key=lambda x: x[1], reverse=True)
            top_chunks = all_chunks[:k_retrieve]

            # 3. Compress chunks to landmarks
            landmarks_batch = []
            scores_batch = []

            for chunk, score in top_chunks:
                # Get chunk embedding
                chunk_emb = torch.from_numpy(chunk.embedding).to(device)

                # Compress to landmark tokens [6, d_model]
                landmark_tokens = self.compressor(chunk_emb.unsqueeze(0))  # [1, 6, d_model]
                landmarks_batch.append(landmark_tokens.squeeze(0))
                scores_batch.append(score)

            # Stack: [num_candidates, 6, d_model]
            landmarks_batch = torch.stack(landmarks_batch)
            scores_batch = torch.tensor(scores_batch, device=device)

            all_landmarks.append(landmarks_batch)
            all_scores.append(scores_batch)

        # Stack batch: [batch, num_candidates, 6, d_model]
        landmarks_tensor = torch.stack(all_landmarks)
        scores_tensor = torch.stack(all_scores)

        # 4. Route: select top-k landmarks
        selected_landmarks, selected_probs, aux_losses = self.router.select_landmarks(
            landmarks_tensor,  # [batch, num_candidates, 6, d_model]
            scores_tensor,     # [batch, num_candidates]
            k=k_select         # Select 4 landmarks
        )

        # Flatten to tokens: [batch, k_select*6, d_model]
        batch_size, k, L, d_model = selected_landmarks.shape
        selected_tokens = selected_landmarks.reshape(batch_size, k * L, d_model)

        # Update stats with routing info
        if aux_losses is not None:
            retrieval_stats.update({
                'aux_losses': aux_losses,
                'selection_stats': {
                    'selection_rate': selected_probs.mean().item(),
                    'avg_prob': selected_probs.mean().item(),
                    'entropy_mean': -(selected_probs * torch.log(selected_probs + 1e-10)).sum(-1).mean().item(),
                    'num_selected': k,
                }
            })

        return selected_tokens, retrieval_stats

    def _initialize_retrieval_system(self, config: ModelConfig):
        """Initialize retrieval system components (US4).

        Args:
            config: ModelConfig with retrieval paths

        Raises:
            FileNotFoundError: If retrieval files not found
            RuntimeError: If retrieval initialization fails
        """
        print(f"\n{'='*60}")
        print("Initializing Retrieval System (US4)")
        print(f"{'='*60}")

        # Pre-flight validation: manifest-driven fail-fast checks (FR-007)
        self.retrieval_registry = None
        if config.retrieval_manifest_path:
            registry = RetrievalRegistry.load(config.retrieval_manifest_path)
            try:
                evaluated_sources = registry.ensure_all_ready()
            except RuntimeError as exc:
                raise RuntimeError(
                    "Retrieval initialization blocked: manifest validation failed."
                ) from exc

            print("Manifest validation results:")
            for source in evaluated_sources.values():
                print(
                    f"  • {source.display_name}: status={source.status.value}"
                )
            self.retrieval_registry = registry

        # Direct path validation for critical artefacts
        required_assets = {
            "dual encoder checkpoint": config.encoder_checkpoint_path,
            "workspace index": config.workspace_index_path,
            "global index": config.global_index_path,
        }
        missing_direct = []
        for label, candidate in required_assets.items():
            if not candidate:
                continue
            resolved = Path(candidate).expanduser()
            if not resolved.exists():
                missing_direct.append(f"{label}: {resolved}")

        if missing_direct:
            raise RuntimeError(
                "Retrieval initialization blocked: missing required assets -> "
                + "; ".join(missing_direct)
            )

        # Load dual encoder
        if config.encoder_checkpoint_path and os.path.exists(config.encoder_checkpoint_path):
            print(f"Loading dual encoder from: {config.encoder_checkpoint_path}")
            checkpoint = torch.load(config.encoder_checkpoint_path, map_location='cpu')

            # Initialize encoder with config from checkpoint
            encoder_config = checkpoint.get('config', {})
            self.dual_encoder = DualEncoder(
                vocab_size=encoder_config.get('vocab_size', 50000),
                d_model=encoder_config.get('d_model', 768),
                num_layers=encoder_config.get('num_layers', 6),
                num_heads=encoder_config.get('num_heads', 12),
                dropout=encoder_config.get('dropout', 0.1),
            )
            self.dual_encoder.load_state_dict(checkpoint['model_state_dict'])
            self.dual_encoder.eval()  # Evaluation mode
            print(f"  ✓ Dual encoder loaded (768-dim embeddings)")
        else:
            print(f"  ⚠ No encoder checkpoint found, creating new encoder")
            self.dual_encoder = DualEncoder(
                vocab_size=50000,
                d_model=768,
                num_layers=6,
                num_heads=12,
                dropout=0.1,
            )

        # Load workspace index
        if config.workspace_index_path and os.path.exists(config.workspace_index_path):
            print(f"Loading workspace index from: {config.workspace_index_path}")
            self.workspace_index = RetrievalIndex.load(config.workspace_index_path)
            stats = self.workspace_index.get_stats()
            print(f"  ✓ Workspace index loaded ({stats.get('num_chunks', 0)} chunks)")
        else:
            print(f"  ⚠ No workspace index found")

        # Load global index
        if config.global_index_path and os.path.exists(config.global_index_path):
            print(f"Loading global index from: {config.global_index_path}")
            self.global_index = RetrievalIndex.load(config.global_index_path)
            stats = self.global_index.get_stats()
            print(f"  ✓ Global index loaded ({stats.get('num_chunks', 0)} chunks)")
        else:
            print(f"  ⚠ No global index found")

        # Initialize landmark compressor
        print("Initializing landmark compressor...")
        self.compressor = LandmarkCompressor(
            embedding_dim=768,  # Dual encoder output
            model_dim=config.d_model,  # Model hidden dim
            num_tokens=config.retrieval_landmark_tokens,  # L=6
        )
        print(f"  ✓ Compressor initialized ({self.compressor.get_num_params()/1e6:.1f}M params)")

        # Initialize Gumbel router
        print("Initializing Gumbel top-k router...")
        self.router = GumbelTopKRouter(
            budget_B=config.router_budget_B,  # B=24
            temperature=config.router_gumbel_temp,  # 0.7
            lambda_sparsity=config.router_lambda_sparsity,  # 2e-4
            lambda_entropy=config.router_lambda_entropy,  # 1e-3
        )
        print(f"  ✓ Router initialized (budget B={config.router_budget_B} tokens)")

        print(f"{'='*60}")
        print("Retrieval system ready!")
        print(f"{'='*60}\n")

    def _check_memory_constraint(self):
        """Validate memory usage doesn't exceed limit.

        Note: 2.98B model with AdamW requires ~36GB for weights+optimizer+gradients

        Raises:
            RuntimeError: If memory exceeds 48GB
        """
        if torch.cuda.is_available():
            allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
            if allocated_mb > 48 * 1024:  # 48GB
                raise RuntimeError(
                    f"Memory exceeded: {allocated_mb/1024:.2f}GB > 48GB\n"
                    f"Try reducing batch_size or seq_len"
                )

    @classmethod
    def from_config(cls, config: ModelConfig) -> "RetNetHRMModel":
        """Create model from config.

        Args:
            config: ModelConfig instance

        Returns:
            RetNetHRMModel instance
        """
        return cls(config)

    def get_num_params(self) -> int:
        """Get total number of parameters.

        Returns:
            Total parameter count
        """
        return sum(p.numel() for p in self.parameters())

    def print_param_count(self):
        """Print parameter count breakdown."""
        total = self.get_num_params()
        retnet = sum(p.numel() for p in self.retnet.parameters())
        output = sum(p.numel() for p in self.output_head.parameters() if p.requires_grad)
        attn_band = sum(p.numel() for p in self.attention_band.parameters()) if self.attention_band is not None else 0

        # HRM/ACT components
        hrm_controller = sum(p.numel() for p in self.hrm_controller.parameters())
        hrm_summarizer = sum(p.numel() for p in self.hrm_summarizer.parameters())
        act_halting = sum(p.numel() for p in self.act_halting.parameters())

        print(f"\n{'='*50}")
        print(f"Model Parameter Count")
        print(f"{'='*50}")
        print(f"RetNet backbone:    {retnet/1e9:.2f}B")
        if self.attention_band is not None:
            print(f"Attention band:     {attn_band/1e6:.2f}M")
        print(f"HRM controller:     {hrm_controller/1e6:.2f}M")
        print(f"HRM summarizer:     {hrm_summarizer/1e6:.2f}M")
        print(f"ACT halting:        {act_halting/1e3:.2f}K")
        print(f"Output head:        {output/1e9:.2f}B")
        print(f"{'='*50}")
        print(f"Total:              {total/1e9:.2f}B")
        print(f"{'='*50}\n")

        # Validate reasonable parameter count
        assert 0.5e9 <= total <= 3.5e9, \
            f"Parameter count {total/1e9:.2f}B outside reasonable 0.5-3.5B range"
