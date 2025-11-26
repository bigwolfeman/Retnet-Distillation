# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]


import torch
import torch.nn.functional as F
from torch import nn
from .rms_norm import RMSNorm

from .multiway_network import MultiwayWrapper

def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\

def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m

def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)

def get_activation_fn(activation):
    if activation == "swish":
        return F.silu
    elif activation == "gelu":
        return F.gelu
    else:
        raise NotImplementedError
    
class MultiScaleRetention(nn.Module):
    def __init__(
        self,
        args,
        embed_dim,
        value_dim,
        num_heads,
        gate_fn="swish",
    ):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.head_dim = self.value_dim // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        
        self.gate_fn = get_activation_fn(activation=str(gate_fn))

        self.q_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=False))
        self.k_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=False))
        self.v_proj = MultiwayWrapper(args, nn.Linear(embed_dim, value_dim, bias=False))
        self.g_proj = MultiwayWrapper(args, nn.Linear(embed_dim, value_dim, bias=False))
        
        self.out_proj = MultiwayWrapper(args, nn.Linear(value_dim, embed_dim, bias=False))

        self.group_norm = MultiwayWrapper(args, RMSNorm(self.head_dim, eps=args.layernorm_eps, elementwise_affine=False))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.g_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=2 ** -1)

    def parallel_forward(self, qr, kr, v, mask, attention_mask=None):
        bsz, tgt_len, embed_dim = v.size()

        vr = v.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        qk_mat = qr @ kr.transpose(-1, -2) # bsz * m * tgt_len * tgt_len
        qk_mat = qk_mat * mask

        # Build combined boolean mask for segment isolation
        # This mask will be used to zero out positions instead of setting to -inf
        combined_mask = None
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                # Boolean mask: True = keep, False = mask
                if attention_mask.ndim == 2:
                    # Padding mask [B, T] - broadcast to [B, 1, 1, T]
                    combined_mask = attention_mask[:, None, None, :]
                elif attention_mask.ndim == 3:
                    # Full attention mask [B, T, T] - broadcast to [B, 1, T, T]
                    combined_mask = attention_mask[:, None, :, :]
                elif attention_mask.ndim == 4:
                    # Pre-expanded mask [B, 1, T, T] or [B, H, T, T]
                    combined_mask = attention_mask
            else:
                # Additive mask: 0 = keep, -inf = mask
                # Convert to boolean: True where finite (keep), False where -inf (mask)
                combined_mask = torch.isfinite(attention_mask)

        # Zero out masked positions BEFORE normalization (instead of -inf)
        # This prevents extreme values (-1e9 / 5e4 = -20,000) that cause NaN
        if combined_mask is not None:
            qk_mat = qk_mat * combined_mask.float()

        # Retention normalization: divide by L1 norm (clamped for stability)
        # Now operates only on valid (non-masked) values, preventing explosion
        qk_mat = qk_mat / qk_mat.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1, max=5e4)

        # Re-apply mask after normalization to ensure masked positions remain exactly 0
        if combined_mask is not None:
            qk_mat = qk_mat * combined_mask.float()

        output = torch.matmul(qk_mat, vr)
        output = output.transpose(1, 2)
        return output

    def recurrent_forward(
        self,
        qr, kr, v,
        decay,
        incremental_state,
        attention_mask=None
    ):
        bsz = v.size(0)

        v = v.view(bsz, self.num_heads, self.head_dim, 1)
        kv = kr * v
        if "prev_key_value" in incremental_state:
            prev_kv = incremental_state["prev_key_value"]
            prev_scale = incremental_state["scale"]
            # Reshape decay for broadcasting: [num_heads] -> [num_heads, 1, 1]
            decay_reshaped = decay.view(self.num_heads, 1, 1)
            scale = prev_scale * decay_reshaped + 1
            # Add numerical stability: clamp scale to avoid division by zero
            scale = scale.clamp(min=1e-6)
            kv_next = prev_kv * (prev_scale.clamp(min=1e-6).sqrt() * decay_reshaped / scale.sqrt()) + kv / scale.sqrt()
            # kv = prev_kv * decay.view(self.num_heads, 1, 1) + kv
            
            # Gate state update based on attention_mask (recurrent: per-step, last token)
            if attention_mask is not None:
                if attention_mask.dtype == torch.bool:
                    # For recurrent mode, mask shape is [B, T] and we use the last position
                    # mask_step: [B] indicating if current token is valid
                    mask_step = attention_mask[:, -1]  # [B]
                    # Expand to match kv shape [B, num_heads, head_dim, 1]
                    mask_step_kv = mask_step.view(bsz, 1, 1, 1)
                    # Keep previous state if padded, otherwise update
                    kv = torch.where(mask_step_kv, kv_next, prev_kv)
                    # For scale: shape is [num_heads, 1, 1], broadcast mask [B] -> need compatible shape
                    # Actually scale is [num_heads, 1, 1], not dependent on batch
                    # The gating should happen per-batch, so we need to handle this differently
                    # For now, only gate kv (which is per-batch)
                    kv = torch.where(mask_step_kv, kv_next, prev_kv)
                    # Scale update: only update if ALL batch items are valid (conservative)
                    if mask_step.all():
                        scale = scale
                    else:
                        scale = prev_scale
                else:
                    # For additive masks, check if the value is -inf (masked)
                    # attention_mask shape: [B, 1, 1, T], take last position
                    mask_val = attention_mask[:, 0, 0, -1]  # [B]
                    mask_step = torch.isfinite(mask_val)  # [B], True if valid
                    mask_step_kv = mask_step.view(bsz, 1, 1, 1)
                    kv = torch.where(mask_step_kv, kv_next, prev_kv)
                    if mask_step.all():
                        scale = scale
                    else:
                        scale = prev_scale
            else:
                kv = kv_next
        else:
            scale = torch.ones_like(decay).view(self.num_heads, 1, 1)

        incremental_state["prev_key_value"] = kv
        incremental_state["scale"] = scale

        output = torch.sum(qr * kv, dim=3)
        return output
    
    def chunk_recurrent_forward(
        self,
        qr, kr, v,
        inner_mask,
        attention_mask=None
    ):
        mask, cross_decay, query_inner_decay, value_inner_decay = inner_mask
        bsz, tgt_len, embed_dim = v.size()
        chunk_len = mask.size(1)
        num_chunks = tgt_len // chunk_len

        assert tgt_len % chunk_len == 0

        qr = qr.view(bsz, self.num_heads, num_chunks, chunk_len, self.key_dim).transpose(1, 2)
        kr = kr.view(bsz, self.num_heads, num_chunks, chunk_len, self.key_dim).transpose(1, 2)
        v = v.view(bsz, num_chunks, chunk_len, self.num_heads, self.head_dim).transpose(2, 3)

        kr_t = kr.transpose(-1, -2)

        # === INNER-CHUNK ATTENTION ===
        qk_mat = qr @ kr_t # bsz * num_chunks * num_heads * chunk_len * chunk_len
        qk_mat = qk_mat * mask

        # Apply attention_mask for segment isolation within chunks
        chunk_combined_mask = None
        if attention_mask is not None:
            chunk_attn_mask = self._prepare_chunk_attention_mask(
                attention_mask, bsz, num_chunks, chunk_len
            )
            # chunk_attn_mask: [bsz, num_chunks, 1, chunk_len, chunk_len]

            if attention_mask.dtype == torch.bool:
                # Boolean: True = keep, False = mask
                chunk_combined_mask = chunk_attn_mask
            else:
                # Additive: 0 = keep, -inf = mask
                # Convert to boolean
                chunk_combined_mask = torch.isfinite(chunk_attn_mask)

            # Zero out masked positions BEFORE normalization (instead of -inf)
            qk_mat = qk_mat * chunk_combined_mask.float()

        inner_scale = qk_mat.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1)
        qk_mat = qk_mat / inner_scale

        # Re-apply mask after normalization
        if chunk_combined_mask is not None:
            qk_mat = qk_mat * chunk_combined_mask.float()

        inner_output = torch.matmul(qk_mat, v) # bsz * num_chunks * num_heads * chunk_len * head_dim

        # === CROSS-CHUNK RECURRENCE ===
        # reduce kv in one chunk
        kv = kr_t @ (v * value_inner_decay)

        # Compute segment boundary indicators for cross-chunk gating
        if attention_mask is not None:
            segment_reset_mask = self._compute_segment_boundaries(
                attention_mask, num_chunks, chunk_len
            )
            # segment_reset_mask: [bsz, num_chunks] boolean
        else:
            segment_reset_mask = None

        kv_recurrent = []
        cross_scale = []
        kv_state = torch.zeros(bsz, self.num_heads, self.key_dim, self.head_dim).to(v)
        kv_scale = torch.ones(bsz, self.num_heads, 1, 1).to(v)

        # accumulate kv by loop with segment-aware gating
        for i in range(num_chunks):
            kv_recurrent.append(kv_state / kv_scale)
            cross_scale.append(kv_scale)

            # Gate state update based on segment boundaries
            if segment_reset_mask is not None:
                # Reset mask: [bsz, 1, 1, 1] for broadcasting
                reset = segment_reset_mask[:, i].view(bsz, 1, 1, 1)
                # If reset, use only current chunk's kv (no carry from previous)
                kv_state = torch.where(reset, kv[:, i], kv_state * cross_decay + kv[:, i])
            else:
                kv_state = kv_state * cross_decay + kv[:, i]

            kv_scale = kv_state.detach().abs().sum(dim=-2, keepdim=True).max(dim=-1, keepdim=True).values.clamp(min=1)

        kv_recurrent = torch.stack(kv_recurrent, dim=1)
        cross_scale = torch.stack(cross_scale, dim=1)

        all_scale = torch.maximum(inner_scale, cross_scale)
        align_inner_scale = all_scale / inner_scale
        align_cross_scale = all_scale / cross_scale

        cross_output = (qr * query_inner_decay) @ kv_recurrent
        output = inner_output / align_inner_scale + cross_output / align_cross_scale
        # output = inner_output / cross_scale + cross_output / inner_scale

        output = output.transpose(2, 3)
        return output

    def _prepare_chunk_attention_mask(self, attention_mask, bsz, num_chunks, chunk_len):
        """
        Convert full attention_mask [B, T, T] to per-chunk format.

        Args:
            attention_mask: [B, T] or [B, T, T] or [B, 1, T, T] or [B, H, T, T]
            bsz, num_chunks, chunk_len: dimensions

        Returns:
            chunk_mask: [bsz, num_chunks, 1, chunk_len, chunk_len]
        """
        # Get actual sequence length from mask, not from chunking params
        # attention_mask shape is [B, ...dims..., T] or [B, ...dims..., T, T]
        tgt_len = attention_mask.shape[-1]  # Use actual mask size

        # Verify it matches expected chunking (or is smaller for last batch/short sequences)
        expected_tgt_len = num_chunks * chunk_len
        if tgt_len != expected_tgt_len:
            # Sequence doesn't perfectly align with chunks - pad or adjust
            if tgt_len < expected_tgt_len:
                # Short sequence: recalculate num_chunks and chunk_len
                # For sequences shorter than chunk_len, treat whole sequence as one chunk
                if tgt_len <= chunk_len:
                    num_chunks = 1
                    chunk_len = tgt_len
                else:
                    # Use actual chunks that fit
                    num_chunks = tgt_len // chunk_len
                    # May have remainder - for now, truncate (should not happen in practice)
                    tgt_len = num_chunks * chunk_len
            else:
                # Mask larger than expected - this shouldn't happen, but handle gracefully
                tgt_len = expected_tgt_len

        if attention_mask.ndim == 2:
            # [B, T] padding mask -> convert to causal mask [B, T, T]
            # Create causal mask: True for positions i >= j
            causal_mask = torch.tril(
                torch.ones(tgt_len, tgt_len, device=attention_mask.device, dtype=torch.bool)
            )
            # Expand padding mask to [B, 1, T] for broadcasting
            pad_mask = attention_mask.unsqueeze(-1)  # [B, T, 1]
            # Combine: can attend if position is valid AND causal
            # attention_mask[b, i] = True means position i is valid
            # We want mask[b, i, j] = True if both i and j are valid AND i >= j
            attention_mask = causal_mask.unsqueeze(0) & pad_mask & attention_mask.unsqueeze(1)
            # attention_mask: [B, T, T]
            attention_mask = attention_mask.unsqueeze(1)  # [B, 1, T, T]
        elif attention_mask.ndim == 3:
            # [B, T, T] -> [B, 1, T, T]
            attention_mask = attention_mask.unsqueeze(1)
        elif attention_mask.ndim == 4:
            pass  # Already [B, 1 or H, T, T]

        # attention_mask: [B, 1 or H, T, T]
        # Reshape to chunks: [B, 1 or H, num_chunks, chunk_len, num_chunks, chunk_len]
        mask_reshaped = attention_mask.view(
            bsz, -1, num_chunks, chunk_len, num_chunks, chunk_len
        )

        # Extract diagonal chunks (chunk i attending to chunk i)
        # mask_reshaped: [B, H, num_chunks, chunk_len, num_chunks, chunk_len]
        # We want: [B, H, num_chunks, chunk_len, chunk_len] where query chunk == key chunk

        # Use einops-style indexing to extract diagonal blocks
        # For each chunk i, extract mask_reshaped[:, :, i, :, i, :]
        # Stack along dim=2 to get [B, H, num_chunks, chunk_len, chunk_len]
        chunk_masks = []
        for i in range(num_chunks):
            chunk_masks.append(mask_reshaped[:, :, i, :, i, :])
        chunk_mask = torch.stack(chunk_masks, dim=2)

        # chunk_mask: [B, 1 or H, num_chunks, chunk_len, chunk_len]
        # Transpose to: [B, num_chunks, 1 or H, chunk_len, chunk_len]
        chunk_mask = chunk_mask.transpose(1, 2)

        return chunk_mask

    def _compute_segment_boundaries(self, attention_mask, num_chunks, chunk_len):
        """
        Identify which chunks start a new segment.

        A chunk i starts a new segment if the first position in chunk i
        cannot attend to the last position in chunk i-1.

        Args:
            attention_mask: [B, T, T] or [B, H, T, T] boolean (True = can attend)
            num_chunks, chunk_len: dimensions

        Returns:
            reset_mask: [bsz, num_chunks] boolean (True = reset state)
        """
        bsz = attention_mask.size(0)
        tgt_len = num_chunks * chunk_len

        if attention_mask.ndim == 4:
            # [B, H, T, T] -> [B, T, T] (take first head, assuming same for all)
            attention_mask = attention_mask[:, 0, :, :]
        elif attention_mask.ndim == 3:
            pass  # [B, T, T]
        elif attention_mask.ndim == 2:
            # [B, T] padding mask - shouldn't happen but handle gracefully
            # No segment boundaries from padding mask alone
            return torch.zeros(bsz, num_chunks, dtype=torch.bool, device=attention_mask.device)

        reset_mask = torch.zeros(bsz, num_chunks, dtype=torch.bool, device=attention_mask.device)

        # First chunk never resets (no previous chunk)
        # For chunk i (i > 0), reset if position[chunk_len * i] cannot attend to position[chunk_len * i - 1]
        for i in range(1, num_chunks):
            first_pos = chunk_len * i
            last_prev_pos = chunk_len * i - 1

            # Check if first position of chunk i can attend to last position of chunk i-1
            # attention_mask[b, query_pos, key_pos]
            can_attend = attention_mask[:, first_pos, last_prev_pos]  # [bsz]

            if attention_mask.dtype == torch.bool:
                # True = can attend, so reset if False
                reset_mask[:, i] = ~can_attend
            else:
                # Additive mask: -inf = masked, so reset if masked
                reset_mask[:, i] = ~torch.isfinite(can_attend)

        return reset_mask
    
    def forward(
        self,
        x,
        rel_pos,
        chunkwise_recurrent=False,
        incremental_state=None,
        attention_mask=None
    ):
        bsz, tgt_len, _ = x.size()
        (sin, cos), inner_mask = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        g = self.g_proj(x)

        k *= self.scaling
        q = q.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        k = k.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)

        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)

        if incremental_state is not None:
            output = self.recurrent_forward(qr, kr, v, inner_mask, incremental_state, attention_mask)
        elif chunkwise_recurrent:
            output = self.chunk_recurrent_forward(qr, kr, v, inner_mask, attention_mask)
        else:
            output = self.parallel_forward(qr, kr, v, inner_mask, attention_mask)
        
        output = self.group_norm(output).reshape(bsz, tgt_len, self.head_dim * self.num_heads)

        output = self.gate_fn(g) * output

        output = self.out_proj(output)

        return output

        
