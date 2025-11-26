"""
Combined loss function per plan.md and research.md.

Implements weighted sum of:
- Main cross-entropy loss (CE) with loss masking
- Format validation loss (BCE)
- Auxiliary head losses (carry, 2D mult, division, schema, selector)
- Tool supervision loss
- Length weighting per example

Default weights: w_ce=1.0, w_fmt=0.1, w_aux=0.2, w_tool=0.5, w_schema=0.2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class LossWeights:
    """Loss component weights per FR-019."""
    w_ce: float = 1.0  # Main cross-entropy
    w_fmt: float = 0.1  # Format validation
    w_carry: float = 0.2  # Carry head
    w_mult_2d: float = 0.2  # 2D multiplication
    w_division: float = 0.2  # Division policy
    w_schema: float = 0.2  # JSON schema
    w_selector: float = 0.5  # Tool selector with KL penalty
    w_tool: float = 0.5  # Tool supervision


@dataclass
class LossOutput:
    """Output from compute_loss function."""
    total_loss: torch.Tensor
    loss_components: Dict[str, torch.Tensor]
    metrics: Dict[str, float]


def compute_masked_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    example_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute masked cross-entropy loss with length weighting.

    Only computes loss on answer/hint tokens (loss_mask=True).
    Applies per-example length weighting: clip(8 / |answer|, 1.0, 4.0).

    Args:
        logits: Model output [batch, seq_len, vocab_size]
        targets: Target token IDs [batch, seq_len]
        loss_mask: Boolean mask [batch, seq_len] (True = compute loss)
        example_weights: Per-token weights [batch, seq_len] (optional)

    Returns:
        Masked CE loss (scalar)
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Compute loss per token (no reduction)
    loss_per_token = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
        reduction='none'
    ).reshape(batch_size, seq_len)

    # Apply loss mask (only answer/hint tokens)
    masked_loss = loss_per_token * loss_mask.float()

    # Apply length weighting if provided
    if example_weights is not None:
        masked_loss = masked_loss * example_weights

    # Normalize by number of answer tokens
    num_answer_tokens = loss_mask.sum()
    if num_answer_tokens > 0:
        return masked_loss.sum() / num_answer_tokens
    else:
        return torch.tensor(0.0, device=logits.device)


def compute_format_loss(
    format_logits: torch.Tensor,
    format_targets: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute format validation loss (binary cross-entropy).

    Predicts whether answer tokens form valid format (integer/decimal/fraction).

    Args:
        format_logits: Format head output [batch, seq_len, 2]
        format_targets: Binary targets [batch, seq_len] (0=invalid, 1=valid)
        loss_mask: Only compute on answer tokens

    Returns:
        Format BCE loss (scalar)
    """
    batch_size, seq_len, _ = format_logits.shape

    # BCE loss per token
    loss_per_token = F.cross_entropy(
        format_logits.reshape(-1, 2),
        format_targets.reshape(-1),
        reduction='none'
    ).reshape(batch_size, seq_len)

    # Mask to answer tokens only
    masked_loss = loss_per_token * loss_mask.float()

    num_answer_tokens = loss_mask.sum()
    if num_answer_tokens > 0:
        return masked_loss.sum() / num_answer_tokens
    else:
        return torch.tensor(0.0, device=format_logits.device)


def compute_carry_loss(
    carry_logits: torch.Tensor,
    carry_targets: torch.Tensor,
    carry_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute carry prediction loss (binary cross-entropy per column).

    Args:
        carry_logits: Carry head output [batch, seq_len, max_columns]
        carry_targets: Binary targets [batch, seq_len, max_columns]
        carry_mask: Optional mask for valid columns

    Returns:
        Carry BCE loss (scalar)
    """
    # Binary cross-entropy with logits
    loss = F.binary_cross_entropy_with_logits(
        carry_logits,
        carry_targets.float(),
        reduction='none'
    )

    # Apply mask if provided
    if carry_mask is not None:
        loss = loss * carry_mask.float()
        num_valid = carry_mask.sum()
    else:
        num_valid = carry_logits.numel()

    if num_valid > 0:
        return loss.sum() / num_valid
    else:
        return torch.tensor(0.0, device=carry_logits.device)


def compute_mult_2d_loss(
    grid_logits: torch.Tensor,
    grid_targets: torch.Tensor,
    grid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute 2D multiplication grid loss.

    Args:
        grid_logits: Mult2D head output [batch, seq_len, grid_size^2, 11]
        grid_targets: Target classes [batch, seq_len, grid_size^2]
        grid_mask: Optional mask for valid cells

    Returns:
        Grid CE loss (scalar)
    """
    batch_size, seq_len, num_cells, num_classes = grid_logits.shape

    # Cross-entropy per cell
    loss = F.cross_entropy(
        grid_logits.reshape(-1, num_classes),
        grid_targets.reshape(-1),
        reduction='none'
    ).reshape(batch_size, seq_len, num_cells)

    # Apply mask if provided
    if grid_mask is not None:
        loss = loss * grid_mask.float()
        num_valid = grid_mask.sum()
    else:
        num_valid = loss.numel()

    if num_valid > 0:
        return loss.sum() / num_valid
    else:
        return torch.tensor(0.0, device=grid_logits.device)


def compute_division_loss(
    quotient_logits: torch.Tensor,
    quotient_targets: torch.Tensor,
    division_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute division policy loss (quotient digit prediction).

    Args:
        quotient_logits: Division head output [batch, seq_len, 10]
        quotient_targets: Target digits 0-9 [batch, seq_len]
        division_mask: Optional mask for valid positions

    Returns:
        Division CE loss (scalar)
    """
    batch_size, seq_len, _ = quotient_logits.shape

    # Cross-entropy per position
    loss = F.cross_entropy(
        quotient_logits.reshape(-1, 10),
        quotient_targets.reshape(-1),
        reduction='none'
    ).reshape(batch_size, seq_len)

    # Apply mask if provided
    if division_mask is not None:
        loss = loss * division_mask.float()
        num_valid = division_mask.sum()
    else:
        num_valid = loss.numel()

    if num_valid > 0:
        return loss.sum() / num_valid
    else:
        return torch.tensor(0.0, device=quotient_logits.device)


def compute_schema_loss(
    schema_logits: torch.Tensor,
    schema_targets: torch.Tensor,
    schema_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute JSON schema validation loss (multi-label BCE).

    Args:
        schema_logits: Schema head output [batch, seq_len, max_keys]
        schema_targets: Binary targets per key [batch, seq_len, max_keys]
        schema_mask: Optional mask for valid keys

    Returns:
        Schema BCE loss (scalar)
    """
    # Multi-label binary cross-entropy
    loss = F.binary_cross_entropy_with_logits(
        schema_logits,
        schema_targets.float(),
        reduction='none'
    )

    # Apply mask if provided
    if schema_mask is not None:
        loss = loss * schema_mask.float()
        num_valid = schema_mask.sum()
    else:
        num_valid = loss.numel()

    if num_valid > 0:
        return loss.sum() / num_valid
    else:
        return torch.tensor(0.0, device=schema_logits.device)


def compute_selector_loss(
    selector_output: Dict[str, torch.Tensor],
    selector_targets: torch.Tensor,
    selector_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute tool selector loss (CE + KL penalty to budget prior).

    Args:
        selector_output: Dictionary from SelectorHead with:
            - selector_logits: [batch, seq_len, n_tools+1]
            - kl_penalty: scalar KL divergence
        selector_targets: Target tool indices [batch, seq_len]
        selector_mask: Optional mask for valid positions

    Returns:
        Selector loss (CE + KL penalty)
    """
    selector_logits = selector_output['selector_logits']
    kl_penalty = selector_output['kl_penalty']

    batch_size, seq_len, _ = selector_logits.shape

    # Cross-entropy for tool selection
    ce_loss = F.cross_entropy(
        selector_logits.reshape(-1, selector_logits.size(-1)),
        selector_targets.reshape(-1),
        reduction='none'
    ).reshape(batch_size, seq_len)

    # Apply mask if provided
    if selector_mask is not None:
        ce_loss = ce_loss * selector_mask.float()
        num_valid = selector_mask.sum()
    else:
        num_valid = ce_loss.numel()

    if num_valid > 0:
        ce_loss = ce_loss.sum() / num_valid
    else:
        ce_loss = torch.tensor(0.0, device=selector_logits.device)

    # Combine CE and KL penalty
    total_loss = ce_loss + kl_penalty

    return total_loss


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    aux_outputs: Dict[str, torch.Tensor],
    aux_targets: Optional[Dict[str, torch.Tensor]] = None,
    example_weights: Optional[torch.Tensor] = None,
    loss_weights: Optional[LossWeights] = None,
) -> LossOutput:
    """
    Compute combined loss with weighted sum of all components.

    Args:
        logits: Main LM output [batch, seq_len, vocab_size]
        targets: Target token IDs [batch, seq_len]
        loss_mask: Boolean mask for answer tokens [batch, seq_len]
        aux_outputs: Dictionary of auxiliary head outputs
        aux_targets: Optional dictionary of auxiliary targets
        example_weights: Optional per-token length weights [batch, seq_len]
        loss_weights: Optional custom loss weights

    Returns:
        LossOutput with total loss, components, and metrics
    """
    if loss_weights is None:
        loss_weights = LossWeights()

    if aux_targets is None:
        aux_targets = {}

    loss_components = {}
    metrics = {}

    # Main cross-entropy loss (always computed)
    ce_loss = compute_masked_ce_loss(logits, targets, loss_mask, example_weights)
    loss_components['ce'] = ce_loss
    metrics['ce_loss'] = ce_loss.item()

    # Format validation loss (if targets provided)
    if 'format' in aux_outputs and 'format' in aux_targets:
        format_loss = compute_format_loss(
            aux_outputs['format'],
            aux_targets['format'],
            loss_mask
        )
        loss_components['format'] = format_loss * loss_weights.w_fmt
        metrics['format_loss'] = format_loss.item()

    # Carry prediction loss (if targets provided)
    if 'carry' in aux_outputs and 'carry' in aux_targets:
        carry_loss = compute_carry_loss(
            aux_outputs['carry'],
            aux_targets['carry'],
            aux_targets.get('carry_mask')
        )
        loss_components['carry'] = carry_loss * loss_weights.w_carry
        metrics['carry_loss'] = carry_loss.item()

    # 2D multiplication loss (if targets provided)
    if 'mult_2d' in aux_outputs and 'mult_2d' in aux_targets:
        mult_2d_loss = compute_mult_2d_loss(
            aux_outputs['mult_2d'],
            aux_targets['mult_2d'],
            aux_targets.get('mult_2d_mask')
        )
        loss_components['mult_2d'] = mult_2d_loss * loss_weights.w_mult_2d
        metrics['mult_2d_loss'] = mult_2d_loss.item()

    # Division policy loss (if targets provided)
    if 'division_policy' in aux_outputs and 'division_policy' in aux_targets:
        division_loss = compute_division_loss(
            aux_outputs['division_policy'],
            aux_targets['division_policy'],
            aux_targets.get('division_mask')
        )
        loss_components['division'] = division_loss * loss_weights.w_division
        metrics['division_loss'] = division_loss.item()

    # Schema validation loss (if targets provided)
    if 'schema' in aux_outputs and 'schema' in aux_targets:
        schema_loss = compute_schema_loss(
            aux_outputs['schema'],
            aux_targets['schema'],
            aux_targets.get('schema_mask')
        )
        loss_components['schema'] = schema_loss * loss_weights.w_schema
        metrics['schema_loss'] = schema_loss.item()

    # Selector loss (if targets provided)
    if 'selector' in aux_outputs and 'selector' in aux_targets:
        selector_loss = compute_selector_loss(
            aux_outputs['selector'],
            aux_targets['selector'],
            aux_targets.get('selector_mask')
        )
        loss_components['selector'] = selector_loss * loss_weights.w_selector
        metrics['selector_loss'] = selector_loss.item()

    # Combine all losses
    total_loss = sum(loss_components.values())
    metrics['total_loss'] = total_loss.item()

    return LossOutput(
        total_loss=total_loss,
        loss_components=loss_components,
        metrics=metrics
    )


# Example usage and testing
if __name__ == "__main__":
    print("=== Testing Loss Functions ===\n")

    # Test configuration
    batch_size = 4
    seq_len = 128
    vocab_size = 49180
    d_model = 512

    # Create dummy inputs
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    loss_mask = torch.rand(batch_size, seq_len) > 0.5  # Random mask
    example_weights = torch.clip(8.0 / torch.randint(1, 10, (batch_size, seq_len)).float(), 1.0, 4.0)

    # Test masked CE loss
    print("Testing compute_masked_ce_loss...")
    ce_loss = compute_masked_ce_loss(logits, targets, loss_mask, example_weights)
    print(f"  CE loss: {ce_loss.item():.4f}")
    assert ce_loss.item() > 0, "CE loss should be positive"
    print("  ✓ Masked CE loss passed\n")

    # Test format loss
    print("Testing compute_format_loss...")
    format_logits = torch.randn(batch_size, seq_len, 2)
    format_targets = torch.randint(0, 2, (batch_size, seq_len))
    format_loss = compute_format_loss(format_logits, format_targets, loss_mask)
    print(f"  Format loss: {format_loss.item():.4f}")
    print("  ✓ Format loss passed\n")

    # Test carry loss
    print("Testing compute_carry_loss...")
    carry_logits = torch.randn(batch_size, seq_len, 32)
    carry_targets = torch.randint(0, 2, (batch_size, seq_len, 32))
    carry_loss = compute_carry_loss(carry_logits, carry_targets)
    print(f"  Carry loss: {carry_loss.item():.4f}")
    print("  ✓ Carry loss passed\n")

    # Test combined loss
    print("Testing compute_loss...")
    aux_outputs = {
        'format': format_logits,
        'carry': carry_logits,
    }
    aux_targets = {
        'format': format_targets,
        'carry': carry_targets,
    }

    loss_output = compute_loss(
        logits, targets, loss_mask,
        aux_outputs, aux_targets, example_weights
    )

    print(f"  Total loss: {loss_output.total_loss.item():.4f}")
    print(f"  Components: {list(loss_output.loss_components.keys())}")
    print(f"  Metrics: {loss_output.metrics}")
    print("  ✓ Combined loss passed\n")

    print("=" * 50)
    print("✓ All loss function tests passed!")
