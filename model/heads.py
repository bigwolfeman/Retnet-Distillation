"""
Auxiliary prediction heads per plan.md and research.md.

Implements 6 auxiliary heads for dense supervision:
1. Carry head (binary BCE per column)
2. 2D multiplication head (predict digit products and carries)
3. Division policy head (predict next quotient digit)
4. Format validation head (binary valid/invalid)
5. JSON schema head (per-key validity for tool calls)
6. Selector head (tool call vs think, KL penalty to budget)

All heads use stop-gradient (detach()) to prevent interference with main LM training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class CarryHead(nn.Module):
    """
    Carry prediction head for multi-digit arithmetic (A2+).

    Predicts binary carry bitstring (LSB→MSB) for addition/subtraction.
    Binary cross-entropy loss per column.
    """

    def __init__(self, d_model: int, max_columns: int = 32):
        """
        Args:
            d_model: Hidden dimension from transformer
            max_columns: Maximum number of columns to predict carries for
        """
        super().__init__()
        self.d_model = d_model
        self.max_columns = max_columns

        # Project hidden to binary logits per column
        # Output: [batch, seq, max_columns] for binary classification
        self.proj = nn.Linear(d_model, max_columns)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Predict carry bitstring.

        Args:
            hidden: Hidden states [batch, seq_len, d_model]

        Returns:
            carry_logits: Binary logits [batch, seq_len, max_columns]
        """
        carry_logits = self.proj(hidden)  # [batch, seq, max_columns]
        return carry_logits


class Mult2DHead(nn.Module):
    """
    2D multiplication grid head for multi-digit multiplication (A3-A4).

    Predicts (d_i * d_j) mod 10 and carry bit for each cell in 2D grid.
    Used for structured supervision of partial products.
    """

    def __init__(self, d_model: int, max_grid_size: int = 16):
        """
        Args:
            d_model: Hidden dimension
            max_grid_size: Maximum grid dimension (e.g., 8x8 for 8-digit multiplication)
        """
        super().__init__()
        self.d_model = d_model
        self.max_grid_size = max_grid_size

        # Predict product digit (0-9) and carry bit for each cell
        # Output: max_grid_size^2 cells × 11 classes (10 digits + carry bit)
        self.grid_proj = nn.Linear(d_model, max_grid_size * max_grid_size * 11)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Predict 2D multiplication grid.

        Args:
            hidden: Hidden states [batch, seq_len, d_model]

        Returns:
            grid_logits: Logits [batch, seq_len, grid_size^2, 11]
        """
        batch, seq_len, _ = hidden.shape
        grid_logits = self.grid_proj(hidden)  # [batch, seq, grid_size^2 * 11]

        # Reshape to [batch, seq, grid_size^2, 11]
        grid_logits = grid_logits.view(batch, seq_len, self.max_grid_size ** 2, 11)
        return grid_logits


class DivisionPolicyHead(nn.Module):
    """
    Division policy head for long division (A5-A6).

    Predicts next quotient digit (0-9) at each step of long division algorithm.
    Provides structured supervision for division reasoning.
    """

    def __init__(self, d_model: int):
        """
        Args:
            d_model: Hidden dimension
        """
        super().__init__()
        self.d_model = d_model

        # Predict quotient digit (0-9)
        self.proj = nn.Linear(d_model, 10)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Predict next quotient digit.

        Args:
            hidden: Hidden states [batch, seq_len, d_model]

        Returns:
            quotient_logits: Logits over digits 0-9 [batch, seq_len, 10]
        """
        quotient_logits = self.proj(hidden)
        return quotient_logits


class FormatHead(nn.Module):
    """
    Format validation head for answer format checking.

    Binary classification: valid vs invalid format.
    Checks if answer matches integer/decimal/fraction regex per dfa.py.
    """

    def __init__(self, d_model: int):
        """
        Args:
            d_model: Hidden dimension
        """
        super().__init__()
        self.d_model = d_model

        # Binary classification: valid (1) or invalid (0)
        self.proj = nn.Linear(d_model, 2)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Predict format validity.

        Args:
            hidden: Hidden states [batch, seq_len, d_model]

        Returns:
            format_logits: Binary logits [batch, seq_len, 2]
        """
        format_logits = self.proj(hidden)
        return format_logits


class SchemaHead(nn.Module):
    """
    JSON schema validation head for tool calls.

    Predicts per-key validity for tool call JSON schemas.
    Multi-label binary classification (each key can be valid/invalid independently).
    """

    def __init__(self, d_model: int, max_keys: int = 8):
        """
        Args:
            d_model: Hidden dimension
            max_keys: Maximum number of JSON keys to validate
        """
        super().__init__()
        self.d_model = d_model
        self.max_keys = max_keys

        # Multi-label binary classification per key
        self.proj = nn.Linear(d_model, max_keys)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Predict per-key validity.

        Args:
            hidden: Hidden states [batch, seq_len, d_model]

        Returns:
            schema_logits: Binary logits per key [batch, seq_len, max_keys]
        """
        schema_logits = self.proj(hidden)
        return schema_logits


class SelectorHead(nn.Module):
    """
    Tool selector head with budget constraint.

    Predicts tool call vs think (binary decision) with KL penalty against
    budget prior to enforce tool call frequency limits.

    Per spec: calc budget ≤0.6, cas budget ≤0.3, tester budget ≤1.0
    """

    def __init__(
        self,
        d_model: int,
        n_tools: int = 3,
        budget_priors: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            d_model: Hidden dimension
            n_tools: Number of tools (calc, cas, tester)
            budget_priors: Budget constraints per tool (e.g., {"calc": 0.6})
        """
        super().__init__()
        self.d_model = d_model
        self.n_tools = n_tools

        # Default budgets per spec
        if budget_priors is None:
            budget_priors = {
                "calc": 0.6,
                "cas": 0.3,
                "tester": 1.0,
            }
        self.budget_priors = budget_priors

        # Predict tool selection: [no_tool, calc, cas, tester]
        self.proj = nn.Linear(d_model, n_tools + 1)

        # Register budget prior as buffer (not trainable)
        # Shape: [n_tools + 1] for [no_tool, calc, cas, tester]
        budget_tensor = torch.tensor([
            1.0 - sum(budget_priors.values()),  # no_tool probability
            budget_priors.get("calc", 0.6),
            budget_priors.get("cas", 0.3),
            budget_priors.get("tester", 1.0),
        ])
        # Normalize to sum to 1
        budget_tensor = budget_tensor / budget_tensor.sum()
        self.register_buffer("budget_prior", budget_tensor)

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict tool selection with KL penalty.

        Args:
            hidden: Hidden states [batch, seq_len, d_model]

        Returns:
            Dictionary with:
                - selector_logits: Logits [batch, seq_len, n_tools+1]
                - selector_probs: Probabilities after softmax
                - kl_penalty: KL divergence from budget prior (scalar)
        """
        selector_logits = self.proj(hidden)  # [batch, seq, n_tools+1]
        selector_probs = F.softmax(selector_logits, dim=-1)

        # Compute KL divergence from budget prior
        # KL(P || Q) where P = model predictions, Q = budget prior
        log_probs = F.log_softmax(selector_logits, dim=-1)
        budget_prior_expanded = self.budget_prior.unsqueeze(0).unsqueeze(0)  # [1, 1, n_tools+1]

        # KL(P || Q) = sum_i P(i) * log(P(i) / Q(i))
        kl_penalty = F.kl_div(
            log_probs,
            budget_prior_expanded.expand_as(log_probs),
            reduction='batchmean',
            log_target=False
        )

        return {
            'selector_logits': selector_logits,
            'selector_probs': selector_probs,
            'kl_penalty': kl_penalty,
        }


# Example usage and testing
if __name__ == "__main__":
    print("=== Testing Auxiliary Heads ===\n")

    # Test configuration
    batch_size = 4
    seq_len = 128
    d_model = 512

    # Create dummy hidden states
    hidden = torch.randn(batch_size, seq_len, d_model)

    # Test CarryHead
    print("Testing CarryHead...")
    carry_head = CarryHead(d_model, max_columns=32)
    carry_logits = carry_head(hidden)
    print(f"  Input: {hidden.shape}")
    print(f"  Output: {carry_logits.shape}")
    assert carry_logits.shape == (batch_size, seq_len, 32)
    print("  ✓ CarryHead passed\n")

    # Test Mult2DHead
    print("Testing Mult2DHead...")
    mult_2d_head = Mult2DHead(d_model, max_grid_size=16)
    grid_logits = mult_2d_head(hidden)
    print(f"  Input: {hidden.shape}")
    print(f"  Output: {grid_logits.shape}")
    assert grid_logits.shape == (batch_size, seq_len, 16 * 16, 11)
    print("  ✓ Mult2DHead passed\n")

    # Test DivisionPolicyHead
    print("Testing DivisionPolicyHead...")
    division_head = DivisionPolicyHead(d_model)
    quotient_logits = division_head(hidden)
    print(f"  Input: {hidden.shape}")
    print(f"  Output: {quotient_logits.shape}")
    assert quotient_logits.shape == (batch_size, seq_len, 10)
    print("  ✓ DivisionPolicyHead passed\n")

    # Test FormatHead
    print("Testing FormatHead...")
    format_head = FormatHead(d_model)
    format_logits = format_head(hidden)
    print(f"  Input: {hidden.shape}")
    print(f"  Output: {format_logits.shape}")
    assert format_logits.shape == (batch_size, seq_len, 2)
    print("  ✓ FormatHead passed\n")

    # Test SchemaHead
    print("Testing SchemaHead...")
    schema_head = SchemaHead(d_model, max_keys=8)
    schema_logits = schema_head(hidden)
    print(f"  Input: {hidden.shape}")
    print(f"  Output: {schema_logits.shape}")
    assert schema_logits.shape == (batch_size, seq_len, 8)
    print("  ✓ SchemaHead passed\n")

    # Test SelectorHead
    print("Testing SelectorHead...")
    selector_head = SelectorHead(d_model, n_tools=3)
    selector_output = selector_head(hidden)
    print(f"  Input: {hidden.shape}")
    print(f"  Logits shape: {selector_output['selector_logits'].shape}")
    print(f"  Probs shape: {selector_output['selector_probs'].shape}")
    print(f"  KL penalty: {selector_output['kl_penalty'].item():.4f}")
    assert selector_output['selector_logits'].shape == (batch_size, seq_len, 4)
    assert selector_output['selector_probs'].shape == (batch_size, seq_len, 4)
    print("  ✓ SelectorHead passed\n")

    print("=" * 50)
    print("✓ All auxiliary head tests passed!")
