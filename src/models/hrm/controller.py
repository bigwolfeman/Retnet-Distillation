"""
HRM (Hierarchical Recurrent Memory) Controller.

The HRM controller maintains a lightweight recurrent state that:
1. Summarizes RetNet hidden states from previous steps
2. Generates queries for retrieval
3. Provides input to the ACT halting head

Uses a simple GRU-like or RWKV-style recurrent cell for efficiency.

References:
- Adaptive Computation Time (Graves): https://arxiv.org/abs/1603.08983
- RWKV architecture for efficient recurrence
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class HRMController(nn.Module):
    """
    Lightweight recurrent controller for adaptive computation.

    Takes RetNet hidden state summaries and maintains a recurrent state
    across ACT steps. The controller state is used for:
    - ACT halting decisions
    - Retrieval query generation
    - Step-wise computation routing

    Args:
        d_model: Model dimension (matches RetNet output)
        d_controller: Controller hidden dimension (default: same as d_model)
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        d_controller: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_controller = d_controller if d_controller is not None else d_model
        self.dropout = dropout

        # Input projection (from RetNet summary to controller dim)
        self.input_proj = nn.Linear(d_model, self.d_controller)

        # GRU-like recurrent cell
        # Gates: reset (r), update (z)
        self.gate_proj = nn.Linear(self.d_controller + self.d_controller, 2 * self.d_controller)
        self.candidate_proj = nn.Linear(self.d_controller + self.d_controller, self.d_controller)

        # Output projections
        self.query_proj = nn.Linear(self.d_controller, d_model)  # For retrieval queries
        self.state_proj = nn.Linear(self.d_controller, d_model)  # For general output

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

        # Layer norm for stability
        self.ln = nn.LayerNorm(self.d_controller)

    def forward(
        self,
        retnet_summary: torch.Tensor,
        prev_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through HRM controller.

        Args:
            retnet_summary: Summary of RetNet hidden states [batch, d_model]
                           (e.g., mean pooling over sequence)
            prev_state: Previous controller state [batch, d_controller]
                       If None, initialized to zeros

        Returns:
            tuple of (new_state, query_vector, output_state):
                - new_state: Updated controller state [batch, d_controller]
                - query_vector: Query for retrieval [batch, d_model]
                - output_state: General output representation [batch, d_model]
        """
        batch_size = retnet_summary.shape[0]

        # Initialize previous state if None
        if prev_state is None:
            prev_state = torch.zeros(
                batch_size, self.d_controller,
                device=retnet_summary.device,
                dtype=retnet_summary.dtype
            )

        # Project input
        x = self.input_proj(retnet_summary)  # [batch, d_controller]
        x = self.dropout_layer(x)

        # GRU-style update
        # Concatenate input and previous state
        combined = torch.cat([x, prev_state], dim=-1)  # [batch, 2*d_controller]

        # Compute gates
        gates = self.gate_proj(combined)  # [batch, 2*d_controller]
        reset_gate, update_gate = gates.chunk(2, dim=-1)
        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)

        # Compute candidate state
        reset_prev = reset_gate * prev_state
        candidate_input = torch.cat([x, reset_prev], dim=-1)
        candidate = self.candidate_proj(candidate_input)
        candidate = torch.tanh(candidate)

        # Update state
        new_state = update_gate * prev_state + (1 - update_gate) * candidate
        new_state = self.ln(new_state)

        # Generate outputs
        query_vector = self.query_proj(new_state)  # For retrieval
        output_state = self.state_proj(new_state)  # General output

        return new_state, query_vector, output_state

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Initialize controller state.

        Args:
            batch_size: Batch size
            device: Device to create tensor on

        Returns:
            Initial state [batch, d_controller]
        """
        return torch.zeros(batch_size, self.d_controller, device=device)


class HRMSummarizer(nn.Module):
    """
    Summarizes RetNet sequence outputs for HRM controller input.

    Supports multiple summarization strategies:
    - mean: Mean pooling over sequence
    - max: Max pooling over sequence
    - last: Last token representation
    - attention: Learnable attention-weighted pooling

    Args:
        d_model: Model dimension
        strategy: Summarization strategy ('mean', 'max', 'last', 'attention')
    """

    def __init__(self, d_model: int, strategy: str = 'mean'):
        super().__init__()
        self.d_model = d_model
        self.strategy = strategy

        if strategy == 'attention':
            # Learnable attention for weighted pooling
            self.attention_proj = nn.Linear(d_model, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Summarize sequence into single vector.

        Args:
            hidden_states: Sequence representations [batch, seq_len, d_model]
            attention_mask: Optional mask [batch, seq_len]

        Returns:
            Summary vector [batch, d_model]
        """
        if self.strategy == 'mean':
            # Mean pooling
            if attention_mask is not None:
                # Masked mean
                mask = attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
                masked_hidden = hidden_states * mask
                sum_hidden = masked_hidden.sum(dim=1)
                sum_mask = mask.sum(dim=1).clamp(min=1e-9)
                return sum_hidden / sum_mask
            else:
                return hidden_states.mean(dim=1)

        elif self.strategy == 'max':
            # Max pooling
            if attention_mask is not None:
                # Masked max
                mask = attention_mask.unsqueeze(-1)
                masked_hidden = hidden_states.masked_fill(~mask.bool(), float('-inf'))
                return masked_hidden.max(dim=1)[0]
            else:
                return hidden_states.max(dim=1)[0]

        elif self.strategy == 'last':
            # Last token
            if attention_mask is not None:
                # Get last valid position per sequence
                lengths = attention_mask.sum(dim=1).long() - 1
                batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
                return hidden_states[batch_indices, lengths]
            else:
                return hidden_states[:, -1]

        elif self.strategy == 'attention':
            # Learnable attention-weighted pooling
            # Compute attention scores
            attn_scores = self.attention_proj(hidden_states).squeeze(-1)  # [batch, seq_len]

            # Apply mask if provided
            if attention_mask is not None:
                attn_scores = attn_scores.masked_fill(~attention_mask.bool(), float('-inf'))

            # Softmax
            attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # [batch, seq_len, 1]

            # Weighted sum
            return (hidden_states * attn_weights).sum(dim=1)

        else:
            raise ValueError(f"Unknown summarization strategy: {self.strategy}")


def test_hrm_controller():
    """Test HRM controller implementation."""
    print("Testing HRM Controller...")

    # Configuration
    batch_size = 2
    seq_len = 1024
    d_model = 2816
    d_controller = 1024

    # Create controller
    controller = HRMController(d_model=d_model, d_controller=d_controller, dropout=0.0)

    # Create summarizer
    summarizer = HRMSummarizer(d_model=d_model, strategy='mean')

    print(f"\nConfiguration:")
    print(f"  d_model: {d_model}")
    print(f"  d_controller: {d_controller}")

    # Test 1: Summarization
    print("\n[Test 1] Summarization")
    hidden_states = torch.randn(batch_size, seq_len, d_model)
    attention_mask = torch.ones(batch_size, seq_len).bool()

    summary = summarizer(hidden_states, attention_mask)
    assert summary.shape == (batch_size, d_model), f"Shape mismatch: {summary.shape}"
    print(f"  Input: {hidden_states.shape}")
    print(f"  Summary: {summary.shape}")
    print("  [PASS]")

    # Test 2: Single step
    print("\n[Test 2] Single controller step")
    new_state, query, output = controller(summary, prev_state=None)

    assert new_state.shape == (batch_size, d_controller)
    assert query.shape == (batch_size, d_model)
    assert output.shape == (batch_size, d_model)
    print(f"  New state: {new_state.shape}")
    print(f"  Query vector: {query.shape}")
    print(f"  Output: {output.shape}")
    print("  [PASS]")

    # Test 3: Multiple steps (ACT simulation)
    print("\n[Test 3] Multiple ACT steps")
    num_steps = 5
    state = None

    for step in range(num_steps):
        # Simulate new RetNet summary each step
        step_summary = torch.randn(batch_size, d_model)

        # Controller update
        state, query, output = controller(step_summary, prev_state=state)

        print(f"  Step {step+1}: state={state.shape}, query={query.shape}")

    print("  [PASS]")

    # Test 4: Different summarization strategies
    print("\n[Test 4] Summarization strategies")
    strategies = ['mean', 'max', 'last', 'attention']

    for strat in strategies:
        summarizer_test = HRMSummarizer(d_model=d_model, strategy=strat)
        summary_test = summarizer_test(hidden_states, attention_mask)
        assert summary_test.shape == (batch_size, d_model)
        print(f"  Strategy '{strat}': {summary_test.shape} [PASS]")

    print("\n" + "="*50)
    print("[PASS] All HRM Controller tests passed!")
    print("="*50)

    # Summary
    print("\nSummary:")
    param_count = sum(p.numel() for p in controller.parameters())
    print(f"  Controller parameters: {param_count / 1e6:.2f}M")
    print(f"  d_controller: {d_controller}")
    print(f"  Recurrent cell: GRU-style")


if __name__ == "__main__":
    test_hrm_controller()
