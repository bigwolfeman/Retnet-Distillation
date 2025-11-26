#!/usr/bin/env python3
"""
Quick 10-step vLLM Teacher Integration Test

Tests the VLLMTeacherClient with a vLLM server.
Runs a minimal distillation training loop to verify:
- Connection works
- Logits are fetched successfully
- Loss computation works
- No errors or crashes
- Actual latency measurements

Usage:
    python scripts/test_vllm_integration.py

Results saved to: vllm_integration_test.json
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.distillation.vllm_teacher_client import VLLMTeacherClient


class SimpleStudentModel(nn.Module):
    """Minimal student model for testing (350M params)."""

    def __init__(self, vocab_size: int = 128256, d_model: int = 1024, n_layers: int = 12):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=d_model * 4,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        x = self.embed(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def generate_synthetic_data(num_samples: int = 10, seq_len: int = 32, vocab_size: int = 128256) -> List[torch.Tensor]:
    """Generate synthetic input sequences for testing."""
    torch.manual_seed(42)

    # Generate random token sequences
    # Avoid special tokens (0-100) and use common tokens (100-10000)
    sequences = []
    for _ in range(num_samples):
        seq = torch.randint(100, 10000, (seq_len,), dtype=torch.long)
        sequences.append(seq)

    return sequences


def compute_distillation_loss(
    student_logits: torch.Tensor,
    teacher_indices: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute distillation loss combining sparse KD and cross-entropy.

    Args:
        student_logits: (batch, seq, vocab)
        teacher_indices: (batch, teacher_seq, k)
        teacher_logprobs: (batch, teacher_seq, k)
        labels: (batch, seq)
        temperature: Temperature for KD
        alpha: Weight for distillation loss

    Returns:
        (total_loss, distill_loss, ce_loss)
    """
    batch_size, seq_len, vocab_size = student_logits.shape
    teacher_seq_len = teacher_indices.shape[1]

    # Handle sequence length mismatch (teacher includes BOS token)
    # Teacher typically has seq_len + 1 due to BOS prepending
    # We skip the first teacher position (BOS) and align with student
    if teacher_seq_len > seq_len:
        # Skip BOS (position 0) in teacher logprobs
        teacher_indices = teacher_indices[:, 1:, :]  # (batch, seq, k)
        teacher_logprobs = teacher_logprobs[:, 1:, :]  # (batch, seq, k)
    elif teacher_seq_len < seq_len:
        # Truncate student to match teacher (rare)
        student_logits = student_logits[:, :teacher_seq_len, :]
        labels = labels[:, :teacher_seq_len]

    # === Distillation Loss (Sparse KL Divergence) ===

    # Gather student logits at teacher's top-k positions
    student_topk_logits = torch.gather(
        student_logits,
        dim=-1,
        index=teacher_indices
    )  # (batch, seq, k)

    # Convert to probabilities with temperature
    T = temperature
    student_topk_log_probs = F.log_softmax(student_topk_logits / T, dim=-1)
    teacher_topk_probs = F.softmax(teacher_logprobs / T, dim=-1)

    # KL divergence
    kl_div = F.kl_div(
        student_topk_log_probs,
        teacher_topk_probs,
        reduction='none',
        log_target=False,
    )  # (batch, seq, k)

    # Sum over k, mean over batch and sequence
    kl_div = kl_div.sum(dim=-1).mean()  # scalar

    # Scale by T^2
    distill_loss = kl_div * (T ** 2)

    # === Cross-Entropy Loss ===

    ce_loss = F.cross_entropy(
        student_logits.view(-1, vocab_size),
        labels.view(-1),
        reduction='mean',
    )

    # === Combined Loss ===

    total_loss = alpha * distill_loss + (1 - alpha) * ce_loss

    return total_loss, distill_loss, ce_loss


def run_integration_test(
    vllm_url: str = "http://localhost:8080",
    vllm_model: str = "meta-llama/Llama-3.2-1B-Instruct",
    vllm_api_key: str = "token-abc123",
    num_steps: int = 10,
    batch_size: int = 2,
    seq_len: int = 32,
    topk: int = 128,
    temperature: float = 2.0,
    alpha: float = 0.5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """
    Run integration test with vLLM teacher server.

    Returns:
        Dict with test results and metrics
    """
    print("=" * 80)
    print("vLLM Teacher Integration Test")
    print("=" * 80)
    print(f"Server: {vllm_url}")
    print(f"Model: {vllm_model}")
    print(f"Steps: {num_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Top-k: {topk}")
    print(f"Device: {device}")
    print("=" * 80)

    results = {
        "config": {
            "vllm_url": vllm_url,
            "vllm_model": vllm_model,
            "num_steps": num_steps,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "topk": topk,
            "temperature": temperature,
            "alpha": alpha,
            "device": device,
        },
        "metrics": {
            "step_times": [],
            "teacher_fetch_times": [],
            "forward_times": [],
            "backward_times": [],
            "losses": [],
            "distill_losses": [],
            "ce_losses": [],
        },
        "errors": [],
        "success": False,
        "timestamp": datetime.utcnow().isoformat(),
    }

    try:
        # 1. Initialize student model
        print("\n[1/5] Initializing student model...")
        student = SimpleStudentModel(vocab_size=128256, d_model=1024, n_layers=12)
        student.to(device)
        student.train()

        param_count = student.count_parameters()
        print(f"  Student params: {param_count / 1e6:.1f}M")
        results["student_params"] = param_count

        # 2. Initialize vLLM teacher client
        print(f"\n[2/5] Connecting to vLLM server at {vllm_url}...")
        teacher_client = VLLMTeacherClient(
            base_url=vllm_url,
            model=vllm_model,
            api_key=vllm_api_key,
            timeout=30.0,
            max_retries=3,
        )

        # Health check
        if teacher_client.health_check():
            print("  ✓ Server is healthy")
        else:
            print("  ✗ Server health check failed!")
            results["errors"].append("Server health check failed")
            return results

        # Get model info
        try:
            model_info = teacher_client.get_model_info()
            print(f"  ✓ Model info: {model_info.get('data', [{}])[0].get('id', 'unknown')}")
        except Exception as e:
            print(f"  ⚠ Could not get model info: {e}")

        # 3. Generate synthetic data
        print(f"\n[3/5] Generating {num_steps} batches of synthetic data...")
        all_batches = []
        for step in range(num_steps):
            batch = generate_synthetic_data(num_samples=batch_size, seq_len=seq_len)
            all_batches.append(batch)
        print(f"  ✓ Generated {num_steps} batches")

        # 4. Initialize optimizer
        print("\n[4/5] Setting up optimizer...")
        optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
        print("  ✓ Optimizer ready")

        # 5. Run training steps
        print(f"\n[5/5] Running {num_steps} training steps...")
        print("=" * 80)

        for step_idx in range(num_steps):
            step_start = time.perf_counter()

            # Get batch
            batch = all_batches[step_idx]
            input_ids = torch.stack(batch).to(device)  # (batch, seq)
            labels = input_ids.clone()

            print(f"\nStep {step_idx + 1}/{num_steps}:")
            print(f"  Input shape: {input_ids.shape}")

            # === Teacher Fetch ===
            teacher_start = time.perf_counter()
            try:
                # Convert to CPU list for API
                input_ids_list = input_ids.cpu().tolist()

                # Get teacher logprobs
                teacher_results = teacher_client.get_prompt_logprobs(
                    input_ids=input_ids_list,
                    topk=topk,
                    temperature=temperature,
                )

                # Process results into tensors
                indices_list = []
                logprobs_list = []

                for result in teacher_results:
                    seq_len_actual = len(result['indices'])
                    max_k = max(len(pos) for pos in result['indices']) if result['indices'] else topk

                    # Pad positions
                    indices_padded = []
                    logprobs_padded = []

                    for pos in range(seq_len_actual):
                        pos_indices = result['indices'][pos]
                        pos_logprobs = result['logprobs'][pos]

                        # Pad to max_k
                        if len(pos_indices) < max_k:
                            pos_indices = pos_indices + [0] * (max_k - len(pos_indices))
                            pos_logprobs = pos_logprobs + [float('-inf')] * (max_k - len(pos_logprobs))

                        indices_padded.append(pos_indices)
                        logprobs_padded.append(pos_logprobs)

                    indices_list.append(torch.tensor(indices_padded, dtype=torch.long))
                    logprobs_list.append(torch.tensor(logprobs_padded, dtype=torch.float32))

                teacher_indices = torch.stack(indices_list).to(device)
                teacher_logprobs = torch.stack(logprobs_list).to(device)

                teacher_time = time.perf_counter() - teacher_start
                print(f"  Teacher fetch: {teacher_time * 1000:.1f}ms")
                print(f"    Logprobs shape: {teacher_logprobs.shape}")

            except Exception as e:
                error_msg = f"Teacher fetch failed at step {step_idx + 1}: {e}"
                print(f"  ✗ {error_msg}")
                results["errors"].append(error_msg)
                raise

            # === Student Forward ===
            forward_start = time.perf_counter()
            try:
                student_logits = student(input_ids)
                forward_time = time.perf_counter() - forward_start
                print(f"  Student forward: {forward_time * 1000:.1f}ms")
                print(f"    Logits shape: {student_logits.shape}")
            except Exception as e:
                error_msg = f"Student forward failed at step {step_idx + 1}: {e}"
                print(f"  ✗ {error_msg}")
                results["errors"].append(error_msg)
                raise

            # === Loss Computation ===
            try:
                total_loss, distill_loss, ce_loss = compute_distillation_loss(
                    student_logits=student_logits,
                    teacher_indices=teacher_indices,
                    teacher_logprobs=teacher_logprobs,
                    labels=labels,
                    temperature=temperature,
                    alpha=alpha,
                )
                print(f"  Losses:")
                print(f"    Total: {total_loss.item():.4f}")
                print(f"    Distill: {distill_loss.item():.4f}")
                print(f"    CE: {ce_loss.item():.4f}")
            except Exception as e:
                error_msg = f"Loss computation failed at step {step_idx + 1}: {e}"
                print(f"  ✗ {error_msg}")
                results["errors"].append(error_msg)
                raise

            # === Backward ===
            backward_start = time.perf_counter()
            try:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                backward_time = time.perf_counter() - backward_start
                print(f"  Backward: {backward_time * 1000:.1f}ms")
            except Exception as e:
                error_msg = f"Backward pass failed at step {step_idx + 1}: {e}"
                print(f"  ✗ {error_msg}")
                results["errors"].append(error_msg)
                raise

            step_time = time.perf_counter() - step_start
            print(f"  Total step time: {step_time * 1000:.1f}ms")

            # Record metrics
            results["metrics"]["step_times"].append(step_time)
            results["metrics"]["teacher_fetch_times"].append(teacher_time)
            results["metrics"]["forward_times"].append(forward_time)
            results["metrics"]["backward_times"].append(backward_time)
            results["metrics"]["losses"].append(total_loss.item())
            results["metrics"]["distill_losses"].append(distill_loss.item())
            results["metrics"]["ce_losses"].append(ce_loss.item())

        # Close teacher client
        teacher_client.close()

        # Mark success
        results["success"] = True

        print("\n" + "=" * 80)
        print("✓ Test completed successfully!")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"✗ Test failed with error: {e}")
        print("=" * 80)
        results["errors"].append(str(e))
        import traceback
        results["traceback"] = traceback.format_exc()

    # Compute summary statistics
    if results["metrics"]["step_times"]:
        metrics = results["metrics"]
        results["summary"] = {
            "avg_step_time_ms": sum(metrics["step_times"]) / len(metrics["step_times"]) * 1000,
            "avg_teacher_fetch_ms": sum(metrics["teacher_fetch_times"]) / len(metrics["teacher_fetch_times"]) * 1000,
            "avg_forward_ms": sum(metrics["forward_times"]) / len(metrics["forward_times"]) * 1000,
            "avg_backward_ms": sum(metrics["backward_times"]) / len(metrics["backward_times"]) * 1000,
            "avg_loss": sum(metrics["losses"]) / len(metrics["losses"]),
            "avg_distill_loss": sum(metrics["distill_losses"]) / len(metrics["distill_losses"]),
            "avg_ce_loss": sum(metrics["ce_losses"]) / len(metrics["ce_losses"]),
            "throughput_tokens_per_sec": (batch_size * seq_len * num_steps) / sum(metrics["step_times"]),
            "teacher_fetch_percentage": sum(metrics["teacher_fetch_times"]) / sum(metrics["step_times"]) * 100,
        }

    return results


def print_summary(results: Dict[str, Any]):
    """Print summary of test results."""
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    if results["success"]:
        print("Status: ✓ SUCCESS")
    else:
        print("Status: ✗ FAILED")
        if results["errors"]:
            print("\nErrors:")
            for error in results["errors"]:
                print(f"  - {error}")

    if "summary" in results:
        summary = results["summary"]
        print("\nPerformance Metrics:")
        print(f"  Avg step time:        {summary['avg_step_time_ms']:.1f}ms")
        print(f"  Avg teacher fetch:    {summary['avg_teacher_fetch_ms']:.1f}ms ({summary['teacher_fetch_percentage']:.1f}%)")
        print(f"  Avg student forward:  {summary['avg_forward_ms']:.1f}ms")
        print(f"  Avg backward:         {summary['avg_backward_ms']:.1f}ms")
        print(f"  Throughput:           {summary['throughput_tokens_per_sec']:.1f} tokens/sec")

        print("\nLoss Metrics:")
        print(f"  Avg total loss:       {summary['avg_loss']:.4f}")
        print(f"  Avg distill loss:     {summary['avg_distill_loss']:.4f}")
        print(f"  Avg CE loss:          {summary['avg_ce_loss']:.4f}")

    print("=" * 80)


def save_results(results: Dict[str, Any], output_path: str = "vllm_integration_test.json"):
    """Save results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="vLLM Teacher Integration Test")
    parser.add_argument("--url", default="http://localhost:8080", help="vLLM server URL")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct", help="vLLM model name")
    parser.add_argument("--api-key", default="token-abc123", help="vLLM API key")
    parser.add_argument("--steps", type=int, default=10, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length")
    parser.add_argument("--topk", type=int, default=128, help="Top-k logits")
    parser.add_argument("--output", default="vllm_integration_test.json", help="Output JSON file")

    args = parser.parse_args()

    # Run test
    results = run_integration_test(
        vllm_url=args.url,
        vllm_model=args.model,
        vllm_api_key=args.api_key,
        num_steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        topk=args.topk,
    )

    # Print summary
    print_summary(results)

    # Save results
    save_results(results, args.output)

    # Exit with appropriate code
    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()
