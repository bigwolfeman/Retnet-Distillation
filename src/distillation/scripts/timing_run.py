"""
M3 CRITICAL GATE: 1k-step timing run to validate training timeline feasibility.

This script performs a 1000-step timing test with full training loop to project
wall-clock time for 60k-80k step training. If projection exceeds 33 days, we must
reduce model size (500M → 350M) and retry.

Tasks: T052-T055 (US1)

Usage:
    python -m src.distillation.scripts.timing_run --variant 350M --output timing_report_350M.json
    python -m src.distillation.scripts.timing_run --variant 500M --output timing_report_500M.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.distillation.student_config import create_student_config, RetNetStudentConfig
from src.distillation.dummy_data import DummyDataGenerator
from src.distillation.dataset import SimpleDataLoader
from src.distillation.optimizer import create_optimizer, create_scheduler
from src.distillation.telemetry import TelemetryLogger, OutputSink, TrainingMetrics
from src.distillation.losses import SparseKLLoss
from src.models.retnet.backbone import RetNetBackbone
from src.distillation.vllm_teacher_client import VLLMTeacherClient


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class MockTeacherClient:
    """Mock teacher client that returns dummy top-k logits for testing.

    This avoids needing a real teacher server for the timing run.
    """

    def __init__(self, vocab_size: int = 128256, topk: int = 128):
        """Initialize mock teacher client.

        Args:
            vocab_size: Vocabulary size for sampling indices
            topk: Number of top-k logits to return
        """
        self.vocab_size = vocab_size
        self.topk = topk
        logger.info(f"MockTeacherClient initialized: vocab_size={vocab_size}, topk={topk}")

    def get_teacher_logits(
        self,
        input_ids: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate dummy teacher logits for timing test.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            device: Device to place tensors on

        Returns:
            Tuple of (indices, values, other_mass)
            - indices: [batch_size, seq_len, k] top-k token indices
            - values: [batch_size, seq_len, k] top-k logit values
            - other_mass: [batch_size, seq_len, 1] probability mass for other tokens
        """
        batch_size, seq_len = input_ids.shape

        # Generate random top-k indices
        indices = torch.randint(
            0, self.vocab_size,
            (batch_size, seq_len, self.topk),
            device=device,
            dtype=torch.long
        )

        # Generate random logit values (simulating teacher distribution)
        # Use realistic range: [-10, 10] for logits
        values = torch.randn(
            batch_size, seq_len, self.topk,
            device=device,
            dtype=torch.float32
        ) * 3.0  # Scale to get reasonable range

        # Other mass: random probability [0, 1] for tail distribution
        # Typically small since top-k captures most probability mass
        other_mass = torch.rand(
            batch_size, seq_len, 1,
            device=device,
            dtype=torch.float32
        ) * 0.1  # Small tail mass

        return indices, values, other_mass


class RealTeacherWrapper:
    """Wrapper around VLLMTeacherClient that provides the same interface as MockTeacherClient."""

    def __init__(
        self,
        teacher_url: str,
        model: str = "meta-llama/Llama-3.2-1B-Instruct",
        vocab_size: int = 128256,
        topk: int = 128,
        api_key: Optional[str] = None,
    ):
        """Initialize real teacher wrapper.

        Args:
            teacher_url: URL to vLLM teacher server
            model: Model identifier
            vocab_size: Vocabulary size
            topk: Number of top-k logits to return
            api_key: Optional API key for authentication
        """
        self.vocab_size = vocab_size
        self.topk = topk
        self.client = VLLMTeacherClient(
            base_url=teacher_url,
            model=model,
            api_key=api_key,
            timeout=60.0,  # Longer timeout for network latency
        )

        # Test connection
        if not self.client.health_check():
            logger.warning(f"Health check failed for {teacher_url}. Server may be down.")
        else:
            logger.info(f"RealTeacherWrapper initialized: url={teacher_url}, model={model}")

    def get_teacher_logits(
        self,
        input_ids: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get teacher logits from real vLLM server.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            device: Device to place tensors on

        Returns:
            Tuple of (indices, values, other_mass)
            - indices: [batch_size, seq_len, k] top-k token indices
            - values: [batch_size, seq_len, k] top-k logit values
            - other_mass: [batch_size, seq_len, 1] probability mass for other tokens
        """
        batch_size, seq_len = input_ids.shape

        # Convert to list format for vLLM API
        input_ids_list = input_ids.cpu().tolist()

        # Query vLLM server
        results = self.client.get_prompt_logprobs(
            input_ids=input_ids_list,
            topk=self.topk,
            temperature=1.0,
        )

        # Process results into tensor format
        # results is a list of dicts with 'indices' and 'logprobs' lists
        all_indices = []
        all_values = []

        for result in results:
            # result['indices'] is List[List[int]] with shape (seq_len, k)
            # result['logprobs'] is List[List[float]] with shape (seq_len, k)
            seq_indices = result['indices']
            seq_logprobs = result['logprobs']

            # Pad to ensure all positions have exactly topk entries
            padded_indices = []
            padded_logprobs = []

            for pos_indices, pos_logprobs in zip(seq_indices, seq_logprobs):
                # Handle BOS token (empty arrays)
                if len(pos_indices) == 0:
                    # Use random fallback for BOS
                    pos_indices = [0] * self.topk
                    pos_logprobs = [-10.0] * self.topk

                # Pad if needed
                if len(pos_indices) < self.topk:
                    padding_needed = self.topk - len(pos_indices)
                    pos_indices = pos_indices + [0] * padding_needed
                    pos_logprobs = pos_logprobs + [-20.0] * padding_needed
                elif len(pos_indices) > self.topk:
                    pos_indices = pos_indices[:self.topk]
                    pos_logprobs = pos_logprobs[:self.topk]

                padded_indices.append(pos_indices)
                padded_logprobs.append(pos_logprobs)

            all_indices.append(padded_indices)
            all_values.append(padded_logprobs)

        # Convert to tensors
        indices = torch.tensor(all_indices, dtype=torch.long, device=device)
        values = torch.tensor(all_values, dtype=torch.float32, device=device)

        # Other mass: assume small tail (5% of probability mass not in top-k)
        other_mass = torch.full(
            (batch_size, seq_len, 1),
            0.05,
            dtype=torch.float32,
            device=device
        )

        return indices, values, other_mass

    def close(self):
        """Close the client connection."""
        self.client.close()


class SimpleStudentWrapper(nn.Module):
    """Wrapper around RetNetBackbone to add language modeling head.

    Provides a complete student model with:
    - RetNet backbone (from src/models/retnet/backbone.py)
    - Language modeling head (vocab projection)
    """

    def __init__(self, config: RetNetStudentConfig):
        """Initialize student model wrapper.

        Args:
            config: Student model configuration
        """
        super().__init__()

        # Create RetNet backbone
        backbone_args = config.to_retnet_backbone_args()
        self.backbone = RetNetBackbone(**backbone_args)

        # Language modeling head (projects hidden states to vocab)
        # Tied with input embeddings if config.tie_word_embeddings
        if config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            self.lm_head.weight = self.backbone.embed.weight
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.config = config
        logger.info(f"SimpleStudentWrapper initialized: variant={config.variant}")
        logger.info(f"  d_model={config.d_model}, n_layers={config.n_layers}, n_heads={config.n_heads}")
        logger.info(f"  tie_word_embeddings={config.tie_word_embeddings}")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]

        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        # Get hidden states from backbone
        hidden_states = self.backbone.forward_train(input_ids)

        # Project to vocabulary
        logits = self.lm_head(hidden_states)

        return logits

    def count_parameters(self) -> int:
        """Count total trainable parameters.

        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_dummy_data(
    output_dir: Path,
    num_examples: int = 1100,  # 1000 train + 100 buffer
    max_length: int = 4096,
    vocab_size: int = 128256,
) -> Path:
    """Create dummy synthetic data for timing test.

    Args:
        output_dir: Directory to save dummy data
        num_examples: Number of examples to generate
        max_length: Maximum sequence length
        vocab_size: Vocabulary size

    Returns:
        Path to generated data file
    """
    logger.info("Generating dummy synthetic data...")
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = DummyDataGenerator(
        vocab_size=vocab_size,
        max_length=max_length,
        min_length=max_length,  # Fixed length for consistent timing
        seed=42,
    )

    data_path = output_dir / "timing_test.jsonl"
    generator.generate_pretokenized(
        output_path=data_path,
        num_examples=num_examples,
        variable_length=False,  # Fixed length for timing accuracy
    )

    logger.info(f"Dummy data generated: {data_path} ({num_examples} examples)")
    return data_path


def run_timing_test(
    variant: str = "350M",
    num_steps: int = 1000,
    log_interval: int = 10,
    output_path: Optional[Path] = None,
    use_real_teacher: bool = False,
    teacher_url: str = "http://localhost:8080",
    teacher_model: str = "meta-llama/Llama-3.2-1B-Instruct",
    teacher_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Run 1000-step timing test with full training loop.

    Args:
        variant: Model variant ("350M" or "500M")
        num_steps: Number of training steps to run (default: 1000)
        log_interval: Steps between logging (default: 10)
        output_path: Path to save timing report JSON
        use_real_teacher: Whether to use real vLLM teacher server (default: False)
        teacher_url: URL to vLLM teacher server (default: http://localhost:8080)
        teacher_model: Teacher model identifier (default: meta-llama/Llama-3.2-1B-Instruct)
        teacher_api_key: Optional API key for teacher server authentication

    Returns:
        Timing report dictionary with M3 gate analysis
    """
    logger.info("=" * 80)
    logger.info(f"M3 CRITICAL GATE: {num_steps}-step timing run")
    logger.info(f"Variant: {variant}")
    logger.info(f"Teacher: {'Real (' + teacher_url + ')' if use_real_teacher else 'Mock'}")
    logger.info("=" * 80)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("⚠️  CUDA not available! Timing will not be accurate.")
    logger.info(f"Device: {device}")

    # Create student config
    config = create_student_config(variant=variant)
    logger.info(f"\nStudent config: {variant}")
    logger.info(f"  d_model: {config.d_model}")
    logger.info(f"  n_layers: {config.n_layers}")
    logger.info(f"  n_heads: {config.n_heads}")
    logger.info(f"  estimated_params: {config.estimate_param_count():,}")

    # Create student model
    logger.info("\nInitializing student model...")
    model = SimpleStudentWrapper(config).to(device)
    actual_params = model.count_parameters()
    logger.info(f"✓ Student model created: {actual_params:,} parameters")

    # Validate parameter count
    try:
        config.validate_actual_params(actual_params)
        logger.info(f"✓ Parameter count within target range: {config.target_param_count_range}")
    except AssertionError as e:
        logger.warning(f"⚠️  Parameter count validation failed: {e}")

    # Set model to BF16 precision
    if config.dtype == "bfloat16" and torch.cuda.is_available():
        model = model.to(dtype=torch.bfloat16)
        logger.info("✓ Model converted to BF16 precision")

    # Create dummy data
    data_dir = Path("data/timing_test")
    data_path = create_dummy_data(
        output_dir=data_dir,
        num_examples=num_steps + 100,  # Buffer for safety
        max_length=4096,
        vocab_size=config.vocab_size,
    )

    # Create data loader
    logger.info("\nSetting up data pipeline...")

    # Create a simple mock tokenizer for pre-tokenized data
    # We don't actually need the real tokenizer since data is already tokenized
    class MockTokenizer:
        def __init__(self):
            self.vocab_size = config.vocab_size
            self.pad_token_id = 0
            self.eos_token = "<|endoftext|>"
            self.bos_token = "<|begin_of_text|>"
            self.pad_token = "<|finetune_right_pad_id|>"

    mock_tokenizer = MockTokenizer()

    dataset = SimpleDataLoader(
        data_path=data_path,
        max_length=4096,
        tokenizer=mock_tokenizer,  # Pass mock tokenizer to avoid HuggingFace download
        use_pretokenized=True,
        return_labels=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Physical batch size = 1 for 32GB VRAM
        shuffle=False,
        collate_fn=SimpleDataLoader.collate_fn,
        num_workers=0,  # Avoid multiprocessing overhead for timing
    )
    logger.info(f"✓ DataLoader created: batch_size=1, {len(dataset)} examples")

    # Create optimizer
    logger.info("\nSetting up optimizer...")
    optimizer = create_optimizer(
        model=model,
        lr=1e-4,
        weight_decay=0.01,
        use_8bit=True,  # Use 8-bit optimizer to save VRAM
    )
    logger.info(f"✓ Optimizer created: {optimizer.__class__.__name__}")

    # Create scheduler
    scheduler = create_scheduler(
        optimizer=optimizer,
        warmup_steps=100,
        T_0=1000,
        T_mult=2,
        eta_min=1e-6,
    )
    logger.info("✓ Scheduler created: CosineAnnealingWarmRestartsWithWarmup")

    # Create loss function
    loss_fn = SparseKLLoss(
        temperature=2.0,
        alpha=0.2,  # 20% hard CE + 80% soft KL
    )
    logger.info(f"✓ Loss function created: {loss_fn}")

    # Create teacher client (mock or real)
    if use_real_teacher:
        teacher = RealTeacherWrapper(
            teacher_url=teacher_url,
            model=teacher_model,
            vocab_size=config.vocab_size,
            topk=128,
            api_key=teacher_api_key,
        )
    else:
        teacher = MockTeacherClient(
            vocab_size=config.vocab_size,
            topk=128,
        )

    # Create telemetry logger
    log_dir = Path("logs/timing_test")
    telemetry = TelemetryLogger(
        log_dir=log_dir,
        log_interval=log_interval,
        sinks=[OutputSink.FILE, OutputSink.CONSOLE],
        enable_wandb=False,
    )
    logger.info(f"✓ Telemetry logger created: log_dir={log_dir}")

    # Training loop
    logger.info("\n" + "=" * 80)
    logger.info(f"Starting {num_steps}-step timing run...")
    logger.info("=" * 80)

    model.train()
    total_tokens = 0
    step_times = []
    teacher_fetch_times = []
    compute_times = []
    vram_usage = []
    losses = []

    start_time = time.time()

    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break

        step_start = time.time()

        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch.get('labels', input_ids).to(device)

        # Get teacher logits (track time separately)
        teacher_start = time.time()
        with torch.no_grad():
            teacher_topk_indices, teacher_topk_values, teacher_other_mass = \
                teacher.get_teacher_logits(input_ids, device)
        teacher_end = time.time()
        teacher_fetch_time = teacher_end - teacher_start
        teacher_fetch_times.append(teacher_fetch_time)

        # Compute phase starts after teacher fetch
        compute_start = time.time()

        # Forward pass
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
            student_logits = model(input_ids)

            # Compute loss
            loss = loss_fn(
                student_logits=student_logits,
                teacher_topk_indices=teacher_topk_indices,
                teacher_topk_values=teacher_topk_values,
                teacher_other_mass=teacher_other_mass,
                hard_targets=labels,
            )

            # MEMORY FIX: Free student logits immediately after loss computation
            del student_logits
            torch.cuda.empty_cache()

        # Backward pass
        loss.backward()

        # Optimizer step
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        # Compute time ends after optimizer step
        compute_end = time.time()
        compute_time = compute_end - compute_start
        compute_times.append(compute_time)

        # Timing and metrics
        step_end = time.time()
        step_time = step_end - step_start
        step_times.append(step_time)

        # VRAM tracking
        if device.type == 'cuda':
            vram_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
            vram_usage.append(vram_allocated)

        # Loss tracking
        losses.append(loss.item())

        # Token count
        num_tokens = input_ids.numel()
        total_tokens += num_tokens

        # Telemetry logging
        telemetry.log_step(
            step=step,
            epoch=0,
            loss=loss.item(),
            learning_rate=scheduler.get_lr()[0],
            grad_norm=grad_norm.item(),
            num_tokens=num_tokens,
            batch_size=input_ids.shape[0],
            step_time=step_time,
        )

    # End timing
    end_time = time.time()
    total_time = end_time - start_time

    logger.info("\n" + "=" * 80)
    logger.info(f"Timing run complete: {num_steps} steps in {total_time:.2f}s")
    logger.info("=" * 80)

    # Compute statistics
    avg_step_time = sum(step_times) / len(step_times)
    avg_teacher_time = sum(teacher_fetch_times) / len(teacher_fetch_times)
    avg_compute_time = sum(compute_times) / len(compute_times)
    avg_vram = sum(vram_usage) / len(vram_usage) if vram_usage else 0.0
    max_vram = max(vram_usage) if vram_usage else 0.0
    avg_loss = sum(losses) / len(losses)

    steps_per_sec = 1.0 / avg_step_time
    tokens_per_sec = total_tokens / total_time
    teacher_overhead_pct = (avg_teacher_time / avg_step_time) * 100
    compute_overhead_pct = (avg_compute_time / avg_step_time) * 100

    logger.info(f"\nTiming Statistics:")
    logger.info(f"  Total time: {total_time:.2f}s")
    logger.info(f"  Avg step time: {avg_step_time:.4f}s")
    logger.info(f"    - Teacher fetch: {avg_teacher_time:.4f}s ({teacher_overhead_pct:.1f}%)")
    logger.info(f"    - Compute (fwd+bwd): {avg_compute_time:.4f}s ({compute_overhead_pct:.1f}%)")
    logger.info(f"  Steps/sec: {steps_per_sec:.2f}")
    logger.info(f"  Tokens/sec: {tokens_per_sec:.1f}")
    logger.info(f"  Avg VRAM: {avg_vram:.2f} GB")
    logger.info(f"  Max VRAM: {max_vram:.2f} GB")
    logger.info(f"  Avg loss: {avg_loss:.4f}")

    # M3 GATE PROJECTION
    logger.info("\n" + "=" * 80)
    logger.info("M3 GATE ANALYSIS: Wall-time projection")
    logger.info("=" * 80)

    # Project to 60k and 80k steps
    steps_60k = 60_000
    steps_80k = 80_000

    # Pure training time
    training_time_60k = steps_60k * avg_step_time
    training_time_80k = steps_80k * avg_step_time

    # Evaluation overhead (every 5k steps, assume 1 minute per eval)
    num_evals_60k = steps_60k // 5000
    num_evals_80k = steps_80k // 5000
    eval_overhead_60k = num_evals_60k * 60  # seconds
    eval_overhead_80k = num_evals_80k * 60

    # Checkpointing overhead (every 5k steps, assume 30 seconds per checkpoint)
    checkpoint_overhead_60k = num_evals_60k * 30  # seconds
    checkpoint_overhead_80k = num_evals_80k * 30

    # Total projected time
    total_time_60k = training_time_60k + eval_overhead_60k + checkpoint_overhead_60k
    total_time_80k = training_time_80k + eval_overhead_80k + checkpoint_overhead_80k

    # Convert to days
    days_60k = total_time_60k / (24 * 3600)
    days_80k = total_time_80k / (24 * 3600)

    logger.info(f"\nProjection for {steps_60k:,} steps:")
    logger.info(f"  Training time: {training_time_60k / 3600:.1f} hours")
    logger.info(f"  Eval overhead: {eval_overhead_60k / 60:.1f} minutes ({num_evals_60k} evals)")
    logger.info(f"  Checkpoint overhead: {checkpoint_overhead_60k / 60:.1f} minutes")
    logger.info(f"  Total time: {total_time_60k / 3600:.1f} hours = {days_60k:.2f} days")

    logger.info(f"\nProjection for {steps_80k:,} steps:")
    logger.info(f"  Training time: {training_time_80k / 3600:.1f} hours")
    logger.info(f"  Eval overhead: {eval_overhead_80k / 60:.1f} minutes ({num_evals_80k} evals)")
    logger.info(f"  Checkpoint overhead: {checkpoint_overhead_80k / 60:.1f} minutes")
    logger.info(f"  Total time: {total_time_80k / 3600:.1f} hours = {days_80k:.2f} days")

    # M3 GATE DECISION
    gate_threshold_days = 33
    gate_status = "PASS" if days_80k <= gate_threshold_days else "FAIL"

    logger.info("\n" + "=" * 80)
    logger.info(f"M3 GATE STATUS: {gate_status}")
    logger.info("=" * 80)

    if gate_status == "PASS":
        logger.info(f"✓ PASS: Projected time {days_80k:.2f} days ≤ {gate_threshold_days} days")
        logger.info(f"  Recommendation: Proceed with {variant} configuration")
    else:
        logger.info(f"✗ FAIL: Projected time {days_80k:.2f} days > {gate_threshold_days} days")
        if variant == "500M":
            logger.info(f"  Recommendation: Reduce model size to 350M and retry")
        else:
            logger.info(f"  Recommendation: Optimize training pipeline or reduce max_steps")

    # Build timing report
    timing_report = {
        "gate_status": gate_status,
        "variant": variant,
        "teacher_config": {
            "type": "real" if use_real_teacher else "mock",
            "url": teacher_url if use_real_teacher else None,
            "model": teacher_model if use_real_teacher else None,
        },
        "model": {
            "d_model": config.d_model,
            "n_layers": config.n_layers,
            "n_heads": config.n_heads,
            "estimated_params": config.estimate_param_count(),
            "actual_params": actual_params,
        },
        "timing": {
            "num_steps": num_steps,
            "total_time_sec": total_time,
            "avg_step_time_sec": avg_step_time,
            "avg_teacher_fetch_sec": avg_teacher_time,
            "avg_compute_sec": avg_compute_time,
            "teacher_overhead_pct": teacher_overhead_pct,
            "compute_overhead_pct": compute_overhead_pct,
            "steps_per_sec": steps_per_sec,
            "tokens_per_sec": tokens_per_sec,
        },
        "resources": {
            "avg_vram_gb": avg_vram,
            "max_vram_gb": max_vram,
            "device": str(device),
        },
        "projection": {
            "60k_steps": {
                "training_time_hours": training_time_60k / 3600,
                "eval_overhead_minutes": eval_overhead_60k / 60,
                "checkpoint_overhead_minutes": checkpoint_overhead_60k / 60,
                "total_time_days": days_60k,
            },
            "80k_steps": {
                "training_time_hours": training_time_80k / 3600,
                "eval_overhead_minutes": eval_overhead_80k / 60,
                "checkpoint_overhead_minutes": checkpoint_overhead_80k / 60,
                "total_time_days": days_80k,
            },
        },
        "gate": {
            "threshold_days": gate_threshold_days,
            "status": gate_status,
            "recommendation": (
                f"Proceed with {variant} configuration"
                if gate_status == "PASS"
                else (
                    "Reduce model size to 350M and retry"
                    if variant == "500M"
                    else "Optimize training pipeline or reduce max_steps"
                )
            ),
        },
        "training": {
            "avg_loss": avg_loss,
            "final_loss": losses[-1] if losses else None,
        },
    }

    # Save timing report
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(timing_report, f, indent=2)
        logger.info(f"\n✓ Timing report saved: {output_path}")

    # Finalize telemetry
    telemetry.finalize()

    # Cleanup teacher client if real
    if use_real_teacher and hasattr(teacher, 'close'):
        teacher.close()
        logger.info("✓ Teacher client closed")

    return timing_report


def main():
    """Main entry point for timing run script."""
    parser = argparse.ArgumentParser(
        description="M3 CRITICAL GATE: 1k-step timing run to validate training timeline"
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["350M", "500M"],
        default="350M",
        help="Model variant to test (default: 350M)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=1000,
        help="Number of training steps to run (default: 1000)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Steps between logging (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save timing report JSON (default: timing_report_{variant}.json)",
    )
    parser.add_argument(
        "--use-real-teacher",
        action="store_true",
        help="Use real vLLM teacher server instead of mock teacher",
    )
    parser.add_argument(
        "--teacher-url",
        type=str,
        default="http://localhost:8080",
        help="URL to vLLM teacher server (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Teacher model identifier (default: meta-llama/Llama-3.2-1B-Instruct)",
    )
    parser.add_argument(
        "--teacher-api-key",
        type=str,
        default=None,
        help="API key for teacher server authentication (optional)",
    )

    args = parser.parse_args()

    # Default output path
    if args.output is None:
        teacher_suffix = "_real_teacher" if args.use_real_teacher else ""
        args.output = Path(f"timing_report_{args.variant}{teacher_suffix}.json")

    # Run timing test
    try:
        timing_report = run_timing_test(
            variant=args.variant,
            num_steps=args.num_steps,
            log_interval=args.log_interval,
            output_path=args.output,
            use_real_teacher=args.use_real_teacher,
            teacher_url=args.teacher_url,
            teacher_model=args.teacher_model,
            teacher_api_key=args.teacher_api_key,
        )

        # Print summary
        print("\n" + "=" * 80)
        print("TIMING RUN SUMMARY")
        print("=" * 80)
        print(f"Variant: {timing_report['variant']}")
        print(f"Model: {timing_report['model']['actual_params']:,} parameters")
        print(f"Steps/sec: {timing_report['timing']['steps_per_sec']:.2f}")
        print(f"Tokens/sec: {timing_report['timing']['tokens_per_sec']:.1f}")
        print(f"Max VRAM: {timing_report['resources']['max_vram_gb']:.2f} GB")
        print(f"\nProjection (80k steps): {timing_report['projection']['80k_steps']['total_time_days']:.2f} days")
        print(f"M3 Gate Status: {timing_report['gate_status']}")
        print(f"Recommendation: {timing_report['gate']['recommendation']}")
        print("=" * 80)

        # Exit with appropriate code
        sys.exit(0 if timing_report['gate_status'] == "PASS" else 1)

    except Exception as e:
        logger.error(f"Timing run failed: {e}", exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
