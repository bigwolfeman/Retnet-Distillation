#!/usr/bin/env python3
"""
Checkpoint evaluation script for manual quality assessment of student outputs.

This script loads a trained student checkpoint alongside the teacher model
and generates comparison outputs on validation samples. Designed for human
quality assessment of distillation progress.

Features:
- Sample caching for reproducible evaluations
- Side-by-side teacher/student output comparison
- Rich terminal formatting with progress bars
- Markdown reports with metrics
- Automated metrics (BLEU, token counts, timing)

FIXES (2025-11-16):
- Fixed model output handling for RetNet/HRM tuple returns
- Fixed prompt generation (truncates to first N tokens instead of using full sequence)
- Added token ID validation to prevent garbage outputs
- Added debug mode for troubleshooting generation
- Improved error handling per sample
- Better prompt vs continuation display in reports

Usage:
    python scripts/evaluate_checkpoint.py --checkpoint runs/direct_mode/checkpoint-10000
    python scripts/evaluate_checkpoint.py --checkpoint runs/direct_mode/checkpoint-10000 --num-samples 50
    python scripts/evaluate_checkpoint.py --checkpoint runs/direct_mode/checkpoint-10000 --debug
"""

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import project modules
from src.distillation.config import load_yaml_config, TrainingConfig
from src.distillation.student_config import create_student_config, RetNetStudentConfig
from src.distillation.dataset import PretokenizedShardDataset, load_llama_tokenizer
try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

# Import rich for terminal formatting
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("Warning: 'rich' library not found. Install with: pip install rich")
    print("Falling back to plain text output.")

# Import BLEU for automated metrics
try:
    from sacrebleu import corpus_bleu
    HAS_BLEU = True
except ImportError:
    HAS_BLEU = False
    print("Warning: 'sacrebleu' library not found. BLEU scores will be skipped.")
    print("Install with: pip install sacrebleu")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_student_model(
    student_config: RetNetStudentConfig,
    checkpoint_path: Path,
    device: torch.device,
    use_bf16: bool = True,
) -> nn.Module:
    """Load student model from checkpoint.

    Args:
        student_config: Student model configuration
        checkpoint_path: Path to checkpoint file
        device: Device to load model to
        use_bf16: Use BF16 precision

    Returns:
        Loaded student model
    """
    logger.info(f"Loading student model from checkpoint: {checkpoint_path}")

    # Check if this is a TitanMAC variant
    if student_config.variant.startswith("titan_mac"):
        from src.models.titans.titan_init import create_titan_mac_model

        model = create_titan_mac_model(
            variant=student_config.variant,
            vocab_size=student_config.vocab_size,
            device=device,
            use_bf16=use_bf16,
        )
    else:
        # Load RetNet model
        from src.models.retnet.backbone import RetNetBackbone, RetNetOutputHead

        model_kwargs = student_config.to_retnet_backbone_args()
        model_kwargs['checkpoint_activations'] = False  # Disable for inference
        model = RetNetBackbone(**model_kwargs)

        # Add output head
        model.add_module('lm_head', RetNetOutputHead(
            d_model=student_config.d_model,
            vocab_size=student_config.vocab_size,
            tie_weights=False,
            embedding_layer=None,
        ))

        # Move to device and set dtype
        model = model.to(device)
        if use_bf16:
            model = model.to(dtype=torch.bfloat16)

        # Tie weights
        model.lm_head.proj.weight = model.embed.weight

    # Load checkpoint
    if checkpoint_path.exists():
        # PyTorch 2.6+ requires weights_only=False for pickled checkpoints
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            # Fallback for older PyTorch versions that don't support weights_only
            checkpoint = torch.load(checkpoint_path, map_location=device)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=True)
        logger.info("Checkpoint loaded successfully")

        # Log checkpoint metadata if available
        if isinstance(checkpoint, dict):
            if 'global_step' in checkpoint:
                logger.info(f"  Checkpoint step: {checkpoint['global_step']}")
            if 'epoch' in checkpoint:
                logger.info(f"  Checkpoint epoch: {checkpoint['epoch']}")
            if 'loss' in checkpoint:
                logger.info(f"  Checkpoint loss: {checkpoint['loss']:.4f}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model.eval()

    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Student model loaded: {param_count:,} parameters")

    return model


def load_teacher_model(
    model_name: str,
    device: torch.device,
    use_bf16: bool = True,
    adapter_path: Optional[str] = None,
) -> nn.Module:
    """Load Hugging Face teacher model for generation."""
    logger.info(f"Loading teacher model: {model_name}")
    dtype = torch.bfloat16 if use_bf16 and torch.cuda.is_available() else torch.float32
    # Load base model
    teacher = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=None,
    )

    # Align embedding size to adapter tokenizer (handles added PAD tokens)
    base_vocab = teacher.get_input_embeddings().weight.size(0)
    target_vocab = base_vocab
    try:
        tok_for_size = AutoTokenizer.from_pretrained(adapter_path or model_name, use_fast=True, local_files_only=True)
        target_vocab = max(
            base_vocab,
            tok_for_size.vocab_size,
            (tok_for_size.pad_token_id + 1) if tok_for_size.pad_token_id is not None else base_vocab,
        )
    except Exception:
        tok_for_size = None

    if target_vocab != base_vocab:
        logger.info(f"Resizing teacher embeddings {base_vocab}->{target_vocab} to match adapter tokenizer")
        teacher.resize_token_embeddings(target_vocab)

    teacher = teacher.to(device)

    if adapter_path:
        if not HAS_PEFT:
            raise RuntimeError("teacher_adapter_path is set but peft is not installed")
        teacher = PeftModel.from_pretrained(teacher, adapter_path)
        logger.info(f"Loaded teacher adapter from {adapter_path}")

    teacher.eval()
    logger.info("Teacher model ready for generation")
    return teacher


def load_or_create_eval_samples(
    data_path: Path,
    cache_path: Path,
    tokenizer: AutoTokenizer,
    num_samples: int = 30,
    prompt_length: int = 128,  # FIX: Use truncated prompts instead of full sequences
    regenerate: bool = False,
) -> List[Dict[str, Any]]:
    """Load cached evaluation samples or create new ones.

    FIXED: Now creates proper prompts by truncating to first N tokens instead of
    using entire training sequences (which are complete solutions).

    Args:
        data_path: Path to validation data
        cache_path: Path to cache file
        tokenizer: Tokenizer for decoding
        num_samples: Number of samples to generate
        prompt_length: Length of prompt to extract from each sample (default: 128 tokens)
        regenerate: Force regenerate cache

    Returns:
        List of evaluation samples with:
            - input_ids: Truncated prompt tokens
            - prompt_text: Decoded prompt text
            - full_text: Full original sequence (for reference)
    """
    if cache_path.exists() and not regenerate:
        logger.info(f"Loading cached evaluation samples from: {cache_path}")
        with open(cache_path, 'r') as f:
            samples = [json.loads(line) for line in f]
        logger.info(f"Loaded {len(samples)} cached samples")
        return samples

    logger.info(f"Generating {num_samples} new evaluation samples (prompt_length={prompt_length})...")

    # Load validation dataset
    dataset = PretokenizedShardDataset(
        data_path=data_path,
        max_length=4096,
    )

    # Randomly sample indices
    total_samples = len(dataset)
    if num_samples > total_samples:
        logger.warning(f"Requested {num_samples} samples but only {total_samples} available")
        num_samples = total_samples

    random.seed(42)  # For reproducibility
    indices = random.sample(range(total_samples), num_samples)

    samples = []
    for idx in indices:
        item = dataset[idx]
        full_input_ids = item['input_ids'].tolist()

        # FIX: Create prompt by taking first N tokens (not full sequence)
        # This creates actual prompts instead of complete solutions
        prompt_ids = full_input_ids[:min(prompt_length, len(full_input_ids))]

        # Decode both prompt and full sequence
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
        full_text = tokenizer.decode(full_input_ids, skip_special_tokens=True)

        samples.append({
            'input_ids': prompt_ids,  # Use truncated prompt
            'prompt_text': prompt_text,
            'full_text': full_text,  # Keep for reference
        })

    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    logger.info(f"Cached {len(samples)} samples to: {cache_path}")

    return samples


@torch.no_grad()
def generate_output(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: torch.device = None,
    tokenizer: AutoTokenizer = None,
    debug: bool = False,
    model_name: str = "model",
) -> tuple[List[int], float]:
    """Generate output from model with robust error handling.

    FIXED: Properly handles different model output formats (tuple, .logits, tensor).
    FIXED: Validates token IDs are in valid vocabulary range.
    FIXED: Adds debug output for troubleshooting.

    Args:
        model: Model to generate from
        input_ids: Input token IDs [seq_len]
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        top_p: Top-p (nucleus) sampling threshold
        device: Device to use
        tokenizer: Tokenizer for validation and debug output
        debug: Enable debug logging
        model_name: Name for debug messages (teacher/student)

    Returns:
        Tuple of (generated_ids, generation_time)

    Raises:
        ValueError: If model output format is unexpected or tokens are invalid
    """
    if device is None:
        device = next(model.parameters()).device

    input_ids = input_ids.to(device)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)  # Add batch dimension

    start_time = time.time()

    # Generate tokens autoregressively
    generated = input_ids[0].tolist()

    if debug:
        logger.info(f"[{model_name}] Starting generation from {len(generated)} prompt tokens")

    for step in range(max_new_tokens):
        # Forward pass
        current_input = torch.tensor([generated], dtype=torch.long, device=device)
        output = model(current_input)

        # FIX: Handle different model output formats
        if isinstance(output, tuple):
            # RetNet/HRM returns hidden states; apply lm_head if present
            logits_or_hidden = output[0]
            if hasattr(model, "lm_head"):
                logits = model.lm_head(logits_or_hidden)
            else:
                logits = logits_or_hidden
            if debug and step == 0:
                logger.info(f"[{model_name}] Model returns tuple, using first element")
        elif hasattr(output, 'logits'):
            # Huggingface-style output
            logits = output.logits
            if debug and step == 0:
                logger.info(f"[{model_name}] Model returns object with .logits attribute")
        elif isinstance(output, torch.Tensor):
            # Raw tensor: for RetNet this is hidden states; project if lm_head exists
            if hasattr(model, "lm_head"):
                logits = model.lm_head(output)
            else:
                logits = output
            if debug and step == 0:
                logger.info(f"[{model_name}] Model returns raw tensor (projecting with lm_head: {hasattr(model, 'lm_head')})")
        else:
            raise ValueError(f"[{model_name}] Unexpected model output type: {type(output)}")

        # FIX: Validate shape
        if logits.dim() != 3:
            raise ValueError(
                f"[{model_name}] Expected 3D logits [batch, seq, vocab], got shape {logits.shape}"
            )

        # Get logits for last position
        next_token_logits = logits[0, -1, :]

        # Apply temperature
        if temperature > 0:
            next_token_logits = next_token_logits / temperature

            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = float('-inf')

            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
        else:
            # Greedy decoding
            next_token = next_token_logits.argmax().item()

        # FIX: Validate token ID is in valid vocabulary range (account for special tokens)
        if tokenizer:
            vocab_size = tokenizer.vocab_size
            special_max = max(
                [tid for tid in [tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.bos_token_id] if tid is not None] or [-1]
            )
            valid_upper = max(vocab_size - 1, special_max)
        else:
            valid_upper = 128256 - 1  # Llama default fallback

        if next_token < 0 or next_token > valid_upper:
            logger.warning(
                f"[{model_name}] Invalid token ID {next_token} at step {step} "
                f"(valid_upper={valid_upper}), stopping generation"
            )
            break

        # FIX: Debug output for first few tokens
        if debug and step < 10:
            if tokenizer:
                token_str = tokenizer.decode([next_token])
                logger.info(f"[{model_name}] Step {step}: token {next_token} = '{token_str}'")
            else:
                logger.info(f"[{model_name}] Step {step}: token {next_token}")

        generated.append(next_token)

        # Stop on EOS token
        eos_token_id = tokenizer.eos_token_id if tokenizer else 128001  # Llama default
        if next_token == eos_token_id:
            if debug:
                logger.info(f"[{model_name}] Hit EOS at step {step}")
            break

    generation_time = time.time() - start_time

    if debug:
        logger.info(
            f"[{model_name}] Generated {len(generated) - len(input_ids[0])} tokens in {generation_time:.2f}s"
        )

    return generated, generation_time


@torch.no_grad()
def generate_teacher_output(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> tuple[List[int], float]:
    """Use HF generate for teacher (handles EOS/padding correctly)."""
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    do_sample = temperature > 0
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        attention_mask=attention_mask,
    )
    start = time.time()
    output_ids = model.generate(input_ids, **gen_kwargs)
    duration = time.time() - start
    return output_ids[0].tolist(), duration


def compute_bleu_score(references: List[str], hypotheses: List[str]) -> float:
    """Compute corpus BLEU score.

    Args:
        references: Reference texts (teacher outputs)
        hypotheses: Hypothesis texts (student outputs)

    Returns:
        BLEU score (0-100)
    """
    if not HAS_BLEU:
        return 0.0

    # sacrebleu expects references as list of lists
    refs = [[ref] for ref in references]

    try:
        bleu = corpus_bleu(hypotheses, refs)
        return bleu.score
    except Exception as e:
        logger.warning(f"BLEU computation failed: {e}")
        return 0.0


def evaluate_checkpoint(
    checkpoint_path: Path,
    config: TrainingConfig,
    num_samples: int = 30,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    prompt_length: int = 128,
    regenerate_cache: bool = False,
    device: str = "cuda",
    debug: bool = False,
    use_bf16: Optional[bool] = None,
) -> Dict[str, Any]:
    """Evaluate checkpoint on validation samples.

    Args:
        checkpoint_path: Path to checkpoint
        config: Training configuration
        num_samples: Number of samples to evaluate
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        prompt_length: Length of prompt to use from samples
        regenerate_cache: Force regenerate sample cache
        device: Device to use
        debug: Enable debug output

    Returns:
        Evaluation results
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Resolve precision override (default to config unless explicitly set)
    if use_bf16 is None:
        use_bf16 = config.use_bf16

    # Create console for rich output
    console = Console() if HAS_RICH else None

    if console:
        console.print("\n[bold cyan]Checkpoint Evaluation[/bold cyan]")
        console.print(f"Checkpoint: {checkpoint_path}")
        console.print(f"Samples: {num_samples}")
        console.print(f"Prompt length: {prompt_length} tokens")
        console.print(f"Max new tokens: {max_new_tokens}")
        console.print(f"Temperature: {temperature}")
        console.print(f"BF16: {use_bf16}")
        console.print(f"Debug mode: {debug}\n")
    else:
        print("\n=== Checkpoint Evaluation ===")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Samples: {num_samples}")
        print(f"Prompt length: {prompt_length} tokens")
        print(f"Max new tokens: {max_new_tokens}")
        print(f"Temperature: {temperature}")
        print(f"BF16: {use_bf16}")
        print(f"Debug mode: {debug}\n")

    # Load tokenizer
    tokenizer = load_llama_tokenizer(
        model_name=config.tokenizer_name,
        adapter_path=getattr(config, "teacher_adapter_path", None),
    )

    # Load or create evaluation samples
    cache_path = Path(f"data/eval_samples_cache_p{prompt_length}.jsonl")
    samples = load_or_create_eval_samples(
        data_path=Path(config.val_data_path),
        cache_path=cache_path,
        tokenizer=tokenizer,
        num_samples=num_samples,
        prompt_length=prompt_length,
        regenerate=regenerate_cache,
    )

    if len(samples) > num_samples:
        samples = samples[:num_samples]

    # Load teacher model
    logger.info("Loading teacher model...")
    teacher = load_teacher_model(
        model_name=config.teacher_model,
        device=device,
        use_bf16=use_bf16,
        adapter_path=getattr(config, "teacher_adapter_path", None),
    )

    # Load student model
    student_config = create_student_config(config.model_variant)
    student = load_student_model(
        student_config=student_config,
        checkpoint_path=checkpoint_path,
        device=device,
        use_bf16=use_bf16,
    )

    # Evaluate samples
    results = []
    teacher_outputs = []
    student_outputs = []

    if console:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        )
        task = progress.add_task("[cyan]Generating outputs...", total=len(samples))
        progress.start()
    else:
        print(f"Generating outputs for {len(samples)} samples...")
        progress = None
        task = None

    for i, sample in enumerate(samples):
        # FIX: Wrap each sample in try-except to prevent single failures from stopping entire run
        try:
            prompt_text = sample.get('prompt_text') or sample.get('text')
            if not prompt_text:
                # Backwards compatibility for older caches that only stored token ids
                legacy_ids = sample.get('input_ids', [])
                prompt_text = tokenizer.decode(legacy_ids, skip_special_tokens=True)

            # Re-tokenize prompt text to guarantee valid vocabulary IDs for both teacher & student
            prompt_encoding = tokenizer(
                prompt_text,
                add_special_tokens=False,
                return_tensors='pt',
            )
            input_ids = prompt_encoding['input_ids'][0]
            if input_ids.numel() == 0:
                logger.warning(f"Sample {i} yielded empty prompt after re-tokenization; skipping")
                continue

            prompt_token_count = input_ids.shape[0]
            full_text = sample.get('full_text', '')

            # Generate teacher output (HF generate)
            try:
                teacher_ids, teacher_time = generate_teacher_output(
                    model=teacher,
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=0.9,
                    tokenizer=tokenizer,
                    device=device,
                )
                prompt_token_count = input_ids.shape[0]
                teacher_output = tokenizer.decode(
                    teacher_ids[prompt_token_count:],
                    skip_special_tokens=True,
                )
            except Exception as e:
                logger.error(f"Teacher generation failed for sample {i}: {e}")
                teacher_output = f"[ERROR: {e}]"
                teacher_ids = input_ids.tolist()
                teacher_time = 0.0

            # Generate student output
            try:
                student_ids, student_time = generate_output(
                    model=student,
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    device=device,
                    tokenizer=tokenizer,
                    debug=debug,
                    model_name="student",
                )
                student_output = tokenizer.decode(
                    student_ids[prompt_token_count:],
                    skip_special_tokens=True
                )
            except Exception as e:
                logger.error(f"Student generation failed for sample {i}: {e}")
                student_output = f"[ERROR: {e}]"
                student_ids = input_ids.tolist()
                student_time = 0.0

            # Compute metrics
            teacher_tokens = len(teacher_ids) - prompt_token_count
            student_tokens = len(student_ids) - prompt_token_count
            length_ratio = student_tokens / max(teacher_tokens, 1)

            # Flags
            incomplete = student_tokens >= max_new_tokens
            empty = student_tokens == 0
            too_short = length_ratio < 0.5
            too_long = length_ratio > 2.0

            result = {
                'sample_id': i,
                'prompt_text': prompt_text,
                'full_text': full_text,  # For reference
                'teacher_output': teacher_output,
                'student_output': student_output,
                'input_tokens': prompt_token_count,
                'teacher_tokens': teacher_tokens,
                'student_tokens': student_tokens,
                'teacher_time': teacher_time,
                'student_time': student_time,
                'length_ratio': length_ratio,
                'incomplete': incomplete,
                'empty': empty,
                'too_short': too_short,
                'too_long': too_long,
            }

            results.append(result)
            teacher_outputs.append(teacher_output)
            student_outputs.append(student_output)

        except Exception as e:
            logger.error(f"Sample {i} processing failed: {e}", exc_info=True)
            # Don't add to results, continue with next sample
            continue

        if progress:
            progress.update(task, advance=1)
        else:
            if (i + 1) % 5 == 0:
                print(f"  Completed {i + 1}/{len(samples)} samples")

    if progress:
        progress.stop()

    # Compute corpus BLEU
    bleu_score = compute_bleu_score(teacher_outputs, student_outputs)

    # Compute summary statistics
    if results:
        summary = {
            'total_samples': len(results),
            'avg_teacher_tokens': np.mean([r['teacher_tokens'] for r in results]),
            'avg_student_tokens': np.mean([r['student_tokens'] for r in results]),
            'avg_length_ratio': np.mean([r['length_ratio'] for r in results]),
            'avg_teacher_time': np.mean([r['teacher_time'] for r in results]),
            'avg_student_time': np.mean([r['student_time'] for r in results]),
            'bleu_score': bleu_score,
            'incomplete_count': sum(r['incomplete'] for r in results),
            'empty_count': sum(r['empty'] for r in results),
            'too_short_count': sum(r['too_short'] for r in results),
            'too_long_count': sum(r['too_long'] for r in results),
        }
    else:
        logger.error("No successful samples! Check debug output above.")
        summary = {
            'total_samples': 0,
            'avg_teacher_tokens': 0.0,
            'avg_student_tokens': 0.0,
            'avg_length_ratio': 0.0,
            'avg_teacher_time': 0.0,
            'avg_student_time': 0.0,
            'bleu_score': 0.0,
            'incomplete_count': 0,
            'empty_count': 0,
            'too_short_count': 0,
            'too_long_count': 0,
        }

    return {
        'checkpoint_path': str(checkpoint_path),
        'config': config.to_dict(),
        'summary': summary,
        'results': results,
    }


def print_results(evaluation: Dict[str, Any], console: Optional[Any] = None):
    """Print evaluation results to terminal.

    Args:
        evaluation: Evaluation results
        console: Rich Console object (optional)
    """
    summary = evaluation['summary']
    results = evaluation['results']

    if console:
        # Print summary table
        table = Table(title="Evaluation Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Total Samples", str(summary['total_samples']))
        table.add_row("Avg Teacher Tokens", f"{summary['avg_teacher_tokens']:.1f}")
        table.add_row("Avg Student Tokens", f"{summary['avg_student_tokens']:.1f}")
        table.add_row("Avg Length Ratio", f"{summary['avg_length_ratio']:.2f}")
        table.add_row("Avg Teacher Time", f"{summary['avg_teacher_time']:.3f}s")
        table.add_row("Avg Student Time", f"{summary['avg_student_time']:.3f}s")
        if HAS_BLEU:
            table.add_row("BLEU Score", f"{summary['bleu_score']:.2f}")
        table.add_row("Incomplete Outputs", str(summary['incomplete_count']))
        table.add_row("Empty Outputs", str(summary['empty_count']))
        table.add_row("Too Short (<0.5x)", str(summary['too_short_count']))
        table.add_row("Too Long (>2x)", str(summary['too_long_count']))

        console.print("\n")
        console.print(table)
        console.print("\n")

        # Print sample comparisons
        for i, result in enumerate(results[:5]):  # Show first 5
            console.print(f"[bold]Sample {i + 1}[/bold]")
            console.print(Panel(result['prompt_text'][:200] + "...", title="Prompt", border_style="blue"))
            console.print(Panel(result['teacher_output'][:300], title="Teacher Output", border_style="green"))
            console.print(Panel(result['student_output'][:300], title="Student Output", border_style="yellow"))
            console.print(f"Teacher: {result['teacher_tokens']} tokens in {result['teacher_time']:.3f}s")
            console.print(f"Student: {result['student_tokens']} tokens in {result['student_time']:.3f}s")
            console.print(f"Length ratio: {result['length_ratio']:.2f}\n")
    else:
        # Plain text output
        print("\n=== Evaluation Summary ===")
        print(f"Total Samples: {summary['total_samples']}")
        print(f"Avg Teacher Tokens: {summary['avg_teacher_tokens']:.1f}")
        print(f"Avg Student Tokens: {summary['avg_student_tokens']:.1f}")
        print(f"Avg Length Ratio: {summary['avg_length_ratio']:.2f}")
        print(f"Avg Teacher Time: {summary['avg_teacher_time']:.3f}s")
        print(f"Avg Student Time: {summary['avg_student_time']:.3f}s")
        if HAS_BLEU:
            print(f"BLEU Score: {summary['bleu_score']:.2f}")
        print(f"Incomplete Outputs: {summary['incomplete_count']}")
        print(f"Empty Outputs: {summary['empty_count']}")
        print(f"Too Short (<0.5x): {summary['too_short_count']}")
        print(f"Too Long (>2x): {summary['too_long_count']}")
        print("\n")


def save_markdown_report(evaluation: Dict[str, Any], output_path: Path):
    """Save evaluation results to markdown file.

    FIXED: Improved format to show prompt vs continuation clearly.

    Args:
        evaluation: Evaluation results
        output_path: Path to output markdown file
    """
    summary = evaluation['summary']
    results = evaluation['results']
    checkpoint_path = evaluation['checkpoint_path']

    # Extract checkpoint info
    checkpoint_name = Path(checkpoint_path).name
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build markdown report
    md = f"""# Checkpoint Evaluation Report

**Checkpoint**: `{checkpoint_name}`
**Path**: `{checkpoint_path}`
**Timestamp**: {timestamp}
**Samples**: {summary['total_samples']}

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Samples | {summary['total_samples']} |
| Avg Teacher Tokens | {summary['avg_teacher_tokens']:.1f} |
| Avg Student Tokens | {summary['avg_student_tokens']:.1f} |
| Avg Length Ratio | {summary['avg_length_ratio']:.2f} |
| Avg Teacher Time | {summary['avg_teacher_time']:.3f}s |
| Avg Student Time | {summary['avg_student_time']:.3f}s |
"""

    if HAS_BLEU:
        md += f"| BLEU Score | {summary['bleu_score']:.2f} |\n"

    md += f"""| Incomplete Outputs | {summary['incomplete_count']} |
| Empty Outputs | {summary['empty_count']} |
| Too Short (<0.5x) | {summary['too_short_count']} |
| Too Long (>2x) | {summary['too_long_count']} |

---

## Sample Comparisons

"""

    # Add individual samples
    for i, result in enumerate(results):
        # FIX: Show prompt vs continuation clearly
        expected_continuation = result.get('full_text', '')[len(result['prompt_text']):]
        if expected_continuation:
            expected_preview = expected_continuation[:200] + "..." if len(expected_continuation) > 200 else expected_continuation
        else:
            expected_preview = "N/A"

        md += f"""### Sample {i + 1}

**Input Prompt** ({result['input_tokens']} tokens):
```
{result['prompt_text']}
```

**Expected Continuation** (reference, from training data):
```
{expected_preview}
```

**Teacher Output** ({result['teacher_tokens']} tokens, {result['teacher_time']:.3f}s):
```
{result['teacher_output']}
```

**Student Output** ({result['student_tokens']} tokens, {result['student_time']:.3f}s):
```
{result['student_output']}
```

**Metrics**:
- Length Ratio: {result['length_ratio']:.2f}
- Incomplete: {result['incomplete']}
- Empty: {result['empty']}
- Too Short: {result['too_short']}
- Too Long: {result['too_long']}

---

"""

    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(md)

    logger.info(f"Markdown report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate student checkpoint quality with teacher comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to student checkpoint file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_direct.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=30,
        help="Number of validation samples to evaluate",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0 = greedy)",
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=128,
        help="Length of prompt to use from samples (default: 128)",
    )
    parser.add_argument(
        "--regenerate-cache",
        action="store_true",
        help="Force regenerate evaluation sample cache",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Run evaluation in FP32 (disable BF16 for student/teacher)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Output directory for markdown reports",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output during generation",
    )

    args = parser.parse_args()

    # Resolve checkpoint path
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Load training config
    config_dict = load_yaml_config(args.config)
    config = TrainingConfig.from_dict(config_dict)

    # Run evaluation
    evaluation = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        config=config,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        prompt_length=args.prompt_length,
        regenerate_cache=args.regenerate_cache,
        device=args.device,
        debug=args.debug,
        use_bf16=None if not args.fp32 else False,
    )

    # Print results to terminal
    console = Console() if HAS_RICH else None
    print_results(evaluation, console)

    # Save markdown report
    checkpoint_name = checkpoint_path.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output_dir) / f"{checkpoint_name}_eval_{timestamp}.md"
    save_markdown_report(evaluation, output_path)

    if console:
        console.print(f"\n[bold green]Evaluation complete![/bold green]")
        console.print(f"Report saved to: {output_path}\n")
    else:
        print(f"\nEvaluation complete!")
        print(f"Report saved to: {output_path}\n")


if __name__ == "__main__":
    main()
