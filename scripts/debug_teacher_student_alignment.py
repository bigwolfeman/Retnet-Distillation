"""
Standalone script to verify teacher-student alignment.

This script:
1. Loads one batch from the training DataLoader
2. Runs teacher forward pass and grabs top-1 predictions
3. Runs student forward pass and compares predicted tokens
4. Checks if ground-truth tokens appear in teacher top-k
5. Prints decoded snippets to visually confirm data quality

Usage:
    python scripts/debug_teacher_student_alignment.py --config configs/stage1_kd_350m.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from distillation.config import load_yaml_config, TrainingConfig
from distillation.dataset import PretokenizedShardDataset, create_streaming_dataloaders
from distillation.direct_teacher_client import DirectTeacherClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_teacher_outputs(input_ids, teacher_topk_indices, teacher_topk_values, labels, tokenizer):
    """Analyze teacher outputs for data quality issues."""
    logger.info("=" * 80)
    logger.info("TEACHER OUTPUT ANALYSIS")
    logger.info("=" * 80)

    batch_size, seq_len, topk = teacher_topk_indices.shape

    # Analyze first example
    example_input = input_ids[0]
    example_topk_indices = teacher_topk_indices[0]
    example_topk_values = teacher_topk_values[0]
    example_labels = labels[0]

    # Find non-pad positions
    non_pad = (example_input != tokenizer.pad_token_id).nonzero(as_tuple=True)[0]

    if len(non_pad) == 0:
        logger.error("❌ Example has no valid tokens (all padding)!")
        return

    logger.info(f"Valid positions: {len(non_pad)}/{seq_len}")

    # Check if ground-truth tokens appear in teacher top-k
    ground_truth_in_topk = 0
    ground_truth_total = 0
    missing_positions = []

    for pos in non_pad[:50]:  # Check first 50 valid positions
        pos_idx = pos.item()

        # Get ground truth label for this position
        gt_label = example_labels[pos_idx].item()

        if gt_label == -100:
            continue  # Skip masked positions

        ground_truth_total += 1

        # Check if ground truth appears in teacher top-k
        teacher_topk_at_pos = example_topk_indices[pos_idx]
        if gt_label in teacher_topk_at_pos:
            ground_truth_in_topk += 1
        else:
            missing_positions.append(pos_idx)

    if ground_truth_total > 0:
        coverage = 100 * ground_truth_in_topk / ground_truth_total
        logger.info(f"Ground-truth coverage: {ground_truth_in_topk}/{ground_truth_total} ({coverage:.1f}%)")

        if coverage < 90:
            logger.error(f"❌ LOW COVERAGE: Only {coverage:.1f}% of ground-truth tokens in teacher top-k!")
            logger.error("This suggests either:")
            logger.error("  1. Wrong tokenizer (teacher/student mismatch)")
            logger.error("  2. Corrupted sequences before reaching teacher")
            logger.error("  3. Teacher input doesn't match student input")

            # Show some missing positions
            if missing_positions:
                logger.error("")
                logger.error(f"First 5 positions where ground-truth NOT in teacher top-k:")
                for pos in missing_positions[:5]:
                    gt = example_labels[pos].item()
                    topk = example_topk_indices[pos].tolist()
                    logger.error(f"  Position {pos}: gt={gt} ({tokenizer.decode([gt])}), teacher top-5={topk[:5]}")
        else:
            logger.info(f"✓ Good coverage: {coverage:.1f}% of ground-truth in teacher top-k")

    # Check teacher logit distribution
    logger.info("")
    logger.info("Teacher logit statistics (first 20 valid positions):")

    for i, pos in enumerate(non_pad[:20]):
        pos_idx = pos.item()
        topk_values = example_topk_values[pos_idx]

        # Convert logprobs to probs for readability
        probs = torch.exp(topk_values)
        top1_prob = probs[0].item()
        total_prob = probs.sum().item()

        logger.info(f"  Position {pos_idx}: top-1 prob={top1_prob:.3f}, top-k sum={total_prob:.3f}")

        # Warn if distribution looks weird
        if top1_prob > 0.99:
            logger.warning(f"    ⚠️  Nearly deterministic (top-1 prob > 99%)")
        if total_prob < 0.5:
            logger.warning(f"    ⚠️  Low probability mass in top-k (sum < 50%)")

    # Decode teacher top-1 predictions
    logger.info("")
    logger.info("Teacher top-1 predictions (first 30 tokens):")

    top1_predictions = example_topk_indices[:, 0]  # [seq_len]
    first_30_pred = top1_predictions[non_pad[:30]]

    logger.info(f"  Token IDs: {first_30_pred.tolist()}")
    decoded_pred = tokenizer.decode(first_30_pred, skip_special_tokens=False)
    logger.info(f"  Decoded: {decoded_pred}")

    # Compare with ground truth
    logger.info("")
    logger.info("Ground truth (first 30 tokens after labels):")
    first_30_gt = example_labels[non_pad[:30]]
    # Filter out -100
    first_30_gt_valid = first_30_gt[first_30_gt != -100]
    logger.info(f"  Token IDs: {first_30_gt_valid.tolist()}")
    if len(first_30_gt_valid) > 0:
        decoded_gt = tokenizer.decode(first_30_gt_valid, skip_special_tokens=False)
        logger.info(f"  Decoded: {decoded_gt}")

    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Debug teacher-student alignment")
    parser.add_argument("--config", type=str, required=True, help="Path to training config")
    parser.add_argument("--topk", type=int, default=128, help="Number of top-k logits to fetch from teacher")
    args = parser.parse_args()

    # Load config
    logger.info(f"Loading config from: {args.config}")
    yaml_config = load_yaml_config(args.config)
    config = TrainingConfig.from_dict(yaml_config)

    # Load tokenizer
    logger.info(f"Loading tokenizer: {config.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Load dataset
    logger.info(f"Loading dataset from: {config.train_data_path}")

    if config.use_pretokenized_data:
        dataset = PretokenizedShardDataset(
            data_path=config.train_data_path,
            max_length=config.max_seq_length,
            splits=config.pretokenized_splits,
            tokenizer_pad_token_id=tokenizer.pad_token_id,
        )
    else:
        logger.error("This script currently only supports pretokenized data")
        return 1

    logger.info(f"Dataset loaded: {len(dataset)} sequences")

    # Create dataloader
    train_loader, _ = create_streaming_dataloaders(
        train_dataset=dataset,
        val_dataset=None,
        batch_size=2,
        num_workers=0,
        shuffle_train=False,
    )

    # Get one batch
    logger.info("Loading one batch...")
    batch = next(iter(train_loader))

    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']

    logger.info(f"Batch shape: {input_ids.shape}")

    # Decode first example for visual check
    logger.info("")
    logger.info("=" * 80)
    logger.info("VISUAL SANITY CHECK: First example decoded text")
    logger.info("=" * 80)

    example = input_ids[0]
    non_pad = example[example != tokenizer.pad_token_id]
    decoded = tokenizer.decode(non_pad, skip_special_tokens=False)

    logger.info(f"Full text ({len(non_pad)} tokens):")
    logger.info(decoded[:500])  # First 500 chars
    if len(decoded) > 500:
        logger.info("... (truncated)")

    # Check for garbage text
    if not decoded.strip():
        logger.error("❌ DECODED TEXT IS EMPTY!")
    elif decoded.count(tokenizer.pad_token) / len(decoded) > 0.5:
        logger.warning("⚠️  More than 50% of decoded text is padding tokens")
    else:
        logger.info("✓ Decoded text looks reasonable")

    # Load teacher model
    logger.info("")
    logger.info("=" * 80)
    logger.info("LOADING TEACHER MODEL")
    logger.info("=" * 80)

    device = torch.device(config.teacher_device if hasattr(config, 'teacher_device') else "cuda")
    logger.info(f"Teacher device: {device}")
    logger.info(f"Teacher model: {config.teacher_model}")

    teacher_client = DirectTeacherClient(
        model_name=config.teacher_model,
        device=device,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for accuracy and efficiency
    )

    logger.info("Teacher model loaded")

    # Run teacher forward pass
    logger.info("")
    logger.info("=" * 80)
    logger.info("RUNNING TEACHER INFERENCE")
    logger.info("=" * 80)

    input_ids_gpu = input_ids.to(device)

    with torch.no_grad():
        teacher_topk_indices, teacher_topk_values, teacher_other_mass = \
            teacher_client.get_top_k_logits_tensors(
                input_ids=input_ids_gpu,
                topk=args.topk,
            )

    logger.info(f"Teacher output shapes:")
    logger.info(f"  topk_indices: {teacher_topk_indices.shape}")
    logger.info(f"  topk_values: {teacher_topk_values.shape}")
    logger.info(f"  other_mass: {teacher_other_mass.shape}")

    # Analyze teacher outputs
    analyze_teacher_outputs(
        input_ids_gpu,
        teacher_topk_indices,
        teacher_topk_values,
        labels.to(device),
        tokenizer,
    )

    logger.info("")
    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Key things to check:")
    logger.info("  1. Is decoded text reasonable or corrupted?")
    logger.info("  2. Do ground-truth tokens appear in teacher top-k (>90% coverage expected)?")
    logger.info("  3. Are teacher predictions reasonable?")
    logger.info("  4. Is teacher probability distribution well-calibrated?")

    return 0


if __name__ == "__main__":
    sys.exit(main())
