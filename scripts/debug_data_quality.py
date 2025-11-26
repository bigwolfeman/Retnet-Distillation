"""
Debug script to inspect data quality and teacher-student alignment.

This script implements the data quality inspection plan:
1. Load a batch from the training DataLoader
2. Decode and display token IDs and text
3. Verify attention mask coverage
4. Check for padding dominance
5. Validate label alignment

Usage:
    python scripts/debug_data_quality.py --config configs/stage1_kd_350m.yaml --num-batches 5
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from distillation.config import load_yaml_config, TrainingConfig
from distillation.dataset import PretokenizedShardDataset, create_streaming_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def inspect_batch(batch, tokenizer, batch_idx):
    """Inspect a single batch and log detailed information."""
    logger.info("=" * 80)
    logger.info(f"BATCH {batch_idx} INSPECTION")
    logger.info("=" * 80)

    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']

    batch_size, seq_len = input_ids.shape

    logger.info(f"Batch shape: {batch_size} x {seq_len}")
    logger.info(f"Input IDs dtype: {input_ids.dtype}")
    logger.info(f"Attention mask dtype: {attention_mask.dtype}")
    logger.info(f"Labels dtype: {labels.dtype}")

    # Check attention mask statistics
    for i in range(min(batch_size, 3)):  # Inspect first 3 examples
        logger.info("")
        logger.info(f"--- Example {i} ---")

        example_input = input_ids[i]
        example_mask = attention_mask[i]
        example_labels = labels[i]

        # Count valid tokens
        valid_tokens = (example_input != tokenizer.pad_token_id).sum().item()
        valid_mask = example_mask.sum().item()
        valid_labels = (example_labels != -100).sum().item()

        logger.info(f"Valid tokens (non-pad): {valid_tokens}/{seq_len} ({100*valid_tokens/seq_len:.1f}%)")
        logger.info(f"Attention mask sum: {valid_mask}/{seq_len} ({100*valid_mask/seq_len:.1f}%)")
        logger.info(f"Valid labels (not -100): {valid_labels}/{seq_len} ({100*valid_labels/seq_len:.1f}%)")

        # Decode first 20 and last 20 tokens
        logger.info("")
        logger.info("First 20 token IDs:")
        logger.info(f"  input_ids:  {example_input[:20].tolist()}")
        logger.info(f"  attn_mask:  {example_mask[:20].tolist()}")
        logger.info(f"  labels:     {example_labels[:20].tolist()}")

        # Find last non-pad position
        non_pad_positions = (example_input != tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
        if len(non_pad_positions) > 0:
            last_valid_pos = non_pad_positions[-1].item()
            start = max(0, last_valid_pos - 10)
            end = min(seq_len, last_valid_pos + 10)

            logger.info("")
            logger.info(f"Around last valid token (pos {last_valid_pos}):")
            logger.info(f"  input_ids:  {example_input[start:end].tolist()}")
            logger.info(f"  attn_mask:  {example_mask[start:end].tolist()}")
            logger.info(f"  labels:     {example_labels[start:end].tolist()}")

        # Decode text
        logger.info("")
        logger.info("Decoded text (first 200 chars):")
        # Decode only non-pad tokens
        if valid_tokens > 0:
            non_pad_ids = example_input[example_input != tokenizer.pad_token_id]
            decoded_text = tokenizer.decode(non_pad_ids, skip_special_tokens=False)
            logger.info(f"  {decoded_text[:200]}")
        else:
            logger.info("  [ALL PADDING - EMPTY SEQUENCE]")
            logger.warning("  ⚠️  WARNING: Found sequence with no valid tokens!")

        # Check for label/mask alignment issues
        logger.info("")
        logger.info("Alignment checks:")

        # Check if attention mask is properly aligned with labels
        # For next-token prediction:
        # - input_ids[i] is the input token at position i
        # - labels[i] should predict input_ids[i+1]
        # - attention_mask[i] should be 1 if we want to compute loss for labels[i]

        # Find positions where label is valid but mask is 0
        misaligned = ((example_labels != -100) & (example_mask == 0)).sum().item()
        if misaligned > 0:
            logger.warning(f"  ⚠️  Found {misaligned} positions with valid label but mask=0")
            # Show first few misaligned positions
            misaligned_positions = ((example_labels != -100) & (example_mask == 0)).nonzero(as_tuple=True)[0][:5]
            logger.warning(f"  First misaligned positions: {misaligned_positions.tolist()}")
        else:
            logger.info(f"  ✓ No misalignment detected (all valid labels have mask=1)")

        # Check for positions where mask is 1 but label is -100
        masked_labels = ((example_labels == -100) & (example_mask == 1)).sum().item()
        if masked_labels > 0:
            logger.warning(f"  ⚠️  Found {masked_labels} positions with mask=1 but label=-100")
        else:
            logger.info(f"  ✓ No masked labels with attention (all mask=1 positions have valid labels)")

    logger.info("=" * 80)


def check_teacher_student_inputs(batch, tokenizer):
    """Verify that teacher would receive the same inputs as student."""
    logger.info("=" * 80)
    logger.info("TEACHER-STUDENT INPUT ALIGNMENT CHECK")
    logger.info("=" * 80)

    input_ids = batch['input_ids']

    # Simulate what would be sent to teacher
    # For DirectTeacherClient, input_ids is passed directly as tensor
    # For VLLMTeacherClient, input_ids is converted to list via .cpu().tolist()

    logger.info(f"Student receives input_ids with shape: {input_ids.shape}")
    logger.info(f"Student input_ids dtype: {input_ids.dtype}")
    logger.info(f"Student input_ids device: {input_ids.device}")

    # Check first example
    example = input_ids[0]
    non_pad = (example != tokenizer.pad_token_id).nonzero(as_tuple=True)[0]

    if len(non_pad) > 0:
        first_tokens = example[:min(10, len(non_pad))].tolist()
        logger.info(f"First 10 tokens sent to both teacher and student: {first_tokens}")
        logger.info(f"Decoded: {tokenizer.decode(first_tokens, skip_special_tokens=False)}")

        # Check for BOS token
        if tokenizer.bos_token_id is not None:
            has_bos = (example[0] == tokenizer.bos_token_id).item()
            logger.info(f"Sequence starts with BOS token: {has_bos}")
            if not has_bos:
                logger.warning("  ⚠️  Sequence does not start with BOS token!")

    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Debug data quality")
    parser.add_argument("--config", type=str, required=True, help="Path to training config")
    parser.add_argument("--num-batches", type=int, default=3, help="Number of batches to inspect")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for inspection")
    args = parser.parse_args()

    # Load config
    logger.info(f"Loading config from: {args.config}")
    yaml_config = load_yaml_config(args.config)
    config = TrainingConfig.from_dict(yaml_config)

    logger.info(f"Training data path: {config.train_data_path}")
    logger.info(f"Max sequence length: {config.max_seq_length}")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {config.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    logger.info(f"BOS token: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")
    logger.info(f"EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
    logger.info(f"PAD token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")

    # Load dataset
    logger.info("")
    logger.info("Loading dataset...")

    if config.use_pretokenized_data:
        dataset = PretokenizedShardDataset(
            data_path=config.train_data_path,
            max_length=config.max_seq_length,
            splits=config.pretokenized_splits,
            tokenizer_pad_token_id=tokenizer.pad_token_id,
        )
    else:
        logger.error("This script currently only supports pretokenized data")
        logger.error("Set use_pretokenized_data: true in config")
        return 1

    logger.info(f"Dataset loaded: {len(dataset)} sequences")

    # Create dataloader
    train_loader, _ = create_streaming_dataloaders(
        train_dataset=dataset,
        val_dataset=None,
        batch_size=args.batch_size,
        num_workers=0,  # Single process for debugging
        shuffle_train=False,  # Don't shuffle for reproducibility
    )

    logger.info(f"DataLoader created with batch_size={args.batch_size}")
    logger.info("")

    # Inspect batches
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= args.num_batches:
            break

        inspect_batch(batch, tokenizer, batch_idx)

        if batch_idx == 0:
            # Do teacher-student alignment check on first batch only
            check_teacher_student_inputs(batch, tokenizer)

    logger.info("")
    logger.info("=" * 80)
    logger.info("DATA QUALITY INSPECTION COMPLETE")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Check the output above for:")
    logger.info("  1. Empty sequences (all padding)")
    logger.info("  2. Misaligned attention masks and labels")
    logger.info("  3. Corrupted or garbled decoded text")
    logger.info("  4. Missing BOS tokens")
    logger.info("  5. Low valid token percentages")

    return 0


if __name__ == "__main__":
    sys.exit(main())
