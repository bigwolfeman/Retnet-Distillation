# Checkpoint Evaluation Script

Manual quality assessment tool for evaluating student model outputs against teacher outputs.

## Overview

`evaluate_checkpoint.py` loads a trained student checkpoint alongside the teacher model and generates comparison outputs on validation samples. This enables human quality assessment of distillation progress beyond automated metrics.

## Features

- **Sample Caching**: First run randomly samples ~30 examples from validation data and caches to `data/eval_samples_cache.jsonl`. Subsequent runs reuse the same samples for reproducible comparisons.
- **Teacher/Student Comparison**: Generates outputs from both models on identical inputs with side-by-side display
- **Rich Terminal Output**: Color-coded, formatted output with progress bars (requires `rich` library)
- **Markdown Reports**: Detailed reports saved to `eval_results/` with full sample comparisons
- **Automated Metrics**: Token counts, BLEU scores, generation time, length ratios
- **Quality Flags**: Automatic detection of incomplete, empty, too-short, or too-long outputs

## Installation

Optional dependencies for enhanced output:

```bash
# For rich terminal formatting (highly recommended)
pip install rich

# For BLEU scores (optional)
pip install sacrebleu
```

## Usage

### Basic Evaluation

Evaluate a checkpoint with default settings (30 samples, temperature 0.7):

```bash
python scripts/evaluate_checkpoint.py --checkpoint runs/direct_mode/checkpoint-10000
```

### Custom Sample Count

Evaluate with more samples for thorough assessment:

```bash
python scripts/evaluate_checkpoint.py --checkpoint runs/direct_mode/checkpoint-10000 --num-samples 50
```

### Regenerate Sample Cache

Force regenerate the evaluation sample cache (useful if validation data changed):

```bash
python scripts/evaluate_checkpoint.py --checkpoint runs/direct_mode/checkpoint-10000 --regenerate-cache
```

### Greedy Decoding

Use greedy decoding (temperature=0) for deterministic outputs:

```bash
python scripts/evaluate_checkpoint.py --checkpoint runs/direct_mode/checkpoint-10000 --temperature 0.0
```

### Custom Config

Use a different training config file:

```bash
python scripts/evaluate_checkpoint.py \
    --checkpoint runs/direct_mode/checkpoint-10000 \
    --config configs/train_cached.yaml
```

### CPU-Only Evaluation

Run on CPU (useful for debugging on machines without GPU):

```bash
python scripts/evaluate_checkpoint.py --checkpoint runs/direct_mode/checkpoint-10000 --device cpu
```

## Output

### Terminal Output

The script displays:

1. **Summary Table**: Aggregate metrics across all samples
   - Total samples evaluated
   - Average token counts (teacher vs student)
   - Average length ratio
   - Average generation time
   - BLEU score (if sacrebleu installed)
   - Quality flags (incomplete, empty, too short, too long)

2. **Sample Comparisons**: First 5 samples with:
   - Input prompt
   - Teacher output
   - Student output
   - Per-sample metrics

### Markdown Report

Saved to `eval_results/checkpoint-{step}_eval_{timestamp}.md`:

- Full checkpoint information (path, step, timestamp)
- Summary statistics table
- Complete outputs for ALL samples (not just first 5)
- Individual metrics for each sample
- Quality flags

See `eval_results/example_checkpoint-10000_eval_20251116_120000.md` for an example report.

## Configuration

The script loads configuration from the training config file (default: `configs/train_direct.yaml`) to determine:

- Teacher model name and adapter path
- Student model variant (350M/500M)
- Tokenizer name
- Validation data path

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--checkpoint` | *required* | Path to student checkpoint file |
| `--config` | `configs/train_direct.yaml` | Path to training config YAML |
| `--num-samples` | `30` | Number of validation samples to evaluate |
| `--max-new-tokens` | `512` | Maximum new tokens to generate per sample |
| `--temperature` | `0.7` | Sampling temperature (0 = greedy) |
| `--regenerate-cache` | `false` | Force regenerate evaluation sample cache |
| `--device` | `cuda` | Device to use (cuda or cpu) |
| `--output-dir` | `eval_results` | Output directory for markdown reports |

## Sample Cache

- **Location**: `data/eval_samples_cache.jsonl`
- **Format**: JSONL with one sample per line
  ```json
  {"input_ids": [1, 2, 3, ...], "text": "decoded prompt"}
  ```
- **Reproducibility**: Same samples used across evaluations unless regenerated
- **Regeneration**: Use `--regenerate-cache` to force new random sample

## Metrics Explained

### Automated Metrics

- **Token Counts**: Number of tokens generated (teacher vs student)
- **Length Ratio**: `student_tokens / teacher_tokens`
- **Generation Time**: Wall-clock time for generation (seconds)
- **BLEU Score**: Corpus-level BLEU score (0-100, higher is better)

### Quality Flags

- **Incomplete**: Student hit `max_new_tokens` limit (generation truncated)
- **Empty**: Student generated zero tokens (complete failure)
- **Too Short**: Length ratio < 0.5 (student significantly shorter than teacher)
- **Too Long**: Length ratio > 2.0 (student significantly longer than teacher)

## Example Workflow

1. **Train for 5K steps**:
   ```bash
   python -m src.distillation.scripts.train --config configs/train_direct.yaml
   ```

2. **Evaluate checkpoint-5000**:
   ```bash
   python scripts/evaluate_checkpoint.py --checkpoint runs/direct_mode/checkpoint-5000
   ```

3. **Review markdown report** in `eval_results/`

4. **Continue training for 5K more steps**

5. **Evaluate checkpoint-10000**:
   ```bash
   python scripts/evaluate_checkpoint.py --checkpoint runs/direct_mode/checkpoint-10000
   ```

6. **Compare reports** to assess improvement

## Troubleshooting

### Out of Memory

If evaluation fails with OOM:

1. Reduce `--num-samples` (e.g., `--num-samples 10`)
2. Reduce `--max-new-tokens` (e.g., `--max-new-tokens 256`)
3. Use CPU: `--device cpu`

### Checkpoint Not Found

Ensure the checkpoint path is correct:

```bash
# List available checkpoints
ls runs/direct_mode/checkpoint_*.pt

# Or use absolute path
python scripts/evaluate_checkpoint.py --checkpoint /absolute/path/to/checkpoint_10000.pt
```

### Teacher Adapter Not Found

If the config specifies `teacher_adapter_path` but the adapter is missing:

1. Remove the adapter path from the config (will use base teacher)
2. Or finetune the teacher first: `make finetune-teacher`

### Import Errors

If you see import errors:

```bash
# Ensure you're in the project root
cd /path/to/000Distill-Titan-Retnet-HRM

# Run with python -m for proper module resolution
python -m scripts.evaluate_checkpoint --checkpoint runs/direct_mode/checkpoint-10000
```

## Notes

- **Primary focus**: Human-readable decoded outputs (not raw logits)
- **Secondary**: Automated metrics for trend tracking
- **Use case**: Manual quality assessment during training
- **Complement to**: Automated eval (perplexity, NIAH) in training loop

## Related Files

- `src/distillation/scripts/train.py` - Main training script
- `src/distillation/evaluation/perplexity.py` - Automated perplexity evaluation
- `configs/train_direct.yaml` - Training configuration
- `eval_results/` - Output directory for markdown reports
