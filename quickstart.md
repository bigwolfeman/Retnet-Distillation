# Quickstart: Data → Tokenization → Training

This walkthrough gets you from an empty checkout to a RetNet-HRM distillation run using the direct-teacher command:

```bash
python -m src.distillation.scripts.train \
    --config configs/train_direct.yaml \
    --weight-decay 0.01 \
    --enable-saddle-interventions \
    --learning-rate 0.001
```

Follow the steps below in order; each section takes ~10–20 minutes on a fast connection/NVMe machine.

---

## 0. Prerequisites

- **Python**: 3.10 or 3.11 with `pip`
- **GPU / CUDA**: NVIDIA GPU with ≥24 GB VRAM, CUDA 11.8+, latest drivers
- **Accounts**:
  - Hugging Face token (needed for `meta-llama/Llama-3.2-1B`)
  - (Optional) Weights & Biases account for logging
- **System packages**: `git`, `wget`, `gzip`, `tmux`/`screen` for long jobs

Set helpful environment variables before you begin:

```bash
export HF_TOKEN="hf_xxx"            # Hugging Face token
export WANDB_API_KEY="..."          # Optional, enables live logging
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## 1. Clone & Install

```bash
git clone <repo-url> 000Distill-Titan-Retnet-HRM
cd 000Distill-Titan-Retnet-HRM

# (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install project + TorchScale fork
pip install -e .
# or: pip install -r requirements.txt
```

Verify PyTorch can see your GPU:

```bash
python - <<'PY'
import torch
print("CUDA:", torch.cuda.is_available(), "device:", torch.cuda.get_device_name(0))
PY
```

---

## 2. Download a Baseline Dataset (≈5 GB)

The fastest way to assemble a balanced PoC corpus is the curated downloader in `data/download_missing_datasets.py`. It fetches ~300 k mixed math/code samples (MetaMath, CodeSearchNet, APPS, MBPP) and puts them under `data/distillation/`.

```bash
cd data
# Streams ~300k samples; Ctrl+C safe between datasets
python download_missing_datasets.py
cd ..
```

You should now see JSONL shards such as `data/distillation/math_reasoning/metamath.jsonl`.

**Need more/other data?**
- `python data/download_curriculum_datasets.py --dry-run` shows the full curriculum plan.
- Pass `--category code` / `--category instruction_chat` to focus on specific domains.

---

## 3. Tokenize & Pack to Parquet

The trainer expects manifest-based parquet shards (input_ids, attention_mask, labels). Use the provided tokenizer pipeline (defaults already point at the Llama‑3.2 tokenizer and 8 K sequences).

```bash
python data/preprocess_to_parquet.py \
    --input data/distillation \
    --output data/distillation_preprocessed \
    --recursive \
    --skip-existing \
    --max-seq-length 8192
```

What this does:
- Walks every JSONL file under `data/distillation/`
- Tokenizes with `meta-llama/Llama-3.2-1B`
- Packs into 8 192-token chunks
- Writes snappy-compressed parquet shards + `manifest.json` in `data/distillation_preprocessed/`

**Verify output**

```bash
ls data/distillation_preprocessed
head data/distillation_preprocessed/manifest.json
```

If you add more raw data later, re-run the command with `--skip-existing` (fast) or `--force` (full re-tokenize).

---

## 4. Point the Training Config at Your Data

`configs/train_direct.yaml` already references `data/distillation_preprocessed` for both train/val paths:

```yaml
train_data_path: "data/distillation_preprocessed"
val_data_path: "data/distillation_preprocessed"
max_seq_length: 8192          # keep in sync with preprocessing
num_workers: 0                # avoids PyArrow cache thrash
use_pretokenized_data: true   # manifest-aware loader
```

If you used a different output directory, update those two lines or symlink your dataset into `data/distillation_preprocessed`.

---

## 5. Finetune the Teacher (Highly Recommended)

**Important:** Finetuning the teacher on your distillation dataset provides a **4.5× convergence speedup** and significantly better logit alignment. The teacher learns your dataset's distribution, producing more informative soft targets for the student. This step is critical for achieving state-of-the-art distillation results.

**Requirements:**
- VRAM: 16 GB (uses LoRA rank-64 adapters)
- Time: ~1 epoch on 5B tokens (2-4 hours on RTX 5090)
- Storage: ~200 MB for adapter weights

**Run teacher finetuning:**

```bash
# Initial training (optimized for speed with Flash Attention 2)
python scripts/finetune_teacher.py --config configs/teacher_ft.yaml --ram-disk

# Or use the Makefile shortcut:
make finetune-teacher

# Resume from checkpoint if training was interrupted:
python scripts/finetune_teacher.py \
    --config configs/teacher_ft.yaml \
    --ram-disk \
    --resume-from teacher_adapters/llama-1b-corrected/checkpoint-2000
```

**What to monitor:**
- Training loss should steadily decrease (target: 10%+ improvement over base)
- Validation perplexity drops by 10%+ vs. base teacher
- Adapter checkpoints saved to `teacher_adapters/`

**Expected outcomes:**
- Merged checkpoint: `teacher_adapters/llama-1b-corrected/`
- LoRA adapters: `teacher_adapters/llama-1b-corrected/adapter_model.bin`
- Validation metrics logged to console and WandB (if enabled)

**Using the finetuned teacher:**

After finetuning completes, update your distillation config to use the adapted teacher:

```yaml
# configs/train_direct.yaml
teacher_model: "teacher_adapters/llama-1b-corrected"  # Use finetuned model
# OR use adapter path (more memory efficient)
teacher_adapter_path: "teacher_adapters/llama-1b-corrected"  # Load adapters on base model
```

**Skip this step if:**
- You need to start training immediately (can finetune later)
- Your dataset is very small (<1B tokens)
- You're doing quick prototyping/debugging

Research shows domain-adapted teachers accelerate convergence by 3-5x (MiniLLM, NeMo-Aligner papers). This 2-4 hour investment pays off across all subsequent training runs.

---

## 6. Launch Direct-Mode Distillation

```bash
python -m src.distillation.scripts.train \
    --config configs/train_direct.yaml \
    --weight-decay 0.01 \
    --enable-saddle-interventions \
    --learning-rate 0.001
```

Key notes:
- Direct mode loads the 1 B teacher + student in the same process (needs ~14 GB VRAM).
- `teacher_topk` defaults to 512; adjust in the config if you need a lighter run.
- Logs stream to stdout and (if `WANDB_API_KEY` is set) to Weights & Biases.
- Checkpoints land in `runs/direct_mode/`.

**First-run checklist**
1. Watch the first few batches for NaNs or exploding loss.
2. Confirm the dataloader speed (no multi-worker stalls because `num_workers=0`).
3. After ~50 steps, expect loss to drop below ~15; if it stalls, inspect data stats and teacher fetch logs.
4. If you used the finetuned teacher, expect faster convergence (5-10% better loss within 1000 steps).

---

## 6b. Alternative: CE Pretraining (No Teacher Required)

If you want to pretrain the student model **without** a teacher—using pure cross-entropy loss on the tokenized data—you can use CE pretrain mode. This is useful for:

- **Bootstrapping**: Train a baseline model before introducing knowledge distillation
- **Resource constraints**: Runs with ~9 GB VRAM (vs ~14 GB for full distillation)
- **Debugging**: Isolate student model issues from teacher-related problems
- **Curriculum learning**: Pretrain on easy data, then distill on harder data

```bash
# CE Pretrain Mode: Train without teacher (~9 GB VRAM)
python -m src.distillation.scripts.train \
    --config configs/train_direct.yaml \
    --pretrain-ce-only \
    --max-steps 10000
```

Key differences from distillation mode:
- No teacher model is loaded (saves ~4 GB VRAM)
- Uses standard cross-entropy loss instead of sparse-KL distillation
- NIAH evaluation is skipped (requires teacher logits)
- Perplexity evaluation still works normally

**Resuming with Knowledge Distillation:**

After CE pretraining, you can resume training with full knowledge distillation:

```bash
# Resume from CE pretrain checkpoint in KD mode
python -m src.distillation.scripts.train \
    --config configs/train_direct.yaml \
    --resume-from runs/direct_mode/checkpoint_latest.pt
```

The system will:
1. Load the pretrained student weights
2. Initialize the teacher model
3. Continue training with sparse-KL distillation loss
4. Log the mode transition for provenance tracking

**When to use CE pretraining:**
- You have limited VRAM and want to start training immediately
- You're iterating on student architecture and don't need teacher feedback yet
- You want to establish a baseline before adding distillation

**When to skip CE pretraining:**
- You have sufficient VRAM for full distillation (~14 GB)
- You want optimal final model quality (teacher guidance from step 1 helps)
- You've already finetuned the teacher (maximizes distillation benefit)

---

## 7. Optional Validation + Monitoring

- **Sanity eval**: `python -m src.distillation.scripts.train --config configs/train_direct.yaml --max-steps 10 --log-interval 1 --eval-interval 0`
- **Perplexity-only pass**: `python -m src.distillation.evaluation.perplexity --config configs/train_direct.yaml --split val`
- **Teacher caching**: Once a run looks good, enable cached logits via `cache_logits: true` + `cache_dir: data/teacher_cache` in the config, warm the cache, then switch to `configs/train_cached.yaml` for cheap replays.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| `RuntimeError: No JSONL files found` when preprocessing | Wrong `--input` path | `ls data/distillation` to confirm download succeeded, rerun with `--recursive` |
| Training hangs between steps | Dataloader cache misses | Ensure `num_workers: 0` (already set) or consolidate shards |
| Teacher fetch takes >1 s | Downloaded model not cached / HF auth failure | Run `huggingface-cli login`, rerun training to let the model cache |
| Loss hovers at 25+ | Tokenization mismatch | Verify `max_seq_length` matches preprocessing; re-tokenize or trim config |

---

### Next Steps

- Sweep `teacher_topk`, temperature, and `distill_alpha` once the baseline run completes.
- Use `python data/preprocess_to_parquet.py --input data/distillation --output data/distillation_preprocessed --recursive --skip-existing --max-seq-length 8192` any time you add new raw JSONL.
- Share the `runs/direct_mode/train.log` + W&B run when submitting hackathon results.

Happy distilling!
