# Quickstart: From Clone to Training

This guide walks you through the complete pipeline: **Clone → Install → Download Data → Tokenize → (Optional) Finetune Teacher → Train**.

**Estimated Time:**
- Setup + Data: ~30 minutes
- Teacher Finetuning: ~2-4 hours (strongly recommended)
- Training: Ongoing (checkpoints save automatically)

---

## Prerequisites

| Requirement | Minimum | Recommended |
|------------|---------|-------------|
| Python | 3.10 | 3.11 |
| GPU VRAM | 16GB | 24GB+ |
| CUDA | 11.8+ | 12.1+ |
| Disk Space | 20GB | 50GB+ |

**Accounts needed:**
- [HuggingFace](https://huggingface.co/) token (for Llama models)
- [Weights & Biases](https://wandb.ai/) account (optional, for logging)

---

## Step 1: Clone & Install

```bash
# Clone the repository
git clone <repo-url> retnet-distillation
cd retnet-distillation

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install the project (includes patched TorchScale)
pip install -e .
```

**Verify GPU access:**

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

---

## Step 2: Set Environment Variables

```bash
# Required: HuggingFace token for Llama models
export HF_TOKEN="hf_your_token_here"

# Optional: Weights & Biases for live logging
export WANDB_API_KEY="your_wandb_key"

# Recommended: Prevent CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## Step 3: Download Data

The curriculum downloader automatically checks what you already have and downloads only what's missing.

```bash
cd data

# See what would be downloaded (dry run)
python download_curriculum_datasets.py --dry-run

# Download instruction/chat data (~35% of curriculum, ~5GB)
python download_curriculum_datasets.py --category instruction_chat

# Or download everything (~25GB total)
python download_curriculum_datasets.py
```

**Output:** JSONL files in `data/distillation/` organized by category.

For more data options, see [data/QUICKSTART.md](data/QUICKSTART.md).

---

## Step 4: Tokenize Data

Convert raw JSONL to tokenized Parquet shards for efficient training.

```bash
# From the data/ directory
python preprocess_to_parquet.py \
    --input distillation \
    --output distillation_preprocessed \
    --recursive \
    --max-seq-length 1024

# Go back to project root
cd ..
```

**What this does:**
- Tokenizes with Llama-3.2 tokenizer
- Packs into fixed-length sequences (1024 tokens)
- Creates Snappy-compressed Parquet shards
- Generates `manifest.json` for the data loader

**Verify:**

```bash
cat data/distillation_preprocessed/manifest.json
```

---

## Step 5: Finetune the Teacher (Strongly Recommended)

> ⚠️ **This step is critical for convergence.** Without teacher finetuning, distillation on pure logits struggles to converge. Microsoft research shows **4.5× faster convergence** when the teacher is adapted to your dataset first.

```bash
python scripts/finetune_teacher.py --config configs/teacher_ft.yaml
```

**Requirements:**
- ~16GB VRAM (uses LoRA adapters)
- ~2-4 hours on RTX 4090/5090

**Output:** Adapters saved to `teacher_adapters/llama-1b-corrected/`

The training config (`configs/train_direct.yaml`) already points to this path:

```yaml
teacher_adapter_path: "teacher_adapters/llama-1b-corrected/final_adapter"
```

**If you skip this step:** The pipeline will work, but expect:
- Much slower convergence
- Higher final loss
- Potentially unstable training

To train without a finetuned teacher, comment out the `teacher_adapter_path` line in the config.

---

## Step 6: Start Training

### Direct Mode (Recommended for Single GPU)

```bash
python -m src.distillation.scripts.train \
    --config configs/train_direct.yaml \
    --learning-rate 1e-4 \
    --weight-decay 0.08
```

**What to watch for:**
- First few batches: Loss should be high (~70-100), then drop rapidly
- After ~100 steps: Loss should be below 20
- After ~1000 steps: Loss should be below 10
- Checkpoints save every 5000 steps to `runs/direct_mode/`

### Cached Mode (Pre-computed Logits)

```bash
# First: Cache teacher logits (one-time)
python scripts/cache_teacher_logits.py \
    --data-path data/distillation_preprocessed \
    --output-dir data/teacher_cache \
    --teacher-model meta-llama/Llama-3.2-1B-Instruct

# Then: Train from cache
python -m src.distillation.scripts.train --config configs/train_cached.yaml
```

### Network Mode (Separate Teacher Server)

```bash
# Terminal 1: Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --host 0.0.0.0 \
    --port 8080 \
    --api-key token-abc123

# Terminal 2: Train
python -m src.distillation.scripts.train --config configs/train_network.yaml
```

Update `configs/train_network.yaml` with your server URL:

```yaml
teacher_url: "http://<your-server-ip>:8080"
```

For 100x faster logit extraction, install the custom `/v1/topk` endpoint. See [README_FAST_ENDPOINT.md](README_FAST_ENDPOINT.md).

---

## Step 7: Monitor Training

### Weights & Biases (Recommended)

If you set `WANDB_API_KEY`, training automatically logs to W&B:
- Loss curves (total, CE, KL)
- Gradient norms
- Learning rate schedule
- Perplexity evaluations
- Saddle point detection events
- Memory usage

### Terminal Logs

Training prints progress every 30 steps:

```
Step 100 | Loss: 15.23 | LR: 1.00e-04 | Grad: 0.45 | Time: 0.12s/step
```

### Checkpoints

Checkpoints save to `runs/direct_mode/`:
- `checkpoint_latest.pt` - Most recent
- `checkpoint_5000.pt`, `checkpoint_10000.pt`, etc.

Resume from checkpoint:

```bash
python -m src.distillation.scripts.train \
    --config configs/train_direct.yaml \
    --resume-from runs/direct_mode/checkpoint_latest.pt
```

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `CUDA out of memory` | VRAM exhausted | Reduce `max_seq_length` or `physical_batch_size` in config |
| Loss stuck at ~25+ | Tokenization mismatch | Ensure `max_seq_length` matches between preprocessing and training |
| Loss NaN/Inf | Gradient explosion | Reduce `learning_rate`, increase `warmup_batches` |
| Teacher fetch slow | Model not cached | Run training once to cache HF model, or use `huggingface-cli login` |
| `No JSONL files found` | Wrong data path | Check `train_data_path` in config matches your preprocessed data |

### Quick Sanity Check

Run 10 steps to verify everything works:

```bash
python -m src.distillation.scripts.train \
    --config configs/train_direct.yaml \
    --max-steps 10 \
    --log-interval 1 \
    --eval-interval 0
```

---

## CE Pretraining (Alternative: No Teacher)

If you want to pretrain the student **without** a teacher (pure cross-entropy on tokens):

```bash
python -m src.distillation.scripts.train \
    --config configs/train_direct.yaml \
    --pretrain-ce-only \
    --max-steps 10000
```

This uses ~9GB VRAM and doesn't load the teacher. Useful for:
- Bootstrapping before distillation
- Limited VRAM situations
- Debugging student architecture

Then resume with full distillation:

```bash
python -m src.distillation.scripts.train \
    --config configs/train_direct.yaml \
    --resume-from runs/direct_mode/checkpoint_latest.pt
```

---

## Next Steps

1. **Evaluate checkpoints**: See `scripts/evaluate_checkpoint.py`
2. **Adjust hyperparameters**: Sweep `teacher_topk`, `distill_alpha`, `temperature`
3. **Scale up**: Increase `max_seq_length` (2048, 4096) for longer context
4. **Try 350M model**: Set `model_variant: "350M"` for faster iteration

---

**Happy Distilling!** 🚀

