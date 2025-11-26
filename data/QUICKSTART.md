# Data Quickstart Guide

**Last Updated:** 2025-11-04


## Data Locations

### Downloaded Raw Data (Ready to Use)
- **Train:** `data/fineweb_large/train/train.jsonl`
- **Val:** `data/fineweb_large/val/val.jsonl`
- **Format:** JSONL with text, id, source, url, score fields

### Test Data (For Development)
- **Location:** `data/train/` and `data/val/`
- **Size:** ~1 MB
- **Use:** Quick testing only

## Quick Commands

### 1. Preprocess Data (Run This First!)
```bash
python data/preprocess_to_parquet.py
```

This will:
- Tokenize all data with Llama-3.2-1B tokenizer
- Pack into 4096-token sequences
- Create 2GB parquet shards
- Save to `data/fineweb_large_preprocessed/`
- Takes ~30-60 minutes

### 2. Verify Preprocessing
```bash
# Check output
ls -lh data/fineweb_large_preprocessed/train/
ls -lh data/fineweb_large_preprocessed/val/

# View manifest
cat data/fineweb_large_preprocessed/manifest.json
```

### 3. Start Training
```bash
# Direct mode (teacher + training in one process)
python src/distillation/scripts/train.py --config configs/train_direct.yaml

# Cached mode (use pre-computed logits)
python src/distillation/scripts/train.py --config configs/train_cached.yaml

# Network mode (separate teacher server)
# Terminal 1: Start teacher server
python src/teacher/server.py --model meta-llama/Llama-70B

# Terminal 2: Start training
python src/distillation/scripts/train.py --config configs/train_network.yaml
```

## Dataset Statistics

### Training Data
- **Samples:** 500,000
- **Size:** 2.4 GB
- **Avg length:** 4,756 chars (773 words, ~778 tokens)
- **Sequences:** ~95,000 (at 4096 max length)
- **Steps/epoch:** ~5,900 (batch=16) or ~2,950 (batch=32)

### Validation Data
- **Samples:** 10,000
- **Size:** 46 MB
- **Avg length:** 4,600 chars (750 words, ~770 tokens)

## Training Time Estimates

Assuming:
- Batch size: 16
- Steps per epoch: 5,900
- Step time: varies by mode

**Direct Mode (CPU):**
- ~2-5 sec/step
- ~3-8 hours/epoch

**Cached Mode (GPU):**
- ~0.5-1 sec/step
- ~1-2 hours/epoch

**Network Mode (GPU):**
- ~1-2 sec/step
- ~2-3 hours/epoch

## Troubleshooting

### If preprocessing fails
```bash
# Check tokenizer is available
python -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B'); print('OK')"

# Check disk space
df -h /mnt/BigAssDrive

# Check input files
ls -lh data/fineweb_large/train/train.jsonl
ls -lh data/fineweb_large/val/val.jsonl
```

### If training fails to load data
```bash
# Verify parquet files exist
ls data/fineweb_large_preprocessed/train/*.parquet
ls data/fineweb_large_preprocessed/val/*.parquet

# Check manifest
cat data/fineweb_large_preprocessed/manifest.json
```

## Scripts Reference

### Download Scripts
- `data/download_fineweb.py` - Small dataset (50k samples)
- `data/download_fineweb_large.py` - Large dataset (500k samples) ✅ Used
- `data/test_tokenization.py` - Test tokenizer compatibility

### Preprocessing Scripts
- `data/preprocess_to_parquet.py` - Main preprocessing script ⏭️ Run Next

### Training Scripts
- `src/distillation/scripts/train.py` - Main training script
- `src/teacher/server.py` - Teacher server (for network mode)

## Next Steps for Tomorrow

1. **Run preprocessing** (~30-60 min):
   ```bash
   python data/preprocess_to_parquet.py
   ```

2. **Verify output** (~1 min):
   ```bash
   ls -lh data/fineweb_large_preprocessed/train/
   cat data/fineweb_large_preprocessed/manifest.json
   ```

3. **Update training configs** to point to new data (if needed)

4. **Start training** with your preferred mode

5. **Monitor with Weights & Biases**:
   ```bash
   wandb login
   # Training will automatically log to W&B
   ```

## Storage Info

- **Total disk:** 15 TB
- **Used:** 6.4 TB
- **Available:** 8.2 TB
- **Data location:** /mnt/BigAssDrive (NVMe)
- **New data size:** ~2.7 GB (raw + preprocessed will be ~5-6 GB total)

Plenty of space for training checkpoints and cached logits!

---

**Ready to train!**
Run preprocessing first, then start training tomorrow.
