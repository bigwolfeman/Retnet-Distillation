# Data Pipeline Quickstart

This guide covers the complete data pipeline: **Download → Verify → Tokenize → Use**.

---

## Overview

The data pipeline consists of three stages:

```
Raw Data (JSONL)  →  Tokenized Parquet  →  Training
     ↓                      ↓                  ↓
download_curriculum    preprocess_to      DataLoader reads
_datasets.py          _parquet.py         manifest.json
```

**Final output:** A `manifest.json` file pointing to Parquet shards that the training script consumes automatically.

---

## Stage 1: Download Data

### Using the Curriculum Downloader

The main downloader intelligently manages your dataset:
- Checks what you already have
- Downloads only missing data
- Organizes by category (instruction, code, math, QA)

```bash
cd data

# See current status and what would be downloaded
python download_curriculum_datasets.py --status-only

# Dry run (show plan without downloading)
python download_curriculum_datasets.py --dry-run

# Download specific category
python download_curriculum_datasets.py --category instruction_chat
python download_curriculum_datasets.py --category code
python download_curriculum_datasets.py --category math_reasoning
python download_curriculum_datasets.py --category factual_qa

# Download everything
python download_curriculum_datasets.py
```

### Dataset Categories

| Category | Target | Description |
|----------|--------|-------------|
| `instruction_chat` | 35% | OpenHermes, UltraChat, OpenOrca, ShareGPT |
| `code` | 25% | StarCoder, The Stack, CodeAlpaca, APPS |
| `math_reasoning` | 20% | NuminaMath, GSM8K, MetaMath, MathInstruct |
| `factual_qa` | 10% | Natural Questions, HotpotQA, SQuAD, TriviaQA |

### Output Structure

```
data/distillation/
├── instruction_chat/
│   ├── openhermes.jsonl
│   ├── openhermes.stats.json
│   ├── ultrachat.jsonl
│   └── ...
├── code/
│   └── ...
├── math_reasoning/
│   └── ...
└── factual_qa/
    └── ...
```

### Verify Download

```bash
# Check file sizes
ls -lh data/distillation/*/*.jsonl

# Count samples per file
wc -l data/distillation/*/*.jsonl

# View a sample
head -1 data/distillation/instruction_chat/openhermes.jsonl | python -m json.tool
```

---

## Stage 2: Tokenize Data

Convert raw JSONL to tokenized Parquet shards.

### Basic Usage

```bash
python preprocess_to_parquet.py \
    --input distillation \
    --output distillation_preprocessed \
    --recursive
```

### Full Options

```bash
python preprocess_to_parquet.py \
    --input distillation \
    --output distillation_preprocessed \
    --recursive \
    --tokenizer meta-llama/Llama-3.2-1B \
    --max-seq-length 1024 \
    --shard-size 2 \
    --skip-existing
```

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | - | Input directory with JSONL files |
| `--output` | - | Output directory for Parquet shards |
| `--recursive` | off | Search subdirectories for JSONL |
| `--tokenizer` | `meta-llama/Llama-3.2-1B` | Tokenizer to use |
| `--max-seq-length` | 8192 | Max tokens per sequence |
| `--shard-size` | 2 | Target shard size in GB |
| `--skip-existing` | off | Skip already-processed files |
| `--force` | off | Re-process everything |

### What Tokenization Does

1. **Loads** each JSONL file
2. **Tokenizes** text with Llama tokenizer
3. **Packs** into fixed-length sequences (concatenates short texts, splits long ones)
4. **Creates** attention masks and labels
5. **Saves** as Snappy-compressed Parquet shards
6. **Generates** `manifest.json` with metadata

### Output Structure

```
data/distillation_preprocessed/
├── manifest.json              # Dataset metadata (required)
├── openhermes/
│   ├── shard_0000.parquet
│   ├── shard_0001.parquet
│   └── openhermes_stats.json
├── ultrachat/
│   └── ...
└── ...
```

### Verify Tokenization

```bash
# Check manifest
cat data/distillation_preprocessed/manifest.json

# Expected format:
# {
#   "tokenizer": "meta-llama/Llama-3.2-1B",
#   "max_seq_length": 1024,
#   "splits": {
#     "openhermes": {
#       "path": "data/distillation_preprocessed/openhermes",
#       "shards": 2,
#       "sequences": 844704,
#       "tokens": 866206327
#     }
#   }
# }

# List parquet files
find data/distillation_preprocessed -name "*.parquet" | head

# Check a shard
python -c "
import pyarrow.parquet as pq
table = pq.read_table('data/distillation_preprocessed/openhermes/shard_0000.parquet')
print(f'Rows: {table.num_rows}')
print(f'Columns: {table.column_names}')
print(f'First input_ids: {table[\"input_ids\"][0].as_py()[:20]}...')
"
```

---

## Stage 3: Use in Training

### Update Training Config

The training script expects `train_data_path` to point to your preprocessed directory:

```yaml
# configs/train_direct.yaml
train_data_path: "data/distillation_preprocessed"
val_data_path: "data/distillation_preprocessed"
use_pretokenized_data: true
```

### Filter Specific Datasets

To train on only certain datasets, use `pretokenized_splits`:

```yaml
# Train only on OpenHermes and NuminaMath
pretokenized_splits: ["openhermes", "numina_cot"]
```

Set to `null` to use all available splits:

```yaml
pretokenized_splits: null  # Use everything
```

### Start Training

```bash
python -m src.distillation.scripts.train --config configs/train_direct.yaml
```

---

## Adding Custom Data

### Step 1: Prepare JSONL

Create a JSONL file with one JSON object per line:

```jsonl
{"text": "Your training text here...", "id": "custom_001", "source": "my_dataset"}
{"text": "Another example...", "id": "custom_002", "source": "my_dataset"}
```

**Required field:** `text`  
**Optional fields:** `id`, `source`, `category`, `metadata`

### Step 2: Place in distillation folder

```bash
mkdir -p data/distillation/custom
mv my_data.jsonl data/distillation/custom/
```

### Step 3: Tokenize

```bash
python preprocess_to_parquet.py \
    --input distillation \
    --output distillation_preprocessed \
    --recursive \
    --skip-existing
```

The `--skip-existing` flag means only your new file will be processed.

### Step 4: Verify

```bash
cat data/distillation_preprocessed/manifest.json
# Should now include your custom dataset in "splits"
```

---

## Troubleshooting

### "No JSONL files found"

```bash
# Check input directory
ls -la data/distillation/

# Use --recursive for nested directories
python preprocess_to_parquet.py --input distillation --recursive
```

### "Tokenizer not found"

```bash
# Login to HuggingFace
huggingface-cli login

# Or set token
export HF_TOKEN="hf_your_token"
```

### Out of Disk Space

```bash
# Check available space
df -h .

# Reduce shard size
python preprocess_to_parquet.py --shard-size 1  # 1GB shards instead of 2GB
```

### Training Can't Find Data

1. Check `train_data_path` in your config matches the output directory
2. Verify `manifest.json` exists in that directory
3. Ensure at least one split has `sequences > 0` in the manifest

---

## Data Format Reference

### Raw JSONL Format

```json
{
  "text": "The actual training text content...",
  "id": "unique_id_001",
  "source": "dataset_name",
  "category": "instruction_chat",
  "metadata": {
    "num_chars": 1234
  }
}
```

### Parquet Schema

```
input_ids:      list<int32>[max_seq_length]   # Token IDs
attention_mask: list<int32>[max_seq_length]   # 1 for real tokens, 0 for padding
labels:         list<int32>[max_seq_length]   # Shifted input_ids, -100 for ignore
```

### Manifest Format

```json
{
  "tokenizer": "meta-llama/Llama-3.2-1B",
  "max_seq_length": 1024,
  "shard_size_gb": 2,
  "input_path": "data/distillation",
  "output_path": "data/distillation_preprocessed",
  "files_processed": 5,
  "splits": {
    "openhermes": {
      "path": "data/distillation_preprocessed/openhermes",
      "shards": 2,
      "sequences": 844704,
      "samples": 2428641,
      "tokens": 866206327
    }
  }
}
```

---

## Quick Reference

```bash
# Download
python download_curriculum_datasets.py --category instruction_chat

# Tokenize
python preprocess_to_parquet.py --input distillation --output distillation_preprocessed --recursive

# Verify
cat data/distillation_preprocessed/manifest.json

# Train
cd .. && python -m src.distillation.scripts.train --config configs/train_direct.yaml
```
