# DualEncoder Training Configurations

This directory contains configuration files for training the dual encoder used in RetNet-HRM's retrieval system.

## Available Configurations

### 1. `baseline.yaml`
Basic training configuration without hard negative mining.

**Use when:**
- Starting encoder training from scratch
- Initial convergence phase
- Limited computational resources

**Expected results:**
- MRR: 0.6-0.7 after 50K steps
- Training time: ~6-8 hours on single A100
- Model size: 77M parameters

**Command:**
```bash
python src/cli/train_encoder.py --config configs/encoder/baseline.yaml
```

### 2. `with_hard_negatives.yaml`
Advanced configuration with FAISS-based hard negative mining.

**Use when:**
- Fine-tuning a pre-trained encoder
- MRR has plateaued on baseline
- You have > 10K validation examples

**Expected results:**
- MRR: 0.7-0.8 (5-10% improvement over baseline)
- Training time: ~3-4 hours on single A100
- Requires FAISS index updates (adds 5-10% overhead)

**Command:**
```bash
# Start from baseline checkpoint
python src/cli/train_encoder.py \
    --config configs/encoder/with_hard_negatives.yaml \
    --resume-from checkpoints/encoder/encoder-best-mrr.pt
```

## Quick Start

### Step 1: Prepare Data

Create training data in JSONL format:

```jsonl
{"query": "function to sort list", "code": "def sort_list(lst):\n    return sorted(lst)"}
{"query": "binary search implementation", "code": "def binary_search(arr, target):..."}
```

Place files at:
- `data/code_pairs/train.jsonl`
- `data/code_pairs/val.jsonl`

Or update paths in the config files.

### Step 2: Train Baseline Model

```bash
python src/cli/train_encoder.py --config configs/encoder/baseline.yaml
```

Monitor training in Wandb. Wait until MRR converges (plateau).

### Step 3: (Optional) Fine-tune with Hard Negatives

```bash
python src/cli/train_encoder.py \
    --config configs/encoder/with_hard_negatives.yaml \
    --resume-from checkpoints/encoder/encoder-best-mrr.pt
```

### Step 4: Use Encoder for Index Building

The best checkpoint will be saved at:
```
checkpoints/encoder/encoder-best-mrr.pt
```

Use this checkpoint for building the FAISS index (T055).

## Configuration Parameters

### Model Parameters

| Parameter | Description | Default | Typical Range |
|-----------|-------------|---------|---------------|
| `vocab_size` | Tokenizer vocabulary size | 50257 | Fixed (match tokenizer) |
| `d_model` | Embedding dimension | 768 | 512-1024 |
| `n_layers` | Number of layers | 6 | 4-12 |
| `n_heads` | Number of attention heads | 12 | 8-16 |
| `max_seq_len` | Max sequence length | 512 | 256-1024 |

### Training Parameters

| Parameter | Description | Default | Typical Range |
|-----------|-------------|---------|---------------|
| `batch_size` | Per-device batch size | 32 | 16-128 |
| `learning_rate` | Peak learning rate | 1e-4 | 5e-5 to 2e-4 |
| `max_steps` | Total training steps | 50000 | 20K-100K |
| `temperature` | Contrastive loss temperature | 0.07 | 0.05-0.1 |
| `use_hard_negatives` | Enable hard negative mining | false | true/false |

### Hardware Requirements

**Baseline training:**
- GPU: 1x A100 (40GB) or V100 (32GB)
- RAM: 32GB
- Disk: 10GB (checkpoints + logs)

**With hard negatives:**
- GPU: 1x A100 (40GB) recommended
- RAM: 64GB (for FAISS index)
- Disk: 20GB

## Monitoring Training

### Key Metrics

1. **train/loss**: Contrastive loss (lower is better)
   - Good: < 0.5 after 10K steps
   - Excellent: < 0.3 after 50K steps

2. **val/mrr**: Mean Reciprocal Rank (higher is better)
   - Good: > 0.6
   - Excellent: > 0.75

3. **val/ndcg@10**: NDCG at 10 (higher is better)
   - Good: > 0.7
   - Excellent: > 0.85

### Troubleshooting

**MRR not improving:**
- Check data quality (are queries matched to correct codes?)
- Try different learning rate (5e-5 or 2e-4)
- Increase batch size for more negatives
- Adjust temperature (0.05 or 0.1)

**High memory usage:**
- Reduce batch_size
- Reduce max_seq_len
- Use fp32 precision (sometimes uses less memory)
- Disable hard_negatives

**Training too slow:**
- Increase batch_size
- Reduce eval_interval
- Disable hard_negatives
- Use fewer num_workers

## Integration with Main Model

After training the encoder, integrate with main model training:

1. **Build FAISS index** (T055):
   ```bash
   python src/cli/build_index.py \
       --encoder-checkpoint checkpoints/encoder/encoder-best-mrr.pt \
       --code-corpus data/knowledge_base/ \
       --output-dir indices/global_kb/
   ```

2. **Configure main model** to use retrieval:
   ```yaml
   # In configs/experiments/your_config.yaml
   retrieval:
     enabled: true
     encoder_checkpoint: checkpoints/encoder/encoder-best-mrr.pt
     faiss_index: indices/global_kb/index.faiss
   ```

3. **Train main model**:
   ```bash
   python src/cli/train.py --config configs/experiments/your_config.yaml
   ```

## References

- Implementation: `src/cli/train_encoder.py`
- Encoder architecture: `src/retrieval_index/dual_encoder.py`
- FAISS builder: `src/retrieval_index/faiss_builder.py`
- Integration guide: `docs/implementation/t054-encoder-training-integration.md`
