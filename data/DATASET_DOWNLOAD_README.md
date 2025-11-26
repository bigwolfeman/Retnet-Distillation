# Knowledge Distillation Dataset Downloader

Comprehensive dataset downloader for knowledge distillation training with the optimal dataset mix for training smaller language models.

## Overview

This downloader manages multiple datasets across four key categories with research-backed ratios:

- **35% General Instruction & Chat** - Multi-turn conversations and instruction following
- **30% Code** - Programming tasks, repository-grounded code, tests
- **25% Math & Logic** - Mathematical reasoning with chain-of-thought
- **10% Factual QA** - Question answering and multi-hop reasoning

## Quick Start

### 1. Install Dependencies

```bash
pip install datasets pyyaml tqdm
```

### 2. Basic Usage (with defaults)

Download 100k samples with default mix:

```bash
python data/download_distillation_datasets.py
```

### 3. Using YAML Configuration

Download using custom configuration:

```bash
python data/download_distillation_datasets.py \
    --config configs/dataset_download.yaml
```

### 4. Download Specific Dataset

Download only one dataset:

```bash
python data/download_distillation_datasets.py \
    --config configs/dataset_download.yaml \
    --dataset gsm8k
```

## Configuration

### YAML Configuration File

The `configs/dataset_download.yaml` file controls all aspects of downloading:

```yaml
# Overall settings
total_samples: 100000
output_dir: "/path/to/data/distillation"

# Category ratios (must sum to 1.0)
category_ratios:
  instruction_chat: 0.35
  code: 0.30
  math_reasoning: 0.25
  factual_qa: 0.10

# Individual datasets
datasets:
  openhermes:
    enabled: true
    category: instruction_chat
    allocation: 0.40  # 40% of instruction_chat samples
    # ... more config
```

### Customizing the Mix

To adjust the dataset mix:

1. **Change category ratios**: Modify `category_ratios` (must sum to 1.0)
2. **Enable/disable datasets**: Set `enabled: false` for any dataset
3. **Adjust allocations**: Change `allocation` within each category (should sum to 1.0 per category)
4. **Set total samples**: Adjust `total_samples` for your needs

Example - More code, less chat:

```yaml
category_ratios:
  instruction_chat: 0.25  # Reduced from 35%
  code: 0.45              # Increased from 30%
  math_reasoning: 0.20    # Reduced from 25%
  factual_qa: 0.10        # Same
```

## Supported Datasets

### Instruction & Chat (35%)

| Dataset | License | Description | Allocation |
|---------|---------|-------------|------------|
| OpenHermes-2.5 | Apache 2.0 | High-quality multi-turn conversations | 40% |
| UltraChat | MIT | Cleaned multi-turn dialogues | 40% |
| Open-Orca | MIT | System-prompted instructions | 20% |

**Special Features:**
- Progressive prompt length filtering (short prompts early, longer later)
- Multi-turn conversation preservation
- Quality filtering

### Code (30%)

| Dataset | License | Description | Allocation |
|---------|---------|-------------|------------|
| The Stack v2 (Python) | Various | Python code with license filtering | 50% |
| The Stack v2 (JavaScript) | Various | JavaScript code with license filtering | 15% |
| StarCoder | Apache 2.0 | Multi-language code with docs | 20% |
| Code Contests | Apache 2.0 | Programming competition problems | 15% |

**Special Features:**
- License allowlist (MIT, Apache 2.0, BSD, MPL)
- Automatic deduplication
- Length filtering (100-100k chars)
- Unit test and documentation inclusion

### Math & Reasoning (25%)

| Dataset | License | Description | Allocation |
|---------|---------|-------------|------------|
| GSM8K | MIT | Grade school math word problems | 30% |
| MATH | MIT | Competition-level math | 30% |
| NuminaMath-CoT | Apache 2.0 | Math with chain-of-thought | 40% |

**Special Features:**
- Teacher can produce short scratchpads
- Metadata for masking rationale tokens during training
- Difficulty level tracking

### Factual QA (10%)

| Dataset | License | Description | Allocation |
|---------|---------|-------------|------------|
| Natural Questions | Apache 2.0 | Google's factual QA dataset | 50% |
| HotpotQA | CC-BY-SA 4.0 | Multi-hop reasoning questions | 50% |

**Special Features:**
- Converted to instruction format
- Support for teacher-written intermediate notes
- Multi-hop reasoning chains

## Output Format

### Directory Structure

```
data/distillation/
├── instruction_chat/
│   ├── openhermes.jsonl
│   ├── openhermes.stats.json
│   ├── ultrachat.jsonl
│   ├── ultrachat.stats.json
│   └── ...
├── code/
│   ├── the_stack_python.jsonl
│   ├── the_stack_python.stats.json
│   └── ...
├── math_reasoning/
│   └── ...
├── factual_qa/
│   └── ...
└── download_summary.json
```

### JSONL Format

Each dataset is saved as JSONL with this structure:

```json
{
  "text": "user: What is 2+2?\nassistant: 2+2 equals 4.",
  "id": "gsm8k_00000123",
  "source": "gsm8k",
  "category": "math_reasoning",
  "metadata": {
    "type": "math_word_problem"
  }
}
```

### Statistics Files

Each dataset includes a `.stats.json` file:

```json
{
  "samples": 15000,
  "total_chars": 45000000,
  "size_mb": 42.9,
  "avg_length": 3000,
  "min_length": 50,
  "max_length": 25000,
  "duplicates_removed": 234,
  "output_file": "/path/to/dataset.jsonl"
}
```

### Summary File

`download_summary.json` contains overall statistics:

```json
{
  "total_samples": 100000,
  "total_size_mb": 450.5,
  "categories": {
    "instruction_chat": {
      "samples": 35000,
      "size_mb": 157.7,
      "datasets": ["openhermes", "ultrachat", "open_orca"]
    },
    ...
  },
  "datasets": {
    "openhermes": { ... },
    ...
  }
}
```

## Advanced Features

### Deduplication

Automatic MD5-based deduplication:
- Computes hash of each sample's text
- Removes exact duplicates within each dataset
- Reports duplicate count in statistics

### License Filtering

For code datasets:
```python
allowed_licenses = {
    'mit',
    'apache-2.0',
    'bsd-3-clause',
    'bsd-2-clause',
    'mpl-2.0'
}
```

Samples without allowed licenses are automatically filtered.

### Quality Filtering

- **Length filters**: Configurable min/max character counts
- **Word count**: Minimum words per sample
- **Empty content**: Automatic removal
- **Malformed data**: Error handling and skipping

### Prompt Length Curriculum

For instruction/chat datasets:
- **First 50% of samples**: Prompts limited to <256 tokens
- **Second 50% of samples**: No prompt length limit

This helps the student model learn short-form responses before tackling longer contexts.

## Command Line Options

```bash
python data/download_distillation_datasets.py [OPTIONS]

Options:
  --config PATH         Path to YAML config file (recommended)
  --total-samples INT   Total samples to download (default: 100000)
  --output-dir PATH     Output directory (default: ./data/distillation)
  --dataset NAME        Download only specific dataset
  --help                Show help message
```

## Examples

### Example 1: Small Test Download

Download 1k samples for testing:

```bash
python data/download_distillation_datasets.py --total-samples 1000
```

### Example 2: Large Production Download

Download 500k samples with custom config:

```yaml
# configs/large_download.yaml
total_samples: 500000
category_ratios:
  instruction_chat: 0.35
  code: 0.30
  math_reasoning: 0.25
  factual_qa: 0.10
```

```bash
python data/download_distillation_datasets.py \
    --config configs/large_download.yaml
```

### Example 3: Code-Only Download

Download only code datasets:

1. Edit `configs/dataset_download.yaml`:
```yaml
datasets:
  openhermes:
    enabled: false
  ultrachat:
    enabled: false
  # ... disable all non-code datasets
  the_stack_python:
    enabled: true
  starcoder:
    enabled: true
```

2. Run:
```bash
python data/download_distillation_datasets.py \
    --config configs/dataset_download.yaml
```

### Example 4: Download Single Dataset

```bash
# Just GSM8K
python data/download_distillation_datasets.py \
    --dataset gsm8k \
    --config configs/dataset_download.yaml

# Just OpenHermes
python data/download_distillation_datasets.py \
    --dataset openhermes \
    --config configs/dataset_download.yaml
```

## Integration with Training Pipeline

### 1. Download Datasets

```bash
python data/download_distillation_datasets.py \
    --config configs/dataset_download.yaml
```

### 2. Tokenize Data

Use the Llama tokenizer to tokenize:

```bash
python data/tokenize_datasets.py \
    --input-dir data/distillation \
    --output-dir data/tokenized \
    --tokenizer meta-llama/Llama-3.2-1B
```

### 3. Generate Teacher Logits (Optional)

For cached distillation:

```bash
python src/distillation/scripts/generate_logits.py \
    --data-dir data/tokenized \
    --teacher-model meta-llama/Llama-3.2-3B-Instruct \
    --output-dir data/teacher_cache
```

### 4. Create Parquet Shards

Convert to efficient parquet format:

```bash
python data/create_parquet_shards.py \
    --input-dir data/tokenized \
    --output-dir data/parquet_shards \
    --shard-size-mb 2048
```

### 5. Train Student Model

```bash
python -m src.distillation.scripts.train \
    --config configs/train_direct.yaml
```

## Troubleshooting

### Issue: HuggingFace Authentication Error

Some datasets require authentication:

```bash
# Login to HuggingFace
huggingface-cli login

# Set token in environment
export HF_TOKEN="your_token_here"
```

### Issue: Out of Disk Space

Monitor download size:

```bash
# Check estimated size before downloading
python data/download_distillation_datasets.py \
    --total-samples 100000 \
    --dry-run  # TODO: implement dry-run mode
```

Reduce total samples or disable large datasets.

### Issue: Slow Download Speed

Use streaming for large datasets (already default):
- Reduces memory usage
- Downloads and processes incrementally
- Better for unstable connections

### Issue: Import Errors

Install all dependencies:

```bash
pip install datasets>=2.14.0 pyyaml>=6.0 tqdm>=4.65.0
```

### Issue: License Compliance

Check licenses before use:

```bash
# View all dataset licenses
grep -r "license:" configs/dataset_download.yaml
```

Review each dataset's license agreement on HuggingFace.

## Best Practices

### For Research/Small Models

- Start with 10-50k samples
- Use all categories for balanced knowledge
- Focus on quality over quantity

### For Production/Larger Models

- Scale to 500k-1M+ samples
- Consider domain-specific adjustments
- Monitor category performance separately

### For Specialized Domains

Adjust ratios for your use case:

```yaml
# Example: Code-focused model
category_ratios:
  instruction_chat: 0.20
  code: 0.60
  math_reasoning: 0.15
  factual_qa: 0.05

# Example: Math-focused model
category_ratios:
  instruction_chat: 0.25
  code: 0.15
  math_reasoning: 0.50
  factual_qa: 0.10
```

## Performance Tips

### Parallel Downloads

The script downloads datasets sequentially. For parallel downloads, run multiple instances:

```bash
# Terminal 1
python data/download_distillation_datasets.py --dataset openhermes &

# Terminal 2
python data/download_distillation_datasets.py --dataset gsm8k &

# Terminal 3
python data/download_distillation_datasets.py --dataset the_stack_python &
```

### Memory Management

For large datasets:
- Use `streaming: true` in config
- Download in batches with smaller `target_samples`
- Process and delete before next batch

### Network Issues

Add retry logic or download in smaller chunks:

```bash
# Download in 10k chunks
for i in {1..10}; do
    python data/download_distillation_datasets.py \
        --total-samples 10000 \
        --output-dir data/distillation/batch_$i
done
```

## License Compliance Summary

| Dataset | License | Commercial Use | Attribution Required |
|---------|---------|----------------|---------------------|
| OpenHermes-2.5 | Apache 2.0 | ✅ Yes | ✅ Yes |
| UltraChat | MIT | ✅ Yes | ✅ Yes |
| Open-Orca | MIT | ✅ Yes | ✅ Yes |
| The Stack v2 | Various | ⚠️ Check per file | ⚠️ Check per file |
| StarCoder | Apache 2.0 | ✅ Yes | ✅ Yes |
| Code Contests | Apache 2.0 | ✅ Yes | ✅ Yes |
| GSM8K | MIT | ✅ Yes | ✅ Yes |
| MATH | MIT | ✅ Yes | ✅ Yes |
| NuminaMath-CoT | Apache 2.0 | ✅ Yes | ✅ Yes |
| Natural Questions | Apache 2.0 | ✅ Yes | ✅ Yes |
| HotpotQA | CC-BY-SA 4.0 | ✅ Yes | ✅ Yes (Share-Alike) |

**Note**: Always review the full license terms before commercial use.

## Contributing

To add a new dataset:

1. Add configuration to `configs/dataset_download.yaml`
2. Implement preprocessing function in `download_distillation_datasets.py`:
   ```python
   def preprocess_mydataset(self, sample: Dict, idx: int, config: DatasetConfig) -> Optional[Dict]:
       # Your preprocessing logic
       return {
           'text': ...,
           'id': ...,
           'source': ...,
           'category': ...,
           'metadata': {...}
       }
   ```
3. Map dataset name to function in `load_config_from_yaml`
4. Test with small sample count
5. Document in this README

## Support

For issues:
1. Check HuggingFace dataset page for known issues
2. Verify dependencies are up to date
3. Check disk space and network connectivity
4. Review error messages for specific dataset failures

## References

- [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5)
- [UltraChat](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
- [Open-Orca](https://huggingface.co/datasets/Open-Orca/OpenOrca)
- [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2)
- [StarCoder](https://huggingface.co/datasets/bigcode/starcoderdata)
- [GSM8K](https://huggingface.co/datasets/gsm8k)
- [MATH](https://huggingface.co/datasets/hendrycks/competition_math)
- [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)
- [Natural Questions](https://huggingface.co/datasets/google-research-datasets/natural_questions)
- [HotpotQA](https://huggingface.co/datasets/hotpot_qa)
