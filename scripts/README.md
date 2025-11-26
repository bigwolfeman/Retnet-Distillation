# Scripts Directory

Collection of utility scripts and training tools for the distillation project.

## Training Scripts

### Teacher Finetuning

**Script**: `finetune_teacher.py`
**Purpose**: Domain-adapt the teacher model using LoRA

```bash
# Basic training
python scripts/finetune_teacher.py --config configs/teacher_ft.yaml

# Save merged model
python scripts/finetune_teacher.py --config configs/teacher_ft.yaml --save-merged

# Resume from checkpoint
python scripts/finetune_teacher.py --config configs/teacher_ft.yaml --resume-from runs/teacher_ft/checkpoint-1000
```

**Features**:
- LoRA adapter training (rank 64, alpha 128)
- BF16 mixed precision
- Gradient checkpointing
- Validation perplexity tracking
- Early stopping
- Token budget limiting
- W&B integration
- Checkpoint management

**Target VRAM**: ≤16 GB

**Expected Duration**: 6-8 hours (1 epoch, RTX 3090/4090)

See `Ai-notes/TEACHER-FINETUNING-IMPLEMENTATION-2025-11-12.md` for full documentation.

---

## Utility Scripts

### Dataset Preparation

- `prepare_datasets.py` - Download and preprocess training data
- `test_dataset_loading.py` - Verify dataset loading works correctly

### Teacher Testing

- `test_vllm_integration.py` - Test vLLM server integration
- `test_simple_teacher.py` - Test direct teacher inference
- `test_fast_endpoint.py` - Benchmark teacher API latency
- `debug_teacher_client.py` - Debug teacher client issues
- `cache_teacher_logits.py` - Pre-cache teacher logits to disk

### Debugging

- `debug_data_quality.py` - Analyze data quality metrics
- `debug_teacher_student_alignment.py` - Check teacher/student output alignment
- `diagnose_act.py` - Diagnose activation issues
- `profile_training.py` - Profile training performance

### Analysis

- `calculate_cache_storage.py` - Estimate cache storage requirements

---

## Configuration Files

All scripts use YAML configs from `/configs`:

- `teacher_ft.yaml` - Teacher finetuning configuration
- `train_direct.yaml` - Direct mode distillation
- `train_cached.yaml` - Cached mode distillation
- `train_network.yaml` - Network mode distillation

---

## Common Patterns

### Loading Config

```python
import yaml
from pathlib import Path

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config('configs/teacher_ft.yaml')
```

### Dataset Loading

```python
from src.distillation.dataset import PretokenizedShardDataset

dataset = PretokenizedShardDataset(
    data_path="data/distillation_preprocessed",
    max_length=8192,
    splits=None,  # Use all splits
)
```

### Model Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
```

---

## Adding New Scripts

When adding new scripts:

1. Add shebang: `#!/usr/bin/env python3`
2. Add docstring with purpose and usage
3. Make executable: `chmod +x scripts/your_script.py`
4. Add entry to this README
5. Create documentation in `Ai-notes/` if significant

---

## Dependencies

Core dependencies for training scripts:

```bash
pip install torch transformers peft datasets bitsandbytes wandb tqdm pyyaml
```

For vLLM integration:

```bash
pip install vllm
```

---

## Help

For detailed help on any script:

```bash
python scripts/script_name.py --help
```

For implementation details, see documentation in `Ai-notes/`.
