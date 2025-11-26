# Data Directory

This directory contains datasets for knowledge distillation training.

## Structure

```
data/
├── unlabeled/           # Prepared datasets (JSONL format)
│   ├── text/           # Text domain (FineWeb-Edu, WikiText)
│   ├── code/           # Code domain (The Stack, Python)
│   ├── math/           # Math domain (OpenWebMath)
│   └── test/           # Test datasets (50 examples each)
│       ├── text.jsonl
│       ├── code.jsonl
│       └── math.jsonl
│
├── train/              # Training split (95%)
│   ├── text/
│   ├── code/
│   └── math/
│
├── val/                # Validation split (5%)
│   ├── text/
│   ├── code/
│   └── math/
│
└── README.md           # This file
```

## Quick Start

### Create Test Dataset

```bash
python scripts/prepare_datasets.py \
    --mode test \
    --output-dir data/unlabeled \
    --num-test-examples 50
```

### Prepare Production Data

```bash
# All domains
python scripts/prepare_datasets.py \
    --mode all \
    --max-examples 1000000 \
    --output-dir data/unlabeled \
    --create-splits

# Or individual domains
python scripts/prepare_datasets.py --mode text --max-examples 1000000 --create-splits
python scripts/prepare_datasets.py --mode code --max-examples 300000 --create-splits
python scripts/prepare_datasets.py --mode math --max-examples 150000 --create-splits
```

## Format

Each JSONL file contains one example per line:

```json
{"text": "content...", "domain": "text|code|math", "num_tokens": 1234}
```


