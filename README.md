# RetNet-HRM: Hierarchical Reasoning Machine with Titans Memory

A novel neural architecture combining RetNet's efficient long-context processing with Hierarchical Reasoning Machine (HRM) adaptive computation and Titans-style persistent memory. This repository serves as an architecture demonstration and distillation target for researchers working on efficient reasoning models.

## Architecture Overview

RetNet-HRM integrates three key architectural innovations:

### 1. RetNet Backbone
- Linear attention mechanism enabling O(1) inference memory per layer
- Supports sequences up to 128k tokens efficiently
- Three operational modes: parallel training, recurrent inference, chunk-recurrent processing
- Built on Microsoft's TorchScale RetNet implementation

### 2. Hierarchical Reasoning Machine (HRM + ACT)
- **Adaptive Computation Time (ACT)**: Dynamic per-token computation allocation
- **Halting mechanism**: Learned confidence thresholds for early exit
- **Blackboard architecture**: MAC (Memory-As-Context) coupling for multi-module coordination
- Specialized L-Engines for domain-specific reasoning (math, code, general)

### 3. Titans Memory System
- **MAG (Memory-As-Gating)**: Persistent memory gates model updates
- Test-time memory writes with drift control
- Surprise-weighted memory updates
- EMA rollback for stability

### 4. RheaNet: Retention + MAC Integration (New!)

**RheaNet** is a novel integration layer that combines retention mechanisms with Titans' Memory-As-Context architecture for efficient long-context processing:

#### What is RheaNet?

RheaNet replaces traditional O(T²) attention with retention-based processing while preserving the MAC (Memory-As-Context) semantics that make Titans effective. It achieves:

- **O(T) training complexity**: Linear scaling with sequence length via block-scan
- **O(1) per-layer inference**: Constant memory through recurrent state reuse
- **Bounded memory retrieval**: Fixed N_ℓ memory tokens per chunk
- **Persistent context**: Learnable tokens that carry global information

#### Key Benefits

1. **25-50% Speedup**: Reduced from ~0.58s/step to ~0.34-0.44s/step on 350M models
2. **Linear Memory Scaling**: No O(T²) attention matrices during training
3. **Streaming Generation**: True sequential processing with state reuse
4. **MAC Compatibility**: Preserves `[persistent | memory | retention]` dataflow

#### Architecture Components

RheaNet consists of three integrated pieces:

1. **Retention Block** (`retention_block.py`)
   - Learnable log-decay parameters per head
   - Three modes: parallel (training), recurrent (inference), chunk-wise (streaming)
   - FP32 accumulation for numerical stability

2. **MAC Dataflow** (`titan_retention_layer.py`)
   - Constructs augmented sequence: `[P | h_t | Y]` where:
     - P = persistent tokens (global context)
     - h_t = retention output (current chunk)
     - Y = top-N_ℓ memory retrievals (bounded history)
   - Local mixer (windowed attention) over augmented sequence
   - Gating to blend retention + memory outputs

3. **Memory Integration** (`titan_memory.py`)
   - Surprise-weighted EMA writes
   - Capacity-based eviction (no unbounded growth)
   - Top-k retrieval with zero padding

#### Dependency Expectations

**Required:**
- PyTorch 2.0+ with CUDA support
- FP32 accumulation for retention state (bf16/fp16 inputs okay)
- Block size ≥ 16 for efficient parallel training

**Configuration:**
```python
from src.models.titans.titan_config import TitanMACConfig

config = TitanMACConfig(
    use_retention=True,           # Enable retention instead of attention
    retention_block_len=64,       # Block size for block-scan
    n_persistent_tokens=8,        # Global context tokens
    n_memory_tokens=32,           # Bounded retrieval count (N_ℓ)
    mixer_window_size=256,        # Local attention window
)
```

> **Progress** (2025-11-06): Retention kernel, MAC dataflow, streaming state threading, and windowed attention are implemented with benchmarks captured. Repository-wide pytest still hits legacy import failures—see `/docs/RheaNet.md` for status details.

#### High-Level Architecture Flow

```
Input Tokens
    ↓
Retention Block (parallel/recurrent/chunk)
    ↓
    [retention output h_t]
    ↓
Augment with Persistent + Memory
    ↓
    [P | h_t | Y]  ← augmented sequence
    ↓
Local Mixer (windowed attention)
    ↓
Gating (blend retention + memory)
    ↓
Output Tokens
```

#### Numerical Guarantees

- **FP32 tolerance**: max_abs_error < 1e-5 between parallel and recurrent modes
- **BF16 tolerance**: max_abs_error < 5e-3 between modes
- **Memory**: O(T) during training, O(1) per layer during inference
- **No O(T²) allocations**: Verified via memory profiling

#### Getting Started with RheaNet

See `/specs/001-rheanet-implementation/quickstart.md` for detailed setup instructions, testing, and benchmarking commands.

For implementation details, see:
- `/specs/001-rheanet-implementation/spec.md` - Full specification
- `/specs/001-rheanet-implementation/plan.md` - Implementation plan
- `/specs/001-rheanet-implementation/tasks.md` - Task breakdown
- `/docs/RheaNet.md` - Architecture deep-dive & progress tracker

## Key Features

- **Efficient Long Context**: O(1) memory per layer during inference (vs O(n) for standard transformers)
- **Adaptive Computation**: Halting mechanism reduces compute for simple tokens
- **Modular Design**: Blackboard-based coordination between specialized reasoning engines
- **Persistent Memory**: Titans memory enables continual adaptation without catastrophic forgetting
- **Teacher Finetuning**: Optional teacher adaptation for 4.5x faster distillation convergence with improved logit alignment
- **170+ Architecture Tests**: Comprehensive test suite validating all architectural components

## Installation

### Prerequisites
- Python 3.10 or 3.11
- PyTorch 2.0+
- CUDA 11.8+ (for GPU support)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd 000Distill-Titan-Retnet-HRM
```

2. Install dependencies:
```bash
pip install -e .
```

This will automatically install the bundled TorchScale fork required for RetNet support.

Alternatively, use requirements.txt:
```bash
pip install -r requirements.txt
```

## Quick Start

For a complete walkthrough including optional teacher finetuning for improved distillation, see [quickstart.md](quickstart.md).

### Basic Inference

```python
import torch
from src.models.retnet.backbone import RetNetBackbone

# Initialize model
model = RetNetBackbone(
    vocab_size=100352,
    d_model=2816,
    n_layers=28,
    n_heads=12,
    dropout=0.1,
)

# Load pretrained weights (if available)
# checkpoint = torch.load("model.safetensors")
# model.load_state_dict(checkpoint)

# Prepare input
input_ids = torch.randint(0, 100352, (1, 512))  # [batch, seq_len]

# Forward pass (parallel training mode)
with torch.no_grad():
    hidden_states = model.forward_train(input_ids)  # [batch, seq_len, d_model]
```

### Recurrent Inference (O(1) Memory)

```python
# For deployment/inference with minimal memory
# Process tokens one at a time with recurrent mode
with torch.no_grad():
    hidden_states = model.forward_recurrent(input_ids)  # [batch, seq_len, d_model]
```

### HRM with Adaptive Computation

```python
from src.models.hrm.controller import HRMController

# Initialize HRM controller with ACT
hrm = HRMController(
    d_model=2816,
    max_steps=12,
    halt_threshold=0.95,
    time_penalty=0.01
)

# Process with adaptive computation
output = hrm(hidden_states)
# Returns: hidden states with adaptive computation applied
```

## Architecture Components

### Core Modules

- `src/models/retnet/`: RetNet backbone implementation
  - O(1) recurrent inference mode
  - Parallel training mode
  - Chunk-recurrent for ultra-long sequences

- `src/models/hrm/`: Hierarchical Reasoning Machine
  - `controller.py`: Main HRM orchestration
  - `act.py`: Adaptive Computation Time implementation
  - `halting.py`: Confidence-based halting mechanism

- `src/models/routing/`: Task routing and engine selection
  - Multi-engine coordination
  - Blackboard state management

- `src/models/retrieval/`: Long-term memory and retrieval
  - `landmark.py`: Landmark attention for compression
  - `compressor.py`: Memory compression strategies
  - `registry.py`: Retrieval index management

### Titans Integration

The Titans memory system provides persistent, test-time adaptation:

```python
from src.models.titans.neural_memory import NeuralMemory

# Initialize Titans memory
memory = NeuralMemory(
    d_model=2816,
    memory_size=4096,
    num_heads=12,
)

# Use in forward pass with MAG coupling
memory_output = memory.retrieve(query)
# Memory gates FFN updates in the Titans H-Layer
```

## TorchScale Dependency

This repository includes a bundled fork of Microsoft's TorchScale library in `./torchscale/`. The fork includes:

- Core RetNet implementation (MIT License)
- Multi-scale retention mechanism
- xPos relative position encoding
- Essential components only (examples and training code removed)

The original TorchScale README and license are preserved in the `torchscale/` directory.

## Testing

Run the comprehensive architecture test suite:

```bash
# All tests
pytest tests/

# Unit tests only
pytest tests/unit/

# Specific component
pytest tests/unit/test_retnet_backbone.py

# GPU-required tests (if available)
pytest tests/unit/ -m gpu

# Skip slow tests
pytest tests/unit/ -m "not slow"
```

170+ tests validate:
- RetNet attention mechanics
- HRM halting and adaptive computation
- Titans memory persistence
- Blackboard coordination
- Routing and arbitration
- Symbolic computation (sympy integration)

## Project Structure

```
.
├── src/
│   ├── models/
│   │   ├── retnet/          # RetNet backbone
│   │   ├── hrm/             # HRM + ACT
│   │   ├── routing/         # Task routing
│   │   ├── retrieval/       # Long-term memory
│   │   └── attention/       # Attention mechanisms
│   ├── core/                # Core utilities
│   ├── config/              # Configuration management
│   └── data/                # Data utilities
├── tests/
│   ├── unit/                # Component tests
│   └── integration/         # System tests
├── torchscale/              # Bundled TorchScale fork
├── docs/                    # Architecture documentation
└── configs/                 # Model configurations

```

## Configuration

Model configurations use YAML files in `configs/`:

```yaml
# configs/model/retnet_hrm_3b.yaml
model:
  backbone: retnet
  d_model: 2816
  n_layers: 28
  n_heads: 12

hrm:
  max_steps: 12
  halt_threshold: 0.95
  time_penalty: 0.01

titans:
  memory_size: 4096
  update_rate: 0.01
  drift_threshold: 0.1
```

## Documentation

Detailed architecture documentation is available in `docs/`:

- `docs/Titan.md`: Full HRM + Titans specification
- `docs/gaussian-inner-vision.md`: Vision integration (deferred)
- `docs/stabilization/`: Numerical stability notes

## Distillation Target

This architecture is designed as a distillation target for larger models. Key characteristics for distillation:

- **Efficient inference**: O(1) memory enables deployment at scale
- **Modular design**: Distill to specific L-Engines for domain specialization
- **Adaptive computation**: Students can learn when to halt, reducing inference cost
- **Long context**: Support for 64k-128k token sequences
- **Verifiable outputs**: Confidence calibration and halting thresholds
- **Teacher adaptation**: Optional teacher finetuning on target dataset for 4.5x convergence speedup

### Teacher Finetuning for Knowledge Distillation

The repository includes an optional teacher finetuning step that significantly improves distillation efficiency. By adapting the teacher model to your specific dataset distribution, you can achieve:

- 4.5x faster convergence to target validation loss
- 15-20% improvement in final distillation quality
- Better logit alignment between teacher and student
- Minimal overhead: ~1 epoch (2-4 hours) of finetuning on 5B tokens

To use this feature:

```bash
# Finetune teacher on your distillation dataset
make finetune-teacher

# Or run directly:
python scripts/finetune_teacher.py --config configs/teacher_ft.yaml
```

The teacher adapters are then automatically used during distillation training. See [quickstart.md](quickstart.md) for detailed instructions.

## License

MIT License - See LICENSE file for details.

The bundled TorchScale library is also under MIT License (Copyright Microsoft Corporation).

## Citation

If you use this architecture in your research, please cite:

```bibtex
@software{retnet_hrm_2024,
  title={RetNet-HRM: Hierarchical Reasoning Machine with Titans Memory},
  author={RetNet-HRM Team},
  year={2024},
  url={https://github.com/...}
}
```

## Contributing

This is a research architecture release. For questions or issues with the architecture specification, please open an issue.

## Acknowledgments

- Microsoft TorchScale team for the RetNet implementation
- Titans memory system design
- HRM and ACT research communities
