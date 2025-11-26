#!/bin/bash
# Training script with PyTorch memory allocator optimization
# Fixes VRAM oscillation from 11GB → 31GB by reducing fragmentation

# Configure PyTorch CUDA allocator to reduce fragmentation
# - expandable_segments: Allows allocator to grow segments dynamically
# - max_split_size_mb:512: Prevents creating tiny fragmented blocks
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

echo "======================================================================"
echo "Starting training with memory fragmentation fix"
echo "PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
echo "======================================================================"
echo ""

# Start training
python -m src.distillation.scripts.train \
    --config configs/train_direct.yaml \
    --weight-decay 0.01 \
    --enable-saddle-interventions \
    "$@"
