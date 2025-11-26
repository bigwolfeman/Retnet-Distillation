#!/bin/bash
# Monitor RAM, swap, and GPU usage during training
# Usage: ./scripts/monitor_training_resources.sh

# Get baseline swap usage
BASELINE_SWAP=$(free -m | awk '/Swap:/ {print $3}')
SWAP_THRESHOLD=100  # Alert if swap increases by 100MB

echo "=========================================="
echo "Training Resource Monitor"
echo "=========================================="
echo "Baseline swap usage: ${BASELINE_SWAP}MB"
echo "Will alert if swap increases by >${SWAP_THRESHOLD}MB"
echo ""
echo "Press Ctrl+C to stop monitoring"
echo "=========================================="
echo ""

# Monitor loop
while true; do
    clear
    echo "=========================================="
    date "+%Y-%m-%d %H:%M:%S"
    echo "=========================================="

    # RAM and Swap
    echo ""
    echo "MEMORY STATUS:"
    free -h | head -2

    # Check swap increase
    CURRENT_SWAP=$(free -m | awk '/Swap:/ {print $3}')
    SWAP_INCREASE=$((CURRENT_SWAP - BASELINE_SWAP))

    echo ""
    if [ $SWAP_INCREASE -gt $SWAP_THRESHOLD ]; then
        echo "⚠️  WARNING: Swap increased by ${SWAP_INCREASE}MB! ⚠️"
        echo "   Consider stopping training (Ctrl+C) and reducing workers"
    else
        echo "✓ Swap increase: ${SWAP_INCREASE}MB (safe)"
    fi

    # GPU status
    echo ""
    echo "GPU STATUS:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | \
        awk -F',' '{printf "  GPU %s: %sMB/%sMB VRAM | %s%% Util | %s°C\n", $1, $3, $4, $5, $6}'

    # Training processes
    echo ""
    echo "TRAINING PROCESSES:"
    ps aux | grep -E "finetune_teacher|DataLoader" | grep -v grep | \
        awk '{printf "  PID %s: %sMB RAM | %s%% CPU | %s\n", $2, int($6/1024), $3, $11}' | head -5

    # /dev/shm usage
    echo ""
    echo "/dev/shm USAGE:"
    df -h /dev/shm | tail -1 | awk '{printf "  %s used / %s total (%s)\n", $3, $2, $5}'

    # Iteration speed (if log exists)
    if [ -f "teacher_adapters/llama-1b-corrected/finetune.log" ]; then
        echo ""
        echo "RECENT TRAINING SPEED:"
        grep -E "s/it|it/s" teacher_adapters/llama-1b-corrected/finetune.log | tail -3 | \
            sed 's/^/  /'
    fi

    echo ""
    echo "=========================================="
    echo "Updating every 3 seconds... (Ctrl+C to stop)"

    sleep 3
done
