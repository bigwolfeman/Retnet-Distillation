#!/bin/bash
# Validate M1 and M2 gates for distillation pipeline
#
# Usage:
#   ./scripts/validate_gates.sh [teacher-url] [output-dir]
#
# Example:
#   ./scripts/validate_gates.sh http://localhost:8000/v1/topk reports/

set -e

# Default values
TEACHER_URL="${1:-http://localhost:8000/v1/topk}"
OUTPUT_DIR="${2:-reports}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "M1/M2 Gate Validation"
echo "========================================"
echo "Teacher URL: $TEACHER_URL"
echo "Output Dir:  $OUTPUT_DIR"
echo "Timestamp:   $TIMESTAMP"
echo ""

# M1 Gate: Network Throughput Test
echo "Running M1 Gate: Network Throughput Test..."
echo "  - Testing 1000 sequences"
echo "  - Measuring latency percentiles (p50, p95, p99)"
echo "  - Measuring throughput (tok/s)"
echo ""

python -m src.distillation.test_network \
    --teacher-url "$TEACHER_URL" \
    --num-sequences 1000 \
    --topk 128 \
    --output "$OUTPUT_DIR/m1_network_${TIMESTAMP}.json" \
    2>&1 | tee "$OUTPUT_DIR/m1_network_${TIMESTAMP}.log"

M1_STATUS=${PIPESTATUS[0]}

echo ""
echo "M1 Gate: $([ $M1_STATUS -eq 0 ] && echo '✓ PASS' || echo '✗ FAIL')"
echo ""

# M2 Gate: Int8 Calibration Test
echo "Running M2 Gate: Int8 Calibration Test..."
echo "  - Testing 128 sequences"
echo "  - Comparing CE loss (fp32 vs int8)"
echo "  - Validating delta ≤ 1e-3"
echo ""

python -m src.distillation.calibrate_int8_topk \
    --teacher-url "$TEACHER_URL" \
    --num-sequences 128 \
    --topk 128 \
    --output "$OUTPUT_DIR/m2_calibration_${TIMESTAMP}.json" \
    2>&1 | tee "$OUTPUT_DIR/m2_calibration_${TIMESTAMP}.log"

M2_STATUS=${PIPESTATUS[0]}

echo ""
echo "M2 Gate: $([ $M2_STATUS -eq 0 ] && echo '✓ PASS' || echo '✗ FAIL')"
echo ""

# Summary
echo "========================================"
echo "Gate Validation Summary"
echo "========================================"
echo "M1 (Network):     $([ $M1_STATUS -eq 0 ] && echo '✓ PASS' || echo '✗ FAIL')"
echo "M2 (Calibration): $([ $M2_STATUS -eq 0 ] && echo '✓ PASS' || echo '✗ FAIL')"
echo ""
echo "Reports saved to:"
echo "  - M1: $OUTPUT_DIR/m1_network_${TIMESTAMP}.json"
echo "  - M2: $OUTPUT_DIR/m2_calibration_${TIMESTAMP}.json"
echo ""

if [ $M1_STATUS -eq 0 ] && [ $M2_STATUS -eq 0 ]; then
    echo "✅ All gates PASSED - Ready to proceed to training!"
    echo ""
    echo "Next steps:"
    echo "  1. Review reports in $OUTPUT_DIR/"
    echo "  2. Proceed to T027-T031 (Student Model Configuration)"
    echo "  3. Run 1k-step timing test (M3 gate)"
    exit 0
else
    echo "❌ Some gates FAILED - Review reports and fix issues"
    echo ""

    if [ $M1_STATUS -ne 0 ]; then
        echo "M1 Gate failed. Possible issues:"
        echo "  - Network latency too high (check bandwidth)"
        echo "  - Server throughput too low (increase batch size or GPUs)"
        echo "  - Review: $OUTPUT_DIR/m1_network_${TIMESTAMP}.json"
    fi

    if [ $M2_STATUS -ne 0 ]; then
        echo "M2 Gate failed. Possible issues:"
        echo "  - Int8 quantization error too high"
        echo "  - Try increasing topk from 128 to 256"
        echo "  - Review: $OUTPUT_DIR/m2_calibration_${TIMESTAMP}.json"
    fi

    exit 1
fi
