#!/bin/bash
# Quick test script to verify the caching deadlock fix

set -e

echo "========================================="
echo "Testing Teacher Logit Caching Fix"
echo "========================================="
echo ""

# Check if teacher server is reachable
echo "1. Checking teacher server health..."
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health | grep -q "200"; then
    echo "   ✓ Teacher server is reachable"
else
    echo "   ✗ Teacher server is not reachable at http://localhost:8080"
    echo "   Please start the teacher server first"
    exit 1
fi
echo ""

# Test sequential mode (workers=1)
echo "2. Testing SEQUENTIAL mode (--workers 1)..."
echo "   This should complete without hanging."
python scripts/cache_teacher_logits.py \
    --test \
    --data-path data/dummy_test/ \
    --output-dir data/test_cache_seq/ \
    --workers 1 \
    --batch-size 4 \
    --max-sequences 10 \
    --timeout 30.0

if [ $? -eq 0 ]; then
    echo "   ✓ Sequential mode completed successfully"
else
    echo "   ✗ Sequential mode failed"
    exit 1
fi
echo ""

# Test parallel mode (workers=4)
echo "3. Testing PARALLEL mode (--workers 4)..."
echo "   This previously would hang - now should work!"
python scripts/cache_teacher_logits.py \
    --test \
    --data-path data/dummy_test/ \
    --output-dir data/test_cache_parallel/ \
    --workers 4 \
    --batch-size 4 \
    --max-sequences 10 \
    --timeout 30.0

if [ $? -eq 0 ]; then
    echo "   ✓ Parallel mode completed successfully"
else
    echo "   ✗ Parallel mode failed"
    exit 1
fi
echo ""

# Cleanup
echo "4. Cleaning up test outputs..."
rm -rf data/test_cache_seq/
rm -rf data/test_cache_parallel/
echo "   ✓ Cleanup complete"
echo ""

echo "========================================="
echo "✓ ALL TESTS PASSED!"
echo "========================================="
echo ""
echo "The caching script is working correctly."
echo "You can now run the full caching with:"
echo ""
echo "  # Sequential (slower, very stable):"
echo "  python scripts/cache_teacher_logits.py --workers 1"
echo ""
echo "  # Parallel (faster, now fixed):"
echo "  python scripts/cache_teacher_logits.py --workers 4"
echo ""
