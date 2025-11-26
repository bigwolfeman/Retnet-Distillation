#!/usr/bin/env python3
"""Debug script to test VLLMTeacherClient with verbose logging."""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.distillation.vllm_teacher_client import VLLMTeacherClient

# Set up verbose logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    print("=" * 80)
    print("Testing VLLMTeacherClient with simple request")
    print("=" * 80)

    # Create client
    client = VLLMTeacherClient(
        base_url="http://192.168.0.71:8080",
        model="meta-llama/Llama-3.2-1B-Instruct",
        api_key="token-abc123",
        topk=128,
        timeout=10.0,  # 10 second timeout
    )

    # Simple test input
    input_ids = [[1, 9906, 1917]]  # "<s> Hello world"

    print(f"\nTest input: {input_ids}")
    print(f"Input length: {len(input_ids[0])} tokens")
    print("\nSending request to server...")
    print("(This should complete in < 1 second)")

    try:
        import time
        start = time.time()

        result = client.get_top_k_logits(input_ids)

        elapsed = time.time() - start
        print(f"\n✅ SUCCESS! Request completed in {elapsed:.2f}s")
        print(f"Result shape: indices={result['indices'].shape}, values={result['values_int8'].shape}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()

if __name__ == "__main__":
    main()
