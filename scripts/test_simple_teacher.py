#!/usr/bin/env python3
"""Simple test of teacher client with actual tokenized input."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.distillation.vllm_teacher_client import VLLMTeacherClient
import time

print("Creating client...")
client = VLLMTeacherClient(
    base_url="http://192.168.0.71:8080",
    model="meta-llama/Llama-3.2-1B-Instruct",
    api_key="token-abc123",
    timeout=10.0,
)

# Simple test: Just 3 tokens
input_ids = [[128000, 9906, 1917]]  # BOS, Hello, world
print(f"\nInput: {input_ids}")
print(f"Length: {len(input_ids[0])} tokens")

print("\nCalling get_prompt_logprobs with topk=128...")
print("This should complete in < 1 second if working correctly.")

start = time.time()
try:
    result = client.get_prompt_logprobs(input_ids, topk=128)
    elapsed = time.time() - start
    print(f"\n✅ SUCCESS in {elapsed:.2f}s")
    print(f"Got result with {len(result)} sequences")
    print(f"First sequence has {len(result[0]['tokens'])} tokens")
except Exception as e:
    elapsed = time.time() - start
    print(f"\n❌ FAILED after {elapsed:.2f}s")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    client.close()
