"""
Sparse streaming knowledge distillation from Llama-3.2-1B to RetNet student.

This module implements v1 of the distillation pipeline:
- Remote teacher server (vLLM) with /v1/topk endpoint
- HTTP client with retry logic and batching
- Sparse-KL divergence loss with int8 quantization
- Training loop with gradient accumulation and checkpointing
- Evaluation infrastructure (perplexity, NIAH)
"""

__version__ = "1.0.0"

__all__ = [
    "teacher_server",
    "teacher_client",
    "losses",
    "dataset",
    "trainer",
    "telemetry",
]
