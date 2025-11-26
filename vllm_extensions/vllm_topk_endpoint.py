#!/usr/bin/env python3
"""Custom /v1/topk endpoint for vLLM that stays within the public engine API."""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any, Iterable, List

import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel, Field

from vllm.inputs.data import TokensPrompt
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

logger = logging.getLogger(__name__)


class TopKRequest(BaseModel):
    """Request schema for /v1/topk endpoint."""

    input_ids: List[List[int]] = Field(
        ...,
        description="Token IDs for input sequences. Shape: (batch_size, seq_len)",
        min_length=1,
    )
    topk: int = Field(
        default=128,
        description="Number of top logits to return per position",
        ge=1,
        le=1024,
    )
    temperature: float = Field(
        default=1.0,
        description="Temperature applied to the returned distribution",
        gt=0.0,
        le=10.0,
    )


class TopKResponse(BaseModel):
    """Response schema for /v1/topk endpoint."""

    indices: List[List[List[int]]] = Field(
        ...,
        description="Top-k token indices. Shape: (batch_size, num_positions, k)",
    )
    values_int8: List[List[List[int]]] = Field(
        ...,
        description="Quantized top-k probabilities (int8). Matches indices shape.",
    )
    scale: List[List[float]] = Field(
        ...,
        description="Per-position scale factors to dequantize int8 values.",
    )
    other_mass: List[List[float]] = Field(
        ...,
        description="Tail probability mass outside top-k.",
    )
    batch_size: int
    num_positions: List[int]
    k: int
    return_dtype: str = "int8"


async def handle_topk_request(request: TopKRequest, engine) -> TopKResponse:
    """Handle /v1/topk request using vLLM's EngineClient interface."""

    if not request.input_ids:
        raise HTTPException(status_code=400, detail="input_ids cannot be empty")

    try:
        outputs = await _collect_prompt_outputs(
            engine=engine,
            batch_input_ids=request.input_ids,
            topk=request.topk,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Failed to collect prompt outputs for /v1/topk")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        response = _build_topk_response(
            outputs=outputs,
            request=request,
        )
    except Exception as exc:  # pragma: no cover
        logger.exception("Failed to build /v1/topk response payload")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    logger.info(
        "Processed /v1/topk request: batch=%d, topk=%d",
        len(request.input_ids),
        request.topk,
    )
    return response


async def _collect_prompt_outputs(
    engine,
    batch_input_ids: List[List[int]],
    topk: int,
):
    """Run vLLM engine to collect prompt logprobs for each sequence."""

    sampling_params = SamplingParams(
        max_tokens=1,  # Must be at least 1 in vLLM 0.5.5
        logprobs=1,
        prompt_logprobs=topk,
        temperature=1.0,
    )

    outputs = []
    for seq in batch_input_ids:
        result = await _gather_single_request(engine, seq, sampling_params)
        outputs.append(result)
    return outputs


async def _gather_single_request(
    engine,
    tokens: List[int],
    sampling_params: SamplingParams,
):
    if not tokens:
        raise HTTPException(
            status_code=400,
            detail="Each sequence must contain at least one token",
        )

    request_id = random_uuid()
    prompt: TokensPrompt = {"prompt_token_ids": tokens}

    final_output = None
    async for output in engine.generate(
        prompt,
        sampling_params,
        request_id,
    ):
        final_output = output

    if final_output is None:
        raise RuntimeError("Engine returned no output for prompt")

    return final_output


def _build_topk_response(outputs, request: TopKRequest) -> TopKResponse:
    indices: List[List[List[int]]] = []
    quantized_values: List[List[List[int]]] = []
    scales: List[List[float]] = []
    tail_mass: List[List[float]] = []
    num_positions: List[int] = []

    for output, tokens in zip(outputs, request.input_ids):
        prompt_logprobs = getattr(output, "prompt_logprobs", None)
        if prompt_logprobs is None:
            raise RuntimeError("vLLM output missing prompt_logprobs")

        seq_indices: List[List[int]] = []
        seq_values: List[List[int]] = []
        seq_scales: List[float] = []
        seq_tail: List[float] = []

        for logprob_map in prompt_logprobs:
            if logprob_map is None:
                seq_indices.append([])
                seq_values.append([])
                seq_scales.append(0.0)
                seq_tail.append(1.0)
                continue

            entries = list(logprob_map.values())
            if not entries:
                seq_indices.append([])
                seq_values.append([])
                seq_scales.append(0.0)
                seq_tail.append(1.0)
                continue

            entries.sort(
                key=lambda entry: (
                    entry.rank if entry.rank is not None else 0,
                    entry.logprob,
                )
            )
            top_entries = entries[: request.topk]

            raw_logits = np.array([entry.logprob for entry in top_entries], dtype=np.float32)
            token_ids = [entry.token_id for entry in top_entries]

            probs = np.exp(raw_logits)
            prob_sum = float(probs.sum())
            other_mass = max(0.0, min(1.0, 1.0 - prob_sum))

            if not math.isclose(request.temperature, 1.0, rel_tol=1e-6):
                temp_logits = raw_logits / request.temperature
                max_logit = float(np.max(temp_logits))
                exp_vals = np.exp(temp_logits - max_logit)
                probs = exp_vals / max(exp_vals.sum(), 1e-12)
                other_mass = max(0.0, min(1.0, 1.0 - float(probs.sum())))

            max_prob = float(probs.max(initial=0.0))
            scale = max(max_prob / 127.0, 1e-12)
            quantized = np.round(probs / scale).clip(0, 127).astype(np.int8)

            seq_indices.append(token_ids)
            seq_values.append(quantized.tolist())
            seq_scales.append(scale)
            seq_tail.append(other_mass)

        indices.append(seq_indices)
        quantized_values.append(seq_values)
        scales.append(seq_scales)
        tail_mass.append(seq_tail)
        num_positions.append(len(tokens))

    return TopKResponse(
        indices=indices,
        values_int8=quantized_values,
        scale=scales,
        other_mass=tail_mass,
        batch_size=len(outputs),
        num_positions=num_positions,
        k=request.topk,
    )


def register_topk_endpoint(app, engine) -> None:
    """Register the /v1/topk endpoint on the provided FastAPI app."""

    @app.post("/v1/topk", response_model=TopKResponse)
    async def topk_endpoint(request: TopKRequest):
        return await handle_topk_request(request, engine)

    logger.info("Registered /v1/topk endpoint")


__all__ = [
    "TopKRequest",
    "TopKResponse",
    "handle_topk_request",
    "register_topk_endpoint",
]
