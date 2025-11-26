"""
API schemas for /v1/topk endpoint.

Defines request/response schemas for sparse top-k logit distribution
with int8 quantization for efficient network transfer during knowledge distillation.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


# Validation limits (F-002, F-004, F-005, F-006)
MAX_SEQUENCE_LENGTH = 32768  # 32k tokens max (F-002: prevent unbounded memory allocation)
MAX_BATCH_SIZE = 128  # Maximum sequences per request (F-004: batch size limit)
MIN_TOPK = 1
MAX_TOPK = 1024  # Reasonable upper limit (F-005: topk validation)


class TopKRequest(BaseModel):
    """Request schema for /v1/topk endpoint.

    Attributes:
        input_ids: Token IDs for the input sequence(s). Shape: (batch_size, seq_len)
        topk: Number of top logits to return per position (default: 128)
        return_dtype: Data type for returned values (default: "int8")
        temperature: Temperature for softmax computation (default: 1.0)
        max_tokens: Maximum number of tokens to generate (default: None, uses seq_len)
    """

    input_ids: List[List[int]] = Field(
        ...,
        description="Token IDs for input sequences. Shape: (batch_size, seq_len)",
        min_length=1
    )

    topk: int = Field(
        default=128,
        description="Number of top logits to return per position"
    )

    return_dtype: Literal["int8", "float16", "float32"] = Field(
        default="int8",
        description="Data type for returned logit values"
    )

    temperature: float = Field(
        default=1.0,
        description="Temperature for softmax computation"
    )

    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of output tokens (None = input length)",
        ge=1
    )

    @field_validator('topk')
    @classmethod
    def validate_topk(cls, v):
        """Validate topk is within reasonable bounds (F-005)."""
        if not (MIN_TOPK <= v <= MAX_TOPK):
            raise ValueError(f"topk must be between {MIN_TOPK} and {MAX_TOPK}, got {v}")
        return v

    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        """Validate temperature is positive."""
        if v <= 0:
            raise ValueError(f"temperature must be positive, got {v}")
        if v > 10.0:
            raise ValueError(f"temperature must be <= 10.0, got {v}")
        return v

    @model_validator(mode='after')
    def validate_input_ids(self):
        """Validate input_ids batch size, sequence lengths, and token IDs (F-002, F-004, F-006)."""
        # F-004: Validate batch size
        if len(self.input_ids) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(self.input_ids)} exceeds maximum {MAX_BATCH_SIZE}"
            )

        if len(self.input_ids) == 0:
            raise ValueError("input_ids cannot be empty")

        # F-002, F-006: Validate sequence lengths and token IDs
        for idx, seq in enumerate(self.input_ids):
            # F-002: Prevent unbounded memory allocation
            if len(seq) > MAX_SEQUENCE_LENGTH:
                raise ValueError(
                    f"Sequence {idx} length {len(seq)} exceeds maximum {MAX_SEQUENCE_LENGTH}"
                )

            if len(seq) == 0:
                raise ValueError(f"Sequence {idx} is empty")

            # F-006: Validate token IDs are non-negative
            for token_id in seq:
                if token_id < 0:
                    raise ValueError(
                        f"Sequence {idx} contains negative token ID: {token_id}"
                    )
                # Upper bound will be validated against model vocab size in server

        return self

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "input_ids": [[1, 2, 3, 4, 5]],
                "topk": 128,
                "return_dtype": "int8",
                "temperature": 2.0,
                "max_tokens": 5
            }
        }
    )


class TopKResponse(BaseModel):
    """Response schema for /v1/topk endpoint.

    Returns sparse top-k logits with int8 quantization for efficient transfer.

    Attributes:
        indices: Top-k token indices per position. Shape: (batch_size, num_positions, k)
                 dtype: int32
        values_int8: Quantized top-k logit values. Shape: (batch_size, num_positions, k)
                     dtype: int8, range [-128, 127]
        scale: Per-position scale factors for dequantization. Shape: (batch_size, num_positions)
               dtype: float32
               Dequantize: logits = values_int8.float() * scale.unsqueeze(-1)
        other_logit: Logit value for tokens outside top-k (logsumexp of tail).
                     Shape: (batch_size, num_positions)
                     dtype: float32
                     Represents the log-sum-exp of all logits for tokens outside top-k
        batch_size: Number of sequences in batch
        num_positions: Number of token positions per sequence
        k: Number of top logits per position
    """

    indices: List[List[List[int]]] = Field(
        ...,
        description="Top-k token indices per position. Shape: (batch_size, num_positions, k)"
    )

    values_int8: List[List[List[int]]] = Field(
        ...,
        description="Quantized top-k logit values (int8). Shape: (batch_size, num_positions, k)"
    )

    scale: List[List[float]] = Field(
        ...,
        description="Per-position scale factors for dequantization. Shape: (batch_size, num_positions)"
    )

    other_logit: List[List[float]] = Field(
        ...,
        description="Logit value for tokens outside top-k. Shape: (batch_size, num_positions)"
    )

    batch_size: int = Field(
        ...,
        description="Number of sequences in batch",
        ge=1
    )

    num_positions: int = Field(
        ...,
        description="Number of token positions per sequence",
        ge=1
    )

    k: int = Field(
        ...,
        description="Number of top logits per position",
        ge=1
    )

    return_dtype: Literal["int8", "float16", "float32"] = Field(
        ...,
        description="Actual data type of returned values (echoed from request)"
    )

    @field_validator('values_int8')
    @classmethod
    def validate_int8_range(cls, v):
        """Validate int8 values are in valid range [-128, 127]."""
        for batch_idx, batch_seq in enumerate(v):
            for pos_idx, pos_values in enumerate(batch_seq):
                for val_idx, val in enumerate(pos_values):
                    if not (-128 <= val <= 127):
                        raise ValueError(
                            f"values_int8[{batch_idx}][{pos_idx}][{val_idx}] = {val} "
                            f"is out of int8 range [-128, 127]"
                        )
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "indices": [[[100, 200, 300, 400, 500]]],
                "values_int8": [[[127, 100, 80, 60, 40]]],
                "scale": [[0.05]],
                "other_logit": [[-4.6]],
                "batch_size": 1,
                "num_positions": 1,
                "k": 5,
                "return_dtype": "int8"
            }
        }
    )


class TopKErrorResponse(BaseModel):
    """Error response schema for /v1/topk endpoint.

    Attributes:
        error: Error type/code
        message: Human-readable error message
        details: Optional additional error details
    """

    error: str = Field(..., description="Error type/code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict] = Field(None, description="Additional error details")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "InvalidInput",
                "message": "input_ids must contain at least one sequence",
                "details": {"field": "input_ids", "value": []}
            }
        }
    )
