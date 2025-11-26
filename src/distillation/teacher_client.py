"""
HTTP client for remote teacher server.

Implements retry logic, batching, and response parsing for /v1/topk endpoint.

⚠️ DEPRECATION NOTICE:
This client is designed for a CUSTOM /v1/topk endpoint that requires server modifications.

For production use, prefer vllm_teacher_client.py which uses STANDARD vLLM endpoints:
- No custom server code needed
- Works with any vLLM server out-of-box
- Uses /v1/completions with prompt_logprobs parameter

See ai-notes/VLLM_INTEGRATION.md for migration guide.
"""

import json
import logging
import time
from typing import List, Optional, Dict, Any

import requests
from requests.exceptions import ConnectionError, Timeout, RequestException

from .schemas import TopKRequest, TopKResponse, TopKErrorResponse


logger = logging.getLogger(__name__)

# Define retryable status codes (5xx server errors)
RETRYABLE_STATUS_CODES = frozenset({500, 502, 503, 504})


class TeacherClientError(Exception):
    """Base exception for teacher client errors."""
    pass


class TeacherServerError(TeacherClientError):
    """Exception raised when teacher server returns an error."""
    pass


class TeacherNetworkError(TeacherClientError):
    """Exception raised for network/connection errors after retries exhausted."""
    pass


class TeacherClient:
    """
    HTTP client for querying remote teacher server.

    Implements:
    - HTTP request/response handling for /v1/topk endpoint
    - Retry logic with exponential backoff
    - Request batching for multiple sequences
    - Response parsing and validation

    Attributes:
        endpoint_url: Full URL to teacher server endpoint (e.g., "http://localhost:8000/v1/topk")
        timeout: Request timeout in seconds (default: 30)
        verify_ssl: Whether to verify SSL certificates (default: True)
        max_retries: Maximum number of retry attempts (default: 3)
        backoff_base: Base backoff duration in seconds (default: 1.0)
    """

    def __init__(
        self,
        endpoint_url: str,
        timeout: float = 30.0,
        verify_ssl: bool = True,
        max_retries: int = 3,
        backoff_base: float = 1.0,
    ):
        """
        Initialize teacher client.

        Args:
            endpoint_url: Full URL to /v1/topk endpoint (e.g., "http://localhost:8000/v1/topk")
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
            max_retries: Maximum retry attempts on network errors (default: 3)
            backoff_base: Base duration for exponential backoff in seconds (default: 1.0)
        """
        if not endpoint_url:
            raise ValueError("endpoint_url cannot be empty")

        self.endpoint_url = endpoint_url
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.max_retries = max_retries
        self.backoff_base = backoff_base

        # Track close state for idempotent cleanup
        self._closed = False

        # Create a session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

        logger.info(
            f"Initialized TeacherClient: endpoint={endpoint_url}, timeout={timeout}s, "
            f"verify_ssl={verify_ssl}, max_retries={max_retries}"
        )

    def query_topk(
        self,
        input_ids: List[List[int]],
        topk: int = 128,
        return_dtype: str = "int8",
        temperature: float = 1.0,
    ) -> TopKResponse:
        """
        Query teacher for top-k logits on input sequences.

        Args:
            input_ids: Token ID sequences. Shape: (batch_size, seq_len)
            topk: Number of top logits to return per position
            return_dtype: Data type for returned values ("int8", "float16", "float32")
            temperature: Temperature for softmax computation

        Returns:
            TopKResponse with sparse top-k logits

        Raises:
            TeacherNetworkError: If network errors persist after max_retries
            TeacherServerError: If server returns an error response
            TeacherClientError: For other client-side errors
        """
        request = TopKRequest(
            input_ids=input_ids,
            topk=topk,
            return_dtype=return_dtype,
            temperature=temperature,
        )

        # Retry loop with exponential backoff
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    self.endpoint_url,
                    json=request.model_dump(),
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                )

                # Check for retryable HTTP server errors (5xx)
                if response.status_code in RETRYABLE_STATUS_CODES:
                    if attempt < self.max_retries - 1:
                        backoff_duration = self.backoff_base * (2 ** attempt)
                        logger.warning(
                            f"Server error {response.status_code} (attempt {attempt + 1}/{self.max_retries}). "
                            f"Retrying in {backoff_duration}s..."
                        )
                        time.sleep(backoff_duration)
                        continue
                    else:
                        # Final attempt failed with retryable server error
                        raise TeacherServerError(
                            f"Server error {response.status_code} after {self.max_retries} retries"
                        )

                # Handle non-retryable HTTP errors (4xx and other errors)
                if response.status_code != 200:
                    try:
                        error_data = response.json()
                    except (ValueError, json.JSONDecodeError) as e:
                        raise TeacherServerError(
                            f"Server error ({response.status_code}): Invalid JSON. Body: {response.text[:200]}"
                        ) from e

                    try:
                        error_response = TopKErrorResponse(**error_data)
                    except Exception:
                        # Failed to parse error response, use raw text
                        raise TeacherServerError(
                            f"Server error ({response.status_code}): {response.text[:200]}"
                        )

                    # Successfully parsed error response
                    raise TeacherServerError(
                        f"Server error ({response.status_code}): {error_response.message}"
                    )

                # Parse successful response
                try:
                    response_data = response.json()
                except (ValueError, json.JSONDecodeError) as e:
                    raise TeacherClientError(
                        f"Server returned invalid JSON. Body: {response.text[:200]}"
                    ) from e

                parsed_response = self._parse_response(response_data)

                # VALIDATE DIMENSIONS
                self._validate_response_dimensions(parsed_response, input_ids, topk)

                return parsed_response

            except (ConnectionError, Timeout) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s
                    backoff_duration = self.backoff_base * (2 ** attempt)
                    logger.warning(
                        f"Network error (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {backoff_duration}s..."
                    )
                    time.sleep(backoff_duration)
                # No else block to avoid duplicate logging
            except RequestException as e:
                # Non-retryable request exceptions
                raise TeacherNetworkError(f"Request failed: {e}") from e

        # If we exit the retry loop, all retries failed
        raise TeacherNetworkError(
            f"Failed after {self.max_retries} retries. Last error: {last_exception}"
        ) from last_exception

    def batch_query(
        self,
        input_ids_list: List[List[List[int]]],
        topk: int = 128,
        return_dtype: str = "int8",
        temperature: float = 1.0,
        batch_size: int = 32,
    ) -> List[TopKResponse]:
        """
        Query teacher for multiple batches with automatic batching.

        Splits input_ids_list into chunks of size batch_size and sends
        separate requests for each chunk. This reduces per-sequence overhead
        by batching multiple sequences into single HTTP requests.

        Args:
            input_ids_list: List of batches of token ID sequences.
                           Each batch has shape (batch_size, seq_len)
            topk: Number of top logits to return per position
            return_dtype: Data type for returned values
            temperature: Temperature for softmax computation
            batch_size: Maximum sequences per HTTP request (default: 32)

        Returns:
            List of TopKResponse objects, one per batch

        Raises:
            TeacherNetworkError: If network errors persist after retries
            TeacherServerError: If server returns an error response
        """
        if not input_ids_list:
            return []

        responses = []
        num_batches = (len(input_ids_list) + batch_size - 1) // batch_size

        logger.info(
            f"Batching {len(input_ids_list)} sequences into {num_batches} requests "
            f"(batch_size={batch_size})"
        )

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(input_ids_list))
            batch = input_ids_list[start_idx:end_idx]

            logger.debug(
                f"Processing batch {batch_idx + 1}/{num_batches}: "
                f"sequences {start_idx} to {end_idx - 1}"
            )

            # Flatten batch into single input_ids list
            # batch is List[List[List[int]]], we need List[List[int]]
            flattened_batch = [seq for sequences in batch for seq in sequences]

            response = self.query_topk(
                input_ids=flattened_batch,
                topk=topk,
                return_dtype=return_dtype,
                temperature=temperature,
            )
            responses.append(response)

        return responses

    def _parse_response(self, response_data: Dict[str, Any]) -> TopKResponse:
        """
        Parse JSON response into TopKResponse dataclass.

        Args:
            response_data: Raw JSON response from server

        Returns:
            Validated TopKResponse object

        Raises:
            TeacherClientError: If response format is invalid
        """
        try:
            # Pydantic handles validation
            response = TopKResponse(**response_data)
            return response
        except Exception as e:
            raise TeacherClientError(
                f"Failed to parse response: {e}. Response keys: {list(response_data.keys())}"
            ) from e

    def _validate_response_dimensions(
        self,
        response: TopKResponse,
        input_ids: List[List[int]],
        topk: int,
    ) -> None:
        """
        Validate response dimensions match request.

        Args:
            response: Parsed response from server
            input_ids: Original request input_ids
            topk: Original request topk

        Raises:
            TeacherClientError: If dimensions don't match
        """
        # Validate batch_size
        expected_batch_size = len(input_ids)
        if response.batch_size != expected_batch_size:
            raise TeacherClientError(
                f"Response batch_size mismatch: expected {expected_batch_size}, "
                f"got {response.batch_size}"
            )

        # Validate k
        if response.k != topk:
            raise TeacherClientError(
                f"Response k mismatch: expected {topk}, got {response.k}"
            )

        # Validate num_positions for each sequence
        for idx, input_seq in enumerate(input_ids):
            expected_positions = len(input_seq)
            if idx < len(response.indices):
                actual_positions = len(response.indices[idx])
                if actual_positions != expected_positions:
                    raise TeacherClientError(
                        f"Response num_positions mismatch for sequence {idx}: "
                        f"expected {expected_positions}, got {actual_positions}"
                    )

        # Validate array lengths are consistent
        if len(response.indices) != response.batch_size:
            raise TeacherClientError(
                f"indices length {len(response.indices)} != batch_size {response.batch_size}"
            )

        if len(response.values_int8) != response.batch_size:
            raise TeacherClientError(
                f"values_int8 length {len(response.values_int8)} != batch_size {response.batch_size}"
            )

        if len(response.scale) != response.batch_size:
            raise TeacherClientError(
                f"scale length {len(response.scale)} != batch_size {response.batch_size}"
            )

        if len(response.other_mass) != response.batch_size:
            raise TeacherClientError(
                f"other_mass length {len(response.other_mass)} != batch_size {response.batch_size}"
            )

    def health_check(self) -> bool:
        """
        Check if teacher server is reachable.

        Returns:
            True if server responds, False otherwise
        """
        try:
            # Try a minimal request to check connectivity
            health_endpoint = self.endpoint_url.replace("/v1/topk", "/health")
            response = self.session.get(
                health_endpoint,
                timeout=5.0,
                verify=self.verify_ssl,
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    def close(self):
        """Close the HTTP session (idempotent)."""
        if self._closed:
            return  # Already closed

        try:
            if hasattr(self, 'session') and self.session:
                self.session.close()
            self._closed = True
            logger.info("Closed TeacherClient session")
        except Exception as e:
            logger.warning(f"Error closing session: {e}")
            self._closed = True

    def __del__(self):
        """Finalizer to ensure session is closed."""
        try:
            self.close()
        except Exception:
            pass  # Suppress exceptions in __del__

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        try:
            self.close()
        except Exception as e:
            logger.warning(f"Error in __exit__ while closing: {e}")
        return False
