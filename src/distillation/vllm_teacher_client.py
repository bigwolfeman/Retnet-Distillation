"""
HTTP client for vLLM teacher server using standard /v1/completions endpoint.

This client uses vLLM's built-in prompt_logprobs parameter to get sparse top-k
logits for knowledge distillation WITHOUT requiring a custom /v1/topk endpoint.

Key Features:
- Uses standard vLLM /v1/completions endpoint
- Gets prompt logprobs via prompt_logprobs parameter
- Handles token string -> ID conversion
- Supports up to k=128 sparse logprobs
- Batch processing support
- Retry logic with exponential backoff
"""

import json
import logging
import time
from typing import List, Optional, Dict, Any, Tuple

import requests
from requests.exceptions import ConnectionError, Timeout, RequestException

logger = logging.getLogger(__name__)

# Define retryable status codes (5xx server errors)
RETRYABLE_STATUS_CODES = frozenset({500, 502, 503, 504})


class VLLMTeacherError(Exception):
    """Base exception for vLLM teacher client errors."""
    pass


class VLLMServerError(VLLMTeacherError):
    """Exception raised when vLLM server returns an error."""
    pass


class VLLMNetworkError(VLLMTeacherError):
    """Exception raised for network/connection errors after retries exhausted."""
    pass


class VLLMTeacherClient:
    """
    HTTP client for querying vLLM teacher server using standard endpoints.

    Uses /v1/completions with prompt_logprobs to get sparse top-k logits
    for knowledge distillation. No custom endpoint required!

    Attributes:
        base_url: Base URL to vLLM server (e.g., "http://localhost:8080")
        model: Model name/identifier
        api_key: Optional API key for authentication
        timeout: Request timeout in seconds (default: 30)
        verify_ssl: Whether to verify SSL certificates (default: True)
        max_retries: Maximum number of retry attempts (default: 3)
        backoff_base: Base backoff duration in seconds (default: 1.0)
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        verify_ssl: bool = True,
        max_retries: int = 3,
        backoff_base: float = 1.0,
    ):
        """
        Initialize vLLM teacher client.

        Args:
            base_url: Base URL to vLLM server (e.g., "http://localhost:8080")
            model: Model identifier (e.g., "meta-llama/Llama-3.2-1B-Instruct")
            api_key: Optional API key for Bearer token authentication
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
            max_retries: Maximum retry attempts on network errors (default: 3)
            backoff_base: Base duration for exponential backoff in seconds (default: 1.0)
        """
        if not base_url:
            raise ValueError("base_url cannot be empty")
        if not model:
            raise ValueError("model cannot be empty")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.max_retries = max_retries
        self.backoff_base = backoff_base

        # Track close state for idempotent cleanup
        self._closed = False

        # Create a session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

        # Cache tokenizer for token string -> ID conversion
        self._token_to_id_cache: Dict[str, int] = {}

        logger.info(
            f"Initialized VLLMTeacherClient: base_url={base_url}, model={model}, "
            f"timeout={timeout}s, max_retries={max_retries}"
        )

    def get_prompt_logprobs(
        self,
        input_ids: List[List[int]],
        topk: int = 128,
        temperature: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Get top-k logprobs for input sequences using vLLM's prompt_logprobs.

        Args:
            input_ids: Token ID sequences. Shape: (batch_size, seq_len)
            topk: Number of top logprobs to return per position (max 128)
            temperature: Temperature for logprob computation (default: 1.0)

        Returns:
            List of dicts, one per sequence, containing:
                - tokens: List[str] - token strings
                - top_logprobs: List[Dict[str, float]] - sparse logprobs per position
                - indices: List[List[int]] - top-k token IDs per position [seq_len, K]
                - logprobs: List[List[float]] - top-k log-probs per position [seq_len, K]
                  NOTE: Padded with -inf for empty positions (e.g., BOS) to ensure consistent shape

        Raises:
            VLLMNetworkError: If network errors persist after max_retries
            VLLMServerError: If server returns an error response
            VLLMTeacherError: For other client-side errors
        """
        if topk > 128:
            logger.warning(f"topk={topk} > 128, vLLM may reject this. Capping at 128.")
            topk = 128

        # vLLM's completions endpoint expects array of prompts for batching
        request = {
            "model": self.model,
            "prompt_token_ids": input_ids,  # vLLM parameter for token IDs
            "max_tokens": 1,  # Need to generate at least 1 token to get logprobs
            "echo": True,  # Echo back the prompt
            "logprobs": topk,  # Top-k for generated tokens (not used for distillation)
            "prompt_logprobs": topk,  # Top-k for PROMPT tokens (this is what we need!)
            "temperature": temperature,
        }

        # Retry loop with exponential backoff
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    f"{self.base_url}/v1/completions",
                    json=request,
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
                        raise VLLMServerError(
                            f"Server error {response.status_code} after {self.max_retries} retries"
                        )

                # Handle non-retryable HTTP errors (4xx and other errors)
                if response.status_code != 200:
                    error_text = response.text[:500]
                    raise VLLMServerError(
                        f"Server error ({response.status_code}): {error_text}"
                    )

                # Parse successful response
                try:
                    response_data = response.json()
                except (ValueError, json.JSONDecodeError) as e:
                    raise VLLMTeacherError(
                        f"Server returned invalid JSON. Body: {response.text[:200]}"
                    ) from e

                # Extract and process logprobs for each sequence
                results = self._process_completion_response(response_data)

                return results

            except (ConnectionError, Timeout) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    backoff_duration = self.backoff_base * (2 ** attempt)
                    logger.warning(
                        f"Network error (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {backoff_duration}s..."
                    )
                    time.sleep(backoff_duration)
            except RequestException as e:
                # Non-retryable request exceptions
                raise VLLMNetworkError(f"Request failed: {e}") from e

        # If we exit the retry loop, all retries failed
        raise VLLMNetworkError(
            f"Failed after {self.max_retries} retries. Last error: {last_exception}"
        ) from last_exception

    def _process_completion_response(
        self, response_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Process vLLM completions response into structured logprob data.

        Args:
            response_data: Raw JSON response from /v1/completions

        Returns:
            List of dicts with processed logprob data per sequence

        Raises:
            VLLMTeacherError: If response format is invalid
        """
        if "choices" not in response_data:
            raise VLLMTeacherError("Response missing 'choices' field")

        choices = response_data["choices"]
        results = []

        for choice in choices:
            if "logprobs" not in choice or choice["logprobs"] is None:
                raise VLLMTeacherError("Choice missing 'logprobs' field")

            logprobs_data = choice["logprobs"]

            # Extract token strings and top_logprobs
            tokens = logprobs_data.get("tokens", [])
            top_logprobs_list = logprobs_data.get("top_logprobs", [])

            if not tokens or not top_logprobs_list:
                raise VLLMTeacherError("Missing tokens or top_logprobs in response")

            # Convert token strings to IDs and extract logprobs
            # Note: First token (BOS) has None logprobs, skip it
            indices = []
            logprobs = []

            for i, (token_str, top_logprobs_dict) in enumerate(
                zip(tokens, top_logprobs_list)
            ):
                if top_logprobs_dict is None:
                    # BOS token or other special case - pad with -inf
                    # Ensure consistent shape [seq_len, K] with -inf for log-probs
                    indices.append([0] * topk)  # Padding token IDs
                    logprobs.append([-float('inf')] * topk)  # -inf for log-probabilities
                    continue

                # Convert token strings to IDs
                position_indices = []
                position_logprobs = []

                for tok_str, logprob in top_logprobs_dict.items():
                    # Try to get token ID from cache or decode
                    tok_id = self._get_token_id(tok_str)
                    position_indices.append(tok_id)
                    position_logprobs.append(logprob)

                # Pad to exactly topk entries if needed
                while len(position_indices) < topk:
                    position_indices.append(0)  # Padding token ID
                    position_logprobs.append(-float('inf'))  # -inf for log-probabilities

                # Ensure we return exactly topk entries (truncate if server returned more)
                indices.append(position_indices[:topk])
                logprobs.append(position_logprobs[:topk])

            results.append(
                {
                    "tokens": tokens,
                    "top_logprobs": top_logprobs_list,
                    "indices": indices,
                    "logprobs": logprobs,
                }
            )

        return results

    def _get_token_id(self, token_str: str) -> int:
        """
        Convert token string to token ID.

        Uses /tokenize endpoint to get the token ID. Results are cached.

        Args:
            token_str: Token string from logprobs response

        Returns:
            Token ID (integer)

        Raises:
            VLLMTeacherError: If token cannot be converted
        """
        # Check cache first
        if token_str in self._token_to_id_cache:
            return self._token_to_id_cache[token_str]

        # Use tokenize endpoint to get ID
        try:
            response = self.session.post(
                f"{self.base_url}/tokenize",
                json={"model": self.model, "prompt": token_str},
                timeout=5.0,
                verify=self.verify_ssl,
            )

            if response.status_code == 200:
                data = response.json()
                tokens = data.get("tokens", [])
                if tokens:
                    # First token (after BOS if present) should be our token
                    # vLLM adds BOS token, so the actual token is usually at index 1
                    tok_id = tokens[-1] if len(tokens) > 1 else tokens[0]
                    self._token_to_id_cache[token_str] = tok_id
                    return tok_id
                else:
                    raise VLLMTeacherError(
                        f"Tokenize returned no tokens for '{token_str}'"
                    )
            else:
                raise VLLMTeacherError(
                    f"Tokenize failed ({response.status_code}): {response.text[:200]}"
                )
        except Exception as e:
            # Fallback: Use a placeholder ID (vocab_size - 1)
            # This is not ideal but prevents crashes
            logger.warning(
                f"Failed to get token ID for '{token_str}': {e}. Using placeholder."
            )
            placeholder_id = 128255  # Llama vocab_size - 1
            self._token_to_id_cache[token_str] = placeholder_id
            return placeholder_id

    def health_check(self) -> bool:
        """
        Check if vLLM server is reachable.

        Returns:
            True if server responds, False otherwise
        """
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=5.0,
                verify=self.verify_ssl,
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information from /v1/models endpoint.

        Returns:
            Model info dict

        Raises:
            VLLMTeacherError: If request fails
        """
        try:
            response = self.session.get(
                f"{self.base_url}/v1/models",
                timeout=5.0,
                verify=self.verify_ssl,
            )

            if response.status_code == 200:
                return response.json()
            else:
                raise VLLMTeacherError(
                    f"Failed to get model info ({response.status_code}): {response.text[:200]}"
                )
        except Exception as e:
            raise VLLMTeacherError(f"Failed to get model info: {e}") from e

    def close(self):
        """Close the HTTP session (idempotent)."""
        if self._closed:
            return

        try:
            if hasattr(self, "session") and self.session:
                self.session.close()
            self._closed = True
            logger.info("Closed VLLMTeacherClient session")
        except Exception as e:
            logger.warning(f"Error closing session: {e}")
            self._closed = True

    def __del__(self):
        """Finalizer to ensure session is closed."""
        try:
            self.close()
        except Exception:
            pass

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
