"""
Async teacher client for overlapping teacher inference with student training.

This module implements a pipeline pattern where teacher logits for batch N+1
are fetched asynchronously while the student trains on batch N, eliminating
idle GPU time during teacher inference.

Features:
- Thread-based async execution via ThreadPoolExecutor
- 1-batch pipeline depth (submit N+1 while processing N)
- Compatible with any teacher client implementing get_prompt_logprobs()
- Integrates seamlessly with CachingTeacherWrapper
- Graceful shutdown with thread cleanup
"""

import logging
import threading
from typing import List, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor, Future
import torch


logger = logging.getLogger(__name__)


class AsyncTeacherClient:
    """Asynchronous wrapper for teacher clients to overlap inference with training.

    This client wraps any teacher client (VLLMTeacherClient, DirectTeacherClient, etc.)
    and executes `get_prompt_logprobs` calls asynchronously in a background thread.
    This allows the training loop to submit batch N+1 while processing batch N,
    eliminating GPU idle time during teacher inference.

    Usage pattern:
        ```python
        # Initialization
        base_teacher = VLLMTeacherClient(...)
        async_teacher = AsyncTeacherClient(base_teacher)

        # Training loop
        for batch_n, batch_n_plus_1 in batch_pairs:
            # Submit next batch asynchronously
            async_teacher.submit(batch_n_plus_1['input_ids'])

            # Retrieve current batch logits (blocks if not ready)
            logits = async_teacher.get()

            # Train on current batch while next batch is being fetched
            train_step(batch_n, logits)

        # Cleanup
        async_teacher.close()
        ```

    Thread safety: This class is designed for single-threaded training loops.
    Do not call submit() or get() from multiple threads concurrently.

    Example:
        >>> teacher = VLLMTeacherClient("http://localhost:8000", "llama-3.2-1B")
        >>> async_teacher = AsyncTeacherClient(teacher, max_queue_depth=4)
        >>>
        >>> # Submit batch for async processing
        >>> async_teacher.submit(input_ids=[[1, 2, 3, 4]])
        >>>
        >>> # Do other work here...
        >>>
        >>> # Retrieve result (blocks if not ready)
        >>> indices, logprobs, other_mass = async_teacher.get()
        >>>
        >>> # Cleanup
        >>> async_teacher.close()
    """

    def __init__(
        self,
        teacher_client: Any,
        max_queue_depth: int = 4,
        max_workers: int = 1,
    ):
        """Initialize async teacher client.

        Args:
            teacher_client: Base teacher client implementing get_prompt_logprobs().
                           Can be VLLMTeacherClient, DirectTeacherClient, CachingTeacherWrapper, etc.
            max_queue_depth: Maximum number of pending requests (default: 4).
                            Higher values increase memory usage but provide more buffering.
            max_workers: Number of worker threads (default: 1).
                        Use 1 to avoid vLLM server contention. Higher values are
                        experimental and may not improve performance.
        """
        self.teacher = teacher_client
        self.max_queue_depth = max_queue_depth
        self.max_workers = max_workers

        # Thread pool for async execution
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="AsyncTeacher"
        )

        # Pending future from most recent submit() call
        self._pending_future: Optional[Future] = None

        # Shutdown flag
        self._shutdown = False

        # Lock for thread-safe access to _pending_future
        self._lock = threading.Lock()

        logger.info(f"Initialized AsyncTeacherClient:")
        logger.info(f"  Max queue depth: {max_queue_depth}")
        logger.info(f"  Max workers: {max_workers}")
        logger.info(f"  Base teacher: {type(teacher_client).__name__}")

    @property
    def device(self):
        """Pass-through device attribute from wrapped client (for async gating logic)."""
        return getattr(self.teacher, 'device', None)

    def submit(
        self,
        input_ids: List[List[int]],
        topk: int = 128,
        temperature: float = 1.0,
    ) -> None:
        """Submit a teacher inference request asynchronously.

        This method submits the request to a background thread and returns immediately.
        The result can be retrieved later via get().

        Warning: Only one pending request is supported at a time. Calling submit()
        before calling get() will overwrite the previous pending request (which is
        acceptable for the training loop pattern where we always want the most recent batch).

        Args:
            input_ids: Batch of input token IDs [[seq1], [seq2], ...]
            topk: Number of top-k logits to return (default: 128)
            temperature: Temperature for softmax (default: 1.0)

        Raises:
            RuntimeError: If client has been shut down
        """
        if self._shutdown:
            raise RuntimeError("AsyncTeacherClient has been shut down")

        # Submit async request to executor
        future = self.executor.submit(
            self._fetch_teacher_logits,
            input_ids,
            topk,
            temperature,
        )

        # Store future (thread-safe)
        with self._lock:
            # If there's a pending future, log a warning (unusual but not fatal)
            if self._pending_future is not None and not self._pending_future.done():
                logger.warning(
                    "Submitting new request before previous request completed. "
                    "Previous request will be ignored (this is normal for training loop)."
                )

            self._pending_future = future

    def submit_tensors(
        self,
        input_ids: torch.Tensor,
        topk: Optional[int] = None,
    ) -> None:
        """Submit a teacher inference request asynchronously using GPU tensors.

        FIX #2: GPU-native async prefetch that avoids CPU/GPU synchronization.
        This method accepts GPU tensors directly and processes them in a background
        thread without .cpu().tolist() conversions, eliminating 64+ CPU/GPU syncs
        per optimizer step.

        This method submits the request to a background thread and returns immediately.
        The result can be retrieved later via get().

        Warning: Only one pending request is supported at a time. Calling submit_tensors()
        before calling get() will overwrite the previous pending request.

        Args:
            input_ids: Token ID tensor on GPU. Shape: [batch_size, seq_len]
            topk: Number of top logits to return (default: wrapped client default)

        Raises:
            RuntimeError: If client has been shut down
            NotImplementedError: If wrapped client doesn't support tensor API
        """
        if self._shutdown:
            raise RuntimeError("AsyncTeacherClient has been shut down")

        # Check if wrapped client has tensor API
        if not hasattr(self.teacher, 'get_top_k_logits_tensors'):
            raise NotImplementedError(
                f"Wrapped client {type(self.teacher).__name__} does not support "
                "get_top_k_logits_tensors(). Use submit() instead for list-based API."
            )

        # Submit async request to executor (tensor-based)
        future = self.executor.submit(
            self._fetch_teacher_logits_tensors,
            input_ids,
            topk,
        )

        # Store future (thread-safe)
        with self._lock:
            # If there's a pending future, log a warning (unusual but not fatal)
            if self._pending_future is not None and not self._pending_future.done():
                logger.warning(
                    "Submitting new tensor request before previous request completed. "
                    "Previous request will be ignored (this is normal for training loop)."
                )

            self._pending_future = future

    def get(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve the result of the most recent submit() call.

        This method blocks until the pending request completes. If no request
        is pending, raises RuntimeError.

        Returns:
            Tuple of:
                - topk_indices: [batch_size, seq_len, k] token indices
                - topk_logprobs: [batch_size, seq_len, k] log probabilities
                - other_mass: [batch_size, seq_len, 1] log prob of remaining mass

        Raises:
            RuntimeError: If no pending request or if client has been shut down
            Exception: Re-raises any exception that occurred during teacher inference
        """
        if self._shutdown:
            raise RuntimeError("AsyncTeacherClient has been shut down")

        # Get pending future (thread-safe)
        with self._lock:
            if self._pending_future is None:
                raise RuntimeError(
                    "No pending request. Call submit() before get()."
                )
            future = self._pending_future
            self._pending_future = None  # Clear after retrieving

        # Block until result is ready
        try:
            result = future.result()
            return result
        except Exception as e:
            logger.error(f"Teacher inference failed: {e}")
            raise

    def _fetch_teacher_logits(
        self,
        input_ids: List[List[int]],
        topk: int,
        temperature: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Internal method to fetch teacher logits (runs in background thread).

        This method calls the base teacher client's get_prompt_logprobs() method
        and converts the result to tensors.

        Args:
            input_ids: Batch of input token IDs
            topk: Number of top-k logits
            temperature: Temperature for softmax

        Returns:
            Tuple of (topk_indices, topk_logprobs, other_mass) as tensors
        """
        # Call base teacher client (this is the blocking call we want to overlap)
        results = self.teacher.get_prompt_logprobs(
            input_ids=input_ids,
            topk=topk,
            temperature=temperature,
        )

        # Convert results to tensors
        # (This logic is copied from trainer.py:_fetch_teacher_logits to maintain compatibility)
        batch_size = len(results)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build tensors from results
        all_indices = []
        all_logprobs = []

        for result in results:
            # indices is List[List[int]], logprobs is List[List[float]]
            indices = result['indices']
            logprobs = result['logprobs']

            # Convert to tensors and pad if needed
            seq_indices = []
            seq_logprobs = []

            for pos_indices, pos_logprobs in zip(indices, logprobs):
                # Handle empty positions (like BOS)
                if not pos_indices:
                    # Use padding values
                    seq_indices.append([0] * topk)
                    seq_logprobs.append([0.0] * topk)
                else:
                    # Pad to topk if needed
                    while len(pos_indices) < topk:
                        pos_indices.append(0)
                        pos_logprobs.append(-float('inf'))
                    seq_indices.append(pos_indices[:topk])
                    seq_logprobs.append(pos_logprobs[:topk])

            all_indices.append(seq_indices)
            all_logprobs.append(seq_logprobs)

        # Convert to tensors
        indices_tensor = torch.tensor(all_indices, dtype=torch.long, device=device)
        logprobs_tensor = torch.tensor(all_logprobs, dtype=torch.float32, device=device)

        # Compute other_mass from top-k logprobs
        probs = torch.exp(logprobs_tensor)  # [batch_size, seq_len, k]
        total_prob = probs.sum(dim=-1, keepdim=True)  # [batch_size, seq_len, 1]

        # Other mass is log(1 - total_prob), clamped to avoid log(0)
        other_prob = torch.clamp(1.0 - total_prob, min=1e-8)
        other_mass = torch.log(other_prob)  # [batch_size, seq_len, 1]

        return indices_tensor, logprobs_tensor, other_mass

    def _fetch_teacher_logits_tensors(
        self,
        input_ids: torch.Tensor,
        topk: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Internal method to fetch teacher logits using GPU tensors (runs in background thread).

        FIX #2: GPU-native async worker that avoids CPU/GPU synchronization.
        This method calls the wrapped client's get_top_k_logits_tensors() directly
        without any tensor-to-list conversions, maintaining GPU-only execution.

        Thread safety: Uses torch.no_grad() to avoid autograd overhead in background thread.
        The wrapped client is responsible for proper device placement.

        Args:
            input_ids: Token ID tensor on GPU. Shape: [batch_size, seq_len]
            topk: Number of top logits to return (None = use client default)

        Returns:
            Tuple of (topk_indices, topk_logprobs, other_mass) as GPU tensors
        """
        # Use no_grad context to avoid autograd overhead in background thread
        with torch.no_grad():
            # Call wrapped client's tensor API (this is the blocking call we want to overlap)
            topk_indices, topk_logprobs, other_mass = self.teacher.get_top_k_logits_tensors(
                input_ids=input_ids,
                topk=topk,
            )

            # Return tensors directly (no conversions)
            return topk_indices, topk_logprobs, other_mass

    def get_prompt_logprobs(
        self,
        input_ids: List[List[int]],
        topk: int = 128,
        temperature: float = 1.0,
    ):
        """Synchronous fallback for backward compatibility.

        This method provides the same interface as the base teacher client,
        allowing AsyncTeacherClient to be used as a drop-in replacement.

        When called directly (without using submit/get pattern), this behaves
        synchronously - it submits and immediately retrieves the result.

        For async operation, use the submit/get pattern instead.

        Args:
            input_ids: Batch of input token IDs
            topk: Number of top-k logits
            temperature: Temperature for softmax

        Returns:
            Same format as base teacher client's get_prompt_logprobs()
        """
        # Call base teacher directly (synchronous)
        return self.teacher.get_prompt_logprobs(
            input_ids=input_ids,
            topk=topk,
            temperature=temperature,
        )

    def has_pending(self) -> bool:
        """Check if there is a pending request.

        Returns:
            True if a request is pending, False otherwise
        """
        with self._lock:
            return self._pending_future is not None and not self._pending_future.done()

    def close(self):
        """Shut down the async client gracefully.

        This method waits for any pending requests to complete, then shuts down
        the thread pool. Call this during training cleanup to ensure threads are
        properly terminated.

        Note: After calling close(), the client cannot be used again.
        """
        if self._shutdown:
            logger.warning("AsyncTeacherClient already shut down")
            return

        logger.info("Shutting down AsyncTeacherClient...")

        # Mark as shutdown
        self._shutdown = True

        # Wait for pending future to complete (if any)
        with self._lock:
            if self._pending_future is not None and not self._pending_future.done():
                logger.info("Waiting for pending request to complete...")
                try:
                    self._pending_future.result(timeout=30.0)
                except Exception as e:
                    logger.warning(f"Pending request failed during shutdown: {e}")

        # Shutdown executor gracefully
        self.executor.shutdown(wait=True)

        logger.info("AsyncTeacherClient shut down complete")

    def get_top_k_logits_tensors(
        self,
        input_ids: torch.Tensor,
        topk: Optional[int] = None,
    ):
        """GPU-native tensor API proxy to wrapped client.

        FIX #2: This method exposes the tensor API of the wrapped client through
        the async wrapper, ensuring the trainer can use the fast GPU-native path
        whether async is enabled or not.

        This eliminates 64 CPU/GPU syncs per optimizer step when using DirectTeacherClient
        wrapped in AsyncTeacherClient.

        Args:
            input_ids: Token ID tensor on GPU. Shape: [batch_size, seq_len]
            topk: Number of top logits to return (default: wrapped client default)

        Returns:
            tuple of 3 tensors (all on GPU):
                - topk_indices: [batch, seq_len, topk]
                - topk_logprobs: [batch, seq_len, topk]
                - other_mass: [batch, seq_len, 1]

        Raises:
            NotImplementedError: If wrapped client doesn't support tensor API
        """
        # Check if wrapped client has tensor API
        if not hasattr(self.teacher, 'get_top_k_logits_tensors'):
            raise NotImplementedError(
                f"Wrapped client {type(self.teacher).__name__} does not support "
                "get_top_k_logits_tensors(). This method requires DirectTeacherClient "
                "or another client implementing the GPU-native tensor API."
            )

        # Proxy directly to wrapped client (synchronous call)
        # NOTE: This bypasses the async submit/get pattern, which is intentional.
        # The tensor API is already fast enough that async prefetch isn't needed.
        return self.teacher.get_top_k_logits_tensors(input_ids=input_ids, topk=topk)

    def __del__(self):
        """Destructor to ensure threads are cleaned up."""
        if not self._shutdown:
            logger.warning("AsyncTeacherClient deleted without calling close()")
            self.close()
