"""
GPU prefetching utilities for overlapping data transfer with compute.

This module provides PrefetchDataLoader, which wraps a PyTorch DataLoader
to prefetch the next batch to GPU while the current batch is being processed.
This overlaps CPU-to-GPU data transfer with GPU compute for better throughput.

Adapted from 00Evolving-Titan project.

Tasks implemented: Data pipeline integration
"""

import torch
from typing import Iterator, Dict, Any


class PrefetchDataLoader:
    """
    Wraps a DataLoader to prefetch next batch to GPU while current batch is processed.

    This overlaps CPU→GPU data transfer with GPU compute, improving throughput.

    Usage:
        >>> base_loader = DataLoader(dataset, batch_size=32)
        >>> prefetch_loader = PrefetchDataLoader(base_loader, device='cuda')
        >>> for batch in prefetch_loader:
        >>>     # batch is already on GPU
        >>>     loss = model(**batch)
    """

    def __init__(
        self,
        dataloader: torch.utils.data.DataLoader,
        device: str = 'cuda'
    ):
        """
        Initialize the prefetch dataloader.

        Args:
            dataloader: Base PyTorch DataLoader to wrap
            device: Target device for data transfer ('cuda' or 'cpu')
        """
        self.dataloader = dataloader
        self.device = torch.device(device)

        # Create CUDA stream for async transfer only if using CUDA
        if self.device.type == 'cuda':
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterate over batches with prefetching.

        Implements double-buffering pattern:
        - Prefetch next batch asynchronously while current batch is being processed
        - Synchronize before yielding to ensure data is ready

        Yields:
            Dict[str, torch.Tensor]: Batch dictionary with all tensors on target device
        """
        base_iter = iter(self.dataloader)

        # Prefetch first batch
        try:
            next_batch = next(base_iter)
        except StopIteration:
            return

        if self.stream is not None:
            # Use CUDA stream for async prefetch
            with torch.cuda.stream(self.stream):
                next_batch = self._to_device(next_batch, non_blocking=True)
        else:
            # CPU device - no async transfer
            next_batch = self._to_device(next_batch, non_blocking=False)

        # Iterate through remaining batches
        for batch in base_iter:
            # Wait for previous prefetch to complete
            if self.stream is not None:
                torch.cuda.current_stream().wait_stream(self.stream)

            # Current batch is ready - yield it
            current_batch = next_batch

            # Start prefetching next batch asynchronously
            if self.stream is not None:
                with torch.cuda.stream(self.stream):
                    next_batch = self._to_device(batch, non_blocking=True)
            else:
                next_batch = self._to_device(batch, non_blocking=False)

            yield current_batch

        # Yield the last prefetched batch
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        yield next_batch

    def _to_device(
        self,
        batch: Dict[str, torch.Tensor],
        non_blocking: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Move all tensors in batch dictionary to target device.

        Args:
            batch: Dictionary containing tensors or nested structures
            non_blocking: If True, async transfer (requires pinned memory)

        Returns:
            Dict[str, torch.Tensor]: Batch with all tensors on target device
        """
        if isinstance(batch, dict):
            return {
                key: self._to_device(value, non_blocking=non_blocking)
                for key, value in batch.items()
            }
        elif isinstance(batch, (list, tuple)):
            return type(batch)(
                self._to_device(item, non_blocking=non_blocking)
                for item in batch
            )
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=non_blocking)
        else:
            # Non-tensor data (e.g., strings, ints) - return as-is
            return batch

    def __len__(self) -> int:
        """
        Return the length of the wrapped dataloader.

        Returns:
            int: Number of batches in the dataloader
        """
        return len(self.dataloader)
