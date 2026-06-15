"""Packed tensor broadcast utilities for NCCL weight synchronisation.

Packs multiple small parameter tensors into a single large uint8 buffer
before broadcasting, reducing NCCL call overhead at the cost of a temporary
GPU allocation.  The target buffer size and number of double-buffers are
tunable via environment variables.

This is the critical hot path for trainer→inference weight sync latency.
The double-buffer design (num_buffers=2 by default) overlaps NCCL broadcast
of buffer N with CPU-side packing of buffer N+1.

Environment variables
---------------------
DOCKYARD_REFIT_BUFFER_MEMORY_RATIO
    Fraction of total GPU memory to use as the pack-buffer target size.
    Default: 0.02 (2%).  Hard cap: 5 GB.

DOCKYARD_REFIT_NUM_BUFFERS
    Number of double-buffers.  Default: 2.  Increase to 3 for very large
    models where packing overhead dominates.
"""

import math
import os
from functools import lru_cache
from typing import Any, Callable, Iterator, List, Tuple
import torch

@lru_cache(maxsize=1)
def get_target_packed_tensor_size() -> int:
    """Return the target packed buffer size in bytes (cached after first call)."""
    ratio      = float(os.environ.get("DOCKYARD_REFIT_BUFFER_MEMORY_RATIO", "0.02"))
    props      = torch.cuda.get_device_properties(torch.device("cuda"))
    total      = props.total_memory
    # Hard cap at 5 GiB to avoid starving the KV cache on inference nodes.
    return min(int(total * ratio), 5 * 1024 ** 3)

@lru_cache(maxsize=1)
def get_num_buffers() -> int:
    """Return the number of double-buffers (cached after first call)."""
    return int(os.environ.get("DOCKYARD_REFIT_NUM_BUFFERS", "2"))

def packed_broadcast_producer(
    iterator:       Iterator[Tuple[str, torch.Tensor]],
    group:          Any,
    src:            int,
    post_iter_func: Callable[[Any], torch.Tensor],
) -> None:
    """Pack and broadcast model parameters from the trainer (producer side).

    Iterates over (name, tensor) pairs from the trainer's parameter iterator,
    packs them into a contiguous uint8 buffer up to the target size, then
    broadcasts the buffer via NCCL.  Double-buffering overlaps packing of
    the next batch with the NCCL send of the current batch.

    Args:
        iterator:       Iterator yielding (name, tensor) pairs from the
                        trainer's prepare_refit_info() + parameter access.
        group:          StatelessProcessGroup (has a .broadcast() method).
        src:            Source rank in the sync communicator (trainer DP leader).
        post_iter_func: Applied to each (name, tensor) pair before packing.
                        Typically handles dtype conversion or TP un-sharding.
    """
    target_size  = get_target_packed_tensor_size()
    num_buffers  = get_num_buffers()
    streams      = [torch.cuda.Stream() for _ in range(num_buffers)]
    buffer_idx   = 0

    packing_list  = [[] for _ in range(num_buffers)]
    packing_sizes = [0  for _ in range(num_buffers)]
    packed        = [
        torch.empty(0, dtype=torch.uint8, device="cuda")
        for _ in range(num_buffers)
    ]

    while True:
        buffer_idx = (buffer_idx + 1) % num_buffers
        streams[buffer_idx].synchronize()

        with torch.cuda.stream(streams[buffer_idx]):  # type: ignore[arg-type]
            try:
                packing_list[buffer_idx]  = []
                packing_sizes[buffer_idx] = 0

                while True:
                    # Post-process (e.g. TP gather) then flatten to uint8.
                    # contiguous() is required: a TP gather can yield a
                    # non-contiguous tensor, which view(torch.uint8) cannot
                    # reinterpret.
                    tensor = (
                        post_iter_func(next(iterator))
                        .contiguous()
                        .view(torch.uint8)
                        .view(-1)
                    )
                    packing_list[buffer_idx].append(tensor)
                    packing_sizes[buffer_idx] += tensor.numel()
                    if packing_sizes[buffer_idx] > target_size:
                        break

                packed[buffer_idx] = torch.cat(packing_list[buffer_idx], dim=0)
                group.broadcast(packed[buffer_idx], src=src)

            except StopIteration:
                if packing_list[buffer_idx]:
                    packed[buffer_idx] = torch.cat(packing_list[buffer_idx], dim=0)
                    group.broadcast(packed[buffer_idx], src=src)
                break

def packed_broadcast_consumer(
    iterator:         Iterator[Tuple[str, Tuple[Any, Any]]],
    group:            Any,
    src:              int,
    post_unpack_func: Callable[[List[Tuple[str, torch.Tensor]]], None],
) -> None:
    """Receive packed parameter tensors from the trainer (consumer side).

    Mirrors packed_broadcast_producer exactly — must consume the same number
    of NCCL broadcast calls in the same order.  After each broadcast, unpacks
    the buffer into individual tensors and calls post_unpack_func (which loads
    them into the vLLM model).

    Args:
        iterator:         Iterator yielding (name, (shape, dtype)) pairs from
                          state_dict_info.items() — matches the trainer's order.
        group:            StatelessProcessGroup with a .broadcast() method.
        src:              Source rank (same as producer's src).
        post_unpack_func: Called with a list of (name, tensor) pairs after
                          each buffer is unpacked.  Typically _load_weights()
                          on VllmInternalWorkerExtension.
    """
    def _unpack(
        packed_tensor: torch.Tensor,
        meta:          List[Tuple[str, Any, Any, int, int]],
    ) -> List[Tuple[str, torch.Tensor]]:
        """Split a packed uint8 tensor into named tensors using stored metadata."""
        sizes    = [m[4] for m in meta]
        chunks   = packed_tensor.split_with_sizes(sizes)
        return [
            (m[0], chunk.view(m[2]).view(*m[1]))
            for m, chunk in zip(meta, chunks)
        ]

    target_size  = get_target_packed_tensor_size()
    num_buffers  = get_num_buffers()
    streams      = [torch.cuda.Stream() for _ in range(num_buffers)]
    buffer_idx   = 0

    meta_data    = [[] for _ in range(num_buffers)]
    packing_size = [0  for _ in range(num_buffers)]
    offsets      = [0  for _ in range(num_buffers)]
    packed       = [
        torch.empty(0, dtype=torch.uint8, device="cuda")
        for _ in range(num_buffers)
    ]

    while True:
        buffer_idx = (buffer_idx + 1) % num_buffers
        streams[buffer_idx].synchronize()

        with torch.cuda.stream(streams[buffer_idx]):  # type: ignore[arg-type]
            meta_data[buffer_idx]    = []
            packing_size[buffer_idx] = 0
            offsets[buffer_idx]      = 0

            try:
                while True:
                    name, (shape, dtype) = next(iterator)
                    tensor_size = math.prod(shape) * dtype.itemsize
                    meta_data[buffer_idx].append(
                        (name, shape, dtype, offsets[buffer_idx], tensor_size)
                    )
                    packing_size[buffer_idx] += tensor_size
                    offsets[buffer_idx]      += tensor_size
                    if packing_size[buffer_idx] > target_size:
                        break

                packed[buffer_idx] = torch.empty(
                    packing_size[buffer_idx], dtype=torch.uint8, device="cuda"
                )
                group.broadcast(packed[buffer_idx], src=src)
                post_unpack_func(_unpack(packed[buffer_idx], meta_data[buffer_idx]))

            except StopIteration:
                if meta_data[buffer_idx]:
                    packed[buffer_idx] = torch.empty(
                        packing_size[buffer_idx], dtype=torch.uint8, device="cuda"
                    )
                    group.broadcast(packed[buffer_idx], src=src)
                    post_unpack_func(
                        _unpack(packed[buffer_idx], meta_data[buffer_idx])
                    )
                break