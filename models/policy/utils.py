import gc
import os
import traceback
from enum import Enum
from typing import Any, Dict, Optional, cast

import requests
import torch
import torch.distributed as dist
import zmq
from torch.multiprocessing.reductions import rebuild_cuda_tensor
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForTextToWaveform,
)

# Deferred — dockyard_rl.distributed.worker_group_utils is not yet written.
try:
    from dockyard_rl.distributed.worker_groups_utils import (
        get_nsight_config_if_pattern_matches,
    )
except ImportError:
    def get_nsight_config_if_pattern_matches(worker_name: str) -> dict:  # type: ignore
        return {}

# AutoModel factory for loading HuggingFace models from the correct class.
#
# Add an entry here when a model (1) uses HF's standard loading path and
# (2) its architecture is not loadable via AutoModelForCausalLM (e.g. VLMs
# using ForConditionalGeneration / ForImageTextToText).
AUTOMODEL_FACTORY: Dict[str, Any] = {
    "qwen2_5_vl": AutoModelForImageTextToText,
    "qwen2_vl": AutoModelForImageTextToText,
    "qwen2_5_omni": AutoModelForTextToWaveform,
    "qwen3_5": AutoModelForImageTextToText,
    "llava": AutoModelForImageTextToText,
    "internvl": AutoModelForImageTextToText,
    "gemma3": AutoModelForImageTextToText,
    "smolvlm": AutoModelForImageTextToText,
    "mistral3": AutoModelForImageTextToText,
    "llama4": AutoModelForImageTextToText,
}

class IPCProtocol(Enum):
    """IPC protocol constants for ZMQ weight streaming."""

    COMPLETE = "complete"
    ACK = "ack"

# Maps default policy worker FQNs to their quantized counterparts when
# quant_cfg is set. Core has no knowledge of ModelOpt internals beyond
# this table — the override is resolved lazily in resolve_policy_worker_cls.
#
# TODO: Replace this hard-coded map with a generic plugin-registration hook
# on Policy (e.g. a worker_cls_overrides registry populated by
# dockyard_rl.modelopt on import) so core has no knowledge of ModelOpt-
# specific worker classes.
POLICY_WORKER_OVERRIDES = {
    "dockyard_rl.models.policy.workers.dtensor_policy_worker.DTensorPolicyWorker": "dockyard_rl.modelopt.models.policy.workers.dtensor_quant_policy_worker.DTensorQuantPolicyWorker",
    "dockyard_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2": "dockyard_rl.modelopt.models.policy.workers.dtensor_quant_policy_worker_v2.DTensorQuantPolicyWorkerV2",
}

def resolve_policy_worker_cls(default_cls: str, config: dict) -> str:
    """Return the quantized policy worker FQN if ``quant_cfg`` is set, else ``default_cls``.

    Safe to call even when ModelOpt is not installed — returns ``default_cls``
    unchanged whenever ``quant_cfg`` is ``None``, so the core policy path stays
    import-free of ModelOpt.
    """
    if config.get("quant_cfg") is None:
        return default_cls
    return POLICY_WORKER_OVERRIDES.get(default_cls, default_cls)

def resolve_model_class(model_name: str) -> Any:
    """Resolve the appropriate HuggingFace AutoModel class for a given model name."""
    return AUTOMODEL_FACTORY.get(model_name.lower(), AutoModelForCausalLM)

def is_vllm_v1_engine_enabled() -> bool:
    """Check if vLLM V1 engine is enabled.

    Returns:
        bool: True if V1 engine is enabled, False otherwise (defaults to True if not set)
    """
    return os.environ.get("DOCKYARD_VLLM_USE_V1", "1") == "1"

def get_gpu_info(model: torch.nn.Module) -> dict[str, Any]:
    """Return information about the GPU being used by this worker."""
    import torch

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device)
    device_count = torch.cuda.device_count()
    memory_allocated = torch.cuda.memory_allocated(device) / (1024**2)  # in MB
    memory_reserved = torch.cuda.memory_reserved(device) / (1024**2)  # in MB
    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # in MB
    peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)  # in MB

    # In distributed training each process only sees its assigned GPU as device 0.
    local_device_id = device
    global_device_id = local_device_id

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if local_rank < len(cuda_visible_devices):
            global_device_id = int(cuda_visible_devices[local_rank])

    # Grab one trainable parameter to confirm CUDA device placement.
    param_info = {}
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if param is not None and param.requires_grad:
                full_name = f"{module_name}.{param_name}"
                param_info[full_name] = {
                    "device": str(param.device),
                    "shape": list(param.shape),
                    "dtype": str(param.dtype),
                }
                break
        if param_info:
            break

    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "local_device_id": local_device_id,
        "global_device_id": global_device_id,
        "device_count": device_count,
        "device_name": device_name,
        "memory_allocated_mb": memory_allocated,
        "memory_reserved_mb": memory_reserved,
        "peak_memory_allocated_mb": peak_memory,
        "peak_memory_reserved_mb": peak_reserved,
        "parameter_sample": param_info,
        "env_vars": {
            k: v
            for k, v in os.environ.items()
            if k.startswith("CUDA") or k in ["LOCAL_RANK", "RANK", "WORLD_SIZE"]
        },
    }

def configure_dynamo_cache() -> None:
    """Disable dynamo autotune_local_cache.

    Dynamo may fail at cached_autotune when there's already a cache with different order of node_bundles.
    Disable autotune_local_cache as a workaround.
    See https://github.com/pytorch/pytorch/issues/153791 for more details.
    """
    torch._inductor.config.autotune_local_cache = False

def get_runtime_env_for_policy_worker(policy_worker_name: str) -> dict[str, Any]:
    """Get runtime environment configuration for policy workers.

    Note: expandable_segments configuration is handled directly in the worker init methods
    to ensure proper GPU detection after CUDA initialization.
    """
    runtime_env = {
        **get_nsight_config_if_pattern_matches(policy_worker_name),
    }
    return runtime_env

def get_handle_from_tensor(tensor: torch.Tensor) -> tuple[Any]:
    """Get IPC handle from a tensor."""
    from torch.multiprocessing.reductions import reduce_tensor

    # skip serializing the function for better refit performance
    return reduce_tensor(tensor.detach())[1:]

def calculate_aligned_size(size_bytes: int, alignment: int = 512) -> int:
    """Calculate aligned size for memory alignment.

    Args:
        size_bytes(int): Size in bytes to align
        alignment(int): Alignment boundary in bytes (default 512)

    Returns:
        Aligned size in bytes(int).
    """
    return int(((size_bytes + alignment - 1) // alignment) * alignment)

def stream_weights_via_ipc_zmq_impl(
    params_generator, buffer_size_bytes: int, zmq_socket, rank: int, worker_name: str
) -> None:
    """Shared implementation for streaming weights via IPC ZMQ with improved memory management.

    Uses ping-pong double buffering to enable overlapping communication while reusing buffers
    to reduce memory allocation overhead and improve stability.

    Args:
        params_generator: Generator yielding (name, tensor) pairs
        buffer_size_bytes: total size of buffer in bytes for batching parameters
        zmq_socket: ZMQ socket for communication
        rank: Worker rank for logging
        worker_name: Name of the worker for logging
    """
    # Divide total buffer size by 2 because we use two individual buffers (ping-pong) for overlapping communication.
    buffer_size_bytes = buffer_size_bytes // 2

    def send_buffer_group_overlap(buffer, param_names, used_bytes, await_recv) -> bool:
        """Send a group of parameters and return new pending_recv state."""
        torch.cuda.current_stream().synchronize()
        cuda_ipc_handle = get_handle_from_tensor(buffer)

        if await_recv:
            zmq_socket.recv()

        payload = (cuda_ipc_handle, param_names, used_bytes)
        zmq_socket.send_pyobj(payload)
        return True  # pending_recv = True

    def allocate_buffer(device):
        """Allocate a new aligned buffer with proper memory alignment."""
        aligned_size = calculate_aligned_size(buffer_size_bytes)
        return torch.empty(
            aligned_size,
            device=device,
            dtype=torch.uint8,
            requires_grad=False,
        )

    def pack_tensor(buffer, tensor, used_bytes) -> int:
        """Pack tensor into buffer and return new used_bytes."""
        tensor_bytes = tensor.nbytes
        buffer[used_bytes : used_bytes + tensor_bytes].data.copy_(
            tensor.data.view(-1).view(dtype=torch.uint8), non_blocking=True
        )
        return used_bytes + calculate_aligned_size(tensor_bytes)

    # Initialize ping-pong double buffering
    buffer_a: torch.Tensor | None = None
    buffer_b: torch.Tensor | None = None
    current_buffer: torch.Tensor | None = None

    used_bytes = 0
    param_names = []
    await_recv = False
    count_of_groups = 0

    try:
        for name, tensor in params_generator:
            if buffer_a is None:
                buffer_a = allocate_buffer(tensor.device)
                buffer_b = allocate_buffer(tensor.device)
                current_buffer = buffer_a

            aligned_size = calculate_aligned_size(tensor.nbytes)
            assert aligned_size <= buffer_size_bytes, (
                f"Parameter {name} too large for buffer: {aligned_size} > {buffer_size_bytes}"
            )

            if used_bytes + aligned_size > buffer_size_bytes:
                await_recv = send_buffer_group_overlap(
                    current_buffer, param_names, used_bytes, await_recv
                )
                count_of_groups += 1
                current_buffer = buffer_b if current_buffer is buffer_a else buffer_a
                used_bytes, param_names = 0, []

            param_names.append(name)
            used_bytes = pack_tensor(current_buffer, tensor, used_bytes)

        if param_names:
            await_recv = send_buffer_group_overlap(
                current_buffer, param_names, used_bytes, await_recv
            )
            count_of_groups += 1

        if await_recv:
            zmq_socket.recv()

        torch.cuda.current_stream().synchronize()
        zmq_socket.send_pyobj(IPCProtocol.COMPLETE)
        zmq_socket.recv()

        if rank == 0:
            print(
                f"{worker_name}: Packed {count_of_groups} groups of tensors", flush=True
            )

    except zmq.Again:
        timeout_ms = zmq_socket.getsockopt(zmq.RCVTIMEO)
        raise TimeoutError(
            f"{worker_name} (rank {rank}): ZMQ communication timeout after {timeout_ms}ms in policy worker side. "
            f"The generation worker may be dead or unresponsive. "
            f"This typically indicates the generation worker has crashed or is not responding to weight streaming."
        ) from None
    except zmq.ZMQError as e:
        raise RuntimeError(
            f"{worker_name} (rank {rank}): ZMQ error during weight streaming: {e} (errno: {e.errno}). "
            f"Error details: {e.strerror}. "
            f"This may indicate network issues or the peer process has terminated unexpectedly.\n"
            f"{traceback.format_exc()}"
        ) from e

    finally:
        if buffer_a is not None:
            del buffer_a
        if buffer_b is not None:
            del buffer_b
        gc.collect()
        torch.cuda.empty_cache()

def rebuild_cuda_tensor_from_ipc(
    cuda_ipc_handle: tuple, device_id: int
) -> torch.Tensor:
    """Rebuild a CUDA tensor from an IPC handle."""
    func = rebuild_cuda_tensor
    args = cuda_ipc_handle[0]
    list_args = list(args)
    list_args[6] = device_id
    return func(*list_args)

def stream_weights_via_http_impl(
    params_generator,
    sglang_url_to_gpu_uuids: dict[str, list[str]],
    rank: int,
    worker_name: str,
    current_device_uuid: str,
) -> None:
    """Stream weights to SGLang servers via HTTP API (update_weights_from_tensor).

    Flow: Each rank creates IPC handler → gather handlers in rank order → send list → SGLang matches by tp_rank index

    Key points:
    - Each rank creates handler on its own GPU
    - Handlers are gathered in rank order: [rank0_handler, rank1_handler, ...]
    - List index = rank = GPU ID
    - SGLang automatically matches: handler = serialized_handlers[tp_rank]

    Args:
        params_generator: Generator yielding (name, tensor) pairs
        sglang_url_to_gpu_uuids: Dict mapping SGLang server URL to list of GPU UUIDs it uses
        rank: Worker rank for logging
        worker_name: Name of the worker for logging
        current_device_uuid: UUID of the current training worker's GPU
    """
    from dockyard_rl.models.generation.sglang.sglang_copied_utils import (
        MultiprocessingSerializer,
    )

    print("[sglang refit details] entering stream_weights_via_http_impl")

    target_urls = [
        url
        for url, uuids in sglang_url_to_gpu_uuids.items()
        if current_device_uuid in uuids
    ]

    if not target_urls:
        raise RuntimeError(
            f"{worker_name} (rank {rank}): No matching SGLang server found for GPU UUID {current_device_uuid}. "
            f"Available servers: {list(sglang_url_to_gpu_uuids.keys())}"
        )

    if len(target_urls) > 1:
        print(
            f"[WARNING] {worker_name} (rank {rank}): GPU UUID {current_device_uuid} matches multiple SGLang servers: {target_urls}. "
            f"Using the first one: {target_urls[0]}"
        )
        target_urls = [target_urls[0]]

    base_url = target_urls[0]
    url = f"{base_url}/update_weights_from_tensor"
    sglang_gpu_uuids = sglang_url_to_gpu_uuids[base_url]

    ipc_gather_group, ipc_gather_src, matching_ranks = _setup_ipc_gather_group(
        rank, current_device_uuid, sglang_gpu_uuids, sglang_url_to_gpu_uuids
    )
    print(
        f"[sglang refit] {worker_name} (rank {rank}): ipc_gather_group={ipc_gather_group}, ipc_gather_src={ipc_gather_src}, matching_ranks={matching_ranks}"
    )
    tensor_count = 0

    try:
        tensor_list = list(params_generator)
        total_tensors = len(tensor_list)

        if rank == ipc_gather_src:
            print(
                f"[sglang refit details] {worker_name}: Starting weight update - "
                f"Total parameters to update: {total_tensors}",
                flush=True,
            )

        for idx, (name, tensor) in enumerate(tensor_list):
            torch.cuda.current_stream().synchronize()
            tensor = tensor.contiguous().cuda()

            named_tensors = [(name, tensor)]
            serialized_handler = MultiprocessingSerializer.serialize(
                named_tensors, output_str=True
            )
            serialized_handler_str = cast(str, serialized_handler)

            gathered_handlers = _gather_ipc_handlers(
                serialized_handler_str,
                ipc_gather_group,
                ipc_gather_src,
                rank,
                matching_ranks,
            )

            if rank == ipc_gather_src and gathered_handlers is not None:
                _send_tensor_to_sglang(
                    url,
                    name,
                    gathered_handlers,
                    tensor.shape,
                    str(tensor.dtype),
                    flush_cache=False,
                )
                tensor_count += 1

            del tensor, serialized_handler
            if rank == ipc_gather_src:
                del gathered_handlers
            torch.cuda.empty_cache()

        if rank == ipc_gather_src:
            print(
                f"[sglang refit details] {worker_name}: Weight update completed - "
                f"Successfully updated {tensor_count}/{total_tensors} parameters to SGLang server: {base_url}",
                flush=True,
            )
            if tensor_count != total_tensors:
                print(
                    f"[sglang refit details] {worker_name}: WARNING - Expected {total_tensors} tensors, "
                    f"but only sent {tensor_count}",
                    flush=True,
                )

    except Exception as e:
        print(
            f"{worker_name} (rank {rank}): Error during HTTP weight streaming: {e}.\n"
            f"{traceback.format_exc()}"
        )
        raise

    finally:
        gc.collect()
        torch.cuda.empty_cache()

def _setup_ipc_gather_group(
    rank: int,
    current_device_uuid: str,
    sglang_gpu_uuids: list[str],
    sglang_url_to_gpu_uuids: dict[str, list[str]],
) -> tuple[Optional[dist.ProcessGroup], Optional[int], Optional[list[int]]]:
    """Setup gather configuration for IPC handlers.

    Returns:
        Tuple of (gather_group, gather_src_rank, matching_ranks)
        - gather_group: None (use default FSDP group)
        - gather_src_rank: The rank that will collect and send to the SGLang server
        - matching_ranks: List of ranks that belong to the same SGLang server
    """
    if not dist.is_initialized():
        return None, None, None

    world_size = dist.get_world_size()

    all_ranks_uuids = [None] * world_size
    dist.all_gather_object(all_ranks_uuids, current_device_uuid)

    matching_ranks = [
        r for r, uuid in enumerate(all_ranks_uuids) if uuid in sglang_gpu_uuids
    ]

    if len(matching_ranks) == 0:
        return None, None, None

    matching_ranks = sorted(matching_ranks)
    gather_src = matching_ranks[0]

    return None, gather_src, matching_ranks

def _gather_ipc_handlers(
    serialized_handler: str,
    gather_group: Optional[dist.ProcessGroup],
    gather_src: Optional[int],
    rank: int,
    matching_ranks: Optional[list[int]] = None,
) -> Optional[list[str]]:
    """Gather IPC handlers from all ranks in the default FSDP group, then filter by server.

    Args:
        serialized_handler: Serialized IPC handler from this rank
        gather_group: Process group (None means use default FSDP group)
        gather_src: Rank that will collect and filter handlers
        rank: Current rank
        matching_ranks: List of ranks that belong to the same SGLang server

    Returns:
        List of serialized handlers in rank order (only on gather_src rank), None otherwise.
        The list contains handlers from matching_ranks only, in rank order.
    """
    if gather_src is None:
        return None

    if not dist.is_initialized():
        return None

    world_size = dist.get_world_size()

    all_handlers: list[Optional[str]] = [None for _ in range(world_size)]
    dist.all_gather_object(all_handlers, serialized_handler)
    all_handlers_str = cast(list[str], all_handlers)

    if rank == gather_src and matching_ranks is not None:
        filtered_handlers: list[str] = [all_handlers_str[r] for r in matching_ranks]
        return filtered_handlers
    else:
        return None

def _send_tensor_to_sglang(
    url: str,
    tensor_name: str,
    gathered_handlers: list[str],
    shape: torch.Size,
    dtype: str,
    flush_cache: bool = False,
) -> None:
    """Send gathered IPC handlers to SGLang server via HTTP.

    Key: gathered_handlers are in rank order [rank0, rank1, ...].
    SGLang will automatically match: handler = serialized_handlers[tp_rank].

    Args:
        url: SGLang server URL
        tensor_name: Name of the tensor
        gathered_handlers: List of serialized IPC handlers in rank order
        shape: Tensor shape
        dtype: Tensor dtype
        flush_cache: Whether to flush cache after this tensor (for last tensor)
    """
    payload = {
        "serialized_named_tensors": gathered_handlers,
        "flush_cache": flush_cache,
    }

    response = None
    try:
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120,
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        error_msg = f"Failed to send tensor '{tensor_name}' to {url}: {e}"
        try:
            if response is not None:
                error_detail = response.text
                error_msg += f"\nResponse status: {response.status_code}"
                error_msg += f"\nResponse body: {error_detail[:500]}"
        except:
            pass
        print(f"[sglang refit] {error_msg}", flush=True)
        raise RuntimeError(error_msg) from e
    except Exception as e:
        raise RuntimeError(
            f"Failed to send tensor '{tensor_name}' to {url}: {e}"
        ) from e