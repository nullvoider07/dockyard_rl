"""Factory for WeightSynchronizer instances.

Selects the transport from the deployment topology (colocated vs.
non-colocated) and the generation backend: SGLang uses HTTP when colocated,
and any non-colocated deployment uses NCCL collectives.

Colocated vLLM would use IPC/ZMQ, but that transport is not wired end to end
and this factory refuses the combination rather than handing back a
synchronizer that raises at first use. See the NotImplementedError below for
what is missing.
"""

from typing import Any, Optional

from dockyard_rl.models.generation.constants import (
    SGLANG_BACKEND,
    VLLM_BACKEND,
)
from dockyard_rl.weight_sync.interfaces import WeightSynchronizer


def create_weight_synchronizer(
    policy: Any,
    generation: Any,
    generation_backend: str,
    colocated: bool,
    *,
    train_cluster: Optional[Any] = None,
    inference_world_size: Optional[int] = None,
    refit_buffer_size_gb: Optional[int] = None,
) -> WeightSynchronizer:
    """Create the WeightSynchronizer for the given deployment.

    Args:
        policy: Policy object (ColocatablePolicyInterface).
        generation: Generation object (GenerationInterface).
        generation_backend: Backend name ("vllm" or "sglang").
        colocated: Whether policy and generation share GPUs.
        train_cluster: Trainer RayVirtualCluster (required for non-colocated).
        inference_world_size: Inference-fleet world size (required for
            non-colocated; computed as inference_nodes * inference_gpus_per_node).
        refit_buffer_size_gb: Optional fixed IPC staging buffer size in GB.

    Returns:
        A WeightSynchronizer for the deployment topology.

    Raises:
        ValueError: For an unknown backend or missing non-colocated arguments.
        NotImplementedError: For an unsupported backend/topology combination —
            non-colocated SGLang, and colocated vLLM (the IPC/ZMQ transport is
            unimplemented; the message names the three missing pieces).
    """
    supported_backends = {VLLM_BACKEND, SGLANG_BACKEND}
    if generation_backend not in supported_backends:
        raise ValueError(
            f"Unknown generation backend {generation_backend!r}. "
            f"Supported backends: {sorted(supported_backends)}"
        )

    if not colocated:
        if generation_backend == SGLANG_BACKEND:
            raise NotImplementedError(
                "SGLang does not support non-colocated inference mode."
            )
        if train_cluster is None or inference_world_size is None:
            raise ValueError(
                "train_cluster and inference_world_size are required for "
                "non-colocated weight synchronization."
            )

        from dockyard_rl.weight_sync.collective_weight_synchronizer import (
            CollectiveWeightSynchronizer,
        )

        return CollectiveWeightSynchronizer(
            policy=policy,
            generation=generation,
            train_cluster=train_cluster,
            inference_world_size=inference_world_size,
        )

    if generation_backend == SGLANG_BACKEND:
        from dockyard_rl.weight_sync.http_weight_synchronizer import (
            HTTPWeightSynchronizer,
        )

        return HTTPWeightSynchronizer(policy=policy, generation=generation)

    if refit_buffer_size_gb is not None and refit_buffer_size_gb <= 0:
        raise ValueError("refit_buffer_size_gb must be > 0")

    # Colocated vLLM selects the IPC/ZMQ transport, which is not wired end to
    # end. Refuse here rather than return a synchronizer that raises on first use:
    # the failure would otherwise surface at the first sync_weights() call, one
    # full rollout after cluster bring-up, model load and engine init, as an
    # AttributeError naming none of the actual causes.
    raise NotImplementedError(
        "Colocated vLLM weight sync (IPC/ZMQ) is not implemented. The receiver "
        "is complete: models/generation/vllm/vllm_backend.py unpacks the "
        "payload that models/policy/utils.py:stream_weights_via_ipc_zmq_impl "
        "already emits, over the same IPCProtocol. Two sender-side pieces are "
        "missing:\n"
        "  1. The policy worker has no ZMQ plumbing. The receiver connects a "
        "REP socket to ipc:///tmp/{device_uuid}.sock (VllmBackend."
        "get_zmq_address), so the sender must bind REQ to the socket for the "
        "device it is colocated with -- which needs colocation discovery. "
        "models/policy/utils.py:_setup_ipc_gather_group does that shape of "
        "UUID matching for the SGLang HTTP path and is the reference, not a "
        "drop-in.\n"
        "  2. DTensorPolicyWorkerImpl.stream_weights_via_ipc_zmq raises "
        "NotImplementedError instead of calling the impl above, and Policy "
        "exposes no dispatch for it; Policy.broadcast_weights_for_collective "
        "is the pattern to copy.\n"
        "Set policy.generation.colocated.enabled=false to use the NCCL "
        "collective transport, which is implemented and tested."
    )
