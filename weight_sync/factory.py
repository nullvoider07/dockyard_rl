"""Factory for WeightSynchronizer instances.

Selects the transport from the deployment topology (colocated vs.
non-colocated) and the generation backend: vLLM uses IPC/ZMQ when colocated,
SGLang uses HTTP when colocated, and any non-colocated deployment uses NCCL
collectives.
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
        NotImplementedError: For an unsupported backend/topology combination.
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
    from dockyard_rl.weight_sync.ipc_weight_synchronizer import (
        IPCWeightSynchronizer,
    )

    return IPCWeightSynchronizer(
        policy=policy,
        generation=generation,
        refit_buffer_size_gb=refit_buffer_size_gb,
    )
