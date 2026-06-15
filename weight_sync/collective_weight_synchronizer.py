"""NCCL collective weight synchronizer for non-colocated deployments.

Transfers weights between policy and generation workers running on separate
GPU clusters via NCCL collectives. The policy broadcasts; generation workers
receive over the established process group.

Lifecycle per sync:
  1. policy.broadcast_weights_for_collective()    -- send via NCCL
     generation.update_weights_from_collective()  -- receive via NCCL
  2. Verify transfer success

No offload/restore is needed: policy and generation run on separate GPUs with
dedicated memory.
"""

from contextlib import nullcontext
from typing import Any, Optional

import ray

from dockyard_rl.utils.timer import Timer
from dockyard_rl.weight_sync.interfaces import WeightSynchronizer


class CollectiveWeightSynchronizer(WeightSynchronizer):
    """Weight synchronizer using NCCL collectives for non-colocated fleets.

    Policy and generation run on separate GPU clusters; weights synchronize
    via NCCL broadcast over a pre-established process group.

    Args:
        policy: Policy object (ColocatablePolicyInterface).
        generation: Generation object (GenerationInterface).
        train_cluster: RayVirtualCluster for the trainer fleet, used for the
            collective master address/port and the trainer world size.
        inference_world_size: World size of the inference fleet. Provided
            explicitly (computed as inference_nodes * inference_gpus_per_node)
            to match the trainer/inference rendezvous layout exactly.
    """

    def __init__(
        self,
        policy: Any,
        generation: Any,
        train_cluster: Any,
        inference_world_size: int,
    ):
        self._policy = policy
        self._generation = generation
        self._train_cluster = train_cluster
        self._inference_world_size = inference_world_size
        self._stale = True

    def sync_weights(
        self,
        *,
        timer: Optional[Timer] = None,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> None:
        timer_context = (
            timer.time("prepare_for_generation/transfer_and_update_weights")
            if timer is not None
            else nullcontext()
        )
        with timer_context:
            futures_train = self._policy.broadcast_weights_for_collective(
                kv_scales=kv_scales
            )
            futures_inference = self._generation.update_weights_from_collective()

            ray.get(futures_train)
            results = ray.get(futures_inference)
            update_success = all(r for r in results if r is not None)

            if not update_success:
                raise RuntimeError(
                    "Weight transfer failed during NCCL collective sync. This "
                    "usually indicates an issue with the NCCL process group or "
                    "the generation backend worker."
                )

        self._stale = False

    @property
    def is_stale(self) -> bool:
        return self._stale

    def mark_stale(self) -> None:
        self._stale = True

    def init_communicator(self) -> None:
        # Order (init_collective then prepare_refit_info) matches the validated
        # dockyard setup() sequence. The two are independent.
        ip, port = self._train_cluster.get_master_address_and_port()
        print(
            f"Using ip: {ip}, port: {port} for collective communication",
            flush=True,
        )
        train_world_size = self._train_cluster.world_size()
        world_size = train_world_size + self._inference_world_size

        futures_train = self._policy.init_collective(
            ip, port, world_size, train_world_size=train_world_size
        )
        futures_inference = self._generation.init_collective(
            ip, port, world_size, train_world_size=train_world_size
        )
        ray.get(futures_train + futures_inference)

        state_dict_info = self._policy.prepare_refit_info()
        if state_dict_info is not None:
            self._generation.prepare_refit_info(state_dict_info)

    def shutdown(self) -> None:
        # The NCCL process group lifecycle is owned by Ray actor teardown: the
        # workers holding the group are destroyed when the cluster shuts down.
        pass
