"""HTTP weight synchronizer for colocated SGLang generation.

Transfers weights between a colocated policy and SGLang generation backend by
streaming over SGLang's HTTP weight-update API.

Lifecycle per sync:
  1. policy.offload_before_refit()                        -- free GPU for staging
  2. generation.prepare_for_generation(tags=["weights"])  -- allocate buffers
  3. generation.invalidate_kv_cache()                     -- clear stale KV cache
  4. policy.stream_weights_via_http()                     -- push via HTTP
  5. policy.offload_after_refit()                         -- restore optimizer state
  6. generation.prepare_for_generation(tags=["kv_cache"]) -- rebuild KV cache
"""

from contextlib import nullcontext
from typing import Any, Optional

import ray

from dockyard_rl.utils.timer import Timer
from dockyard_rl.weight_sync.interfaces import WeightSynchronizer


class HTTPWeightSynchronizer(WeightSynchronizer):
    """Weight synchronizer using HTTP for colocated SGLang deployments.

    Policy and generation share GPUs; weights stream to SGLang servers via
    their HTTP weight-update API.

    Args:
        policy: Policy object (ColocatablePolicyInterface).
        generation: SGLangGeneration instance exposing
            get_sglang_url_to_gpu_uuids().
    """

    def __init__(self, policy: Any, generation: Any):
        self._policy = policy
        self._generation = generation
        self._stale = True

    def sync_weights(
        self,
        *,
        timer: Optional[Timer] = None,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> None:
        self._policy.offload_before_refit()
        self._generation.prepare_for_generation(tags=["weights"])

        sync_succeeded = False
        try:
            timer_context = (
                timer.time("prepare_for_generation/transfer_and_update_weights")
                if timer is not None
                else nullcontext()
            )
            with timer_context:
                sglang_url_to_gpu_uuids = (
                    self._generation.get_sglang_url_to_gpu_uuids()
                )

                flush_success = self._generation.invalidate_kv_cache()
                if not flush_success:
                    print(
                        "SGLang KV cache invalidation failed before weight update.",
                        flush=True,
                    )

                futures_train = self._policy.stream_weights_via_http(
                    sglang_url_to_gpu_uuids=sglang_url_to_gpu_uuids,
                )
                ray.get(futures_train)
            sync_succeeded = True
        finally:
            self._policy.offload_after_refit()
            self._generation.prepare_for_generation(tags=["kv_cache"])

        self._stale = not sync_succeeded

    @property
    def is_stale(self) -> bool:
        return self._stale

    def mark_stale(self) -> None:
        self._stale = True

    def init_communicator(self) -> None:
        state_dict_info = self._policy.prepare_refit_info()
        if state_dict_info is not None:
            self._generation.prepare_refit_info(state_dict_info)

    def shutdown(self) -> None:
        pass
