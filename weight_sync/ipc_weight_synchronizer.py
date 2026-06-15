"""IPC (ZMQ) weight synchronizer for colocated vLLM generation.

Transfers weights between a colocated policy and vLLM generation backend via
CUDA IPC handles over ZMQ sockets. Primary transport for colocated vLLM.

Lifecycle per sync:
  1. policy.offload_before_refit()                        -- free GPU for staging
  2. generation.prepare_for_generation(tags=["weights"])  -- allocate buffers
  3. policy.stream_weights_via_ipc_zmq()                  -- send via ZMQ
     generation.update_weights_via_ipc_zmq()              -- receive
  4. policy.offload_after_refit()                         -- restore optimizer state
  5. generation.prepare_for_generation(tags=["kv_cache"]) -- rebuild KV cache
"""

import os
from contextlib import nullcontext
from typing import Any, Optional

import ray

from dockyard_rl.utils.timer import Timer
from dockyard_rl.weight_sync.interfaces import WeightSynchronizer


class IPCWeightSynchronizer(WeightSynchronizer):
    """Weight synchronizer using ZMQ IPC for colocated vLLM deployments.

    Policy and generation share GPUs; weights transfer via CUDA IPC handles
    over ZMQ, avoiding network overhead.

    Args:
        policy: Policy object (ColocatablePolicyInterface).
        generation: VllmGeneration instance (GenerationInterface).
        refit_buffer_size_gb: Fixed staging-buffer size in GB. If None, sized
            dynamically from free GPU memory and
            DOCKYARD_REFIT_BUFFER_MEMORY_RATIO.
    """

    def __init__(
        self,
        policy: Any,
        generation: Any,
        refit_buffer_size_gb: Optional[int] = None,
    ):
        self._policy = policy
        self._generation = generation
        self._refit_buffer_size_gb = refit_buffer_size_gb
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
                buffer_size_bytes = self._compute_buffer_size()

                futures_train = self._policy.stream_weights_via_ipc_zmq(
                    buffer_size_bytes=buffer_size_bytes
                )
                futures_inference = self._generation.update_weights_via_ipc_zmq()

                ray.get(futures_train)
                results = ray.get(futures_inference)
                update_success = all(r for r in results if r is not None)

                if not update_success:
                    raise RuntimeError(
                        "Weight transfer failed during IPC/ZMQ sync. This "
                        "usually indicates an issue with cuda-ipc or the vLLM "
                        "worker."
                    )
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

    def _compute_buffer_size(self) -> int:
        if self._refit_buffer_size_gb is not None:
            if self._refit_buffer_size_gb <= 0:
                raise ValueError("refit_buffer_size_gb must be > 0")
            return self._refit_buffer_size_gb * (1024**3)

        memory_ratio_raw = os.getenv("DOCKYARD_REFIT_BUFFER_MEMORY_RATIO", "0.3")
        try:
            memory_ratio = float(memory_ratio_raw)
        except ValueError as exc:
            raise ValueError(
                "DOCKYARD_REFIT_BUFFER_MEMORY_RATIO must be a valid float, got "
                f"{memory_ratio_raw!r}"
            ) from exc
        if memory_ratio <= 0:
            raise ValueError(
                f"DOCKYARD_REFIT_BUFFER_MEMORY_RATIO must be > 0, got {memory_ratio}"
            )
        return int(self._policy.get_free_memory_bytes() * memory_ratio)
