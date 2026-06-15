"""Weight synchronization interface for Project Dockyard.

WeightSynchronizer decouples weight-transfer logic from both the policy and
the generation backend. It owns the transfer of model weights from the
trainer fleet into the inference fleet.

Transport-specific implementations (IPC/ZMQ for colocated vLLM, HTTP for
colocated SGLang, NCCL collective for non-colocated fleets) each encapsulate
the full transfer lifecycle, so the GRPO orchestrator never branches on
backend type or topology.

Colocated transports (IPC, HTTP) own GPU phase transitions internally
(offload before refit, prepare_for_generation, offload after refit) as part
of sync_weights(). The NCCL collective transport is a pure data mover: policy
and generation run on separate GPU clusters, so no phase transitions occur.

This interface assumes global weight updates: all generation workers are
updated atomically and always share the same weight version. In async GRPO,
heterogeneous weight ages are tracked at the sample level (the replay buffer's
target_weight_versions), not at the synchronizer level.
"""

from abc import ABC, abstractmethod
from typing import Optional

from dockyard_rl.utils.timer import Timer


class WeightSynchronizer(ABC):
    """Abstract base for weight synchronization between policy and generation.

    Implementations handle weight transfer for one transport mechanism (ZMQ
    IPC, HTTP, NCCL collective). The orchestrator calls sync_weights() /
    mark_stale() / is_stale without knowing the transport or whether the
    fleets are colocated.

    Colocated transports (IPC, HTTP) own phase transitions internally. The
    NCCL collective transport is a pure data mover.
    """

    @abstractmethod
    def sync_weights(
        self,
        *,
        timer: Optional[Timer] = None,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> None:
        """Transfer the latest policy weights into the generation backend.

        Encapsulates the full sync lifecycle:
          1. Prepare the policy side (e.g. offload optimizer state).
          2. Prepare the generation side (e.g. allocate weight buffers).
          3. Transfer weights over the transport.
          4. Verify the transfer succeeded.
          5. Restore both sides to their ready state.

        Steps 1-2 and 5 (phase transitions) run only for colocated transports
        (IPC, HTTP); the NCCL collective transport skips them. On failure,
        implementations raise RuntimeError.

        Args:
            timer: Optional Timer for profiling the transfer phase.
            kv_scales: Optional FP8 KV-cache scales. Honored only by the NCCL
                collective transport, which forwards them to
                policy.broadcast_weights_for_collective(); IPC and HTTP ignore it.

        Raises:
            RuntimeError: If the weight transfer fails.
        """
        ...

    @property
    @abstractmethod
    def is_stale(self) -> bool:
        """Whether the generation backend's weights are out of date.

        True after mark_stale() and before the next successful sync_weights().
        """
        ...

    @abstractmethod
    def mark_stale(self) -> None:
        """Mark the generation weights stale after a training step.

        Call after every training step so the orchestrator knows a sync is
        needed before the next generation phase. Applies globally: all
        generation workers are stale and updated atomically on the next
        sync_weights().
        """
        ...

    @abstractmethod
    def init_communicator(self) -> None:
        """Initialize communication channels needed for weight transfer.

        Called once during setup, after policy and generation workers are
        constructed. Colocated transports prepare refit metadata; the NCCL
        collective transport additionally initializes the process group.
        """
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Release all communication resources."""
        ...
