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

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from dockyard_rl.utils.timer import Timer

logger = logging.getLogger(__name__)


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


def _invalidate_kv_cache_after_refit(
    generation: Any, recompute_kv_cache: bool, transport: str
) -> None:
    """Drop KV blocks computed with the pre-refit weights.

    Shared by the two vLLM transports (collective, IPC). The HTTP/SGLang
    transport invalidates before its transfer instead, and unconditionally,
    because that call is SGLang's *only* flush: its finish_generation
    deliberately preserves the prefix cache between rollout batches to keep
    RadixAttention warm (sglang_generation.py:439).

    vLLM's finish_generation does invalidate on every batch, which covers the
    synchronous training loop -- there, sync_weights runs on an already-empty
    cache and this call is a no-op. It does not cover the asynchronous loop:
    that cycle is prepare_for_refit -> sync_weights -> resume_after_refit
    (grpo.py:2885-2896) with no finish_generation, so generation resumed on a
    cache built from the previous weights. That is the gap this closes, and it
    is why the config key lives in AsyncGRPOConfig.

    Failure is logged, not raised: the weights are already in place, and
    aborting the step would leave the engine mid-refit. A stale cache degrades
    sample quality; a half-applied refit breaks the run.

    Args:
        generation: Generation object exposing invalidate_kv_cache().
        recompute_kv_cache: Whether invalidation is enabled for this run.
        transport: Transport name, for log messages.
    """
    if not recompute_kv_cache:
        logger.warning(
            "Skipping KV cache invalidation after %s weight sync "
            "(recompute_kv_cache_after_weight_updates is false). Generation "
            "may reuse KV blocks computed with the previous weights.",
            transport,
        )
        return

    invalidate = getattr(generation, "invalidate_kv_cache", None)
    if invalidate is None:
        logger.warning(
            "Generation backend exposes no invalidate_kv_cache(); KV blocks "
            "from the previous weights may be reused after %s weight sync.",
            transport,
        )
        return

    if not invalidate():
        logger.warning(
            "KV cache invalidation reported failure after %s weight sync.",
            transport,
        )
