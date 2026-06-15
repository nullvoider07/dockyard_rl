"""Policy interfaces for Project Dockyard.

Defines the TypedDicts and abstract base classes that all trainer-side
policy workers must implement.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Iterator, Optional, TypedDict
import ray
import torch
from dockyard_rl.distributed.batched_data_dict import BatchedDataDict
from dockyard_rl.models.generation.interfaces import GenerationDatumSpec

# Timing context manager
class Timer:
    """Lightweight wall-clock timer for profiling policy worker steps."""

    def __init__(self) -> None:
        self._times: dict[str, float] = {}
        self._starts: dict[str, float] = {}

    @contextmanager
    def time(self, key: str) -> Iterator[None]:
        self._starts[key] = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - self._starts.pop(key, 0.0)
            self._times[key] = self._times.get(key, 0.0) + elapsed

    def get(self, key: str) -> float:
        return self._times.get(key, 0.0)

    def reset(self) -> None:
        self._times.clear()
        self._starts.clear()

    def as_dict(self) -> dict[str, float]:
        return dict(self._times)

# Output TypedDicts for policy forward methods
class LogprobOutputSpec(TypedDict):
    """Per-token log-probabilities from the active policy."""

    logprobs: torch.Tensor   # (batch, seq_len)

class ReferenceLogprobOutputSpec(TypedDict):
    """Per-token log-probabilities from the frozen reference policy."""

    reference_logprobs: torch.Tensor   # (batch, seq_len)

class ScoreOutputSpec(TypedDict):
    """Scalar reward / value scores from a critic or reward model."""

    scores: torch.Tensor   # (batch,)

class TopkLogitsOutputSpec(TypedDict):
    """Per-position top-k logits and global token indices.

    Shape: (batch, seq_len - 1, k) for both tensors.
    The seq_len - 1 reflects next-token prediction alignment: position i
    holds the logit distribution used to predict token i+1.
    """

    topk_logits:  torch.Tensor   # (batch, seq_len-1, k)
    topk_indices: torch.Tensor   # (batch, seq_len-1, k)

# Trainer-side policy configuration. Resolved from YAML/Hydra at runtime (an
# OmegaConf DictConfig) and indexed with arbitrary, sometimes nested keys across
# backends, so it is typed as a permissive str->Any mapping rather than a closed
# TypedDict. This matches how every call site accesses it and how the config is
# actually constructed; a TypedDict would raise false "unknown/optional key"
# errors on legitimate accesses.
PolicyConfig = dict[str, Any]

# Abstract policy interface
class PolicyInterface(ABC):
    """Abstract base class for all trainer-side policy workers.

    Every concrete policy worker (DTensor and future backends) must
    implement all abstract methods.  The worker_group attribute is attached
    by the policy wrapper after the policy's workers are constructed.
    """

    # Set by the policy wrapper after worker construction.
    worker_group = None

    @abstractmethod
    def get_logprobs(
        self,
        data:             BatchedDataDict[GenerationDatumSpec],
        micro_batch_size: Optional[int] = None,
        timer:            Optional[Timer] = None,
    ) -> BatchedDataDict[LogprobOutputSpec]:
        """Compute per-token log-probabilities for a batch of sequences.

        Convention: logprob of the first token is 0.0; logprob of token i
        is stored at position i.  This keeps the output shape (batch, seq_len)
        consistent with the input without shifting.

        Args:
            data:             BatchedDataDict with input_ids and input_lengths.
            micro_batch_size: Override for micro-batch size during forward.
            timer:            Optional timer for profiling sub-steps.

        Returns:
            BatchedDataDict with "logprobs" tensor of shape (batch, seq_len).
        """

    @abstractmethod
    def get_reference_policy_logprobs(
        self,
        data:             BatchedDataDict[GenerationDatumSpec],
        micro_batch_size: Optional[int] = None,
        timer:            Optional[Timer] = None,
    ) -> BatchedDataDict[ReferenceLogprobOutputSpec]:
        """Compute log-probabilities from the frozen reference policy.

        Switches the model to reference-policy mode (EMA weights or a
        separate frozen copy depending on the backend), runs a forward
        pass, then restores the active policy.

        Returns:
            BatchedDataDict with "reference_logprobs" tensor (batch, seq_len),
            moved to CPU to free GPU memory before the training step.
        """

    @abstractmethod
    def get_topk_logits(
        self,
        data:             BatchedDataDict[GenerationDatumSpec],
        k:                int,
        micro_batch_size: Optional[int] = None,
        timer:            Optional[Timer] = None,
    ) -> BatchedDataDict[TopkLogitsOutputSpec]:
        """Return top-k logits and token indices at each sequence position.

        Used for process reward model (PRM) training and speculative
        decoding calibration.

        Args:
            data:             Input BatchedDataDict.
            k:                Number of top logits to return per position.
            micro_batch_size: Micro-batch size override.
            timer:            Optional timer.

        Returns:
            BatchedDataDict with "topk_logits" and "topk_indices",
            both of shape (batch, seq_len - 1, k).
        """

    @abstractmethod
    def train(
        self,
        data:      BatchedDataDict,
        loss_fn:   Any,            # LossFunction from algorithms/loss/interfaces.py
        eval_mode: bool = False,
        *,
        gbs:   Optional[int] = None,
        mbs:   Optional[int] = None,
        timer: Optional[Timer] = None,
    ) -> dict[str, Any]:
        """Run one training step (forward + backward + optimiser step).

        Args:
            data:      Global batch — will be split into micro-batches
                       inside the worker according to gbs/mbs.
            loss_fn:   Loss function from algorithms/loss/interfaces.py.
            eval_mode: When True, run forward only (no gradient update).
                       Used for validation loss logging.
            gbs:       Global batch size override.
            mbs:       Micro batch size override.
            timer:     Optional timer.

        Returns:
            Dict of scalar training metrics (loss, grad_norm, lr, etc.).
        """

    @abstractmethod
    def calibrate_qkv_fp8_scales(
        self,
        data:             BatchedDataDict[GenerationDatumSpec],
        micro_batch_size: Optional[int] = None,
        percentile:       float = 99.9,
        margin:           float = 1.05,
        include_q:        bool = False,
    ) -> dict[str, Any]:
        """Calibrate FP8 Q/K/V activation scales for KV-cache quantization.

        Args:
            data:             Calibration data (representative input batch).
            micro_batch_size: Micro-batch size for calibration forward pass.
            percentile:       Percentile for per-tensor amax estimation.
            margin:           Safety margin multiplier applied to amax.
            include_q:        Also compute scale for Q (K/V always included).

        Returns:
            Dict with overall config and per-layer float scales.
        """

    @abstractmethod
    def prepare_for_training(self, *args: Any, **kwargs: Any) -> None:
        """Pre-training hook (e.g. transition from inference to train mode)."""

    @abstractmethod
    def finish_training(self, *args: Any, **kwargs: Any) -> None:
        """Post-training hook (e.g. sync EMA weights, release grad buffers)."""

    @abstractmethod
    def save_checkpoint(self, *args: Any, **kwargs: Any) -> None:
        """Persist model state to disk."""

    @abstractmethod
    def shutdown(self) -> bool:
        """Release all resources held by this worker.

        Returns:
            True if shutdown completed cleanly.
        """

# Colocation / weight-refit coexistence contract
class ColocatablePolicyInterface:
    """Lifecycle contract for a policy that colocates with the generation
    backend on shared GPUs.

    A colocatable policy must hand the device back and forth between training
    and generation (prepare_for_*/finish_*), offload its optimizer and buffers
    around a weight refit (offload_before/after_refit), and drive the collective
    weight-sync transport that pushes updated weights to the inference fleet
    (init_collective / prepare_refit_info / broadcast_weights_for_collective).

    Non-abstract mix-in: the driver-side Policy handle and the trainer-side
    worker each override the subset they need via MRO; the default bodies raise
    NotImplementedError so an unsupported transition surfaces clearly. It is
    deliberately NOT derived from PolicyInterface — the two concrete implementors
    satisfy different subsets of that ABC, so inheriting its abstract methods
    here would make them uninstantiable.
    """

    # Train <-> generation device handoff
    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        raise NotImplementedError

    def prepare_for_training(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def prepare_for_lp_inference(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        raise NotImplementedError

    def finish_training(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def invalidate_kv_cache(self, *args: Any, **kwargs: Any) -> bool:
        raise NotImplementedError

    # Memory offload around a weight refit
    def offload_before_refit(self) -> None:
        raise NotImplementedError

    def offload_after_refit(self) -> None:
        raise NotImplementedError

    # Collective weight-sync transport (trainer -> inference fleet)
    def init_collective(
        self, ip: str, port: int, world_size: int, *, train_world_size: int
    ) -> list[ray.ObjectRef]:
        raise NotImplementedError

    def prepare_refit_info(
        self, state_dict_info: Optional[Any] = None
    ) -> Optional[dict[str, Any]]:
        raise NotImplementedError

    def broadcast_weights_for_collective(
        self, kv_scales: Optional[dict[str, float]] = None
    ) -> list[ray.ObjectRef]:
        raise NotImplementedError
