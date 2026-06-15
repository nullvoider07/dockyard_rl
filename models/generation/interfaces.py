"""Generation interfaces for Project Dockyard.

Defines the data specs (GenerationDatumSpec, GenerationOutputSpec),
the abstract GenerationInterface that VllmGeneration implements, and
the padding-verification helper used throughout the vLLM worker layer.

Design notes
------------
- Collocated inference is explicitly unsupported in Dockyard.  The
  ColocatedConfig TypedDict and the ``collocated`` field in
  GenerationConfig that appear in the upstream source have been removed.
  VllmGeneration is always non-collocated (dedicated inference fleet).
- The ``_pad_token_id`` field in GenerationConfig is populated by
  configure_generation_config() in models/generation/__init__.py before
  the config is passed to VllmGeneration.
"""

from abc import ABC, abstractmethod
from typing import Any, NotRequired, TypedDict, Union
import ray
import torch
from dockyard_rl.distributed.batched_data_dict import BatchedDataDict

# Padding verification
def verify_right_padding(
    data: Union[
        BatchedDataDict["GenerationDatumSpec"],
        BatchedDataDict["GenerationOutputSpec"],
    ],
    pad_value: int = 0,
    raise_error: bool = True,
) -> tuple[bool, Union[str, None]]:
    """Verify that a BatchedDataDict is right-padded as declared by its lengths.

    Accepts either GenerationDatumSpec (input_ids + input_lengths) or
    GenerationOutputSpec (output_ids + unpaddded_sequence_lengths).

    Args:
        data:        BatchedDataDict to verify.
        pad_value:   Expected padding token id (default: 0).
        raise_error: Raise ValueError on the first violation when True;
                     return (False, message) when False.

    Returns:
        (True, None) when padding is correct, (False, message) otherwise.
    """
    assert isinstance(data, BatchedDataDict), (
        f"data must be a BatchedDataDict, got type: {type(data)}"
    )
    assert pad_value is not None, (
        "pad_value must not be None.  Pass _pad_token_id from the config."
    )

    if "input_ids" in data and "input_lengths" in data:
        tensor  = data["input_ids"]
        lengths = data["input_lengths"]
    elif "output_ids" in data and "unpaddded_sequence_lengths" in data:
        tensor  = data["output_ids"]
        lengths = data["unpaddded_sequence_lengths"]
    elif "output_ids" in data and "unpaddded_sequence_lengths" not in data:
        # Tolerate the key name used in vllm_worker.py
        # (unpadded_sequence_lengths without double-d).
        tensor  = data["output_ids"]
        lengths = data.get("unpadded_sequence_lengths") or data.get(
            "unpaddded_sequence_lengths"
        )
        if lengths is None:
            msg = (
                f"Could not find unpaddded_sequence_lengths in output data. "
                f"Got keys: {list(data.keys())}"
            )
            if raise_error:
                raise ValueError(msg)
            return False, msg
    else:
        msg = (
            "Could not find the required field pairs. Expected either "
            "(input_ids, input_lengths) or "
            "(output_ids, unpaddded_sequence_lengths). "
            f"Got keys: {list(data.keys())}"
        )
        if raise_error:
            raise ValueError(msg)
        return False, msg

    if tensor.ndim != 2:
        msg = (
            f"Expected 2-D tensor for padding check, got shape {tensor.shape}"
        )
        if raise_error:
            raise ValueError(msg)
        return False, msg

    batch_size, seq_len = tensor.shape

    if lengths.shape[0] != batch_size:
        msg = (
            f"Batch-size mismatch: tensor={batch_size}, "
            f"lengths={lengths.shape[0]}"
        )
        if raise_error:
            raise ValueError(msg)
        return False, msg

    for i in range(batch_size):
        length = lengths[i].item()
        if length > seq_len:
            msg = (
                f"Length {length} at index {i} exceeds tensor seq-dim {seq_len}"
            )
            if raise_error:
                raise ValueError(msg)
            return False, msg
        if length < seq_len and not torch.all(
            tensor[i, length:] == pad_value
        ):
            non_pad = (
                torch.where(tensor[i, length:] != pad_value)[0] + length
            )
            msg = (
                f"Non-padding values after length at index {i}: "
                f"positions {non_pad.tolist()}"
            )
            if raise_error:
                raise ValueError(msg)
            return False, msg

    return True, None

# TypedDicts
class GenerationConfig(TypedDict):
    """Top-level configuration shared by all generation backends."""

    backend:         str
    max_new_tokens:  int
    temperature:     float
    top_p:           float
    top_k:           int | None
    model_name:      NotRequired[str]
    stop_token_ids:  list[int] | None
    stop_strings:    list[str] | None
    # Populated by configure_generation_config(), not by the caller.
    _pad_token_id:   NotRequired[int]

class StructuredSpec(TypedDict):
    """Per-sample constrained-decoding request (structured tool-use, fork 2).

    Threaded through GenerationDatumSpec into the per-sample sampling params so
    the vLLM worker can set ``SamplingParams.structured_outputs``. Built by the
    structured-tool-use protocol's prompt/registry layer (slice 7), never by the
    generation worker itself. When this key is absent or None the sampling params
    are byte-identical to the unconstrained path.

    Fields:
        structural_tag: JSON-serialized vLLM structural-tag payload (the
            ``{"structures": [...], "triggers": [...]}`` shape — see
            ``utils.build_structural_tag``). Assigned verbatim to
            ``StructuredOutputsParams(structural_tag=...)``.
        constrained_tools: Tool names the structural tag constrains. Carried for
            the rollout's structural-submask derivation and for diagnostics; not
            consumed by the engine.
    """

    structural_tag:    str
    constrained_tools: list[str]


class GenerationDatumSpec(TypedDict):
    """Input data required by generation models.

    Tokens are right-padded; input_lengths carries the unpadded lengths.
    Use verify_right_padding() to assert the invariant at worker boundaries.
    """

    input_ids:     torch.Tensor   # (batch, padded_seq_len)
    input_lengths: torch.Tensor   # (batch,)
    stop_strings:  NotRequired[list[str]]
    # Per-sample constrained-decoding spec (structured tool-use). Absent/None on
    # the free-generation path → sampling params byte-identical to today.
    structured_spec: NotRequired["StructuredSpec | None"]
    __extra__:     Any

class GenerationOutputSpec(TypedDict):
    """Output data returned by generation models.

    output_ids includes the full prompt+response sequence, right-padded.
    logprobs are zero for prompt positions; non-zero for generated positions.
    """

    output_ids:                  torch.Tensor   # (batch, padded_total_len)
    generation_lengths:          torch.Tensor   # (batch,) — response only
    unpadded_sequence_lengths:   torch.Tensor   # (batch,) — prompt + response
    logprobs:                    torch.Tensor   # (batch, padded_total_len)
    truncated:                   NotRequired[torch.Tensor]  # (batch,) bool
    # Per-token structural-scaffolding submask aligned to output_ids (1 where a
    # token is grammar-forced structural scaffolding, 0 otherwise), present only
    # when a structured_spec drove the request. The trainer intersects it with
    # the role loss-mask to exclude grammar-forced tokens from the GRPO ratio
    # (design fork 2). Only the <tool_call>/</tool_call> delimiter spans are
    # CPU-derivable; the exact per-token forced set inside the JSON is HV-gated
    # (vLLM 0.21 does not surface the matcher bitmask on CompletionOutput).
    structural_token_mask:       NotRequired[torch.Tensor]  # (batch, padded_total_len)
    __extra__:                   Any

# Abstract interface
class GenerationInterface(ABC):
    """Abstract base class for all generation backends."""

    @abstractmethod
    def init_collective(
        self,
        ip: str,
        port: int,
        world_size: int,
        *,
        train_world_size: int,
    ) -> list[ray.ObjectRef]:
        """Initialise the NCCL weight-sync communicator on inference workers."""

    @abstractmethod
    def generate(
        self,
        data: BatchedDataDict["GenerationDatumSpec"],
        greedy: bool,
    ) -> BatchedDataDict["GenerationOutputSpec"]:
        """Synchronous batch generation."""

    @abstractmethod
    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Pre-generation hook (e.g. reset KV cache)."""

    @abstractmethod
    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Post-generation hook (e.g. reset prefix cache)."""

    @property
    def requires_kv_scale_sync(self) -> bool:
        """True when KV cache scales must be synced after weight refit."""
        return False

    def prepare_refit_info(
        self,
        state_dict_info: dict[str, Any],
    ) -> None:
        """Send weight metadata to inference workers before broadcast."""
        raise NotImplementedError

    def update_weights_via_ipc_zmq(self) -> list[ray.ObjectRef]:
        """Update weights via CUDA IPC handles (collocated-only path)."""
        raise NotImplementedError

    def update_weights_from_collective(self) -> list[ray.ObjectRef]:
        """Update weights via NCCL collective (non-collocated path)."""
        raise NotImplementedError

    def invalidate_kv_cache(self) -> bool:
        """Invalidate vLLM prefix/KV cache after weight update."""
        return False

    def clear_logger_metrics(self) -> None:
        """Clear telemetry counters at the start of a training step."""

    def get_logger_metrics(self) -> dict[str, Any]:
        """Return telemetry metrics from vLLM workers."""
        return {}