"""EP-aware refit: native GroupedExperts -> per-expert HF weight expansion.

The native ``GroupedExperts`` (B.4) fuses an MoE layer's experts into stacked 3D
params (``w1_EFD`` ``[E,F,D]`` gate, ``w3_EFD`` ``[E,F,D]`` up, ``w2_EDF``
``[E,D,F]`` down), sharded ``Shard(0)`` over the EP axis. The inference backend
(vLLM / SGLang) loads the standard HF checkpoint layout — one tensor per expert,
named ``...mlp.experts.{i}.{gate,up,down}_proj.weight`` — and fuses internally.

Refit must therefore RESHARD across the trainer->inference boundary: gather the
EP-sharded stack to a full tensor, then unbind dim 0 into per-expert HF-named
tensors. This module is the pure name/shape/unbind logic; the EP gather itself
(``DTensor.full_tensor()``) is device-bound and lives in the v2 worker.

Expert ordering: ``full_tensor()`` over ``Shard(0)``-on-ep reconstructs experts
in global index order (the EP partition is contiguous — see ``mesh.py``
``experts_for_rank``), so ``stack[i]`` is global expert ``i``. This depends on
the EP-axis ordering tracked as HV-1.
"""

from __future__ import annotations

from typing import Callable, Iterable, Iterator, TypeVar

import torch

# Native fused-expert param suffix -> HF per-expert projection name. The SwiGLU
# convention matches Mixtral / Qwen3-MoE: w1=gate (silu), w3=up (multiply),
# w2=down (output).
_GROUPED_EXPERT_SUFFIXES: dict[str, str] = {
    "w1_EFD": "gate_proj",
    "w3_EFD": "up_proj",
    "w2_EDF": "down_proj",
}


def is_grouped_expert_key(name: str) -> bool:
    """Whether ``name`` is a native fused GroupedExperts param (w1/w2/w3_E**)."""
    return name.rsplit(".", 1)[-1] in _GROUPED_EXPERT_SUFFIXES


def expand_grouped_expert_key(name: str, expert_index: int) -> str:
    """Map a fused-expert key + expert index to its per-expert HF name.

    ``...mlp.experts.w1_EFD`` + ``i`` -> ``...mlp.experts.{i}.gate_proj.weight``.
    """
    prefix, suffix = name.rsplit(".", 1)
    proj = _GROUPED_EXPERT_SUFFIXES[suffix]
    return f"{prefix}.{expert_index}.{proj}.weight"


def expand_grouped_expert_info(
    name: str, shape: tuple[int, ...]
) -> list[tuple[str, tuple[int, ...]]]:
    """Expand fused-expert metadata into per-expert ``(name, shape)`` entries.

    ``shape`` is the global fused shape ``(E, *per_expert)``; the result has one
    entry per expert with the leading expert dim dropped.
    """
    num_experts = shape[0]
    per_expert_shape = tuple(shape[1:])
    return [
        (expand_grouped_expert_key(name, i), per_expert_shape)
        for i in range(num_experts)
    ]


T = TypeVar("T")


def iter_expanded_refit_tensors(
    items: Iterable[tuple[str, T]],
    materialize: Callable[[T], "object"],
) -> Iterator[tuple[str, object]]:
    """Flat-map a refit state-dict stream, expanding fused-expert params.

    For each ``(name, tensor)``:
      - fused-expert param -> ``materialize`` it (EP gather to a full tensor),
        then yield one ``(per_expert_name, stack[i])`` per expert;
      - any other param -> passed through unchanged (NOT materialized, so the
        caller's existing per-item materialization still applies).

    ``materialize`` performs the device-bound EP gather (``full_tensor()``); it
    is injected so this function stays pure/testable. The unbound per-expert
    tensors are views into the gathered stack.
    """
    for name, tensor in items:
        if is_grouped_expert_key(name):
            full = materialize(tensor)
            for i in range(full.shape[0]):  # type: ignore[attr-defined]
                yield expand_grouped_expert_key(name, i), full[i]  # type: ignore[index]
        else:
            yield name, tensor


# Router state the inference engine holds in fp32 regardless of the model dtype.
# The aux-loss-free correction bias is added to the router scores before top-k
# and steps by the load-balance coefficient (order 1e-3) per update, so casting
# it to a bf16 refit dtype quantizes it at roughly one step and the rollout
# engine can rank experts differently from the trainer that produced it. vLLM
# creates it as an fp32 nn.Parameter on every noaux_tc model and narrows it only
# on the ROCm AITER path (deepseek_v2.py:317, :395).
_FP32_REFIT_SUFFIXES: frozenset[str] = frozenset({"e_score_correction_bias"})

# Trainer-side router state with no inference-side counterpart. dockyard's
# MoEBlock registers expert_bias_E when load_balance_coeff is set; vLLM's
# Qwen3-MoE router -- the family apply_moe_surgery supports -- carries no
# correction-bias parameter at all, and AutoWeightsLoader raises on a name it
# cannot place (model_executor/models/utils.py:395).
_UNTRANSFERABLE_REFIT_SUFFIXES: dict[str, str] = {
    "expert_bias_E": (
        "policy.dtensor_cfg.moe_parallelizer.load_balance_coeff is set, which "
        "registers an aux-loss-free routing bias on every MoEBlock. The "
        "inference engine has no parameter to receive it: vLLM's Qwen3-MoE "
        "router carries no correction bias, so refit would abort inside vLLM's "
        "weight loader -- and dropping the key instead would let the rollout "
        "engine route without the bias the trainer routes with, which is the "
        "train/generation divergence the bias itself is meant to avoid. Set "
        "load_balance_coeff to null for runs that refit a generation backend."
    ),
}


def refit_tensor_dtype(
    name: str, tensor: torch.Tensor, default_dtype: torch.dtype
) -> torch.dtype:
    """Dtype to stream ``name`` in: ``default_dtype``, except fp32 router state.

    The single place the refit dtype is decided. Both the tensor-streaming
    transports and the ``prepare_refit_info`` metadata maps read it, because the
    IPC receiver slices its buffer using that metadata -- a name streamed at one
    width and declared at another is read at the wrong offset, not merely at the
    wrong precision.
    """
    if tensor.dtype != torch.float32:
        return default_dtype
    if name.rsplit(".", 1)[-1] in _FP32_REFIT_SUFFIXES:
        return tensor.dtype
    return default_dtype


def check_refit_names(names: Iterable[str]) -> None:
    """Refuse a refit stream carrying trainer-only state, naming the config key.

    Called where the refit metadata is prepared, so an unsupported combination
    fails before the first rollout rather than inside the inference engine's
    weight loader after one.
    """
    for name in names:
        reason = _UNTRANSFERABLE_REFIT_SUFFIXES.get(name.rsplit(".", 1)[-1])
        if reason is not None:
            raise ValueError(f"refit cannot transfer {name!r}: {reason}")
